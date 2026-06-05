# Slot-registry audit — 2026-06-05

Phase 1 of the memory-cluster fix plan (`MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md`)
adds a compile-time slot-conflict registry. This document lists every
existing op that legitimately shares a structural slot with another op
and therefore needs a `slot_share=(<kind>,)` annotation, plus the
slot-id derivation rationale.

## Registry surface

The registry is implemented in
`c4_release/neural_vm/unified_compiler/slot_registry.py`. It records
every `(layer_idx, slot_id)` claim derived from each placed op and
raises `SlotConflictError` when two ops claim the same slot at the same
layer without a `slot_share` opt-out.

### Slot ids

| Tuple                       | Meaning                                           |
|-----------------------------|---------------------------------------------------|
| `("ffn",)`                  | Whole `block.ffn` module replacement              |
| `("attn",)`                 | Whole `block.attn` module replacement / resize    |
| `("attn", "head", k)`       | Attention head index `k`                          |
| `("attn", "Wq" / "Wk" / "Wv" / "Wo")` | Per-matrix claim (non-head-aligned)     |
| `("post_ops", i)`           | Explicit `block.post_ops[i]` slot                 |
| `("post_ops_append",)`      | `.append(...)` / `.insert(0, ...)` post_op claim  |
| `("ffn_units", start, end)` | Per-unit-range FFN claim (`end` exclusive)        |

### Slot kinds (used by `slot_share`)

`"ffn"`, `"attn"`, `"attn_head"`, `"attn_matrix"`, `"post_ops"`,
`"post_ops_append"`, `"ffn_units"`.

### Derivation rules (`derive_slot_ids_for_op`)

1. Topology-anchor ops (`declarative_authority="topology_anchor"`)
   contribute no claims — their bake is a no-op by contract.
2. `produces["__module_replacement"]` sentinel:
   - `"L*.ffn[*]"` → `("ffn",)`
   - `"L*.attn[*]"` → `("attn",)`
   - `"L*.post_ops[*]"` → `("post_ops_append",)`
3. `compiler_ir.layers[i].attention.rules[j].spec.head_idx`
   → `("attn", "head", head_idx)`
4. `ffn_units_used` (when `kind` is `"ffn"` or `"block"`):
   → `("ffn_units", 0, ffn_units_used)`
5. Coarse fallback by `kind`:
   - `"ffn"` → `("ffn",)`
   - `"attn"` → `("attn",)`

The range claim is coarse — it always covers `[0, N)` rather than the
actual `[start, end)` the bake writes into. A future refinement could
read the per-op start unit from the layer-allocator layout table; for
now the gap is covered by `slot_share=("ffn_units",)` annotations on
ops that share FFN unit ranges with each other.

## Slot-share annotation audit (efficient + lookup mode)

The first compile under the registry surfaced six conflict groups.
Each was investigated; all six are legitimate sharing (the ops sequence
or co-bake a single composite module). The fix is `slot_share=(<kind>,)`
on every participant.

### Group 1: L2 FFN unit-range share

| Op | Range | Status |
|---|---|---|
| `layer2_mem_byte_flags` (`l2_ops.py:322`) | units 0..7 | needs `slot_share=("ffn_units",)` |
| `layer2_initial_pc_bake_cancel` (`l2_ops.py:504`) | units 8..9 | needs `slot_share=("ffn_units",)` |

Both ops bake non-overlapping unit ranges of `block[2].ffn`. The coarse
range derivation `[0, ffn_units_used)` overlaps; the actual ranges
(`[0, 8)` vs `[8, 10)`) are disjoint.

### Group 2: L8 FFN unit-range share

| Op | Range | Status |
|---|---|---|
| `layer8_multibyte_routing` (`l8_ops.py:1536`) | units 0..2054 | needs `slot_share=("ffn_units",)` |
| `layer8_sp_gathered_sentinel` (`l8_ops.py:2598`) | unit 2055 | needs `slot_share=("ffn_units",)` |

The sentinel op adds one trailing unit after the multibyte router.
Disjoint ranges, coarse derivation overlaps.

### Group 3: L14 FFN unit-range share

| Op | Range | Status |
|---|---|---|
| `layer14_alu_nocarry_ax_bytes_zero` (`l14_ops.py:2145`) | units 0..1873 | needs `slot_share=("ffn_units",)` |
| `layer14_demo_phase6_wave7` (`l14_ops.py:2299`) | unit 1874 | needs `slot_share=("ffn_units",)` |

`_L14_CLEANUP_CHAIN_LAYOUT` pins ALU-nocarry up to unit 1873; the demo
adds one more unit at 1874.

### Group 4: L11 `block.ffn[FlattenedALUMul]` co-assembly

Ten ops sequentially install stages of a single `FlattenedALUMul`
composite onto `block[11].ffn` via `_ensure_l11_mul_module` (idempotent
get-or-install). All ten carry `produces={'__module_replacement':
'L11.ffn[FlattenedALUMul]'}` which the derivation maps to `("ffn",)`.
All ten need `slot_share=("ffn",)`:

- `efficient_l11_alumul_wrap` (`alu_ops.py:812`)
- `l11_alu_mul_bdtoge` (`alu_ops.py:897`)
- `l11_alu_mul_schoolbook` (`alu_ops.py:938`)
- `l11_alu_mul_carrypass1` (`alu_ops.py:978`)
- `l11_alu_mul_carrypass2` (`alu_ops.py:1019`)
- `l11_alu_mul_carrypass3` (`alu_ops.py:1061`)
- `l12_alu_mul_genprop` (`alu_ops.py:1102`)
- `l12_alu_mul_binarylookahead` (`alu_ops.py:1143`)
- `l12_alu_mul_finalcorrection` (`alu_ops.py:1183`)
- `l12_alu_mul_getobd` (`alu_ops.py:1248`)

(The `l12_*` names are historical — they target the same L11 block.)

### Group 5: L10 `block.post_ops[FlattenedDivMod]` co-assembly

Five ops sequentially install stages of the divmod composite into
`block[10].post_ops`. Each carries either a
`'L10.post_ops[FlattenedDivMod]'` or `'L10.post_ops[+6 structural FFNs]'`
sentinel (both map to `("post_ops_append",)`). All need
`slot_share=("post_ops_append",)`:

- `l10_post_op_attach` (`l10_ops.py:6912`)
- `l10_alu_divmod_bdtoge` (`alu_ops.py:1362`)
- `l10_alu_divmod_longdiv` (`alu_ops.py:1400`)
- `l10_alu_divmod_getobd` (`alu_ops.py:1468`)
- `l10_alu_divmod_install` (`alu_ops.py:1603`)

### Group 6: L15 `block.attn` resize + bake

| Op | Role | Status |
|---|---|---|
| `layer15_memory_lookup` (`l15_ops.py:594`) | head bake | needs `slot_share=("attn",)` |
| `l15_attention_resize` (`l15_ops.py:1867`) | structural resize after bake | needs `slot_share=("attn",)` |

The resize op runs `requires={"after": "layer15_nibble_copy"}` and
replaces `block[15].attn.W_q/W_k/W_v/W_o` wholesale to change
`num_heads`. The memory lookup re-bakes against the resized module.
Legitimate post-bake structural mutation, not a silent overwrite.

## Summary

- 18 ops gained a `slot_share` annotation across `alu_ops.py`,
  `l2_ops.py`, `l8_ops.py`, `l10_ops.py`, `l14_ops.py`, `l15_ops.py`.
- Zero new conflicts remain on `alu_mode=efficient` or `alu_mode=lookup`.
- Zero op was found to be a real silent-overwrite bug (V4 was the
  hypothesized one; the L13 head-0 contest only surfaces under the
  V4-style "pin layer13_mem_addr_gather to L13 + leave layer10_carry_relay
  at L13" placement, which the production schedule does not adopt).

## V4 retry verification

Manually claiming `(13, ("attn", "head", 0))` for both
`layer10_carry_relay_bake` and `layer13_mem_addr_gather` produces the
exact error the V4 attempt was missing:

```
Slot-conflict registry detected unauthorized conflicts:
  layer=13 slot=('attn', 'head', 0):
    'layer10_carry_relay_bake' (kind='block') vs
    'layer13_mem_addr_gather' (kind='block')
Resolve by (a) routing one op to a different slot, or
(b) adding slot_share=(<kind>,) on the Operation when the shared claim
is legitimate (see docs/SLOT_REGISTRY_AUDIT_2026_06_05.md).
```

Phases 2-5 of the memory cluster fix plan can now proceed with the
silent-contest class of bug closed.
