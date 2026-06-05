# Memory cluster fix Phase 2 — BLOCKER (2026-06-05)

Per `docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` §"Phase 2", the goal
was to decouple `layer10_carry_relay` from the future L13.attn.head_0
collision with `layer13_mem_addr_gather` so that Phase 4 can pin
`_layer13_attn_dep_anchor` to physical L13 and land mem_addr_gather
on block[13].attn without silently overwriting carry-relay's head_0
weights.

Both options described in the plan were attempted (no compile + smoke
budget left to attempt a third). Both regressed smoke beyond the
"smoke ≥ 46/52 baseline" guard, so both were reverted with no edits
committed beyond this doc.

## Smoke baseline

```
$ PYTHONPATH=. python -m pytest c4_release/tests/test_smoke.py --no-header -q --tb=no
45 passed, 6 failed, 1 deselected
```

Failures: `test_lea_basic` + 5 memory tests
(`test_si_li_{roundtrip,multiple_stores,overwrite,16bit_value}`,
`test_sc_lc_roundtrip`). Baseline = 45/51 = 88%.

## Option A — pin `layer10_carry_relay` to a non-L13 layer

The brief suggested `layer_idx=10` ("its name's original intent").
Direct attempt:

```python
# l10_ops.py:make_layer10_carry_relay_op
layer_idx=10,
requires={"after": "layer9_marker_suppress"},
```

**Result: compile fails with SlotConflictError before smoke even
runs.** The slot-conflict registry (Phase 1's
`slot_registry.py`) catches:

```
layer=10 slot=('ffn_units', 0, 3405): 'layer9_alu' (kind='block') vs
                                       'layer10_alu' (kind='block')
```

**Root cause**: `layer9_alu` block op resolves to physical L10 (via its
own dep chain) and owns L10.ffn for 3405 units. `layer10_alu` block op
binds via `target_op_name="layer10_carry_relay"` — so when carry_relay
moves to L10, layer10_alu also moves to L10, wanting 1846 units of
L10.ffn. Sum 5251 > 4096 unit budget, and the two `block.ffn` slots
overlap regardless of the budget. Phase 1's registry refuses.

Same failure mode would apply to `layer_idx=11` (the cascade also
pulls layer10_alu's 1846 units to L11, where `_layer11_ffn_dep_anchor`
sits with no current FFN claim, but the L11 attn slot already has
`layer10_psh_stack0_passthrough` — that's a different conflict).

### A variant — pin to `layer_idx=12`

L12 in baseline is structurally empty (only the attn anchor
`layer10_stack0_byte_relay` is there; no block ops resolve to L12).
Trying `layer_idx=12`:

* Compile passes — the L10 family migrates cleanly: all 6 L10 `_bake`
  block ops move to block[12], `layer10_alu` ffn moves to L12.ffn,
  the rest of the cascade re-flows (L11 mul at L13, L12 mul_combine
  at L14, mem_addr_gather at L15). Heads 0/1/2 of block[13].attn are
  now free — Phase 4's goal is structurally achievable.
* **Smoke: 21/51 pass (30 fail).** Regression of 24 tests from
  baseline. Every AX-write test (`add_*`, `sub_*`, bitwise, MUL, SHL,
  SHR), the 16-bit cascade, comparison-truthy, lea_basic, and the
  memory cluster all regress.

The regression is the same "AX-write residual chain collapsed" mode
V3/V4 documented as Blocker 4 in
`SMOKE_MEMORY_FIX_ATTEMPT_V4_20260605.md` §"Why B2 didn't help" — the
L10 carry/byte/sp/stack0 heads must fire at the right physical block
to populate AX_CARRY_LO/HI before downstream readers (L11+ mul, L13
shifts, L14 cleanups). Moving them one layer earlier (L13 → L12) is
not byte-identical even though no slot-conflict registry warning
fires.

`requires["after"]` declarations do not capture this — the legacy
ordering relied on the dep-graph slot-increment landing the L10 family
at L13 to satisfy the implicit "after L11 ffn anchor's downstream
ALU_LO write" ordering. Pin L10 family earlier and that ordering
breaks silently (within the dep graph's rules; the failure surfaces
only at runtime AX-byte residual mismatch).

Reverted: `git checkout -- c4_release/neural_vm/unified_compiler/ops/l10_ops.py`.

## Option B — different head index for carry_relay (or mem_addr_gather)

The brief suggested "head 1 or 4, whichever is free" for carry_relay.
**This option is geometrically infeasible without architectural
change** — the L10 attention family already occupies all 8 head slots
at whatever layer it lands:

```
L10 family at block[X].attn heads:
  head_0  layer10_carry_relay_bake          (ADD/SUB byte carry)
  head_1  layer10_byte_passthrough_bake     (AX byte passthrough)
  head_2  layer10_sp_byte_passthrough_bake  (SP byte passthrough)
  head_3  layer10_psh_stack0_passthrough_bake (PSH STACK0)
  head_4  layer10_stack0_byte_relay_bake    (bitwise stack-byte relay)
  head_5  layer10_stack0_byte_relay_bake    (non-bitwise stack-byte relay)
  head_6  layer10_stack0_byte_relay_bake    (STACK0 persistence)
  head_7  layer10_bp_byte_passthrough_bake  (BP byte passthrough)
```

`layer13_mem_addr_gather` uses heads 0, 1, 2 (per
`_L13_HEAD_LAYOUT`). Phase 4 lands it on block[13].attn. The L10
family is also on block[13].attn (in current baseline). The
3 mem_addr_gather heads (0,1,2) collide with the corresponding L10
heads — there is no free slot at heads 3-7 for renumbering, because
the L10 family covers them all.

So "different head_idx for carry_relay" doesn't help: it would only
free head_0; heads 1 and 2 would still collide with byte_passthrough
and sp_byte_passthrough. The fix would require renumbering carry_relay
+ byte_passthrough + sp_byte_passthrough — but with only 8 head slots
on the layer and 8 L10 ops claiming them, there is no admissible
renumbering that frees heads 0-2 simultaneously.

The only Option B variant that could work would be to move
mem_addr_gather (not carry_relay) onto heads 3, 4, 5 — but those are
owned by `layer10_psh_stack0_passthrough_bake` (head 3) and
`layer10_stack0_byte_relay_bake` (heads 4, 5, 6). Same problem in the
mirror direction.

## Why Phase 2 needs Phase 3 first

The Phase 2 brief assumed L13.attn was *primarily* a head_0 contest
between carry_relay and mem_addr_gather. **Reality**: L13.attn is fully
owned by the L10 family (all 8 heads, plus the L10 FFN at L13.ffn).
Moving carry_relay alone (Option B) does not solve the contest; the
other 7 L10 attn bakes follow carry_relay via `target_op_name` and
remain at L13.

Moving the entire L10 family off block[13] (Option A) is what's
structurally needed — but the current L10 family co-binds attn AND
FFN bakes through one anchor (`layer10_carry_relay`). Pinning the
anchor moves *all* L10 bakes, including the 1846-unit `layer10_alu`
FFN, which then competes for FFN budget at the pinned layer.

The clean fix is **Phase 3**: split `layer10_carry_relay` into
separate attn and FFN anchors so the attn family can migrate
independently of the FFN family. Then a future Phase 2 retry can pin
the attn anchor to L12 (or another empty-attn layer) without dragging
the FFN at 1846 units alongside.

Specifically:

1. Add a new attn-only L10 anchor `_layer10_attn_anchor` pinned to a
   non-L13 attn-free layer (e.g., L12 — currently hosts only a single
   noop anchor `layer10_stack0_byte_relay`, plus the cascade dependent
   `_layer12_ffn_dep_anchor`).
2. Repoint the 6 attn-bake ops (`layer10_carry_relay_bake`,
   `layer10_byte_passthrough_bake`, `layer10_sp_byte_passthrough_bake`,
   `layer10_psh_stack0_passthrough_bake`,
   `layer10_stack0_byte_relay_bake`,
   `layer10_bp_byte_passthrough_bake`) to `target_op_name=
   _layer10_attn_anchor`.
3. Keep `layer10_alu` and the other L10 FFN bakes targeting
   `layer10_carry_relay` (which now anchors only FFN scheduling).
4. Verify the dep-graph cascade still places L11 mul / L12 mul_combine
   / L13 mem_addr_gather correctly under Phase 4's pin.

That structural change is the missing prerequisite. **Phase 2 cannot
land without it.**

## Recommendation

Skip Phase 2 in its current form. Reorder the plan so Phase 3 (anchor
split) lands first; then a revised Phase 2 (attn-only anchor pin to a
non-L13 layer) can land smoke-neutral, after which Phase 4's L13 pin
finds clean head slots for mem_addr_gather.

If Phase 3's anchor split is too large to attempt directly,
investigate whether the L10 family's "must run at one layer" assumption
can be partially relaxed — e.g., the head_6 STACK0 persistence head
is the load-bearing one for downstream MUL; the other 7 heads may be
relocatable independently. That investigation needs the verifier
substrate (`decl_verifier.py`) per the project's
`feedback_single_rule_fixes_are_zero_sum.md` constraint, not
single-rule patches.

## Budget consumed

* 1 baseline smoke run (45/51 pass, 6 fail) — confirmed baseline.
* 1 compile probe for Option A `layer_idx=10` — SlotConflictError
  (no smoke needed).
* 1 compile + 1 smoke for Option A variant `layer_idx=12` (21/51 pass,
  30 fail) — REVERTED.
* 0 compiles for Option B — geometric infeasibility ruled it out
  before code change.

## Files referenced

* `c4_release/docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` — Phase 2
  description.
* `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V4_20260605.md` — V4's
  head_0 contest diagnosis (motivated Phase 2 brief).
* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1917-1956` —
  `layer10_carry_relay` anchor (the Phase 2 edit target).
* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:37-47` —
  `_L10_HEAD_LAYOUT` (all 8 heads consumed by L10 family).
* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:31-36` —
  `_L13_HEAD_LAYOUT` (mem_addr_gather heads 0/1/2).
* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:394-459` —
  `_layer13_attn_dep_anchor` (Phase 4's pin target).
* `c4_release/neural_vm/unified_compiler/slot_registry.py` — Phase 1
  registry that surfaced Option A's FFN collision.
* `c4_release/neural_vm/unified_compiler/layer_compiler.py:1997-2134` —
  `_assign_layers` (the layer-pin enforcement logic).
