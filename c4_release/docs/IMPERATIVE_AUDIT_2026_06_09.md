# Imperative Weight-Write Audit (2026-06-09)

Audit of all remaining imperative attention/FFN weight writes in
`c4_release/neural_vm/`. Patterns matched:

- `attn.W_q.data[...] = ...` / `attn.W_q[...] = ...` (and `_k`, `_v`, `_o`)
- `ffn.W_up[...] = ...` / `ffn.W_gate[...]` / `ffn.W_down[...]`
- `ffn.b_up[...] = ...` / `ffn.b_gate[...]`

Counts are raw `grep` line counts (so multi-line lvalues such as
`attn.W_k.data[\n    base + 40,\n    ...\n] = X` count as one line).
"Live" means the helper is referenced from the compile path (factories
under `unified_compiler/ops/`, the dispatcher, or the
declarative_bake_fn closure). "Dead" means only tests or docs reference
the helper.

## Summary by file (sorted desc)

| Count | File | Live? | Suggested priority |
|------:|------|:-----:|--------------------|
| 2133 | `neural_vm/vm_step.py` | mixed | DEFER — legacy bake hooks; track per-helper |
| 735 | `neural_vm/unified_compiler/compiler.py` | live | low — orchestration layer; `attn.W_*` are dispatch, not bake |
| 229 | `neural_vm/unified_compiler/ops/l15_ops.py` | live | **HIGH** — `memory_lookup` runtime fragments + `lev` heads |
| 167 | `neural_vm/setup_helpers_l10.py` | live | HIGH — multiple `_set_layer10_*` still called from L10 ops |
| 158 | `neural_vm/unified_compiler/ops/l14_ops.py` | live | (this commit drops ~120 → ~38 by migrating override) |
| 102 | `neural_vm/setup_helpers_l5.py` | mostly dead | low — most cut by Phase 7.C |
| 102 | `neural_vm/setup_helpers_l14.py` | live | medium — chain helpers wired into L14 cleanup factories |
| 79 | `neural_vm/unified_compiler/ops/l8_ops.py` | live | medium — `sp_gather` + multibyte_fetch fragments |
| 77 | `neural_vm/unified_compiler/primitives.py` | live | n/a — the lowering primitives themselves |
| 68 | `neural_vm/alu/ops/bitwise.py` | live | medium — ALU bake primitives |
| 62 | `neural_vm/setup_helpers_l6.py` | mostly dead | low |
| 51 | `neural_vm/setup_helpers_l2.py` | live | medium — `_set_layer2_mem_byte_flags` chain |
| 34 | `neural_vm/setup_helpers_l13.py` | live | medium — addr_gather |
| 32 | `neural_vm/alu/ops/shift.py` | live | medium |
| 30 | `neural_vm/setup_helpers_l1.py` | live | medium |
| 28 | `neural_vm/alu_optimized.py` | live | medium |
| 27 | `neural_vm/alu/ops/mul_efficient.py` | live | medium |
| 24 | `neural_vm/setup_helpers_l9.py` | live | medium |
| 17 | `neural_vm/alu_integration.py` | live | low |
| 15 | `neural_vm/setup_helpers_l3.py` | live | low |
| 14 | `neural_vm/unified_compiler/ops/l9_ops.py` | live | medium — `_set_pop_low8_collapse` style |
| 14 | `neural_vm/alu/ops/div.py` | live | medium |
| 13 | `neural_vm/unified_compiler/ops/l4_ops.py` | live | low |
| 12 | `neural_vm/efficient_integration.py` | dead | drop |
| 12 | `neural_vm/alu/ops/cmp.py` | live | low |
| 8 | `neural_vm/setup_helpers_l15.py` | dead | drop |
| 7 | `neural_vm/unified_compiler/ops/l11_ops.py` | live | low |
| 7 | `neural_vm/setup_helpers_l4.py` | live | low |
| 7 | `neural_vm/setup_helpers_l12.py` | live | low |
| 7 | `neural_vm/setup_helpers_l11.py` | live | low |
| 6 | `neural_vm/alu/ops/mul.py` | live | low |
| 6 | `neural_vm/alu/ops/mul_swiglu.py` | live | low |
| 6 | `neural_vm/alu/ops/divmod_longdiv.py` | live | low |
| 5 | `neural_vm/setup_helpers_l7.py` | live | low |
| 4 | `neural_vm/alu/ops/floor_nibble_extract.py` | live | low |
| 3 | `neural_vm/unified_compiler/ops/l6_ops.py` | live | low |
| 2 | `neural_vm/contracts.py` | live | n/a — schema |

## Notes

- `vm_step.py` (2133 writes) is dominated by 22 `_set_layerN_*` helpers.
  Most are now only kept for test compatibility — the compile path uses
  `migrated=True` ops with `compiler_ir_factory=`. Migration target:
  shrink toward zero by cutting helpers as their tests migrate to the
  declarative IR comparator (`compare_symbolic_to_lowered_attn`).
- `compiler.py` (735) is mostly module wiring (`attn.W_q` references via
  `getattr`/`setattr`), not direct bakes. Excluded from real migration
  priority.
- `l15_ops.py` (229) is the next highest-impact migration target after
  this commit lands. `memory_lookup_lev_heads_4_11` and the
  `_set_layer15_*` post-bake patches both inline imperative writes.
- `l14_ops.py` migration (this commit): drops
  `_clear_l14_mem_generation_overbroad_sp_suppression` from the bake
  path. The helper remains importable for the four
  `test_l14_output_cleanup.py` tests that call it directly after the
  legacy `_set_layer14_mem_generation` helper.

## Methodology

```bash
grep -rn "\.W_q\.data\[\|\.W_k\.data\[\|\.W_v\.data\[\|\.W_o\.data\[" \
    c4_release/neural_vm/ | grep -v test
grep -rn "attn\.W_q\[\|attn\.W_k\[\|attn\.W_v\[\|attn\.W_o\[" \
    c4_release/neural_vm/ | grep -v test
grep -rn "ffn\.W_[a-z]*\[\|ffn\.b_[a-z]*\[" \
    c4_release/neural_vm/ | grep -v test
```

Live vs dead was determined by `grep -rn "<helper_name>\b"` against
`c4_release/` excluding `vm_step.py` and `setup_helpers*.py` self-refs.
