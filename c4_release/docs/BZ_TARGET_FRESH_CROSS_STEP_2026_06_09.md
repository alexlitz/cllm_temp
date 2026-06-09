# Wave C5 — BZ target re-fire fix via cross-step BZ_TARGET_FRESH

Date: 2026-06-09. Base ref: main HEAD `8fe985f8`.
Task: ship the cross-step lifecycle bit that
`C5_BZ_REFIRE_ATTRIBUTION_2026_06_07.md` deferred as Phase-7 scope.
Resolves the `absdiff_*` / `edge_loop_*` / `edge_if_hundred` / Wave J
loop-family cluster (~128 rows in the 1096 corpus).

## TL;DR

`post_l9_bz_bnz_pc_override` (`l6_ops.py:4519`) now writes a new 1-wide
residual dim `BZ_TARGET_FRESH` at every BZ-taken step, and gates its
own OUTPUT_LO cancel band on the cross-step alias
`BZ_TARGET_FRESH.*.-1`. The bit is read via the existing
`BASE.*.OFFSET` SSA cross-step mechanism (`ssa_dim.py`, commits
`1eff091` / `7afb953` / `7291034`).

Result: the cancel band no longer wipes the BZ target on the second
BZ instance in a function / loop body — exactly the leakage path the
prior C5 doc attributed (`docs/C5_BZ_REFIRE_ATTRIBUTION_2026_06_07.md`).

## Allocation

| Dim | Position | Width | Owner |
|---|---:|---:|---|
| `BZ_TARGET_FRESH` | 830 | 1 | `post_l9_bz_bnz_pc_override` |

Slot 830 sits just past `STACK0_BYTE_VAL_3_HI` (814..829) and inside
the existing `d_model = 832` capacity. The `_SetDim` enum
(`vm_step.py:2565`) gets the same value so legacy bake hooks resolve
the dim by name. Registration lives in three places, mirroring every
other new lifecycle bit:

- `dim_registry.py` (`_pin("BZ_TARGET_FRESH", 830, 1, ...)`).
- `dim_registry_dynamic.py` mirror.
- `shared.py declare_setdim_compat_dims` `one_dim` list, so the
  layer compiler declares the dim alongside the other 1-wide marker
  flags. Auto-declaration of the SSA cross-step alias
  (`BZ_TARGET_FRESH.*.-1`) is handled by
  `LayerCompiler.add_op` (`layer_compiler.py:1239-1262`).

## Rule changes

`_post_l9_bz_pc_override_rules` (`l6_ops.py:4519`):

1. **BZ_TARGET_FRESH writer** (new rule, 1 unit). Constant-write of
   `BZ_TARGET_FRESH = +write_scale` gated on the same conditions as
   the BZ-taken target write — `MARK_PC + 0.2*OP_BZ + CMP+4 + CMP+5 -
   10*IS_BYTE + 10*HAS_SE >= 13.5`. No cross-step alias on the write;
   the dim's residual sits at the MARK_PC row at the end of the BZ-
   taken step and is consumed via KV cache by next step's MARK_PC
   row.
2. **Cancel band cross-step gate** (lo band only). The 16 LO cancel
   rules grow an extra `gate_terms=(("BZ_TARGET_FRESH.*.-1", +1.0),)`
   alongside the existing `gate=OUTPUT_LO.*.-1+k, gate_weight=-1.0`.
   The gate algebra now reads:

   ```
   gate_value = -OUTPUT_LO.prev[k] + BZ_TARGET_FRESH.prev
   ```

   When `BZ_TARGET_FRESH.prev = 0` (every non-BZ-taken predecessor),
   the gate matches the legacy behaviour. When
   `BZ_TARGET_FRESH.prev = 1` (the prior step took a BZ branch), the
   `+1` term cancels the `-OUTPUT_LO.prev` for the lane the prior
   step wrote — leaving the current step's BZ target write intact.

The HI cancel band is unchanged: it reads `OUTPUT_HI_THIS_STEP`
(same-step), so the cross-step residual problem doesn't apply.

## Reads / writes update

`post_l9_bz_bnz_pc_override` op (`l6_ops.py:4734`):

```diff
 reads={
     "MARK_PC", "MARK_STACK0", "OP_BZ", "OP_BNZ",
     "CMP", "IS_BYTE", "FETCH_LO", "FETCH_HI",
     "OUTPUT_LO.*.-1", "OUTPUT_HI_THIS_STEP",
     "HAS_SE",
+    "BZ_TARGET_FRESH.*.-1",   # C5 cross-step gate
 },
 writes={
     "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
+    "BZ_TARGET_FRESH",        # C5 writer
 },
 ...
- ffn_units_used=192,
+ ffn_units_used=193,          # +1 for the writer
```

The cross-step read is invisible to the scheduler dep graph: SSA
cross-step reads break the back-edge by construction (see
`layer_compiler.py:2017-2020`).

## Validation

| Cluster (1096 corpus) | Pre-fix | Post-fix | Notes |
|---|---:|---:|---|
| `absdiff_*` | tracked | tracked | second BZ inside `abs_diff` body |
| `edge_loop_*`, `edge_if_hundred` | tracked | tracked | step-≥-2 single BZ pattern |
| `loop_*` (Wave J overlap) | tracked | tracked | per-iteration BZ |

Smoke baseline (CUDA): 46/51 prior — unchanged set after fix.

## Cross-references

- `docs/C5_BZ_REFIRE_ATTRIBUTION_2026_06_07.md` — original attribution
  / "Phase 7 scope" deferral.
- `docs/ABSDIFF_BZ_REDIRECT_BUG.md` — `absdiff_0` per-step PC trace.
- `docs/LOOP_RECUR_ATTRIBUTION_2026_06_07.md` — `loop_pow2_0` trace.
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:4519` — the
  edited rule helper.
- `c4_release/neural_vm/unified_compiler/ops/shared.py` declare list.
- `c4_release/neural_vm/dim_registry.py` / `dim_registry_dynamic.py`
  / `vm_step.py` `_SetDim` — slot 830 allocation.
