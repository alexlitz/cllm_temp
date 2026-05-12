# BD = _SetDim Hardcode Audit (Phase 1 Agent C)

Per `ARCH_LEAKAGE_FIX_PLAN.md` Phase 1 Agent C. Read-only enumeration of all
remaining `BD = _SetDim` and `_SetDim.<NAME>` hardcoded references in bake
functions / runtime modules.

Two reference fixes establish the threading pattern:

- A1 commit `5fc519d` — `BinaryOpByteZeroingPostOp` (IMM 240/255 corruption fix).
- P2 commit `619a14d` — `CarryPropagationPostOp`, `BitwiseBytePropagationPostOp`,
  `ComparisonCombine` (introduces `_resolve_bd(dim_positions)` helper).

Both threaded `dim_positions=None` through the PureFFN constructor, stashed it
via `object.__setattr__(self, '_pending_dim_positions', dim_positions)` before
calling `super().__init__()`, then resolved BD inside `_bake_weights` via
`_resolve_bd(getattr(self, '_pending_dim_positions', None))` which falls back to
`_SetDim` when None.

The proxy implementation is `c4_release/neural_vm/unified_compiler/ops/shared.py::_as_setdim_proxy`.

## Collision shape

Legacy `_SetDim` positions (sample) vs compact `pin_io_only=True` (`d_model=728`):

| Dim         | Legacy | Compact |
| ----------- | ------ | ------- |
| ALU_LO      | 360    | 330     |
| ALU_HI      | 376    | 346     |
| AX_CARRY_LO | 328    | 362     |
| AX_CARRY_HI | 344    | 378     |
| OP_MUL      | 289    | 210     |
| OUTPUT_LO   | 174    | 69      |
| OUTPUT_HI   | 190    | 85      |
| MARK_AX     | 1      | 1       |

A `_bake_weights` that pins to legacy positions writes / reads on the wrong
columns under compact (and may *silently coincide* with an unrelated dim) —
the exact failure mode that produced the IMM 240/255 / 0xFA bug in A1's
commit message.

## Inventory

Grep used (re-runnable):

```
grep -rn "BD = _SetDim\|BD: type = _SetDim\|bd = _SetDim\|from .* import _SetDim\|_SetDim\." \
  c4_release/neural_vm/ c4_release/neural_vm/unified_compiler/ \
  | grep -v ".pyc"
```

`c4_release/neural_vm/vm_step.py.current` is a tracked snapshot file (not a
`.py` module, never imported); excluded from the priority counts below.
`c4_release/neural_vm/DEBUGGING_QUICK_REFERENCE.md` is documentation; excluded.

### Already threaded (A1 + P2)

| File | Lines | Class | Notes |
| ---- | ----- | ----- | ----- |
| `c4_release/neural_vm/vm_step.py` | 489-687 | `ComparisonCombine` | `_resolve_bd(self._pending_dim_positions)` (P2 619a14d). |
| `c4_release/neural_vm/vm_step.py` | 613-688 | `BinaryOpByteZeroingPostOp` | A1 5fc519d. Refactored to `_resolve_bd` in P2 619a14d. |
| `c4_release/neural_vm/vm_step.py` | 689-810 | `CarryPropagationPostOp` | P2 619a14d. |
| `c4_release/neural_vm/vm_step.py` | 811-883 | `BitwiseBytePropagationPostOp` | P2 619a14d. |

### HIGH priority (production PureFFN / nn.Module; runs every forward)

#### H1: `_ensure_l11_mul_module` (lookup-mode MUL pipeline)

- File: `c4_release/neural_vm/unified_compiler/ops/shared.py`
- Lines: 157-170 (definition).
- Callers: `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:371,394,417,441,466,490,514,537,562` (9 phase-ordered install ops at phases 11.0-12.3).
- Signature: `def _ensure_l11_mul_module(block, S):` — does **not** accept `dim_positions`.
- Body:
  ```python
  from ...vm_step import _SetDim
  ...
  module = FlattenedALUMul(S, _SetDim)
  ```
- Dims affected: every dim read/written by `FlattenedALUMul` substages
  (`BDToGEConverter`, `_BDToGEStage`, `_MulCombineStage`, `_GEToBDStage`):
  `ALU_LO`, `ALU_HI`, `AX_CARRY_LO`, `AX_CARRY_HI`, `OP_MUL`, `OP_ADD`,
  `OP_SUB`, `OP_OR`, `OP_XOR`, `OP_AND`, `OP_SHL`, `OP_SHR`, `OP_DIV`,
  `OP_MOD`, `OUTPUT_LO`, `OUTPUT_HI`, `MARK_AX`. See
  `c4_release/neural_vm/efficient_alu_neural.py:36-72` (BDToGEConverter).
- Layout collision: legacy `ALU_LO=360` vs compact `ALU_LO=330`. Compact
  `OP_MUL=210` lies inside the OP-flag block legacy `_SetDim` reserves at
  262-296, so an MUL-baked weight at legacy index 289 writes to a dim that
  has unrelated semantics in the compact layout (and vice-versa).
- Fix pattern: change signature to
  `_ensure_l11_mul_module(block, S, dim_positions=None)`, change body to
  `BD = _as_setdim_proxy(dim_positions) if dim_positions is not None else _SetDim`,
  and update all 9 alu_ops.py callers to pass `dim_positions=dim_positions`.
  Symmetric to the existing `_make_alu_postop_attach_op` (same file,
  lines 102-153) which already uses the proxy.
- Priority: **HIGH** — runs on every step that exercises MUL. Currently masked
  in CI because `quick_runner` uses `trust_neural_alu=True` -> `alu_mode='efficient'`
  which goes through `_make_alu_postop_attach_op` (already proxied). The
  `alu_mode='lookup'` path that hits this site is hit by `test_runtime_vanilla.py
  ::test_compile_full_vm_default_mode_audit` and any non-trust caller, and
  `test_mul_basic` currently times out (orthogonal Phase 2 issue, but this
  hardcode would mask any progress).

### MEDIUM priority (compile-time bake helpers)

All `_set_layer*` helpers in `c4_release/neural_vm/vm_step.py` and
`c4_release/neural_vm/setup_helpers.py` take `BD` as a parameter — already
properly threaded by the compiler op call sites (`_as_setdim_proxy(dim_positions)`).
The remaining hardcoded BD = _SetDim assignments in those files are either
unreachable (dead callers) or fallbacks behind an explicit `BD=None` guard.
They are listed for completeness because Agent C should not change them, but
they should be deleted (or downgraded to a `_resolve_bd` call) when their
dead-caller status is confirmed by a follow-up.

#### M1: `_set_carry_forward_attn` (dead — no callers) — ✅ DELETED

- File: `c4_release/neural_vm/vm_step.py:2519-2566` (now removed).
- Line 2537: `BD = _SetDim` (hardcoded, no `BD` parameter).
- Callers: **none** in the production graph. The only live caller was the
  byte-equivalence test `c4_release/tests/test_primitives_l3_carry_equivalence.py`
  which existed *specifically* to pin the legacy helper against the new
  `Primitives.carry_forward_attention`. Production wiring goes through
  `c4_release/neural_vm/unified_compiler/primitives.py::Primitives.carry_forward_attention`
  (takes `bd=None`).
- Resolution (cleanup-bd-setdim-medium): deleted the function from
  `vm_step.py` and the now-purposeless equivalence test. Replaced with a
  brief sentinel comment in `vm_step.py` so future re-introduction is
  immediately visible in code review. Doc references in `l3_ops.py` and
  `primitives.py` updated to call out the deletion rather than point at
  the old line numbers.

#### M2: `_set_cs_threshold_attn` (dead — no callers) — ✅ DELETED

- File: `c4_release/neural_vm/setup_helpers.py:196-210` (now removed).
- Line 203: `BD = _SetDim` (hardcoded, no `BD` parameter).
- Callers: **none** (only an import-only re-export at `vm_step.py:2329`; no
  `_set_cs_threshold_attn(...)` call site existed in the active source).
- Resolution (cleanup-bd-setdim-medium): deleted the function from
  `setup_helpers.py` and removed the re-export from `vm_step.py`. A short
  sentinel comment is left in place so re-introduction is visible in
  review.

#### M3: `_set_stack0_carry_attn` BD=None fallback — ✅ HARDENED

- File: `c4_release/neural_vm/setup_helpers.py:214-260`.
- Lines 227-229: previously `if BD is None: from .vm_step import _SetDim; BD = _SetDim`.
- Callers: `c4_release/neural_vm/unified_compiler/ops/l3_ops.py:93`
  (always passes `BD=proxy`). The fallback was reachable only from legacy
  hand-set callers that no longer exist.
- Resolution (cleanup-bd-setdim-medium): signature changed to require `BD`
  (still accepted as the 4th positional / kwarg, defaulting to `None` for
  the explicit error). The implicit `_SetDim` fallback is replaced with a
  `TypeError` so missing-BD bugs surface at bake time instead of silently
  pinning to the legacy layout. Docstring updated to spell out the new
  contract.

#### M4: `_set_threshold_attn` BD=None fallback — ✅ HARDENED

- File: `c4_release/neural_vm/vm_step.py:2232-2264`.
- Line 2245: previously `if BD is None: BD = _SetDim`.
- Callers: `unified_compiler/ops/l0_ops.py:64`, `l1_ops.py:43,56`,
  `l2_ops.py:129`. The audit originally claimed *all* of these passed
  `BD=proxy`; in fact only `l0_ops.py` did. `l1_ops.py` and `l2_ops.py`
  silently relied on the legacy `_SetDim` fallback, so under
  `pin_io_only=True` they would bake into the wrong residual columns.
- Resolution (cleanup-bd-setdim-medium):
  1. `_set_threshold_attn` now raises `TypeError` when `BD is None`
     (previously it pinned to the legacy `_SetDim` layout).
  2. The three legacy-fallback call sites in `l1_ops.py` and `l2_ops.py`
     were updated to explicitly forward `BD=proxy`. This is functionally
     equivalent under the default layout (CONST / IS_MARK / MARKS are
     IO-pinned) but unblocks any future non-IO-pinned compact layout.

#### M5: `compact_moe` `BD = _SetDim` resolver fallback — ✅ DOCUMENTED

- File: `c4_release/neural_vm/vm_step.py:1300-1344`.
- Line 1302: `BD = _SetDim` then `D(name)` falls back to `getattr(BD, name)`.
- Callers: `c4_release/neural_vm/vm_step.py::AutoregressiveVM.compact_moe`
  is called from `_expand_wrapper_blocks` and runtime code paths that
  already pass `self.dim_positions`. The fallback is only used when
  `dim_positions` is not a dict.
- Resolution (cleanup-bd-setdim-medium): structurally correct — the
  resolver checks `isinstance(self.dim_positions, dict)` first and only
  falls back to `_SetDim` when no compact layout is in scope. An
  inline comment block at the `BD = _SetDim` line now spells out the
  contract so future readers don't mistake it for a legacy pin.

### MEDIUM priority — Production helpers that still hardcode `_SetDim`

#### M6: `setup_helpers.py::_set_cs_threshold_attn` — already covered (M2).

#### M7: `efficient_alu_neural.py::PureNeuralALU` baking constructor — ✅ CONFIRMED

- File: `c4_release/neural_vm/efficient_alu_neural.py:281`
  (`def __init__(self, S, BD, operations='add_sub')`).
- Verification (cleanup-bd-setdim-medium): `BD` is a positional required
  constructor parameter; the class is no longer actively instantiated in
  the production graph (the split ALU implementations
  `efficient_alu_addsub_split.py` / `efficient_alu_divmod_split.py` are
  used instead). The remaining references are docstrings calling these
  out as the historical equivalent. The split implementations themselves
  thread `BD` through their constructors and are called from
  `_make_alu_postop_attach_op` with `_as_setdim_proxy(dim_positions)`.
  **Properly threaded; no change needed.**

### LOW priority (debug / test / dead code)

These hardcode `_SetDim` but are not on the production forward path:

| File | Lines | Reason |
| ---- | ----- | ------ |
| `c4_release/neural_vm/contracts.py` | 19-21 | `DimensionContract` validator — debug utility, called only from `DEBUGGING_QUICK_REFERENCE.md` examples. |
| `c4_release/neural_vm/debugger.py` | 25-27 | `VMExecutionTracer` — debug utility. |
| `c4_release/neural_vm/step_debugger.py` | 21-24 | `StepDebugger` — debug utility. |
| `c4_release/neural_vm/eviction_policy.py` | 15 | KV-cache eviction policy; imported in `test_archive/test_eviction_policy.py` only. |
| `c4_release/neural_vm/weighted_eviction.py` | 18 | KV-cache eviction; not used in production runner. |
| `c4_release/neural_vm/score_based_eviction.py` | 10 | KV-cache eviction; not used in production runner. |
| `c4_release/neural_vm/efficient_add.py` | 240 | Inside `if __name__ == '__main__':` block. |
| `c4_release/neural_vm/dim_registry.py` | 528 | `_build_setdim_lookup` for debug dim name reverse-map. |
| `c4_release/neural_vm/weight_modules/base.py` | 154 | `get_dimension_registry()` — used only by `weight_modules/{embedding,function_calls}.py`, themselves used only in `c4_release/tests/test_ent_lev_neural.py`. |
| `c4_release/neural_vm/tests/test_neural_embedding.py` | 33, 63, 89, 123, 150, 180, 213, 256 | Tests. |
| `c4_release/neural_vm/tests/test_opcodes.py` | 895-926 (multiple) | Tests. |
| `c4_release/neural_vm/tests/test_dimension_dataflow.py` | 148-149 | Tests. |
| `c4_release/neural_vm/tests/test_dim_registry.py` | 222 | Tests. |
| `c4_release/neural_vm/vm_step.py` | 933, 1018, 1044, 1112, 1131 | `DivModModule` — class definition only, **never instantiated in production** (only in `test_archive/test_divmod_modes.py`). DIV/MOD runs through `_FlattenedDivModBuilder` / `FlattenedDivMod`. |
| `c4_release/neural_vm/unified_compiler/ops/l9_ops.py` | 247 | `from ...vm_step import _SetDim as BD_DEFAULT` — dead import; the same function on line 258 immediately rebinds `BD = _as_setdim_proxy(dim_positions)`. |
| `c4_release/neural_vm/unified_compiler/primitives.py` | 26 | `from ..vm_step import _SetDim as BD` — most methods now take `bd=None` and use `spec = bd if bd is not None else BD`; remaining direct `BD.` refs are inside primitives whose only callers pre-resolve via the proxy at the call site. Worth a follow-up sweep but each individual site is masked today. |

## Count summary

```
Total hardcoded refs (production path): 14
  HIGH (production PureFFN / nn.Module): 1   (H1: _ensure_l11_mul_module — open)
  MEDIUM (bake helpers / fallbacks):     0 remaining
    M1, M2 ........................ DELETED (cleanup-bd-setdim-medium)
    M3, M4 ........................ HARDENED — BD now required, TypeError on None
    M5 ............................ DOCUMENTED — structurally correct (resolver
                                     consults dim_positions dict first)
    M7 ............................ CONFIRMED — PureNeuralALU takes BD as required
                                     ctor param; class is no longer actively
                                     instantiated (split ALUs are used instead)
  LOW (tests / debug / dead):           ~18  (see table)
  Already threaded (A1, P2):             4   (ComparisonCombine, BinaryOpByteZeroingPostOp,
                                              CarryPropagationPostOp, BitwiseBytePropagationPostOp)
```

Plus `c4_release/neural_vm/unified_compiler/primitives.py` has ~30 individual
`BD.<NAME>` references inside methods that are partially threaded (some take
`bd=None`, some don't). Those should be enumerated in a follow-up; for the
purposes of this audit they are LOW because each currently-reachable caller
pre-resolves the dim positions before calling into Primitives.

## Spot-check / validation

Tried to construct a runtime repro for H1 (`_ensure_l11_mul_module`). The MUL
smoke test `test_smoke.py::TestSmokeBasic::test_mul_basic` is already failing
(timeout) under the headline runner (`pure_neural=True, trust_neural_alu=True`),
which routes MUL through the efficient (proxy-correct) path. The lookup path
that exercises H1 is not currently exercised by a fast smoke test, so a layout-
collision repro for MUL needs to be built as part of the H1 fix; deferred.

Layout collision shape is confirmed via dump:

```
Legacy _SetDim                     Compact pin_io_only=True (d_model=728)
  ALU_LO       = 360                 ALU_LO       = 330
  ALU_HI       = 376                 ALU_HI       = 346
  AX_CARRY_LO  = 328                 AX_CARRY_LO  = 362
  AX_CARRY_HI  = 344                 AX_CARRY_HI  = 378
  OP_MUL       = 289                 OP_MUL       = 210
  OUTPUT_LO    = 174                 OUTPUT_LO    = 69
  OUTPUT_HI    = 190                 OUTPUT_HI    = 85
  MARK_AX      = 1                   MARK_AX      = 1
```

Every dim except `MARK_AX` differs; a bake against legacy `ALU_LO=360` reads
from `d_model[..., 360]` in a 728-wide tensor, which exists but addresses an
unrelated dim under compact, identical in shape to the A1 IMM 240/255 bug.

## Recommended next action

1. **H1 fix** (the only remaining HIGH item): add `dim_positions=None` to
   `_ensure_l11_mul_module` and route through `_as_setdim_proxy`. Mirror the
   threading in `_make_alu_postop_attach_op` in the same file. Update the 9
   callers in `alu_ops.py` to forward `dim_positions`. Re-bake test:
   `test_smoke.py::test_mul_basic` (will still time out from orthogonal
   Phase 2 issues, but `test_runtime_vanilla` should stay green).
2. ~~**M1, M2 deletion**~~ — done in cleanup-bd-setdim-medium.
3. ~~**M3, M4** defensive fallback tightening~~ — done in
   cleanup-bd-setdim-medium; both now raise `TypeError` when `BD is None`
   and all live callers explicitly forward `BD=proxy`.

## Gate suite

```
timeout 400 python -m pytest \
  c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit \
  c4_release/tests/test_runtime_vanilla.py \
  c4_release/tests/test_layer_idx_consistency.py \
  c4_release/tests/test_compile_determinism.py \
  -v --tb=line --timeout=300
```

Result: **11 passed** (291.65s) on this branch — read-only audit, no changes.
