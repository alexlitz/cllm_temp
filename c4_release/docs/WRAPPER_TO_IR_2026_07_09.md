# Wrapper FFN blocks → IR coverage: opaque_skipped 5 → 0 (faithful interpreter now covers 100 % of production)

**Date:** 2026-07-09  **Branch:** `wrapper-to-ir`  **Golden (unchanged):** `e50521f3`

## Problem (from the derivation/interpreter audit)

The **efficient / production-mode** model (`compile_full_vm_dynamic(alu_mode="efficient")`,
the model `DSLInterpreterVerdictRunner` builds by default) installs **5 campaign
WRAPPER FFN block types** that are neither a `PureFFN` (they expose no
`W_up`/`W_gate`/`W_down`) nor members of the interpreter's composite-fragment set:

| block idx (efficient) | wrapper class | install site |
|---|---|---|
| 12 | `LoadedOperandAddHi15ClearFFN` | `make_loaded_operand_add_hi15_clear_op` |
| 17 | `MulOperandSeRecoverFFN`        | `make_efficient_l11_alumul_wrap_op` / `l11_ops.py` |
| 21 | `CmpOperandSeRecoverFFN`        | `make_efficient_l10_andorxor_wrap_op` |
| 29 | `BitwiseOperandSeRecoverFFN`    | `make_efficient_l10_andorxor_wrap_op` |
| 33 | `ShiftOutputClearFFN`           | `alu_ops.py` shift composite build |

Each wraps an `inner` `PureFFN`, does an in-place operand/output **cell edit** in its
`forward(x)` (x is `[B, N, D]`), then delegates to `inner`. So it has no SwiGLU rule
form of its own.

Consequence in `neural_vm/verification/faithful_interpreter.py`:

* `block_ffn_coverage(model)` classified these 5 blocks as `opaque_skipped`
  (they are not in `COMPOSITE_ALU_FFN` and `hasattr(block.ffn, "W_up")` is False) →
  **`opaque_skipped = [12, 17, 21, 29, 33]` (5)** — the faithful DSL interpreter did
  NOT cover production.
* `IRBlockForward.__init__` / `CachedFaithfulForward.__init__` / `IRBlockForward`
  (the `DSLInterpreterVerdictRunner` vehicle) took the `else` branch and read
  `ffn.W_up` →
  **`AttributeError: 'LoadedOperandAddHi15ClearFFN' object has no attribute 'W_up'`**
  → `DSLInterpreterVerdictRunner` **crashed on the production efficient model**.

Both were reproduced on this branch's base (`59a9de19`) before the fix.

## Fix (approach *a* — the composite-fragment pattern, the cleanest option)

Register the 5 wrapper class names into `COMPOSITE_ALU_FFN` in
`neural_vm/verification/faithful_interpreter.py`. That is the SAME mechanism the 4
composite ALU blocks (`AddSub5StageBlock` / `FlattenedALUMul` / `FlattenedDivMod` /
`ALUShiftComposite`) already use: a block whose class name is in `COMPOSITE_ALU_FFN`
is executed through a `CompositeFFNFragment` (built by `composite_ffn_ir`), which runs
`block_ffn(x.unsqueeze(0))[0]` — i.e. it **invokes the exact deployed `nn.Module`**, so
it is **byte-identical to the deployed block by construction**. The wrappers already
take `[B, N, D]` and return `[B, N, D]` (they explicitly document honouring "the same
composite-FFN contract `AddSub5StageBlock` / `FlattenedDivMod` honour"), so the
`[S, D]` ↔ `[B, N, D]` adapter in `composite_ffn_ir` needs no change.

No new IR node type, no re-expression of the imperative cell edit as SwiGLU rules
(approach *b* was unnecessary and would not have been byte-exact — the edit is a masked
`torch.where`, not affine). The one edit is the `COMPOSITE_ALU_FFN` tuple + its
docstring.

Effect on the three consumers (`block_ffn_coverage`, `IRBlockForward`,
`CachedFaithfulForward`): all now route the 5 wrappers through
`composite_ffn_ir` → `FaithfulInterpreter._apply_ffn_op`'s `composite` branch,
instead of `opaque_skipped` / the `W_up` `else`.

Tooling touched (model is untouched — see golden below):
* `neural_vm/verification/faithful_interpreter.py` — add 5 names to `COMPOSITE_ALU_FFN`.
* `tools/faithful_interpreter_full_validate.py` — `build_model` / `main` gain
  `--alu-mode {lookup,efficient}` + `--no-disk-cache` so `--coverage` can be run on the
  **efficient/production** model (it previously only built the lookup model).

## Proof

### 1. opaque 5 → 0, 100 % IR-executable (efficient/production model)

`tools/faithful_interpreter_full_validate.py --alu-mode efficient --no-disk-cache --coverage`:

```
BEFORE (base 59a9de19):  opaque_skipped = [12, 17, 21, 29, 33]   (5)
AFTER  (this branch):    opaque_skipped = []                     (0)
  n_blocks           = 61
  n_ir_executable    = 61        # 61/61 = 100 %
  n_composite (ALU)  = 8  at [12, 13, 17, 21, 29, 30, 31, 33]   # 3 ALU composites + 5 wrappers
```

### 2. `IRBlockForward` / `DSLInterpreterVerdictRunner` no longer crash on production

```
BEFORE: IRBlockForward(model) -> AttributeError: 'LoadedOperandAddHi15ClearFFN' has no 'W_up'
AFTER:  DSLInterpreterVerdictRunner(model=efficient) constructed OK; inner._dsl = IRBlockForward
```

### 3. Byte-identity held (the model was NOT changed)

* **Golden hash unchanged:** `tools/_isa_golden_hash.py` = `e50521f3…` (== target `e50521f3`),
  identical before and after — the change is confined to the verification/tooling layer.
* **Per-block residual identity (incl. the 5 wrappers, executed via the IR fragment):**
  `--alu-mode efficient` Layer-A = **40/40** programs (each wrapper block's IR-fragment
  output matches the deployed `block(x)` to fp exactness).
* **Full-tape argmax identity vs `model.forward`:** `CachedFaithfulForward` and
  `IRBlockForward` = **2310/2310** positions across the 14 smoke programs
  (add/sub/mul/div/mod/and/or/xor/eq/…, which exercise every wrapper block), and
  the validator's Layer-B (`CachedFaithfulForward` argmax vs `model.forward`) =
  **40/40** on the 40-program corpus sample, PC/AX per-step decode = **40/40**.
* **End-to-end verdict = CPU-neural:** `DSLInterpreterVerdictRunner` (efficient model,
  `IRBlockForward` vehicle) vs `BatchedPureNeuralRunner` CPU-neural agree on
  status **and** first-divergence step (e.g. `mul` fail@step-3 on both) — the full
  autoregressive framing-drift verdict, byte-for-byte.
* `pytest tests/test_faithful_interpreter.py` = **5/5 pass**.

## Bottom line

The 5 efficient/production-mode wrapper FFN blocks are now first-class IR-executable
ops (via `CompositeFFNFragment`), so `opaque_skipped` goes **5 → 0** on the production
model, `IRBlockForward` / `DSLInterpreterVerdictRunner` no longer crash, and the result
is byte-identical to `model.forward`. The faithful DSL interpreter now covers **100 % of
production**, with the model left byte-identical (golden `e50521f3`).
