# spec_k=8 vs spec_k=0 divergence — investigation + fix (2026-06-11)

## TL;DR

The premise ("spec_k=8 diverges from spec_k=0 on the smoke path") is **false
for the path the smoke gate uses**. The smoke gate runs
`BatchedPureNeuralRunner` and it is **byte-identical at spec_k=0 and
spec_k=8** (verified: 14/14 CMP+MEM+FUNC tests, including `test_ne_true` and
the SI/LI/SC/LC memory tests that exercise the DraftVM-trusted MEM offsets).
There is no KV-cache or speculation bug in the smoke path. (`use_kv_cache`
defaults to off in the batched runner, so both spec_k values take identical
fresh `model.forward(padded)` passes; the only spec-path delta is the
`_UNSAFE_OFFSETS` (26..33) trust, which does not diverge on any test.)

The divergence agents reported (`test_ne_true` returning 16/20 instead of 1)
came from the **serial `AutoregressiveVMRunner`** — the conftest
`_pure_neural_runner_model` fixture path — and it is **not** a spec/KV bug.

## Root cause (serial runner)

`AutoregressiveVMRunner.run()`'s non-speculative loop (`spec_k=0`) runs a
`draft_divergence` early-bail heuristic (`enable_divergence_bail=True`): per
emitted token it compares the model's argmax against a `DraftVM` reference
and bails when >50% of a 50-token window mismatches.

`DraftVM.draft_tokens()` emits a rigid 35-token step (`[REG_PC, 4b, REG_AX,
4b, REG_SP, 4b, REG_BP, 4b, STACK0, 4b, MEM, 4b addr, 4b val, STEP_END]`).
The model's emitted step is **not rigidly 35 tokens** — it occasionally
emits a stray boundary token — so after step 0 the compare position
(`draft_step_pos`) slips permanently one position out of phase with the
DraftVM reference. The result: ~half the per-token compares spuriously
mismatch on short correct programs (measured 0.42–0.49, right at the 0.50
threshold), randomly tripping a **false** `draft_divergence` bail that
truncates execution before the result is computed — returning the raw
operand (e.g. `ne_true` → 20 instead of 1).

The serial `spec_k>0` path (`_run_speculative`) has **no** divergence-bail,
so it ran to completion and happened to be correct — hence the apparent
"spec_k=8 right, spec_k=0 wrong" inversion.

### Evidence (serial, `ne_true` and friends), exit codes

| test | s0 bail ON (before) | s0 bail OFF | s8 | batched s0 (ground truth) |
|---|---|---|---|---|
| eq_true | 42 ✗ | 0 | 0 | 0 |
| ne_true | 20 ✗ | 1 | 1 | 1 |
| lt/gt/le/ge_true | raw operand ✗ | 1 | 1 | 1 |
| si_li_roundtrip | 42 | 42 | 42 | 42 |

`s0 bailON != s0 bailOFF` ⇒ the bail (not spec/KV) is the divergence source.

## Fix

`c4_release/neural_vm/run_vm.py` (+23 lines, serial loop only): re-anchor
`draft_step_pos = 0` whenever the model emits `Token.REG_PC` (the
unambiguous start-of-step marker, always draft offset 0). This keeps the
per-token compare in phase with the model's real step boundaries. Genuine
divergence (wrong VALUE bytes after the marker) still accumulates mismatches
and trips the window, so the timeout guard is preserved.

## Results

- Serial `spec_k=0` vs `spec_k=8`: diverging tests dropped from **7 → 1**.
  The 6 fixed (`eq_true, lt_true, ne_true, gt_true, le_true, ge_true`) are
  now byte-identical and match the batched ground truth.
- The 1 remainder (`eq_false`: s0=0, s8=17) is an independent unspec-vs-spec
  VALUE difference on a test that FAILS the smoke gate in all paths
  (batched=17, expected 0); out of scope for the false-bail fix.
- **Smoke gate (batched, spec_k=0) unchanged**: TestSmokeComparison +
  TestSmokeMemory = 11 passed / 2 failed, identical to baseline. The 2
  failures (`eq_true`, `eq_false`) are pre-existing weight-level failures
  documented in `PROBE_GROUNDTRUTH_2026_06_10.md`, not caused by this change.
  The change touches only `AutoregressiveVMRunner.run`; the batched runner
  has no reference to `draft_step_pos`/`draft_mismatch_history`.

## Constraints honoured

- No model-weight / `ops/` changes — runner-layer (bail heuristic) only.
- Smoke gate (batched `spec_k=0`) behaviour byte-for-byte unchanged.
- No forward hooks; production model untouched.
