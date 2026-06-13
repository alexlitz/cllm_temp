# `strict_trace` (token-identity) criterion — 2026-06-13

Adds a third pass criterion to the 1096 canonical runner: **token-identity**
fail-fast, i.e. the literal "autofail on the first divergent token from the
DraftVM."

This is a **measurement/runner change only** — no model weights, no model ops,
no GPU build for the core work. The correctness gate below is code review +
CPU-only sanity checks; the GPU validation is **deferred** (see "Deferred GPU
validation").

## Criteria ladder (strictly increasing strength)

```
exit_code  ⊆  full_trace  ⊆  strict_trace
```

- **`exit_code`** (default, suite-exact): `neural_exit == decl_exit`.
- **`full_trace`** (`--fail-fast`): after each VM step, the model's decoded
  `(PC, AX)` must match the DraftVM oracle's `(pc, ax)`.
- **`strict_trace`** (new): after each completed 35-token VM step, the model's
  emitted token at **every SAFE offset** must equal the DraftVM
  `draft_tokens()` reference token at that offset. Fail on the FIRST mismatch.

`strict_trace ⊇ full_trace` because the PC/AX value bytes (offsets 1–4, 6–9)
are all SAFE offsets, so any `(PC, AX)` divergence is also a token divergence.
strict_trace *additionally* catches SP/BP/STACK0 byte drift and STEP_END/HALT
framing drift (offsets 10–24, 34) that full_trace ignores.

## 35-token step layout (`DraftVM.draft_tokens`)

| offset | field | safe? |
|---|---|---|
| 0 | `PC_marker` | ✅ |
| 1–4 | `PC[0..3]` | ✅ |
| 5 | `AX_marker` | ✅ |
| 6–9 | `AX[0..3]` | ✅ |
| 10 | `SP_marker` | ✅ |
| 11–14 | `SP[0..3]` | ✅ |
| 15 | `BP_marker` | ✅ |
| 16–19 | `BP[0..3]` | ✅ |
| 20 | `STACK0_marker` | ✅ |
| 21–24 | `STACK0[0..3]` | ✅ |
| 25 | `MEM_marker` | ✅ |
| **26–29** | `MEM_ADDR[0..3]` | ❌ UNSAFE |
| **30–33** | `MEM_VAL[0..3]` | ❌ UNSAFE |
| 34 | `STEP_END/HALT` | ✅ |

SAFE = `{0..25, 34}` (27 offsets). UNSAFE = `_UNSAFE_OFFSETS = range(26, 34)`
(the MEM addr/val metadata bytes) — the embedding's MEM-metadata injection
makes these unreadable from a flat-forward argmax, so the DraftVM is trusted
there and they are SKIPPED by the compare. This is the SAME set already used by
the speculative accept/correct loop.

## What changed (code)

### `neural_vm/batched_pure_neural.py`
- `_oracle_pc_ax_steps(..., with_tokens=False)`: when `with_tokens=True`,
  returns `(steps, tokens)` where `tokens[i]` is the full 35-token
  `draft_tokens()` reference for step `i`, captured from the SAME fresh DraftVM
  at the SAME point (right after `vm.step()`) as the `(pc, ax)` pair — so the
  token oracle is byte-aligned with the `(pc, ax)` oracle. `with_tokens=False`
  (default) preserves the legacy `List[(pc, ax)]` return — full_trace unchanged.
- `_ElementState`: new fields `ff_oracle_tokens`, `ff_div_offset`,
  `ff_div_offset_name`, `ff_expected_tok`, `ff_got_tok`.
- `run_batch_fail_fast(..., criterion="full_trace")`: validates the knob,
  builds the token oracle only when `criterion == "strict_trace"`, threads the
  knob to `_run_fail_fast`, and adds `divergence_offset`,
  `divergence_offset_name`, `expected_tok`, `got_tok` to each result dict.
- `_run_fail_fast(..., criterion="full_trace")`: threads the knob to the two
  `_ff_check_new_steps` call sites.
- `_ff_check_new_steps(s, *, criterion="full_trace")`: dispatches to
  `_ff_check_new_steps_strict` when `criterion == "strict_trace"`; the
  full_trace `(PC, AX)` path is byte-for-byte unchanged.
- `_ff_check_new_steps_strict(s)` (new): per completed step, compares context
  tokens vs `ff_oracle_tokens` at every SAFE offset, fails on first mismatch
  recording step/offset/name/expected_tok/got_tok (+ `(PC, AX)` context). Keeps
  the "ran more steps than the oracle declares" failure (reported as a
  STEP_END/HALT-offset divergence).

### `tools/run_1096_canonical.py`
- `--criterion` gains the `strict_trace` choice. `--criterion strict_trace`
  implies `--fail-fast`. `criterion_name` / `trace_criterion` resolution
  added; `trace_criterion` threads through `_run_chunks_streaming` →
  `_run_one_chunk_with_oom_retry` → `run_batch_fail_fast` → `_score_fail_fast_results`.
- `_score_fail_fast_results(..., criterion=)`: populates the new
  `ProgramResult` offset fields and builds a strict_trace-flavored error
  string (`strict-trace token divergence at step N offset O (NAME): ...`).
- Per-fail stderr sample line for strict_trace shows
  `offset=O(NAME) expected_tok=… got_tok=… [got(pc,ax) oracle(pc,ax)]`.
- `--output` JSON `criterion` description string adds the strict_trace variant.

### `tools/run_1096_fast.py`
- `ProgramResult` gains `divergence_offset`, `divergence_offset_name`,
  `expected_tok`, `got_tok` (serialized via `asdict` to the checkpoint sidecar
  and `--output` JSON).

### `tests/test_batched_pure_neural.py`
- `test_ff_check_new_steps_strict_token_identity` (CPU, no model): pass case,
  SAFE-offset fail (AX[1]), UNSAFE-offset ignored, SP-drift full_trace-ignores
  / strict_trace-catches.
- `test_run_batch_fail_fast_strict_trace_integration` (fake CPU model):
  end-to-end pass + AX[0] divergence through the spec accept/correct loop.

## CPU-only sanity checks (done, no GPU)

All green:
- Offset-name mapping correct for all 35 offsets; SAFE set = `{0..25,34}` (27).
- `_oracle_pc_ax_steps(with_tokens=True)` returns the SAME `(pc,ax)` steps as
  the legacy call and 35-wide token lists byte-identical to an independent
  `draft_tokens()` trace.
- `_ff_check_new_steps_strict`: passes a perfect trace; fails at AX[1] with
  correct offset/name/tokens; ignores UNSAFE corruption; catches SP drift that
  full_trace ignores.
- `run_batch_fail_fast(criterion="bogus")` raises `ValueError`.
- canonical-runner criterion resolution table (6 cases + conflict) correct;
  `argparse` exposes the `strict_trace` choice.
- The 2 pre-existing full_trace CPU tests still pass (path unperturbed).

## Deferred GPU validation (DO NOT run here — would OOM the other agents)

When a GPU frees, the orchestrator should run (from `c4_release/c4_release`):

1. **strict_trace ⊇ full_trace on a known full_trace PASS** — pick an id that
   PASSES full_trace (e.g. a simple edge/`add` id) and confirm it STILL passes
   strict_trace (every SAFE token matches), proving strict_trace does not
   regress clean programs:

   ```
   python tools/run_1096_canonical.py --fail-fast --criterion full_trace  --ids <PASS_ID>
   python tools/run_1096_canonical.py --fail-fast --criterion strict_trace --ids <PASS_ID>
   # both report PASS
   ```

2. **strict_trace catches the add_0 AX byte-1 carry** — id 0 (`add_0`) is a
   known AX byte-1 divergence; confirm strict_trace FAILS at **step 1, the AX
   byte-1 offset (offset 7, name `AX[1]`)**:

   ```
   python tools/run_1096_canonical.py --fail-fast --criterion strict_trace --ids 0
   # FAIL: strict-trace token divergence at step 1 offset 7 (AX[1])
   ```

3. **Set-monotonicity spot check** — on a small id slice, confirm
   `strict_trace PASS set ⊆ full_trace PASS set ⊆ exit_code PASS set`:

   ```
   python tools/run_1096_canonical.py                       --ids <SLICE> --output /tmp/exit.json
   python tools/run_1096_canonical.py --criterion full_trace  --ids <SLICE> --output /tmp/full.json
   python tools/run_1096_canonical.py --criterion strict_trace --ids <SLICE> --output /tmp/strict.json
   # pass(strict) <= pass(full) <= pass(exit) for the slice
   ```

No model build was performed for this change; (1)–(3) are the only steps that
require the GPU.
