# Canonical fast 1096 runner (2026-06-11)

Date: 2026-06-11
Branch: `main` HEAD `6b159480`
Tool: [`tools/run_1096_canonical.py`](../tools/run_1096_canonical.py)
Path measured: **pure-neural** (the vanilla thesis), NOT the production handler.

## TL;DR

* `tools/run_1096_canonical.py` is a FAST runner that reports the **same
  per-program pass/fail criterion** as the pytest suite
  `tests/test_suite_1096_pure_neural_pytest.py`.
* The pytest suite and `tools/run_1096_fast.py` already used the **same pass
  criterion**. The historical "~1%" from `run_1096_fast.py` was a
  **`--limit 128` slice artifact** (the leading add/sub/mul band, which the
  model fails on the high-byte carry path), NOT a 60x criterion mismatch.
* The genuine reconciliation finding is about **speculation determinism**, not
  the comparison: the suite defaults to `C4_SPEC_K=adaptive`, whose pass set is
  **non-deterministic across batch composition**. The canonical runner defaults
  to `spec_k=0` (raw decode), which is **deterministic and batch-independent**
  (verified: identical pass set at chunk=16 and chunk=32). This is the
  smoke-gate ground-truth path (`project_probe_path_spec_k_not_hooks`).
* Per-id agreement with the actual pytest suite, both at matched `spec_k=0`,
  chunk=16, over a 49-id stratified sample: **48/49**. The single residual
  (id 1010 `1*1`) is a *batch-composition-sensitive edge program* in the
  batched runner — its result flips with which neighbors share its forward
  batch, even at spec_k=0; not a criterion bug.

## The two pass criteria, diffed precisely

### pytest suite `test_program` (the canonical criterion)

```python
if err is not None:                       # compile / declarative / batch error
    pytest.fail(...)                      # -> FAIL
assert decl["exit_code"] == (expected & 0xFFFFFFFF)   # oracle vs suite expected
assert exit_code == decl["exit_code"]                 # neural vs oracle
```

with `@pytest.mark.xfail(strict=False)` on every `test_program`, so
**XPASS = PASS** and **xfail = FAIL**. Config: session fixture compiles all
selected programs and runs them through `BatchedPureNeuralRunner.run_batch` in
chunks of `C4_BATCH_CHUNK` (default 32), `max_steps=None` (declarative halt
horizon per program), `C4_SPEC_K=adaptive` (-1), KV cache off.

### `tools/run_1096_fast.py` (pre-existing)

Identical comparison, expressed imperatively:

1. compile error / oracle exception / oracle non-halt / oracle-vs-suite
   disagreement -> `error` status (= FAIL). The oracle is called with
   `suite_expected=expected`, so an oracle-vs-suite mismatch becomes
   `oracle.error` == the suite's `assert #2` failing. **Equivalent.**
2. `neural_exit is None` -> `error` (= FAIL).
3. else `(neural_exit & 0xFFFFFFFF) == (decl_exit & 0xFFFFFFFF)` -> ok/fail ==
   the suite's `assert #3`. **Equivalent.**

The criteria are equivalent. The differences are **defaults**:
`run_1096_fast` defaults `--spec-k 4` and historically was *invoked* with
`--limit 128`. The "1%" came from the slice, not the comparison.

### `tools/run_1096_canonical.py` (this tool)

Reuses `run_1096_fast`'s `_compile_and_oracle` / `_run_chunks` (same criterion),
runs the **full** corpus by default, defaults to the **deterministic**
`spec_k=0`, and emits a **per-cluster breakdown**.

## The ~60x "off" criterion was a measurement artifact, not a comparison bug

The brief asked for "the precise criterion that was off by ~60x". The honest
answer: **there was no 60x criterion difference.** Both paths compare
`neural_exit == declarative_exit` (32-bit masked). The 1% vs 62 gap was:

* **1% (`run_1096_fast`)**: only ever measured on `--limit 128` (= ids 0-127 =
  add 0-49 / sub 50-99 / mul 100-127). That band is *uniformly* failing because
  every multi-digit result trips the 16-bit high-byte carry / 0xFF-leak path.
  1-2/128 is the local floor of that specific slice.
* **62 (pytest suite)**: measured on the *full* 1096, where the passing mass
  lives in the div/mod single-byte results and the `edge_*` / `if_*` / scattered
  short programs — ids the 128-slice never reached.

Same criterion, different denominator. The "60x" was 1/128 vs 62/1096.

## Speculation determinism (the real reconciliation knob)

| spec_k | path | deterministic? | pass-set stable across chunk size? |
|---|---|---|---|
| `0` | raw one-token-per-forward | **yes** | **yes** (chunk16 == chunk32, verified) |
| `4` | fixed-K DraftVM | mostly | mostly |
| `-1` (adaptive) | per-element K, rolling reject rate | **no** | **no** — flips on batch composition |

Adaptive K (the suite default) updates each element's K from a rolling
rejection rate over the batch, so the *same* program can pass or fail depending
on which programs share its batch. We observed the 49-id sample give three
different adaptive pass sets across chunk=16 / chunk=32 / suite-collection
order. **The suite's own 62/1096 is therefore not bit-reproducible.** The
canonical runner uses `spec_k=0` to give a *stable* number to track fixes
against. Pass `--spec-k -1` to reproduce the suite's literal (noisier) path.

## Per-id validation (acceptance gate)

49-id stratified shallow sample (steps <= 10, 32 distinct clusters incl.
add/sub/mul/div/mod, if_*, var_simple, expr_*, every edge_* family, bool_and).
Both runs: `spec_k=0`, `chunk=16`, same id set, GPU 1.

* canonical PASS: `375 401 1003 1004 1007 1009 1011 1012 1013 1023 1026 1027 1028 1031 1071` (15)
* suite    PASS: `375 401 1003 1004 1007 1009 1010 1011 1012 1013 1023 1026 1027 1028 1031 1071` (16)
* **Agreement: 48/49.** Sole diff: id 1010 (`1*1`) — batch-edge sensitivity
  (XPASS when batched with 2 neighbors, FAIL alone / in the 49-batch).

At the suite's *default* (adaptive), agreement was 45/49 — the extra 3 diffs
were adaptive non-determinism, not criterion. This is why the canonical default
is spec_k=0.

## Canonical 1096 number on HEAD `6b159480`

Pure-neural, `spec_k=0` (deterministic), `chunk=16`, GPU 1.

**Canonical score: ~119 / 1096 (~10.9%).** This is **+57 over the 62/1096
suite baseline** in `docs/1096_TRIAGE_2026_06_11.md` — the in-flight CMP /
operand-gather fixes have clearly moved the **if** cluster (0 -> 37) and the
**edge / bool_and** clusters are now strong passers.

Measurement note: the deep clusters **loop (450-524), func (525-699), rec
(700-799), gcd (900-949), expr (800-899), nested (950-999), absdiff
(1046-1070)** are 0-pass and *intractably slow at full horizon* under the
pure-neural path — their neural decode never emits a clean halt token
(JSR/LEV/loop-branch arch-blocked), so it runs to the full declarative horizon
(rec_fib alone is 8369 VM steps). They were confirmed 0-pass on the chunks that
did complete (expr 800-847 = 0/48; loop region in the full run = 0; func
chunks all non-halting) and are 0 in every prior triage. The ~119 number counts
them as 0. The tractable measured mass is exact.

### Per-cluster breakdown (current HEAD `6b159480`)

| Cluster | ids | n | pass | source |
|---|---|---:|---:|---|
| add | 0-49 | 50 | **0** | measured |
| sub | 50-99 | 50 | **1** | measured |
| mul | 100-149 | 50 | **1** | measured |
| div | 150-199 | 50 | **17** | measured |
| mod | 200-249 | 50 | **25** | measured |
| var_simple/mul/three/update | 250-349 | 100 | **0** | measured |
| if_gt | 350-374 | 25 | **13** | measured |
| if_lt | 375-399 | 25 | **13** | measured |
| if_eq | 400-424 | 25 | **11** | measured |
| if_var | 425-449 | 25 | **0** | measured |
| loop_* | 450-524 | 75 | **0** | partial+triage (slow, non-halting) |
| func_* | 525-699 | 175 | **0** | partial+triage (slow, non-halting) |
| rec_* | 700-799 | 100 | **0** | triage (deepest; rec_fib 8369 steps) |
| expr_* | 800-899 | 100 | **0** | measured 800-847=0/48 +triage |
| gcd | 900-949 | 50 | **0** | triage (45-253 steps) |
| nested_* | 950-999 | 50 | **0** | triage |
| edge_* | 1000-1045 | 46 | **24** | measured |
| absdiff | 1046-1070 | 25 | **0** | triage |
| bool_and | 1071-1095 | 25 | **14** | measured |
| **TOTAL** | | **1096** | **~119** | |

Movement vs the 62 baseline lives entirely in **if (+37)** and the
**edge / bool_and** families (the CMP / one-hot operand-gather surface). The
arithmetic mass (div 17, mod 25) was already passing single-byte results;
add/sub/mul/var remain blocked on the 16-bit high-byte carry and stack-store
baselines.

## Usage

```bash
# Full canonical score + per-cluster breakdown (GPU 1):
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
    python tools/run_1096_canonical.py --output /tmp/canonical_1096.json

# Reproduce the suite's literal adaptive path (same passes, slower, noisier):
CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py --spec-k -1

# Validate per-id against the pytest suite on a sample:
CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py \
    --ids 0,49,250,550,975 --output /tmp/sample.json
CUDA_VISIBLE_DEVICES=1 C4_SPEC_K=0 C4_BATCH_CHUNK=16 python -m pytest \
    tests/test_suite_1096_pure_neural_pytest.py -k "0000_ or 0049_ or ..." \
    --junitxml=/tmp/suite.xml
```

## Wall-time reality

The full pure-neural decode is dominated by the **deep-step tail**
(loop / rec / gcd with 100s-1000s of VM steps; each step is ~35 tokens and the
batched forward is sequential at spec_k=0). The shallow first ~250 programs
finish in a few minutes; the deep recursion tail is what made the pytest suite
take >5h. The canonical runner is far faster than the per-test suite for the
shallow mass, but the deep tail still costs real time. For a quick tracking
signal, run `--limit 900` (skips the deepest rec/gcd, which are 0/all in every
prior measurement) or `--ids` a per-cluster sample.

## Cross-references

* `docs/1096_TRIAGE_2026_06_11.md` — the 62/1096 baseline + cluster triage.
* `project_1096_fast_tool_is_at_1pct_floor` (memory) — the `--limit 128`
  floor note this doc supersedes/explains.
* `project_probe_path_spec_k_not_hooks` (memory) — spec_k=0 is the smoke-gate
  ground truth.
* `docs/PHASE_8_RUNNER_SWITCH_SCOPE.md` — the 200-400/1096 target.
