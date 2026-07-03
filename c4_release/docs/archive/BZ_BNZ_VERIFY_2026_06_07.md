# BZ/BNZ Verify — 2026-06-07

Verification of whether `763ff75c` (`fix(removal-bz-bnz): L4 FFN PC
carry-forward model-side replaces 44709e13`) fixes the 21-test
BZ/BNZ + control-flow failure cluster identified in
[`TEST_CATEGORY_SWEEP_2026_06_06.md`](TEST_CATEGORY_SWEEP_2026_06_06.md).

Base: `main` HEAD `77ed77f0` (`cache(sparse): COO sidecar storage`),
which includes `763ff75c`.

## TL;DR

The BZ/BNZ override-removal commit **does not fix the 21-test
cluster**. Test counts are byte-identical to the sweep baseline.

| | Pass | Fail | xFail | Total |
|---|---:|---:|---:|---:|
| Sweep baseline (78f5433a)               | 1 | 21 | 6 | 28 |
| After `763ff75c` (77ed77f0)             | 1 | 21 | 6 | 28 |
| Delta                                   | 0 | 0  | 0 | 0  |

Wall time: 896.74 s (CPU; CUDA OOM on this host).

## Run

```
CUDA_VISIBLE_DEVICES="" pytest \
    tests/test_bz_bnz_neural.py tests/test_control_flow_neural.py \
    --tb=line -q --timeout=180
```

Result: `21 failed, 1 passed, 6 xfailed, 3 warnings`.

## Remaining failures

### test_bz_bnz_neural.py — 13 failed (unchanged)

All fail with the same pattern the sweep flagged: `assert 0 == 42`
(BZ/BNZ does not reach the IMM-42 target), or `assert 1 == 42` for
edge cases where AX ends up at 1 instead.

- `TestBZWithHandler::test_bz_{zero_branches, nonzero_falls_through, with_arithmetic_result}` — `assert 0 == 42`
- `TestBNZWithHandler::test_bnz_{nonzero_branches, zero_falls_through, large_value}` — `assert 0 == 42`
- `TestBranchLoops::test_countdown_loop_bnz` — pytest-timeout 180 s (`vm_step.py:551`); infinite loop because BNZ never falls through
- `TestBranchLoops::test_search_loop_bz` — `assert 0 == 42`
- `TestBranchEdgeCases::test_bz_at_boundary` — pytest-timeout 180 s (`vm_step.py:503`)
- `TestBranchEdgeCases::test_bnz_negative_like` — `assert 0 == 42`
- `TestBranchEdgeCases::test_bz_comparison_result` — `assert 1 == 42`
- `TestBranchEdgeCases::test_bnz_comparison_result` — `assert 1 == 42`
- `TestCombinedBranches::test_bz_bnz_sequence` — `assert 1 == 42`
- `TestCombinedBranches::test_nested_branches` — `assert 0 == 42`

xfailed (unchanged): 2 cases on the known negative-value path; not in
the 21 cluster.

### test_control_flow_neural.py — 8 failed (unchanged)

- `TestJMPBasic::test_jmp_step1_works` — `JMP should skip to IMM 42, got 0`
- `TestBZBasic::test_bz_{zero_branches, nonzero_continues}` — `BZ should branch/NOT branch ... got 0`
- `TestBNZBasic::test_bnz_{nonzero_branches, zero_continues}` — `BNZ should branch/NOT branch ... got 0`
- `TestLoopPatterns::test_simple_while_loop` — `Expected sum=10, got 65504` (negative wrap)
- `TestLoopPatterns::test_if_else` — `Expected 42 (condition true), got 10`
- `TestLoopPatterns::test_if_else_false` — `Expected 99 (condition false), got 3`

All three `TestLoopPatterns` cases emit
`AutoregressiveVMRunner.run: divergence detected — draft_divergence:
25 of last 50 tokens disagree with DraftVM` before bailing.

## Why the override removal didn't help

`763ff75c`'s commit message claims:

> Verified: smoke 45/51 passing (same as baseline), with
> `test_bz_branch` and `test_bnz_branch` now green via the model-side
> fix alone.

That covers two **smoke** tests
(`tests/test_smoke.py::test_bz_branch`, `test_bnz_branch`), not the
broader `test_bz_bnz_neural.py` / `test_control_flow_neural.py`
suites. The L4 FFN PC carry-forward fix in `l6_ops.py` evidently
handles the smoke single-step BZ/BNZ case but does not extend to the
multi-step branching / loop / nested-branch scenarios these two suites
exercise.

The `JMP should skip to IMM 42, got 0` failure on
`test_control_flow_neural.py::TestJMPBasic::test_jmp_step1_works`
is particularly informative: JMP was previously cited (in the
`763ff75c` commit message) as "works correctly through a separate L6
first-step relay path." That separate path either regresses under the
test-suite harness or has its own gap that the smoke suite doesn't
exercise.

## Root-cause overlap with Removal 2

The remaining cluster is **not** the same surface as Removal 1
(IMM AX literal override, fixed in `66638578`) or Removal 2 (L7 K-side
attention head, see
[`REMOVAL_2_DEEP_DIVE_2026_06_06.md`](REMOVAL_2_DEEP_DIVE_2026_06_06.md)).
Both Removal 1 and Removal 2 surface as IMM/ALU drift; the cluster
here is purely the BZ/BNZ/JMP branch-target emission path, which lives
in L4/L6 FFN PC carry-forward.

Hypotheses for the 21-test gap (not investigated; out of scope for
this verification):

1. The new `_append_pc_byte0_imm_to_byte_addr_rules` lowering helper
   in `l6_ops.py` only covers the **first** branch in a program. Once
   the VM has executed any prior step, the helper's input source may
   no longer be `FETCH_LO+k` but a different residual slot.
2. The L6 first-step relay path that handles JMP works for smoke
   (`test_jmp` is currently green) but fails under
   `TestJMPBasic::test_jmp_step1_works`'s harness — different
   `max_steps` / `data` shape may exercise a different code path.
3. The three `TestLoopPatterns` cases all emit `draft_divergence: 25
   of last 50 tokens disagree with DraftVM` — the DraftVM disagreement
   is a separate symptom that may share the BZ/BNZ root cause but
   needs a different fix surface (DraftVM PC update path).

## Next step

This cluster needs its own deep-dive. The
`OVERRIDE_REMOVAL_STATUS_2026_06_06.md` matrix marks the BZ/BNZ
override as a 2-test cost (only the smoke-level branch tests), which
is consistent with this finding: the override was masking a smoke-only
gap, not the broader test suite's gap. A follow-up agent should:

1. Run `tests/test_bz_bnz_neural.py::TestBZWithHandler::test_bz_zero_branches`
   under `decl_verifier.py` with the L4 FFN PC carry-forward trace
   enabled, to localize whether the byte-addr conversion is firing on
   the right step.
2. Compare with the smoke `test_bz_branch` trace to identify which
   condition differs.
3. Check `TestLoopPatterns::test_simple_while_loop`'s draft divergence:
   if the DraftVM PC update path also needs the byte-addr conversion,
   that's a second site to fix.
