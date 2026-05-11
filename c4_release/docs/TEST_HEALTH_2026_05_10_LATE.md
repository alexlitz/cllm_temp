# Test Health Snapshot — 2026-05-10 (Late)

Status snapshot of all 7 pure-neural phase test files after Wave 1 + Wave 2
compiler migrations landed on `main`. Snapshot run on branch
`docs-test-health-2026-05-10` (worktree off `main` at commit `1b3f802` — "Document Phase 6 pure-neural IO test status").

Compared to the earlier baseline at commit `356e460` (`cleanup-alu-aliases`).

## Setup

- GPU: `CUDA_VISIBLE_DEVICES=0` (idle GPU at start; shared with another concurrent claude session running `test_pure_neural_psh_add.py` during Phase 2 / Phase 3 windows — may have inflated per-test runtimes for those phases).
- Python 3.11.8, pytest 8.4.1, pytest-timeout 2.4.0, torch CUDA 12.6.
- Per-test timeout: 120s for Phase 1, 60s thereafter (tight budget — see Caveats).
- Wall-clock budget: 120 minutes total. Used ~100 minutes.

## Summary

| Phase | File | Collected | Reported P | F | X | XP | E | Completion |
|---|---|---|---|---|---|---|---|---|
| 1 | `test_pure_neural_pc.py` | 13 | 0 | 13 | 0 | 0 | 0 | Complete |
| 2 | `test_pure_neural_psh_add.py` | 16* | 0 | 13 | 0 | 0 | 0 | Partial — pytest INTERNALERROR after 13th timeout (last 3 not reported) |
| 3 | `test_pure_neural_multibyte.py` | 9 | 0 | 0 | 2 | 0 | 0 | Partial — INTERNALERROR after 2nd xfail (last 7 not reported) |
| 4 | `test_pure_neural_jmp_bz.py` | 12 | 0 | 0 | 3 | 0 | 0 | Partial — INTERNALERROR after 3rd xfail (last 9 not reported) |
| 5 | `test_pure_neural_jsr_ent_lev.py` | 7 | 0 | 0 | 7 | 0 | 0 | Complete |
| 6 | `test_pure_neural_io.py` | 7 | 0 | 0 | 3 | 0 | 0 | Partial — INTERNALERROR after 3rd xfail (last 4 not reported) |
| 7 | `test_pure_neural_heap_div.py` | 18† | 0 | 0 | 3 | 0 | 0 | Partial — only `-k "test_mul_small or test_div"` ran; INTERNALERROR after the 3 xfail-marked `test_mul_small` parametrizations, before reaching non-xfail `test_div_small` |

\* Phase 2 collected 16 but only 13 reported before INTERNALERROR.
† Phase 7 -k filter selected 18 of 25 (per pytest "15 deselected" from 25 minus the executed 3); see Phase 7 note below.

Legend: P=Passed, F=Failed, X=XFAIL (expected fail), XP=XPASS, E=Error.

### Reported XPASS

**None observed.** No XFAIL-marked test crossed the line to XPASS in this snapshot.

### Reported real-FAIL outside the expected pattern

- Phase 1: 13/13 timed out at 120s. All in `divmod_longdiv.py` / `efficient_alu_neural.py` / torch tensor paths. Same pattern as earlier baseline (Phase 1 fails are real, not xfails).
- Phase 2: 13/16 timed out at 60s. Pattern matches Phase 1 (compute-bound hangs).
- Phase 7 `test_div_small[*]`: not xfail-marked; first parametrization hung past 60s and triggered the pytest INTERNALERROR that aborted the session.

## Comparison to earlier baseline (commit `356e460`)

The earlier baseline characterized:

- Phase 1: 0P / 5+F / 0X / 0XP → **today: 0P / 13F** (worse coverage of timeouts, same character: all fail/hang).
- Phase 4: 11 xfail + 1 xpass → **today: only 3 XFAIL recorded** before INTERNALERROR. The previously-reported XPASS (likely `test_bz_not_taken[imm]` per the xfail-reason wording) was **not reached** in this run because pytest crashed early. Cannot confirm or deny regression.
- Phase 6 (per dedicated 2026-05-10 doc `PHASE_6_STATUS_2026_05_10.md`): 7 XFAIL / 0 XPASS / 0 FAIL with 400s per-test timeout in 45min wall-clock. **Today's run** caught only 3 XFAIL (`test_putchar_one_char`, `test_putchar_two_chars`, `test_prtf_simple`) before INTERNALERROR; consistent with the prior doc (no XPASS observed), but coverage was incomplete due to budget.

**Bottom line for comparison:** This run could not produce a definitive verdict on improvement vs. regression because pytest's INTERNALERROR aborts the session after the first per-test timeout in `--tb=no` mode (see Caveats). The 5 phases for which a full count was either complete (Phases 1, 5) or partially captured (Phases 2, 3, 4, 6, 7) show the same xfail pattern as the earlier baseline, with **no new XPASSes observed**.

## Per-XPASS recommendation

None observed — no action needed. Confirm the 2026-05-10 Phase 4 baseline XPASS (`test_bz_not_taken`) by running just that test in isolation in a follow-up.

## Per-FAIL note

- **Phase 1 + Phase 2 timeouts:** consistent with the pre-existing reality that IMM/PSH/ADD pure-neural paths hang in `divmod_longdiv.py` due to dispatch hot loops. No new failure mode introduced by the Wave 1/2 compiler migrations.
- **Phase 7 `test_div_small`:** non-xfail-marked, hangs past 60s. Since the same test is not marked xfail and there's no evidence it ever passed, the missing xfail marker is either an oversight or a planned aspirational test. Recommend adding `@pytest.mark.xfail(reason="DIV: not yet wired in pure-neural")` if div is not yet expected to work.

## Caveats

1. **pytest INTERNALERROR pattern:** Multiple phases ended with
   `INTERNALERROR ... TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'`
   originating in `_pytest/_code/code.py:212` (`return self._rawentry.tb_lineno - 1`).
   This triggers after the first per-test timeout when `--tb=no` is set: pytest
   can't generate a traceback for the timeout-killed frame, the report path
   crashes, and the rest of the session aborts. Result: only tests that
   completed (passed/xfailed) before the first timeout get reported.
2. **GPU contention:** Phase 2 and Phase 3 had a concurrent pytest job from
   another claude session sharing GPU 0 (running `test_pure_neural_psh_add.py`
   with `-k "bitwise or and_ or or_ or xor"`). Per-test runtimes here may
   include GPU contention slowdown.
3. **Wall-clock budget:** Phase 1 alone consumed 26 minutes (13 tests * 120s).
   Combined with INTERNALERROR truncation, no individual phase reached its
   full collected test count except Phases 1 and 5.

## Recommendation for next snapshot

- Use `--tb=line` (not `--tb=no`) when timeouts are likely; this lets pytest
  build a traceback header and avoids the INTERNALERROR cascade that aborts
  the session.
- Or, drop `pytest-timeout` and use `--forked` with a wall-clock per-test
  cap so that timeouts don't poison the session-level traceback machinery.
- Run Phase 7 unfiltered (or with a narrower `-k` that only selects xfail-
  marked tests) so a single non-xfail hang doesn't abort the remaining tests.
- Allocate at least 3 hours for a full run with `pure_neural` mode.
