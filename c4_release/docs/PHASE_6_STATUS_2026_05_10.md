# Phase 6 IO Pure-Neural Status — 2026-05-10

Status snapshot of `c4_release/tests/test_pure_neural_io.py` after the
`fix-phase6-prtf-args-read-open` integration (commit `334b135`, merged via
`3bfbfbc`).

Tests run on branch `verify-phase6-xpasses` (rebased off the integration
commit) with `CUDA_VISIBLE_DEVICES=0` and per-test timeout 400s.

## Summary

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | TestPureNeuralPutchar::test_putchar_one_char        | XFAIL | No neural OUTPUT path for PUTCHAR in pure_neural; handlers skipped, no autoregressive byte-emit weight. |
| 2 | TestPureNeuralPutchar::test_putchar_two_chars       | XFAIL | Same root cause as #1. |
| 3 | TestPureNeuralPrtf::test_prtf_simple                | XFAIL | PRTF skips `_syscall_prtf`; no neural TOOL_CALL emit + DATA-section walk to externalize the format string. |
| 4 | TestPureNeuralPrtf::test_prtf_with_arg              | XFAIL | Needs neural format-string interpreter + integer-to-decimal emit. |
| 5 | TestPureNeuralGetchar::test_getchar_returns_byte    | XFAIL | GETCHAR neural USER_INPUT attention head not baked; `_inject_getchar` bypassed in pure_neural. |
| 6 | TestPureNeuralFileOps::test_open_close_cycle_xfail  | XFAIL | OPEN/CLOS are pure tool-boundary; no neural filename addressing. |
| 7 | TestPureNeuralRead::test_read_from_stdin            | XFAIL | READ buffer addressing requires neural memory head; both that and neural USER_INPUT consumption missing. |

## Aggregate

- Collected: 7
- XFAIL (expected): 7
- XPASS (test passed despite xfail marker): 0
- FAILED (real failure): 0
- Runtime: 2716.13s (0:45:16)

## Conclusion

The runner-side shims added in commit `334b135`
(`_neural_prtf_emit` with `%d/%s/%c/%x/%%` format support, `_neural_open_emit`,
`_neural_clos_emit`, `_neural_read_emit`) are not sufficient on their own to
make any of the 7 Phase 6 pure-neural IO tests pass: in `pure_neural=True`
mode the `_syscall_handlers` path is skipped entirely, so these runner-side
emit shims never fire. The tests' xfail reasons remain accurate — what is
needed are the neural-side baked weights (OUTPUT path for PUTCHAR/PRTF,
USER_INPUT attention head for GETCHAR/READ, `_set_layer7_memory_heads` for
filename and buffer addressing).

No xfail markers were removed.
