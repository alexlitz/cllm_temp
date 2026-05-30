# Batch 685-821 Status — Complete

## Task
Re-run the 1096 pure-neural diagnostic shard for IDs 685-821 (137 tests covering
recursion + tail-of-functions). Capture `[1096-summary]` and per-id divergence
rows.

## Worktree Setup
- Worktree was originally branched from commit `1bdb721` (Phase 4 BNZ-branch
  fix), which predates the diagnostic test file at
  `c4_release/tests/test_1096_neural_declarative_diagnostic.py`.
- Reset HEAD to the requested base `speedup-cache-and-buckets` @ `4d069f7`
  so the diagnostic test was present.

## Execution
- First attempt with `LIMIT=137 CHUNK=1` ran for the full 60-min timeout under
  heavy multi-agent GPU contention and produced no progress output (the canonical
  recipe does not export `C4_1096_PROGRESS`, so the runner stays silent until
  the final summary). See `diag.log` (truncated, killed by `timeout`).
- Re-ran in two halves with `C4_BATCH_CHUNK=8` and `C4_1096_PROGRESS=1`:
  - `diag_h1.log` — OFFSET=685 LIMIT=64  (ids 685-748), passed in 29:46.
  - `diag_h2.log` — OFFSET=749 LIMIT=73  (ids 749-821), passed in 17:39.

## Combined Result
| half | selected | ok | divergences | errors | suite_mismatches |
| ---- | -------- | -- | ----------- | ------ | ---------------- |
| h1 (685-748) | 64 | 5 | 59 | 0 | 0 |
| h2 (749-821) | 73 | 3 | 70 | 0 | 0 |
| **total**    | **137** | **8** | **129** | **0** | **0** |

`[1096-summary]` lines (from each half):
```
[1096-summary] mode=final-output selected=64 ok=5 divergences=59 errors=0 suite_mismatches=0
[1096-summary] mode=final-output selected=73 ok=3 divergences=70 errors=0 suite_mismatches=0
```

## Passing IDs (8 total)
- h1: `700, 701, 703, 717, 733`
- h2: `785, 787, 818`

## Diverging IDs (129 total)
See per-row `[1096-diag]` entries in `diag_h1.log` and `diag_h2.log`. Every row
has `suite_decl=match` (declarative VM matches expected) but
`status=neural-divergence` (the neural runner produces a different final output
value than the declarative ground truth).

Key clusters observed:
- 685-699: `func_min_*` family — neural=0 vs expected min(a,b).
- 702, 704-716: `func_max_*` family — neural=0/None vs expected max(a,b).
- 718-732: `rec_factorial_*` (skipping 717=pass, 733=pass) — neural=1 vs
  expected factorial value.
- 725-748: `rec_fib_*` family — neural=0 vs expected fibonacci.
- 749-782: `func_inc_*` / `func_dbl_*` / loop bodies — varied neural values.
- 783-817: `expr_add_mul_*` family — neural close-to-correct (off by small
  amounts) for some, None for many. Examples: id=815 neural=315 vs expected=304;
  id=820 neural=195 vs expected=193.

`decl_steps` for the recursive-fib programs ranges up to 8369 VM steps
(`rec_fib_9: fib(12)`), explaining the long wall-time of the recursion-heavy
batches.

## Files
- `diag.log` — original truncated run (60-min timeout, no progress emitted).
- `diag_h1.log` — half-1 full run with progress + summary.
- `diag_h2.log` — half-2 full run with progress + summary.
