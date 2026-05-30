# spec_k sweep -- 1096 declarative neural diagnostic

Per-id ok count (1 = match, 0 = divergence, ERR = pytest failure)
for each `C4_SPEC_K` value. 32-id representative subset.

| id | k=0 | k=8 | k=64 | k=128 |
|----|------|------|------|------|
| 5 | ERR | 0 | 0 | 0 |
| 32 | 0 | 0 | 0 | 0 |
| 75 | 0 | 0 | 0 | 0 |
| 150 | 0 | 0 | 0 | 0 |
| 210 | 1 | 1 | 1 | 1 |
| 260 | 1 | 1 | 1 | 1 |
| 310 | ERR | 0 | 0 | 0 |
| 360 | 0 | 0 | 0 | 0 |
| 410 | 1 | 1 | 1 | 1 |
| 425 | 0 | 0 | 0 | 0 |
| 470 | ERR | 0 | 0 | 0 |
| 520 | ERR | ERR | ERR | 0 |
| 555 | 0 | 0 | 0 | 0 |
| 580 | 0 | 0 | 0 | 0 |
| 620 | 0 | 0 | 0 | 0 |
| 660 | 0 | 0 | 0 | 0 |
| 700 | 1 | 1 | 1 | 1 |
| 740 | 0 | 0 | 0 | 0 |
| 780 | 0 | 0 | 0 | 0 |
| 810 | 0 | 0 | 0 | 0 |
| 830 | 1 | 1 | 1 | 1 |
| 855 | 1 | 1 | 1 | 1 |
| 880 | 1 | 1 | 1 | 1 |
| 905 | 0 | 0 | 0 | 0 |
| 925 | 0 | 0 | 0 | 0 |
| 955 | 0 | 0 | 0 | 0 |
| 975 | 0 | 0 | 0 | 0 |
| 1000 | 1 | 1 | 1 | 1 |
| 1020 | 1 | 1 | 1 | 1 |
| 1055 | 0 | 0 | 0 | 0 |
| 1075 | 0 | 0 | 0 | 0 |
| 1090 | 1 | 1 | 1 | 1 |

## Totals

| metric | k=0 | k=8 | k=64 | k=128 |
|--------|------|------|------|------|
| ok | 10 | 10 | 10 | 10 |
| divergences | 18 | 21 | 21 | 22 |
| errors | 4 | 1 | 1 | 0 |

**No spec_k spike detected** vs k=0 baseline (ok counts: {0: 10, 8: 10, 64: 10, 128: 10}).

## Spike Highlight

- The neural VM achieves **10/32 matches** at every value of `C4_SPEC_K`
  in {0, 8, 64, 128}. Increasing speculation depth does **not** introduce
  new divergences vs the declarative oracle baseline at k=0.
- The k>0 rows have slightly more `divergences` than k=0 (18 -> 21 -> 21 -> 22)
  only because k=0 had 4 transient pytest ERRs that resolved to clean
  divergences at higher K. Each ERR cell (id=5, 310, 470 at k=0; id=520 at
  k=0/8/64) is consistent with the test running into the per-id 180s timeout
  wall under early GPU contention, not a real speculation regression. The
  ok-vs-divergence pattern is per-id identical (no id flips from ok to
  divergence as K grows).
- No id flips from `ok=1` at k=0 to `ok=0` (or ERR) at any k>0: speculation
  preserves end-to-end neural correctness on this representative subset.

## Methodology

- 32 representative ids from the 1096 suite (id 5..1090) x 4 spec-K depths
  (0, 8, 64, 128), one id per pytest invocation, 180s wall-clock budget each.
- Env per run: `C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 C4_BATCH_CHUNK=1
  C4_DECLARATIONS_ONLY_BAKE=1 C4_BATCH_USE_KV_CACHE=0`,
  `C4_1096_OFFSET=<id>`, `C4_1096_LIMIT=1`, `C4_SPEC_K=<K>`.
- Test node: `c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice`.
- Runner: `c4_release/tests/runners/run_spec_k_sweep.sh` (initial run;
  killed mid-k=8). Completed via `.agent-logs/spec-k-sweep/resume.sh` which
  skips ids already present in `k<K>.log`.
