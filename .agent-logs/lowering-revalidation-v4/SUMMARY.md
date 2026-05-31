# Lowering Revalidation v4 (B7-9)

Re-run of the 1096 teacher-forced lowering audit on
`integration/batch-merge-full-v4` (HEAD `10ab316`), covering all four 274-row
slices that span the full 1096-row corpus.

## Run parameters

All four slices were invoked with:

- `C4_1096_LOWERING_AUDIT=1`
- `C4_1096_LOWERING_ASSERT_MODE=off`
- `C4_1096_LOWERING_PRINT_MODE=drift`
- `C4_1096_LOWERING_MIN_MARGIN=0.5`
- `C4_1096_LOWERING_OUTPUT_BAND_MIN_MARGIN=0.5`
- `C4_1096_LOWERING_MAX_FAILURES=64`
- `C4_1096_LOWERING_DETAIL_LIMIT=64`
- `C4_1096_LOWERING_MAX_TRACE_TOKENS=40000`
- `C4_DECLARATIONS_ONLY_BAKE=1`

Logs live under `.agent-logs/lowering-revalidation-v4/`. Aggregated per-slice
summaries produced via `aggregate_lowering_audit.py` (copied from the
2026-05-27 lowering-audit run) are written next to each raw log.

## Per-slice summaries

| Slice | Selected | Fatal rows | Info-only | Fatal failures | Info failures | Wrong-token failures | Errors | Skipped | Runtime |
|------|---------:|-----------:|----------:|---------------:|--------------:|---------------------:|-------:|--------:|--------:|
| 0-273    | 274 | 256 | 18 |  1,508 |   318 |  1,826 | 0 | 0 | 1h01m |
| 274-547  | 274 | 273 |  1 |  6,861 | 4,440 | 11,287 | 0 | 0 | 1h33m |
| 548-821  | 274 | 266 |  0 | 10,170 | 5,516 | 15,681 | 0 | 8 | 1h26m |
| 822-1095 | 274 | 274 |  0 |  8,349 | 3,958 | 12,299 | 0 | 0 | 0h27m |
| **Total**| 1096 | 1069 | 19 | 26,888 | 14,232 | 41,093 | 0 | 8 |  |

The first re-run of slice 274-547 was killed by the 7200s timeout because of
parallel-agent GPU contention, so it was re-launched with `timeout 14400` and
completed in 5634s. The succeeded log is the one summarized above.

## Aggregated first-fatal-slot histogram (v4 vs 2026-05-27 baselines)

The baselines come from the pre-batch (2026-05-27) lowering-audit run; the
worktree only contains slices 0-273, 274-547, and 548-821 for that run, so the
"baseline" numbers below already exclude slice 822-1095. The post-batch v4 run
covers all four slices, including the new slice 822-1095. For an apples-to-apples
comparison we report both the full-corpus v4 totals **and** the slice
0-821-only v4 totals.

| Fatal slot | Baseline (2026-05-27, 822 rows) | v4 0-821 | v4 0-1095 |
|---|---:|---:|---:|
| SP_byte0     | 461 | 677 | 893 |
| STACK0_byte0 | 198 |  21 |  22 |
| AX_byte0     | 121 |  29 |  39 |
| STACK0_byte2 |   - |  25 |  25 |
| PC_byte0     |   - |  42 |  42 |
| PC_byte1     |   - |   0 |  46 |
| AX_byte1     |   - |   1 |   2 |

### Per-slice first-fatal-slot breakdown (v4)

```
slice 0-273:
  SP_byte0:225  AX_byte0:24  STACK0_byte0:6  AX_byte1:1

slice 274-547:
  SP_byte0:229  STACK0_byte2:25  STACK0_byte0:15  AX_byte0:4

slice 548-821:
  SP_byte0:223  PC_byte0:42  AX_byte0:1

slice 822-1095:
  SP_byte0:216  PC_byte1:46  AX_byte0:10  AX_byte1:1  STACK0_byte0:1
```

### First-info-slot histogram (v4 totals across 0-1095)

```
MEM_addr1:544  MEM_addr0:489
```

## Key observations

- **STACK0_byte0 collapsed from 198 fatals to 22** across the whole corpus
  (an 11x reduction). On slices 0-821 only it drops from 198 to 21 (a 9x
  reduction). The remaining 22 are scattered across `var_simple`,
  `var_update`, `if_var`, `loop_countdown`, and `edge_loop_never`.
- **AX_byte0 collapsed from 121 fatals to 39** across the whole corpus, and
  from 121 to 29 on the comparable 0-821 slices (a 4x reduction). Most of the
  residual AX_byte0 fatals are in `expr_mod`, `gcd`, `if_*`, and `edge_*`
  rows; the long-running `add/sub/mul/div` train no longer dominates AX_byte0
  failures.
- **SP_byte0 has grown** from 461 to 893 (corpus-wide) or 677 on the matching
  0-821 slices. The growth concentrates in two new failure modes that did
  not exist (or were masked) in the pre-batch run:
  - `step1:STACK0_byte2` (25 cases) — exclusive to `var_three_*` rows in
    slice 274-547.
  - `step2:SP_byte0` (200 cases in slice 548-821, 75 cases in 822-1095)
    — almost all `func_*`, `rec_*`, `nested_*`, and `absdiff_*` rows now
    first diverge on the SP byte0 lane at step 2 rather than later. This
    suggests the structural-dim fixes have exposed an SP-lane lowering issue
    that was previously masked by earlier STACK0/AX failures.
- The four added/changed buckets (`STACK0_byte2`, `PC_byte0`, `PC_byte1`,
  `AX_byte1`) have not been observed in the baseline run; together they
  account for 115 fatals concentrated in three families
  (`gcd`, `rec_fib`, `rec_power`, and `var_three`).
- 1069 of 1096 rows still produce at least one fatal failure; only 27 rows
  are info-only/clean (19 info-only + 8 skipped/errored).

## Files

- `slice_0_273.log`, `slice_274_547.log`, `slice_548_821.log`,
  `slice_822_1095.log` — raw audit logs.
- `aggregate_0_273.txt`, `aggregate_274_547.txt`,
  `aggregate_548_821.txt`, `aggregate_822_1095.txt` — per-slice structured
  aggregates produced via `aggregate_lowering_audit.py`.
