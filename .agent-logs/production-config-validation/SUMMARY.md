# B5-H Production Config Validation -- integration/batch-merge-full @ 5726cd3

## Scope

Re-run pre-batch fast-shards recipe (production config -- handlers + speculation + KV cache enabled, NO C4_DECLARATIONS_ONLY_BAKE=1) against integration/batch-merge-full HEAD 5726cd3.

Test entry: c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice

Env: C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 C4_BATCH_CHUNK=8 C4_1096_PROGRESS=1. Each shard limited by timeout 5400 (90 min).

Run started: 2026-05-30 10:19 EDT, all-done: 2026-05-30 18:25 EDT (~8 hours, with restarts due to harness kills + heavy concurrent sub-agent GPU contention).

## Per-shard results

| Shard | Pre-batch baseline (fast-shards-20260527-141401) | Current (integration/batch-merge-full) | Delta |
|---|---|---|---|
| 0-136 | 131/137 (96%) | 36/137 (26%) | **-95 REGRESSION** |
| 137-273 | 113/137 (82%) | 74/137 (54%) | **-39 REGRESSION** |
| 274-410 | 31/137 (23%) | 53/137 (39%) | **+22 improvement** |
| 411-547 | 5/137 (4%) | killed at 40/137 partial (no summary) | indeterminate |
| 548-684 | 9/137 (7%) | 16/137 (12%) | **+7 improvement** |
| 685-821 | never completed | 7/137 (5%) | new data |
| 822-958 | 56/137 (41%) | killed at 88/137 partial (no summary) | indeterminate |
| 959-1095 | never run | 33/137 (24%) | new data |

### Totals for comparable completed shards (0-136, 137-273, 274-410, 548-684)

| | Baseline | Current | Delta |
|---|---|---|---|
| Tests passing | 284/548 (52%) | 179/548 (33%) | **-105** |

Compared on the 4 shards where BOTH baseline and current produced complete `1096-summary` lines.

### Killed shards detail

- **411-547**: Both attempts killed by harness (rc=137 SIGKILL on first; rc=137 on second). Got to ~40/137 rows before kill. Heavy `max_steps=451` batches caused per-batch timing to exceed harness patience. Cannot compare to baseline 5/137.
- **822-958**: First run timed out (rc=124) at ~88/137 (timeout 5400s = 90 min). Heavy `max_steps=175/227+` batches near end of shard. Cannot directly compare to baseline 56/137 -- partial passed-row count was not captured because the test never emitted the `1096-summary` line before being killed mid-batch.

## Key observations

1. **Severe regression on shard 0-136 (-95 tests)**: Baseline 131/137 -> integration 36/137. This is the cleanest signal of a production-handler-path regression. The early/easy tests that the handler+speculation path covered at 96% no longer pass at integration HEAD. The handler dispatch on shard 0-136 needs investigation -- the changes in this batch (batched KV eviction, declarative lowering, etc.) appear to have either bypassed the handler fast-path or broken its semantics.

2. **Regression on shard 137-273 (-39 tests)**: Same pattern as 0-136 but smaller magnitude. Likely same root cause -- handler-fast-path coverage degraded.

3. **Improvements where declarative lowering helps (+22 on 274-410, +7 on 548-684)**: These are bands where B2-H declarative lowering fixes actively help. This matches the prior batch reports.

4. **New coverage on shards 685-821 and 959-1095 that never previously completed**: 7/137 and 33/137 respectively. These give us new visibility into the back half of 1096 even if the totals are modest.

5. **Two shards killed without summaries** (411-547, 822-958): The timeout 5400 is insufficient when a shard contains batches with `max_steps >= 150`. The 822 shard was actually making progress (88/137 with no diag line summary captured because the summary is emitted only at full-shard completion). The 411 shard hit memory/resource pressure and was SIGKILLed. To resolve these would require either (a) running with a longer timeout (3+ hours per heavy shard) or (b) reducing C4_BATCH_CHUNK and accepting longer total wall time but smaller batches that fit in memory.

## Verdict

**REGRESSION: production -105 tests on the 4 directly-comparable shards (0-136, 137-273, 274-410, 548-684); investigate handler dispatch on shard 0-136.**

Even though the +22 and +7 improvements on shards 274-410 and 548-684 confirm B2-H declarative-lowering work helps, the much larger -95 loss on shard 0-136 (and -39 on 137-273) demonstrates that the production handler fast-path that covered 96% of the easy/early tests has been broken or bypassed in this integration. The handler-dispatch path on shard 0-136 is the highest-priority investigation target.

## Files

- `.agent-logs/production-config-validation/shard_0_136.log`     -- completed (ok=36)
- `.agent-logs/production-config-validation/shard_137_273.log`   -- completed (ok=74)
- `.agent-logs/production-config-validation/shard_274_410.log`   -- completed (ok=53)
- `.agent-logs/production-config-validation/shard_411_547.log`   -- killed (partial, no summary)
- `.agent-logs/production-config-validation/shard_548_684.log`   -- completed (ok=16)
- `.agent-logs/production-config-validation/shard_685_821.log`   -- completed (ok=7)
- `.agent-logs/production-config-validation/shard_822_958.log`   -- timed out (88/137 partial, no summary)
- `.agent-logs/production-config-validation/shard_959_1095.log`  -- completed (ok=33)
- `.agent-logs/production-config-validation/run.log`             -- orchestrator log

## Run scripts

- `run_production_shards.sh`        -- initial attempt (killed mid-run)
- `run_remaining_shards.sh`         -- second attempt (killed mid-run)
- `run_remaining_shards2.sh`        -- third attempt (nohup-detached; completed)
