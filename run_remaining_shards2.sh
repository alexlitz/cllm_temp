#!/bin/bash
set -u
# Use nohup-style decoupling to survive parent kill
cd /home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-a57fe05c0b3dd3619

for spec in "274 137" "411 137" "548 137" "685 137" "822 137" "959 137"; do
  read -r off lim <<< "$spec"
  end=$((off + lim - 1))
  log=".agent-logs/production-config-validation/shard_${off}_${end}.log"
  echo "[$(date -Iseconds)] Starting shard $off-$end (rerun2)" >> .agent-logs/production-config-validation/run.log
  C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 \
    C4_1096_OFFSET=$off C4_1096_LIMIT=$lim \
    C4_BATCH_CHUNK=8 C4_1096_PROGRESS=1 \
    PYTHONPATH=.:c4_release PYTHONUNBUFFERED=1 \
    timeout 5400 python -u -m pytest -q -s \
    c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice \
    --tb=no > "$log" 2>&1
  rc=$?
  summary=$(grep "1096-summary" "$log" | tail -1)
  echo "[$(date -Iseconds)] Done shard $off-$end rc=$rc summary=$summary" >> .agent-logs/production-config-validation/run.log
done

echo "[$(date -Iseconds)] ALL DONE (rerun2)" >> .agent-logs/production-config-validation/run.log
