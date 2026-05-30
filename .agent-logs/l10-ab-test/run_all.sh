#!/bin/bash
set -u
cd /home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-a3a5056f42a23ba07

LOGDIR=.agent-logs/l10-ab-test
PROGRESS="$LOGDIR/_progress.log"
: > "$PROGRESS"

log_progress() {
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROGRESS"
}

run_smoke() {
  local variant=$1
  local outfile="$LOGDIR/${variant}.log"
  : > "$outfile"
  local head_sha=$(git rev-parse HEAD | cut -c1-7)
  echo "=== $variant on $head_sha ===" >> "$outfile"
  log_progress "BEGIN variant=$variant head=$head_sha"
  for spec in "0 32" "200 50" "425 25" "550 25" "700 25"; do
    read -r off lim <<< "$spec"
    echo "--- offset=$off limit=$lim ---" >> "$outfile"
    log_progress "  run variant=$variant offset=$off limit=$lim"
    local start=$(date +%s)
    set +e
    C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 \
      C4_1096_OFFSET=$off C4_1096_LIMIT=$lim \
      C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K=0 C4_BATCH_USE_KV_CACHE=0 \
      C4_BATCH_CHUNK=8 C4_1096_PROGRESS=1 \
      PYTHONPATH=.:c4_release PYTHONUNBUFFERED=1 \
      timeout 2400 python -u -m pytest -q -s \
      c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice \
      --tb=no > "$LOGDIR/raw_${variant}_${off}_${lim}.out" 2>&1
    local rc=$?
    set -e
    local end=$(date +%s)
    local dur=$((end - start))
    if [ $rc -eq 124 ]; then
      echo "[1096-summary] TIMEOUT after ${dur}s" >> "$outfile"
      log_progress "  TIMEOUT variant=$variant offset=$off limit=$lim dur=${dur}s"
    else
      grep "1096-summary" "$LOGDIR/raw_${variant}_${off}_${lim}.out" >> "$outfile" || echo "[1096-summary] NO_OUTPUT rc=$rc dur=${dur}s" >> "$outfile"
      log_progress "  done variant=$variant offset=$off limit=$lim dur=${dur}s rc=$rc"
    fi
  done
  log_progress "END variant=$variant"
}

log_progress "STARTING all variants"

# 1. Baseline (no fix)
git reset --hard origin/speedup-cache-and-buckets >> "$PROGRESS" 2>&1
run_smoke baseline

# 2. B2-A
git reset --hard origin/speedup-cache-and-buckets >> "$PROGRESS" 2>&1
git fetch origin fix/l10-ent-discrimination >> "$PROGRESS" 2>&1
git cherry-pick FETCH_HEAD >> "$PROGRESS" 2>&1
run_smoke b2a

# 3. B3-delta
git reset --hard origin/speedup-cache-and-buckets >> "$PROGRESS" 2>&1
git fetch origin fix/rec-premature-exit >> "$PROGRESS" 2>&1
git cherry-pick FETCH_HEAD >> "$PROGRESS" 2>&1
run_smoke b3d

# 4. B4-A
git reset --hard origin/speedup-cache-and-buckets >> "$PROGRESS" 2>&1
git fetch origin fix/l10-e0-coordinated >> "$PROGRESS" 2>&1
git cherry-pick FETCH_HEAD >> "$PROGRESS" 2>&1
run_smoke b4a

log_progress "ALL VARIANTS COMPLETE"
echo "DONE" > "$LOGDIR/_complete.marker"
