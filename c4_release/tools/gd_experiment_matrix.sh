#!/usr/bin/env bash
# Orchestrates the GD stability/effectiveness experiment matrix.
# Each run rebuilds a FRESH in-memory copy of the hand-built VM (no disk cache,
# no production weights touched) and trains it under AdamW. Results -> JSON.
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash tools/gd_experiment_matrix.sh <outdir>
#
# Sized for a SHARED 24 GB GPU: ~96-program balanced sample, batch 4, chunked
# reporting + eval forwards. The per-record full_trace eval over the sample is
# the wall-clock bottleneck, so steps/eval-every are tuned so the (fast) GD
# collapse is densely sampled early without paying for many late evals.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/gd_results}"
mkdir -p "$OUT"

# ~96-program balanced sample covering all clusters:
#   PASSING-heavy: add(12) div(12) mod(12) if_eq(12) if_lt(8)
#   FAILING:       var_simple(12) expr_mul_div(12) func_identity(12) edge(13)
SAMPLE="0-11,150-161,200-211,400-411,375-382,250-261,850-861,550-561,1031-1043"

COMMON="--batch 4 --loss-chunk 4 --eval-chunk 8 --trainable all \
        --supervision pc_ax --max-oracle-steps 60"

run () {
  local label="$1"; shift
  local out="$OUT/${label}.json"
  if [[ -f "$out" ]]; then
    echo "[matrix] SKIP $label (exists)"; return 0
  fi
  echo "[matrix] === RUN $label ==="
  python tools/gd_experiment_run.py --output "$out" --label "$label" "$@" \
    2>"$OUT/${label}.log" || echo "[matrix] $label FAILED (see ${label}.log)"
  echo "[matrix] done $label"
}

# --- Experiment 1: LR grid (stability + effectiveness) -----------------------
# 150 steps, eval every 25 -> points {0,25,50,75,100,125,150}. The collapse (if
# any) shows in the first 1-2 evals; 150 steps is ample headroom to see whether
# a stable LR ALSO learns any failing program.
run "exp1_lr1e-6" --ids "$SAMPLE" --lr 1e-6 --steps 150 --eval-every 25 $COMMON
run "exp1_lr1e-5" --ids "$SAMPLE" --lr 1e-5 --steps 150 --eval-every 25 $COMMON
run "exp1_lr1e-4" --ids "$SAMPLE" --lr 1e-4 --steps 150 --eval-every 25 $COMMON

# --- Forgetting curve tail: ultra-low LR (is there ANY stable+effective LR?) --
run "exp3_lr1e-7" --ids "$SAMPLE" --lr 1e-7 --steps 150 --eval-every 25 $COMMON

# --- Raw fragility: no stabilization (expected: NaN by step 1) ----------------
run "exp0_nostab_lr1e-6" --ids "0-9,1031-1040" --lr 1e-6 --steps 12 \
    --eval-every 3 --batch 4 --loss-chunk 4 --eval-chunk 8 --trainable all \
    --supervision pc_ax --max-oracle-steps 40 --no-stabilize

echo "[matrix] MATRIX COMPLETE"
