#!/usr/bin/env bash
# Run the 1096 declarations-only neural diagnostic as fast pass-rate shards.
#
# Defaults prioritize throughput over detailed diagnostics:
#   - large speculation (C4_SPEC_K=128)
#   - moderate per-pytest chunking (C4_BATCH_CHUNK=16)
#   - KV cache enabled, forced incremental KV, KV verification disabled
#   - residual/band tracing disabled
#   - optional sorting by symbolic step count to reduce padded batch work
#
# By default this reports final-output pass/fail.  Set
# C4_SPEC_FAIL_ON_CORRECTION=1 to run the stricter first-safe-token divergence
# mode; those counts are intentionally reported under a separate mode label.
#
# Use the slower run_1096_diag_chunk.sh for traced failure investigation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TOTAL="${C4_FAST_TOTAL:-1096}"
SHARD_SIZE="${C4_FAST_SHARD_SIZE:-137}"
OUT_DIR="${1:-${C4_FAST_OUT_DIR:-.agent-logs/fast-shards}}"
GPU_LIST="${C4_FAST_GPUS:-0 1}"

mkdir -p "${OUT_DIR}"

read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "C4_FAST_GPUS must name at least one GPU" >&2
  exit 2
fi

run_shard() {
  local gpu="$1"
  local offset="$2"
  local limit="$3"
  local mode
  mode="$(comparison_mode)"
  local log="${OUT_DIR}/gpu${gpu}_${offset}_${limit}_${mode}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:--p no:cacheprovider}" \
  C4_1096_DIAG=1 \
  C4_1096_OFFSET="${offset}" \
  C4_1096_LIMIT="${limit}" \
  C4_BATCH_CHUNK="${C4_BATCH_CHUNK:-16}" \
  C4_DECLARATIONS_ONLY_BAKE="${C4_DECLARATIONS_ONLY_BAKE:-1}" \
  C4_SPEC_K="${C4_SPEC_K:-128}" \
  C4_BATCH_CONTEXT_WINDOW="${C4_BATCH_CONTEXT_WINDOW:-512}" \
  C4_BATCH_MODEL_MAX_SEQ_LEN="${C4_BATCH_MODEL_MAX_SEQ_LEN:-4096}" \
  C4_BATCH_USE_KV_CACHE="${C4_BATCH_USE_KV_CACHE:-1}" \
  C4_BATCH_FORCE_INCREMENTAL_KV="${C4_BATCH_FORCE_INCREMENTAL_KV:-1}" \
  C4_BATCH_KV_VERIFY="${C4_BATCH_KV_VERIFY:-0}" \
  C4_1096_TRACE_FAILURES="${C4_1096_TRACE_FAILURES:-0}" \
  C4_1096_DIAG_ASSERT="${C4_1096_DIAG_ASSERT:-0}" \
  C4_1096_BAND_PROJECTION_DIAG="${C4_1096_BAND_PROJECTION_DIAG:-0}" \
  C4_1096_SORT_BY_STEPS="${C4_1096_SORT_BY_STEPS:-1}" \
  C4_1096_PROGRESS="${C4_1096_PROGRESS:-1}" \
  python -m pytest -q \
    c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice \
    -s > "${log}" 2>&1
}

comparison_mode() {
  case "${C4_SPEC_FAIL_ON_CORRECTION:-}" in
    1|true|TRUE|yes|YES|on|ON) echo "strict-first-safe-token" ;;
    *) echo "final-output" ;;
  esac
}

summary_field() {
  local log="$1"
  local key="$2"
  local line
  line="$(rg '^\[1096-summary\]' "${log}" | tail -1 || true)"
  if [[ -z "${line}" ]]; then
    echo "0"
    return
  fi
  awk -v key="${key}" '{
    for (i = 1; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == key) {
        print kv[2]
        exit
      }
    }
    print "0"
  }' <<< "${line}"
}

worker() {
  local worker_idx="$1"
  local gpu="${GPUS[worker_idx]}"
  local offset="${worker_idx}"
  offset=$((offset * SHARD_SIZE))
  local stride=$((${#GPUS[@]} * SHARD_SIZE))

  while [[ "${offset}" -lt "${TOTAL}" ]]; do
    local remaining=$((TOTAL - offset))
    local limit="${SHARD_SIZE}"
    if [[ "${remaining}" -lt "${limit}" ]]; then
      limit="${remaining}"
    fi
    echo "RUN gpu=${gpu} offset=${offset} limit=${limit}"
    run_shard "${gpu}" "${offset}" "${limit}"
    local mode
    mode="$(comparison_mode)"
    local log="${OUT_DIR}/gpu${gpu}_${offset}_${limit}_${mode}.log"
    local selected ok divergences errors suite_mismatches
    selected="$(summary_field "${log}" selected)"
    ok="$(summary_field "${log}" ok)"
    divergences="$(summary_field "${log}" divergences)"
    errors="$(summary_field "${log}" errors)"
    suite_mismatches="$(summary_field "${log}" suite_mismatches)"
    local runtime
    runtime="$(rg 'passed in' "${log}" | tail -1 || true)"
    echo "DONE gpu=${gpu} offset=${offset} limit=${limit} mode=${mode} selected=${selected} ok=${ok} divergences=${divergences} errors=${errors} suite_mismatches=${suite_mismatches} ${runtime}"
    offset=$((offset + stride))
  done
}

pids=()
for idx in "${!GPUS[@]}"; do
  worker "${idx}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

completed=0
passed=0
divergences=0
errors=0
suite_mismatches=0
mode="$(comparison_mode)"
for log in "${OUT_DIR}"/gpu*_*_*_"${mode}".log; do
  [[ -f "${log}" ]] || continue
  if rg -q 'passed in' "${log}"; then
    selected="$(summary_field "${log}" selected)"
    ok="$(summary_field "${log}" ok)"
    div="$(summary_field "${log}" divergences)"
    err="$(summary_field "${log}" errors)"
    suite="$(summary_field "${log}" suite_mismatches)"
    completed=$((completed + selected))
    passed=$((passed + ok))
    divergences=$((divergences + div))
    errors=$((errors + err))
    suite_mismatches=$((suite_mismatches + suite))
  fi
done
echo "SUMMARY mode=${mode} completed=${completed}/${TOTAL} ok=${passed} divergences=${divergences} errors=${errors} suite_mismatches=${suite_mismatches}"

exit "${status}"
