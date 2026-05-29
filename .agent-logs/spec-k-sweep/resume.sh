#!/usr/bin/env bash
# Resume the spec_k sweep by skipping IDs already logged per K.
# Appends to existing k<K>.log files in LOG_DIR.

set -euo pipefail

LOG_DIR="${1:-.agent-logs/spec-k-sweep}"
mkdir -p "${LOG_DIR}"

IDS=(
  5 32 75 150 210 260 310 360 410 425 470 520 555 580 620 660
  700 740 780 810 830 855 880 905 925 955 975 1000 1020 1055 1075 1090
)
SPEC_KS=(0 8 64 128)

TEST_NODE="c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice"

for K in "${SPEC_KS[@]}"; do
  log="${LOG_DIR}/k${K}.log"
  touch "${log}"
  for ID in "${IDS[@]}"; do
    # Skip if already logged
    if grep -qE "^id=${ID} " "${log}" 2>/dev/null; then
      continue
    fi
    line=$(
      C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 \
      C4_1096_OFFSET="${ID}" C4_1096_LIMIT=1 \
      C4_1096_TRACE_FAILURES=0 C4_BATCH_CHUNK=1 \
      C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K="${K}" C4_BATCH_USE_KV_CACHE=0 \
      PYTHONPATH=.:c4_release \
      timeout 180 python -m pytest -q -s "${TEST_NODE}" --tb=no 2>&1 \
        | grep "1096-summary" || echo "[1096-summary] mode=final-output selected=0 ok=ERR divergences=0 errors=1 suite_mismatches=0"
    )
    printf 'id=%s %s\n' "${ID}" "${line}" >> "${log}"
  done
done
