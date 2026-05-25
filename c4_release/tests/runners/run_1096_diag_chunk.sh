#!/usr/bin/env bash
# Run a single 1096 declarations-only neural diagnostic chunk.
#
# Usage:
#   tests/runners/run_1096_diag_chunk.sh <OFFSET> <LIMIT> [LOG_PATH] [GPU_ID]
#
# Defaults:
#   - declarations-only bake on, spec_k=0, kv cache off
#   - trace_failures on with a small trace cap so logs stay short
#
# Must be invoked from the c4_release directory (the directory that contains
# the tests/ package); the script does not cd for you on purpose.

set -euo pipefail

OFFSET="${1:?offset required}"
LIMIT="${2:?limit required}"
LOG_PATH="${3:-/tmp/diag-1096-scan-next/chunk_${OFFSET}_${LIMIT}.log}"
GPU_ID="${4:-0}"

mkdir -p "$(dirname "${LOG_PATH}")"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
C4_1096_DIAG=1 \
C4_1096_OFFSET="${OFFSET}" \
C4_1096_LIMIT="${LIMIT}" \
C4_DECLARATIONS_ONLY_BAKE="${C4_DECLARATIONS_ONLY_BAKE:-1}" \
C4_SPEC_K="${C4_SPEC_K:-0}" \
C4_BATCH_USE_KV_CACHE="${C4_BATCH_USE_KV_CACHE:-0}" \
C4_1096_TRACE_FAILURES="${C4_1096_TRACE_FAILURES:-1}" \
C4_1096_TRACE_LIMIT="${C4_1096_TRACE_LIMIT:-5}" \
C4_1096_DIAG_ASSERT="${C4_1096_DIAG_ASSERT:-0}" \
python -m pytest -q \
  tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice \
  -s > "${LOG_PATH}" 2>&1
