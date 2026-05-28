#!/usr/bin/env bash
# Run the pre-smoke declarative lowering gate.
#
# This catches declaration-to-weight bugs before the neural smoke/1096 paths:
#   1. cheap CompilerIR symbolic-vs-lowered FFN equivalence,
#   2. symbolic byte-signature invariants for known hard programs,
#   3. L10 tail correction symbolic/lowered regression coverage,
#   4. optional teacher-forced neural support audit on a small 1096 slice.
#
# Defaults are intentionally CPU/cheap and must pass.  Set
# C4_LOWERING_GATE_NEURAL=1 to also run the opt-in teacher-forced neural audit
# with its fatal-only gate enabled by default.  Use
# C4_1096_LOWERING_ASSERT_MODE=off for a survey-only neural audit, or
# C4_1096_LOWERING_ASSERT_MODE=all for strict support-drift gating.
#
# The neural audit accepts C4_LOWERING_GATE_NEURAL_SLICES as a comma/space
# separated list of offset:limit pairs, e.g. "0:32,274:274".  Its defaults
# use a bounded context window so raising C4_1096_LOWERING_MAX_TRACE_TOKENS
# for long shards does not force a single huge full-trace forward.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/c4_release${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:--p no:cacheprovider}"

SCOPE="${C4_LOWERING_GATE_SCOPE:-fast}"

run_step() {
  local name="$1"
  shift
  echo "RUN lowering-gate step=${name}"
  "$@"
  echo "DONE lowering-gate step=${name}"
}

case "${SCOPE}" in
  fast)
    run_step compiler-ir \
      python -m pytest -q \
        c4_release/tests/test_compiler_ir.py \
        -k "compare_symbolic_to_lowered_ffn or lower_ffn_accumulates_duplicate_terms_like_symbolic_ffn" \
        --tb=short
    run_step symbolic-byte-signatures \
      python -m pytest -q \
        c4_release/tests/test_symbolic_byte_signature.py \
        --tb=short
    run_step l10-tail-focused \
      python -m pytest -q \
        c4_release/tests/test_l10_tail_correction.py \
        -k "${C4_LOWERING_GATE_TAIL_K:-missing_stack_high}" \
        --tb=short
    ;;
  full)
    run_step marked-lowering-suite \
      python -m pytest -q -m lowering \
        c4_release/tests/test_compiler_ir.py \
        c4_release/tests/test_symbolic_byte_signature.py \
        c4_release/tests/test_1096_teacher_forced_lowering_audit.py \
        --tb=short
    run_step l10-tail-focused \
      python -m pytest -q \
        c4_release/tests/test_l10_tail_correction.py \
        -k "${C4_LOWERING_GATE_TAIL_K:-missing_stack_high}" \
        --tb=short
    ;;
  *)
    echo "C4_LOWERING_GATE_SCOPE must be 'fast' or 'full' (got '${SCOPE}')" >&2
    exit 2
    ;;
esac

if [[ "${C4_LOWERING_GATE_NEURAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  NEURAL_SLICES="${C4_LOWERING_GATE_NEURAL_SLICES:-${C4_1096_OFFSET:-0}:${C4_1096_LIMIT:-16}}"
  NEURAL_SLICES="${NEURAL_SLICES//,/ }"
  for slice in ${NEURAL_SLICES}; do
    offset="${slice%%:*}"
    if [[ "${slice}" == *:* ]]; then
      limit="${slice#*:}"
    else
      limit="${C4_1096_LIMIT:-16}"
    fi
    if [[ -z "${offset}" || -z "${limit}" ]]; then
      echo "Invalid C4_LOWERING_GATE_NEURAL_SLICES entry '${slice}'" >&2
      exit 2
    fi
    run_step "teacher-forced-neural-audit-${offset}-${limit}" \
    env \
      C4_1096_LOWERING_AUDIT=1 \
      C4_1096_OFFSET="${offset}" \
      C4_1096_LIMIT="${limit}" \
      C4_1096_LOWERING_ASSERT_MODE="${C4_1096_LOWERING_ASSERT_MODE:-fatal}" \
      C4_1096_LOWERING_PRINT_MODE="${C4_1096_LOWERING_PRINT_MODE:-drift}" \
      C4_1096_LOWERING_DETAIL_LIMIT="${C4_1096_LOWERING_DETAIL_LIMIT:-2}" \
      C4_1096_LOWERING_MAX_FAILURES="${C4_1096_LOWERING_MAX_FAILURES:-2}" \
      C4_1096_LOWERING_MAX_TRACE_TOKENS="${C4_1096_LOWERING_MAX_TRACE_TOKENS:-20000}" \
      C4_1096_LOWERING_CONTEXT_WINDOW="${C4_1096_LOWERING_CONTEXT_WINDOW:-512}" \
      C4_1096_LOWERING_CHUNK_TOKENS="${C4_1096_LOWERING_CHUNK_TOKENS:-512}" \
      C4_BATCH_MODEL_MAX_SEQ_LEN="${C4_BATCH_MODEL_MAX_SEQ_LEN:-4096}" \
      C4_DECLARATIONS_ONLY_BAKE="${C4_DECLARATIONS_ONLY_BAKE:-1}" \
      python -m pytest -q -s \
        c4_release/tests/test_1096_teacher_forced_lowering_audit.py \
        --tb=short
  done
fi

echo "SUMMARY lowering-gate scope=${SCOPE} neural=${C4_LOWERING_GATE_NEURAL:-0} status=passed"
