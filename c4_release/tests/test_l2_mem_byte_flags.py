"""Per-op audit harness for Layer 2 (MEM byte flags + STACK0 byte index).

Layer 2 owns:

* ``layer2_threshold_attn`` — single threshold attention head writing
  the L2H0 marker-distance flag.
* ``layer2_mem_byte_flags`` — FFN that lights up MEM_VAL_B0..B3 and the
  BYTE_INDEX_0..3 (+ STACK0_BYTE1..3) flags used by every downstream
  byte-aware op.
* ``layer2_initial_pc_bake_cancel`` — pair of FFN units that cancel
  the REG_PC token-embedding ``PC_OFFSET`` injection at step 1+ (gated
  by MARK_PC AND HAS_SE).
* ``layer2_lookback_detection_head`` — conversational-I/O attention
  head 1, gated on ``enable_conversational_io=True``. The default
  verifier build uses the False branch so this op declares no claims;
  tracked below as a known-absent op so a future flip is loud.

See ``test_l0_marker_transitions.py`` for the shared fixture rationale.
"""

import pytest

from ._per_op_audit import assert_no_drift, assert_op_absent, assert_op_fires


L2_OPS_WITH_CLAIMS = (
    "layer2_threshold_attn",
    "layer2_mem_byte_flags",
    "layer2_initial_pc_bake_cancel",
)

# Gated by ``enable_conversational_io`` (default False) — declares no
# claims in the default build, so should not appear in the report.
L2_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "layer2_lookback_detection_head",
)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L2_OPS_WITH_CLAIMS)
def test_l2_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L2", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L2_OPS_WITH_CLAIMS)
def test_l2_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L2", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L2_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l2_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L2", op_name)
