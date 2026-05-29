"""Per-op audit harness for Layer 0 (marker-transition detector).

Layer 0 owns the residual-stream "Phase A" book-keeping that every
other layer leans on:

* ``layer0_threshold_attn`` — 8 threshold attention heads writing the
  H0..H7 marker-distance flags (PC/AX/SP/BP/MEM/SE/CS).
* ``phase_a_ffn`` — block-pinned FFN that reads H0..H4 and writes the
  NEXT_PC/AX/SP/BP/STACK0/MEM/SE markers consumed by the L1+ embedding
  chain.

Both ops are reachable transitively from the 1096 lowering suite, but
when their declarations drift the audit can only finger them by slot,
not by op name. This module gives each L0 op its own
claim-verification gate so a future drift is named directly in the
failing test rather than buried in a 1096 backtrace.

The ``static_claims_report`` fixture lives in ``conftest.py`` at
session scope so the ~25-60s verifier bake runs once per pytest
session regardless of how many per-layer harnesses are present.
"""

import pytest

from ._per_op_audit import assert_no_drift, assert_op_fires


L0_OPS_WITH_CLAIMS = (
    "layer0_threshold_attn",
    "phase_a_ffn",
)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L0_OPS_WITH_CLAIMS)
def test_l0_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L0", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L0_OPS_WITH_CLAIMS)
def test_l0_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L0", op_name)
