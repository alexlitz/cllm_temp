"""Per-op audit harness for Layer 4 (PC relay + fetch).

Layer 4 owns the PC-marker → AX-marker relay that drives the L5 fetch:

* ``layer4_pc_relay`` — attention heads 0 + 1 that copy the prior PC
  marker's EMBED_LO/HI (plus ADDR_KEY[32:48]) into the AX marker
  (head 0) and the byte positions' TEMP slot (head 1). Load-bearing
  for L5's first-step opcode fetch.
* ``layer4_sp_to_addr_key`` — SP-to-ADDR_KEY staging for the
  STACK0_VIA_MEM_ATTENTION_PLAN. Registered in ``all_core_ops`` with
  ``enable=False`` because its L8 reader runs before the current
  neural ADDR_KEY decode (so MEM value tokens do not yet carry stable
  byte-address keys). With ``enable=False`` the op authors zero
  per-cell claims, so the verifier skips it — tracked below as a
  known-absent op so flipping the enable flag will trip a loud test
  failure that migrates it into the drift-checked list.
* ``layer4_ffn`` — Phase-A PC+1/+2/+3/+4 nibble rotation chain. Ships
  with empty ``claims`` today (544 hidden units spec'd via
  ``ffn_units_used`` only) so the verifier skips it. Tracked below as
  a known-absent op so a future per-cell claim map is loud.

See ``test_l0_marker_transitions.py`` for the shared fixture rationale.
"""

import pytest

from ._per_op_audit import assert_no_drift, assert_op_absent, assert_op_fires


L4_OPS_WITH_CLAIMS = (
    "layer4_pc_relay",
)

# Ops that intentionally ship with empty ``claims`` in the default
# build (no per-cell map authored or ``enable=False`` gate). Listed so
# adding claims later requires updating this constant.
L4_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "layer4_ffn",
    # STACK0_VIA_MEM_ATTENTION_PLAN registers this op with
    # ``enable=False`` until MEM value-byte ADDR_KEY producers exist
    # before L8. The op's ``_claims`` set is gated on ``enable`` so it
    # ships empty by default and the verifier skips it. When the plan
    # lands and the call site flips to ``enable=True``, move this back
    # into ``L4_OPS_WITH_CLAIMS`` (the 48-cell map is already authored).
    "layer4_sp_to_addr_key",
)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L4_OPS_WITH_CLAIMS)
def test_l4_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L4", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L4_OPS_WITH_CLAIMS)
def test_l4_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L4", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L4_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l4_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L4", op_name)
