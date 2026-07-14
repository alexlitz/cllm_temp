"""Per-op audit harness for Layer 8.

L8 carries the largest declarative-op surface area in the upstream
arithmetic stack: ALU lo nibble (ADD/SUB/LEA/ADJ/CMP), SP gather to the
ADDR_B* staging band, multi-byte IMM fetch+routing, the OP_IMM AX-byte
relay, the head-6 AX_CARRY refresh (gated off in production), and the
mem-to-ALU mem-attention head (also gated off). This harness extends the
existing per-op audit pattern (see ``test_l13_mem_addr_gather.py``,
``test_l0_marker_transitions.py``) with a single module covering every
L8 op:

1. ``static_claims_report`` drift checks for every L8 op that ships with a
   populated claim map (``layer8_multibyte_fetch_bake``,
   ``layer8_sp_gather_bake``, ``layer8_op_imm_relay``). The remaining L8
   ops ship with empty ``claims`` (either FFN ops without a per-cell
   author pass, or ``enable=False`` gated bakes) and are tracked as
   known-absent so a future claim-map addition surfaces loudly here.

2. Fires-during-bake — each claim-bearing L8 op must dispatch and emit
   at least one observable write under the shared verifier.

3. **SP-gather symbolic forwards**: confirm ``layer8_sp_gather_bake``
   heads 0-2 fire at BOTH MARK_STACK0 (baseline) AND MARK_SP, writing the
   gathered SP byte to ADDR_B0_LO/HI. The MARK_SP path closes the B6-G
   Section 3.1 design gap (``.agent-logs/l7-l9-structural-audit/REPORT.md``)
   that previously forced L10 tail rules to read residual MARK_STACK0
   leakage at MARK_SP rows.

4. ADD lo-nibble carry-out symbolic forward — drive the L8 ALU lookup
   path with operands that overflow the lo nibble (a + b >= 16) and
   confirm the CARRY+0 dim is set at the AX marker. Sibling regression
   to ``test_l8_addsub_stage_ownership.py`` which covers the efficient
   AddSub5StageBlock; this test pins the legacy lookup-mode ALU bake.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureAttention, PureFFN  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402
from tests.oracles.vm_step_layer_bakes import (  # noqa: E402
    _set_layer8_alu,
    _set_layer8_sp_gather,
)

from ._per_op_audit import (  # noqa: E402
    assert_no_drift,
    assert_op_absent,
    assert_op_fires,
)


# ---------------------------------------------------------------------------
# L8 op inventory (kept in sync with c4_release/neural_vm/unified_compiler/
# ops/l8_ops.py and ops/all_core_ops.py registration order).
# ---------------------------------------------------------------------------


# Ops that ship a populated ``claims`` set in the default build, so the
# session-scoped ``static_claims_report`` fixture will produce a result
# entry for each.
L8_OPS_WITH_CLAIMS = (
    "layer8_multibyte_fetch_bake",
    "layer8_sp_gather_bake",
    "layer8_op_imm_relay",
)


# Ops that intentionally ship with an empty ``claims`` set (FFN ops that
# never had a per-cell map authored, or ``enable=False`` gated bakes that
# clear their claim list). The verifier skips ops with empty claims, so
# these MUST stay absent from the static report. If a future change adds
# claims, this list needs to migrate the op into ``L8_OPS_WITH_CLAIMS``.
L8_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "layer8_alu",
    "layer8_multibyte_fetch",       # kind="attn" dep anchor, no-op bake
    "layer8_multibyte_routing",
    "layer8_sp_gather",             # kind="attn" dep anchor, no-op bake
    "layer8_head6_ax_carry_refresh",  # enable=False
    "layer8_mem_to_alu",            # enable=False -> claims set is empty
    "format_position_counter",      # convo-io flag off by default
)


# ---------------------------------------------------------------------------
# 1. Drift checks via ``static_claims_report``
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L8_OPS_WITH_CLAIMS)
def test_l8_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L8", op_name)


# ---------------------------------------------------------------------------
# 2. Fires-during-bake for every claim-bearing L8 op
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L8_OPS_WITH_CLAIMS)
def test_l8_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L8", op_name)


# ---------------------------------------------------------------------------
# 3. Known-absent ops (intentionally empty ``claims`` in default build)
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L8_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l8_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L8", op_name)


# ---------------------------------------------------------------------------
# 4. B6-G structural-gap symbolic forward.
# ---------------------------------------------------------------------------


def _build_sp_gather_attn(num_heads: int = 8, dim: int = 512) -> PureAttention:
    """PureAttention with only L8 SP gather heads 0-2 baked."""
    attn = PureAttention(dim=dim, num_heads=num_heads)
    HD = dim // num_heads
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        _set_layer8_sp_gather(attn, 100.0, _SetDim, HD)
    return attn


# Residual stream layout for SP-gather tests. Pos 0-2 carry SP bytes 0/1/2;
# pos 3 carries the MARK_SP token; pos 4 is the configurable query position.
_QUERY_POS = 4
_SP_I = 2  # _SetDim.H1 + _SP_I is the SP-area marker the gather K-side reads.


def _sp_byte_layout(
    *,
    sp_byte_values: tuple[int, int, int] = (0xF8, 0x00, 0x00),
    query_position: str = "STACK0",
    dim: int = 512,
) -> torch.Tensor:
    """Synthesize the minimal residual stream the L8 SP gather attends to.

    Layout:
      pos 0: SP byte 0  (H1[SP]=1, BYTE_INDEX_0=1, CLEAN_EMBED carries the byte)
      pos 1: SP byte 1  (H1[SP]=1, BYTE_INDEX_1=1)
      pos 2: SP byte 2  (H1[SP]=1, BYTE_INDEX_2=1)
      pos 3: MARK_SP    (H1[SP]=0, MARK_SP=1)
      pos 4: query      (either MARK_STACK0=1 or MARK_SP=1 depending on caller)
    """
    x = torch.zeros(1, 5, dim)

    byte_idx_dims = [_SetDim.BYTE_INDEX_0, _SetDim.BYTE_INDEX_1, _SetDim.BYTE_INDEX_2]
    for k, byte_val in enumerate(sp_byte_values):
        x[0, k, _SetDim.H1 + _SP_I] = 1.0
        x[0, k, byte_idx_dims[k]] = 1.0
        x[0, k, _SetDim.CLEAN_EMBED_LO + (byte_val & 0xF)] = 1.0
        x[0, k, _SetDim.CLEAN_EMBED_HI + ((byte_val >> 4) & 0xF)] = 1.0

    # MARK_SP at pos 3 (the bake's documented Q-side blocker target).
    x[0, 3, _SetDim.MARK_SP] = 1.0

    # Query position: either MARK_STACK0 (canonical) or MARK_SP (B6-G probe).
    if query_position == "STACK0":
        x[0, _QUERY_POS, _SetDim.MARK_STACK0] = 1.0
    elif query_position == "MARK_SP":
        # Stage MARK_SP + H1[SP_I]: B6-G's load-bearing observation is that
        # the bake's Q-side actively suppresses these with -L coefficients,
        # so the gather refuses to fire even though L10 rule A expects it to.
        x[0, _QUERY_POS, _SetDim.MARK_SP] = 1.0
        x[0, _QUERY_POS, _SetDim.H1 + _SP_I] = 1.0
    else:
        raise ValueError(f"unknown query_position {query_position!r}")

    x[0, :, _SetDim.CONST] = 1.0
    return x


def _addr_b0_argmax(y: torch.Tensor, pos: int) -> tuple[int, int]:
    """Return (lo_argmax, hi_argmax) for the ADDR_B0 band at residual row pos."""
    lo = y[0, pos, _SetDim.ADDR_B0_LO:_SetDim.ADDR_B0_LO + 16]
    hi = y[0, pos, _SetDim.ADDR_B0_HI:_SetDim.ADDR_B0_HI + 16]
    return int(lo.argmax().item()), int(hi.argmax().item())


def _addr_b0_band_energy(y: torch.Tensor, pos: int) -> float:
    """Sum of absolute magnitudes in the ADDR_B0 band at row pos.

    Used to confirm "head did not write anything meaningful here" vs
    "head wrote a recognisable one-hot here".
    """
    lo = y[0, pos, _SetDim.ADDR_B0_LO:_SetDim.ADDR_B0_LO + 16]
    hi = y[0, pos, _SetDim.ADDR_B0_HI:_SetDim.ADDR_B0_HI + 16]
    return float((lo.abs().sum() + hi.abs().sum()).item())


def test_l8_sp_gather_fires_at_mark_stack0_baseline() -> None:
    """Sanity baseline: at MARK_STACK0 the gather DOES fire and writes ADDR_B0.

    Establishes the working leg of B6-G's finding: the gather is fully
    functional when the query position is MARK_STACK0, so any failure of
    the MARK_SP-probe test below is unambiguously the documented gap and
    not a bake bug.
    """
    attn = _build_sp_gather_attn()
    x = _sp_byte_layout(query_position="STACK0")
    with torch.no_grad():
        y = attn(x)
    lo, hi = _addr_b0_argmax(y, _QUERY_POS)
    assert lo == 0x8, (
        f"L8 SP gather at MARK_STACK0 should write ADDR_B0_LO argmax=8 for "
        f"SP byte 0 = 0xF8 but got {lo}; gather may be miswired."
    )
    assert hi == 0xF, (
        f"L8 SP gather at MARK_STACK0 should write ADDR_B0_HI argmax=15 for "
        f"SP byte 0 = 0xF8 but got {hi}; gather may be miswired."
    )


def test_l8_sp_gather_fires_at_mark_sp() -> None:
    """L8 SP gather heads 0-2 fire at MARK_SP as well as MARK_STACK0.

    ADDR_B0/B1/B2 at MARK_SP rows must carry a fresh in-step SP-derived
    address rather than residual leakage from MARK_STACK0 — the L10
    tail-correction family at ``l10_ops.py:3450-4100`` consumes these
    bands at MARK_SP rows. Without the MARK_SP Q-side gate the gather
    only fires at MARK_STACK0 and L10 reads stale residue (B6-G Section
    3.1, ``.agent-logs/l7-l9-structural-audit/REPORT.md``).
    """
    attn = _build_sp_gather_attn()
    x = _sp_byte_layout(query_position="MARK_SP")
    with torch.no_grad():
        y = attn(x)
    lo, hi = _addr_b0_argmax(y, _QUERY_POS)
    assert lo == 0x8 and hi == 0xF, (
        f"L8 SP gather did not fire at MARK_SP: ADDR_B0 argmax=({lo}, {hi}); "
        f"expected (8, 15) for SP byte 0=0xF8. ADDR_B0 band energy at "
        f"MARK_SP row: {_addr_b0_band_energy(y, _QUERY_POS):.4f} (vs "
        f"MARK_STACK0 baseline ~2.0). Check the AP(0, MARK_SP, 2*L) Q-gate "
        f"in _layer8_sp_gather_head_specs and _set_layer8_sp_gather."
    )


# ---------------------------------------------------------------------------
# 5. ADD lo-nibble carry-out symbolic forward.
# ---------------------------------------------------------------------------


def _build_alu_ffn(d_model: int = 512, hidden_dim: int = 4096) -> PureFFN:
    """Build a PureFFN with only the L8 ALU lookup-mode units baked."""
    ffn = PureFFN(dim=d_model, hidden_dim=hidden_dim)
    with torch.no_grad():
        ffn.W_up.data.zero_()
        ffn.b_up.data.zero_()
        ffn.W_gate.data.zero_()
        ffn.b_gate.data.zero_()
        ffn.W_down.data.zero_()
        _set_layer8_alu(ffn, 100.0, _SetDim)
    return ffn


def _alu_input(*, op_dim: int, lhs: int, rhs: int) -> torch.Tensor:
    """One-row residual stream at the AX marker with ALU operands staged.

    ``lhs`` is read from ALU_LO (operand A in the L8 ALU's 3-way AND);
    ``rhs`` is read from AX_CARRY_LO (operand B). Only the lo nibble
    matters for the L8 ALU path -- hi nibble lives at L9.
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, op_dim] = 1.0
    x[0, 0, _SetDim.ALU_LO + (lhs & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (rhs & 0xF)] = 1.0
    return x


@pytest.mark.parametrize(
    ("a", "b", "expect_carry"),
    [
        # No carry-out: a + b < 16.
        (0x3, 0x4, False),
        (0xF, 0x0, False),
        # Carry-out: a + b >= 16.
        (0xF, 0x1, True),
        (0x8, 0x8, True),
        (0xF, 0xF, True),
    ],
)
def test_l8_add_lo_nibble_carry_out_from_alu_ffn(a, b, expect_carry):
    """L8 ADD lo nibble produces CARRY+0 iff a + b >= 16.

    This pins the load-bearing inter-byte carry signal that L9 consumes
    to propagate ADD carry into the hi nibble. The lookup ALU bake
    (``_set_layer8_alu``) allocates 120 "carry detection" units, one per
    (a, b) pair where a + b >= 16 (see ``_set_layer8_alu`` ADD carry
    block at vm_step.py:5551). Each unit gates on OP_ADD and writes
    CARRY+0 with a small positive coefficient. The contract here is:

      - For non-carry pairs, CARRY+0 must stay near zero.
      - For carry pairs, CARRY+0 must rise above a recognisable
        threshold (the bake normalizes gate->1, so a positive non-zero
        write is the load-bearing observable).

    Sibling regression to ``test_l8_addsub_stage_ownership.py`` which
    validates the efficient ``AddSub5StageBlock`` post-op pipeline; this
    test pins the legacy lookup-mode path the L8 ALU bake still owns
    when ``alu_mode='lookup'``.
    """
    ffn = _build_alu_ffn()
    x = _alu_input(op_dim=_SetDim.OP_ADD, lhs=a, rhs=b)
    with torch.no_grad():
        y = ffn(x)
    carry = float(y[0, 0, _SetDim.CARRY + 0].item())
    output_lo = int(
        y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16].argmax().item()
    )
    expected_lo = (a + b) % 16
    assert output_lo == expected_lo, (
        f"L8 ADD lo nibble: OUTPUT_LO argmax={output_lo} != expected "
        f"{expected_lo} for a=0x{a:X}, b=0x{b:X}"
    )
    if expect_carry:
        assert carry > 0.1, (
            f"L8 ADD lo nibble carry-out: expected CARRY+0 > 0.1 for "
            f"a=0x{a:X} + b=0x{b:X} = {a + b} (>=16), got CARRY+0={carry:.4f}. "
            f"This indicates the 120-unit carry detection block in "
            f"_set_layer8_alu is not gating the (a, b) pair correctly."
        )
    else:
        assert carry < 0.1, (
            f"L8 ADD lo nibble: spurious CARRY+0={carry:.4f} for "
            f"a=0x{a:X} + b=0x{b:X} = {a + b} (<16); the carry-detection "
            f"block should only fire when a+b>=16."
        )
