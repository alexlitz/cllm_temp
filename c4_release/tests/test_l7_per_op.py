"""Per-op audit harness for Layer 7.

L7 is the operand gather + memory-head fanout layer:

* ``layer7_operand_gather`` -- attention heads 0/1; gathers STACK0 byte 0
  into ALU at the AX marker (head 0) plus the OUTPUT byte for LEA/ADJ/ENT
  destination addressing (head 1). Ships with a 64-cell W_v claim grid.
* ``layer7_memory_heads`` -- attention heads 2-7; gathers prev AX bytes
  into ADDR_B*_LO/HI staging dims (heads 2-4), broadcasts OP_LI/LC/JSR/
  bitwise/SI/SC relays (head 5), PSH/ENT relays + STACK0 marker (head 6),
  and the MEM marker flag broadcast (head 7). 108-cell claim grid.
* ``layer7_sp_byte0_is_f8`` -- B7-2's new head-6 extension that writes
  ``SP_BYTE0_IS_F8 = 1.0`` only when SP byte 0 == 0xF8 at MARK_SP rows;
  consumed by L10's ``tail_sp_marker_byte0_f8_from_initial_stack_exact``.
  4-cell claim grid.
* ``format_pointer_extraction`` -- attention head 7 extension for
  conversational-I/O format-string pointer extraction
  (``enable_conversational_io=False`` default; ships without claims).
* ``format_position_counter`` -- the format-position FFN counter
  (registered at L7 phase 7.6 by the convo-I/O bundle but pinned to
  ``layer_idx=8``; convo-I/O off by default, ships without claims).
* ``convo_io_prtf_capture`` -- prtf transport hook for conversational
  I/O; ships without claims by default.

This file mirrors the ``test_l8_per_op.py`` / ``test_l15_per_op.py``
patterns:

  1. ``static_claims_report``-backed drift checks for every L7 op with a
     populated claim map. If anyone empties a claim grid or wires a new
     dim outside the declared cells the audit fails with a layer-labelled
     error before it shows up in the 1096 backtrace.
  2. ``assert_op_fires`` smokes for every claim-bearing L7 op -- the
     bake must dispatch and emit at least one observable write.
  3. ``assert_op_absent`` watchdogs for the empty-claims ops -- if a
     future commit adds claims to e.g. ``format_pointer_extraction``,
     the watchdog flips loud and the maintainer must migrate the op
     into the drift-checked list.
  4. Symbolic forward for ``layer7_operand_gather`` head 0: drive a
     two-position residual with the prior STACK0 byte 0 staged and a
     MARK_AX query; assert ALU_LO/HI at the AX row pick up the byte 0
     CLEAN_EMBED nibbles.
  5. Symbolic forward for ``layer7_memory_heads`` head 7: drive a
     MARK_MEM key/query and confirm MEM_STORE / OP_JSR are broadcast
     to the MEM byte positions (head 7's stated job).
  6. Symbolic forward for ``layer7_sp_byte0_is_f8``: drive a single
     MARK_SP row with various carry-forwarded SP byte 0 values via
     EMBED_LO/HI and confirm ``SP_BYTE0_IS_F8`` == 1.0 only when the
     full byte == 0xF8 in CLEAN_EMBED context (half-matches cap at
     0.5; non-matches stay at 0.0).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureAttention  # noqa: E402
from neural_vm.unified_compiler.ops.l7_ops import (  # noqa: E402
    _layer7_memory_head_specs,
    _layer7_operand_gather_head_specs,
    _layer7_sp_byte0_is_f8_spec,
)
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402

from ._per_op_audit import (  # noqa: E402
    assert_no_drift,
    assert_op_absent,
    assert_op_fires,
)


# ---------------------------------------------------------------------------
# L7 op inventory. Kept in sync with c4_release/neural_vm/unified_compiler/
# ops/l7_ops.py and ops/all_core_ops.py registration order (phases 7, 7.5,
# 7.6). ``verify_claims_static`` only inspects ops with non-empty claims,
# so the no-claims set must remain absent from the static report.
# ---------------------------------------------------------------------------


L7_OPS_WITH_CLAIMS = (
    "layer7_operand_gather",
    "layer7_memory_heads",
    "layer7_sp_byte0_is_f8",
)


# Ops registered at L7 phases that ship without claims in the default build:
#   - ``format_pointer_extraction``  (enable_conversational_io=False -> bake
#     is a no-op; the Operation never authored a per-cell claim map).
#   - ``format_position_counter``    (convo-I/O off by default; ships
#     without claims even though registered at L7 phase 7.6).
#   - ``convo_io_prtf_capture``      (enable_conversational_io=False -> the
#     prtf capture bake is a no-op and ships without claims).
L7_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "format_pointer_extraction",
    "format_position_counter",
    "convo_io_prtf_capture",
)


# ---------------------------------------------------------------------------
# 1. Drift checks via ``static_claims_report``
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L7_OPS_WITH_CLAIMS)
def test_l7_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L7", op_name)


# ---------------------------------------------------------------------------
# 2. Fires-during-bake for every claim-bearing L7 op
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L7_OPS_WITH_CLAIMS)
def test_l7_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L7", op_name)


# ---------------------------------------------------------------------------
# 3. Known-absent ops (empty-claims by design in the default build)
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L7_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l7_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L7", op_name)


# ---------------------------------------------------------------------------
# 4. Symbolic forward: layer7_operand_gather head 0 -- STACK0 byte 0 ->
#    ALU_LO/HI at the AX marker.
# ---------------------------------------------------------------------------
#
# Head 0 Q at slot 0: MARK_AX (+L) - OP_LEA/ADJ/ENT (-L)
# Head 0 Q at slot 33: MARK_AX (+L) - CONST (-L/2) - {OP_LEA,OP_ADJ,OP_ENT}*10L
# Head 0 K at slot 0: STACK0_BYTE0 (+L)
# Head 0 K at slot 33: CONST (+L)
# Head 0 V slot 1+k: CLEAN_EMBED_LO+k
# Head 0 V slot 17+k: CLEAN_EMBED_HI+k
# Head 0 O routes V slot 1+k -> ALU_LO+k (weight 6.0) and 17+k -> ALU_HI+k.
#
# So the gather fires only for ADD/SUB/LI/etc. (non-address ops), pulling
# the prev STACK0 byte 0 nibbles into ALU_{LO,HI} at the AX row.


_NUM_HEADS = 8
_DIM = 512
_HD = _DIM // _NUM_HEADS


def _build_operand_gather_attn() -> PureAttention:
    """PureAttention with ONLY the L7 operand-gather head 0 + head 1 wired.

    ``PureAttention`` from base_layers ships without alibi_slopes (only the
    vm_step variant has them). Tests that need the alibi-shaped softmax can
    still verify Q->K dispatch because the bake's Q/K coefficients carry the
    discriminator weight on their own; the ALiBi slope only sharpens the
    softmax in production.
    """
    attn = PureAttention(dim=_DIM, num_heads=_NUM_HEADS)
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        Primitives.generate_attention_heads(
            attn, _layer7_operand_gather_head_specs(_SetDim), _HD
        )
    return attn


def _operand_gather_input(byte_value: int) -> torch.Tensor:
    """Two-position residual: pos 0 = prev STACK0 byte 0; pos 1 = AX query.

    Pos 0 carries the STACK0_BYTE0 K-side flag plus CLEAN_EMBED_LO/HI
    nibbles (so the head's V slots 1+k / 17+k can read them out).
    Pos 1 carries MARK_AX (Q-side) so the gather fires here.
    """
    assert 0 <= byte_value <= 0xFF
    lo = byte_value & 0xF
    hi = (byte_value >> 4) & 0xF
    x = torch.zeros(1, 2, _DIM)
    # Bias for both Q slot 33 and K slot 33.
    x[0, :, _SetDim.CONST] = 1.0
    # Pos 0: prev STACK0 byte 0 source.
    x[0, 0, _SetDim.STACK0_BYTE0] = 1.0
    x[0, 0, _SetDim.CLEAN_EMBED_LO + lo] = 1.0
    x[0, 0, _SetDim.CLEAN_EMBED_HI + hi] = 1.0
    # Pos 1: AX-marker query (a non-address op like OP_ADD so head 0
    # is not blocked).
    x[0, 1, _SetDim.MARK_AX] = 1.0
    x[0, 1, _SetDim.OP_ADD] = 1.0
    return x


@pytest.mark.parametrize(
    "byte_value",
    [0x00, 0x42, 0xAB, 0xF8, 0xFF],
    ids=lambda b: f"byte=0x{b:02X}",
)
def test_l7_operand_gather_head0_writes_stack0_byte0_to_alu_at_ax(
    byte_value: int,
) -> None:
    """Head 0 gathers prev STACK0 byte 0 into ALU_{LO,HI} at MARK_AX.

    The argmax across ALU_LO and ALU_HI at the AX row must match the
    encoded byte's lo/hi nibbles. This is the load-bearing path for the
    operand A delivery into the L8 ALU lookup -- ``layer7_operand_gather``
    declares ``produces={"ALU_LO": "AX_byte0", "ALU_HI": "AX_byte0"}``
    per the Phase 3 staleness invariants.
    """
    attn = _build_operand_gather_attn()
    x = _operand_gather_input(byte_value)
    with torch.no_grad():
        y = attn(x)
    delta = y - x
    lo_delta = delta[0, 1, _SetDim.ALU_LO:_SetDim.ALU_LO + 16]
    hi_delta = delta[0, 1, _SetDim.ALU_HI:_SetDim.ALU_HI + 16]
    expected_lo = byte_value & 0xF
    expected_hi = (byte_value >> 4) & 0xF
    assert int(lo_delta.argmax().item()) == expected_lo, (
        f"L7 operand_gather head 0: expected ALU_LO argmax={expected_lo} "
        f"for byte=0x{byte_value:02X}; got argmax="
        f"{int(lo_delta.argmax().item())} with deltas={lo_delta.tolist()}"
    )
    assert int(hi_delta.argmax().item()) == expected_hi, (
        f"L7 operand_gather head 0: expected ALU_HI argmax={expected_hi} "
        f"for byte=0x{byte_value:02X}; got argmax="
        f"{int(hi_delta.argmax().item())} with deltas={hi_delta.tolist()}"
    )
    # The fired ALU_LO/HI nibbles must actually move the residual.
    assert lo_delta[expected_lo].item() > 0.1, (
        f"ALU_LO+{expected_lo} delta too small ({lo_delta[expected_lo]:.4f}); "
        f"head 0 may be softmaxing too flatly."
    )
    assert hi_delta[expected_hi].item() > 0.1, (
        f"ALU_HI+{expected_hi} delta too small ({hi_delta[expected_hi]:.4f})."
    )


# NOTE: head 0 Q-side address-op blocker invariants live in
# ``test_l7_operand_gather.py::test_l7_stack0_gather_is_fully_blocked_for_address_ops``.
# They are intentionally NOT duplicated here -- the per-op harness's
# add-on value is the forward-direction smoke above plus the
# static_claims_report drift gates at the top of this file.


# ---------------------------------------------------------------------------
# 5. Symbolic forward: layer7_memory_heads head 7 -- MEM flag broadcast.
# ---------------------------------------------------------------------------
#
# Head 7 is the simplest head in the memory-heads bundle: Q/K both gate on
# MARK_MEM (with -L blockers on H1[AX/SP/BP] and H4[BP] so adjacent
# markers don't steal attention). V slot 1 reads MEM_STORE, slot 2 reads
# MEM_ADDR_SRC, slot 3 OP_JSR, slot 4 OP_ENT. O routes each back to its
# own dim. The bake bumps alibi_slopes[7] = 5.0 so the head attends back
# multiple positions.


def _build_memory_heads_attn() -> PureAttention:
    """PureAttention with ONLY the L7 memory-heads (heads 2-7) wired."""
    attn = PureAttention(dim=_DIM, num_heads=_NUM_HEADS)
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        Primitives.generate_attention_heads(
            attn, _layer7_memory_head_specs(_SetDim), _HD
        )
    return attn


_MEM_I = 4


def test_l7_memory_heads_head7_broadcasts_mem_store_at_mark_mem() -> None:
    """Head 7: a MARK_MEM source row with MEM_STORE=1 propagates the flag
    back to itself via the head's self-attention.

    The two-position residual sets up:
      pos 0: MARK_MEM source carrying MEM_STORE=1 and OP_JSR=1.
      pos 1: a second position (also MARK_MEM, no flags) so the head
             sees a non-trivial K-side option set.

    Self-attention at pos 0 should pump MEM_STORE and OP_JSR back into
    the same row. The K-side gate is MARK_MEM, so both positions are
    eligible; the V-source choice is dominated by pos 0's flag values.
    """
    attn = _build_memory_heads_attn()
    x = torch.zeros(1, 2, _DIM)
    x[0, :, _SetDim.CONST] = 1.0
    # Pos 0: MARK_MEM with flags to broadcast.
    x[0, 0, _SetDim.MARK_MEM] = 1.0
    x[0, 0, _SetDim.H3 + _MEM_I] = 1.0  # Q-side gate
    x[0, 0, _SetDim.MEM_STORE] = 1.0
    x[0, 0, _SetDim.OP_JSR] = 1.0
    # Pos 1: MARK_MEM neighbour -- the head must still pick up the
    # flagged source.
    x[0, 1, _SetDim.MARK_MEM] = 1.0
    x[0, 1, _SetDim.H3 + _MEM_I] = 1.0
    with torch.no_grad():
        y = attn(x)
    delta = y - x
    # Head 7 writes MEM_STORE/OP_JSR back at the query rows. We only
    # require that at LEAST ONE of the MARK_MEM rows receives a
    # positive delta on MEM_STORE and OP_JSR -- that confirms the head
    # is wired Q->K->V->O correctly. Position selection between two
    # equal-K rows is dominated by ALiBi/numerical tiebreak which is
    # not the load-bearing property.
    mem_store_delta = delta[0, :, _SetDim.MEM_STORE]
    op_jsr_delta = delta[0, :, _SetDim.OP_JSR]
    assert mem_store_delta.max().item() > 0.1, (
        f"L7 memory_heads head 7 should broadcast MEM_STORE to MARK_MEM "
        f"rows; max MEM_STORE Δ across positions={mem_store_delta.tolist()}"
    )
    assert op_jsr_delta.max().item() > 0.1, (
        f"L7 memory_heads head 7 should broadcast OP_JSR to MARK_MEM "
        f"rows; max OP_JSR Δ across positions={op_jsr_delta.tolist()}"
    )


def test_l7_memory_heads_head7_blocks_non_mem_query_rows() -> None:
    """Head 7 Q at slot 0 subtracts -L for H1[AX/SP/BP] and H4[BP].

    A row carrying ``H1+AX_I`` (an AX-marker neighbour, not a MEM marker)
    must not see MEM_STORE/OP_JSR broadcast into it -- otherwise head 7
    would leak the MEM flag onto AX-context rows and break downstream
    flag-gated logic.
    """
    attn = _build_memory_heads_attn()
    x = torch.zeros(1, 2, _DIM)
    x[0, :, _SetDim.CONST] = 1.0
    # Pos 0: real MARK_MEM source carrying the flag.
    x[0, 0, _SetDim.MARK_MEM] = 1.0
    x[0, 0, _SetDim.H3 + _MEM_I] = 1.0
    x[0, 0, _SetDim.MEM_STORE] = 1.0
    # Pos 1: an AX-neighbour row that must NOT receive the broadcast.
    AX_I = 1
    x[0, 1, _SetDim.H1 + AX_I] = 1.0
    with torch.no_grad():
        y = attn(x)
    delta = y - x
    # The AX-neighbour row's MEM_STORE delta must be near-zero.
    ax_mem_delta = float(delta[0, 1, _SetDim.MEM_STORE].item())
    assert ax_mem_delta < 0.1, (
        f"L7 memory_heads head 7 leaked MEM_STORE to an AX-neighbour "
        f"row (H1+AX_I=1, no MARK_MEM); MEM_STORE Δ={ax_mem_delta:.4f}. "
        f"This would corrupt MEM-flag gating downstream."
    )


# ---------------------------------------------------------------------------
# 6. Symbolic forward: layer7_sp_byte0_is_f8 -- SP_BYTE0_IS_F8 producer.
# ---------------------------------------------------------------------------
#
# B7-2's new head-6 V/O slot extension. SP_BYTE0_IS_F8 must read 1.0 IFF
# both the LO nibble of SP byte 0 == 8 AND the HI nibble == 15 (==0xF8)
# in the CLEAN_EMBED context the L7-stage residual carries on MARK_SP
# rows after L3 head 2's carry-forward.
#
# The B7-2 op composes on top of the head-6 PSH/CMP relays already baked
# by ``layer7_memory_heads``, but the new V/O slots 6 and 7 only read
# EMBED_LO+8 / EMBED_HI+15 and only write SP_BYTE0_IS_F8 (weight 0.5
# each, summing to 1.0 on full match). Test_l7_sp_byte0_is_f8.py already
# covers the V/O wiring + spec invariants; here we add a forward smoke
# that confirms the end-to-end signal at the per-op level after both
# ops bake on the same head.


@pytest.fixture(scope="module")
def sp_byte0_is_f8_attn() -> PureAttention:
    """PureAttention with the L7 memory-heads head 6 baked AND the B7-2
    SP_BYTE0_IS_F8 V/O slot extension applied on top.

    The order here mirrors the production bake order (memory_heads first
    at phase=7, then SP_BYTE0_IS_F8 at phase=7.6) so any drift between
    the two-stage bake and a single-stage equivalent surfaces in the
    forward delta. Module-scoped because the tests below are read-only
    (each builds a fresh ``x`` and runs ``attn(x)``) -- sharing the
    baked weights avoids redundant generate_attention_head calls.
    """
    attn = PureAttention(dim=_DIM, num_heads=_NUM_HEADS)
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        # Stage 1: bake head 6 from memory_heads (Q/K + V slots 1..5).
        all_specs = _layer7_memory_head_specs(_SetDim)
        head6_spec = next(s for s in all_specs if s.head_idx == 6)
        Primitives.generate_attention_head(attn, head6_spec, _HD)
        # Stage 2: overlay B7-2's SP_BYTE0_IS_F8 V/O slots 6 and 7.
        Primitives.generate_attention_head(
            attn, _layer7_sp_byte0_is_f8_spec(_SetDim), _HD,
        )
    return attn


_SP_I = 2


def _sp_byte0_residual(sp_byte0: int) -> torch.Tensor:
    """Single MARK_SP row whose CLEAN_EMBED encodes the SP byte 0 value.

    The B7-2 op reads ``EMBED_LO+8`` and ``EMBED_HI+15`` (not CLEAN_EMBED).
    Per the production setup, L3 head 2 carry-forwards SP byte 0's
    EMBED_LO/HI nibbles onto MARK_SP rows. We simulate that here by
    staging the bytes directly in EMBED_LO/HI. The head's Q (MARK_SP +
    H1[SP_I]) and K (MARK_SP + MARK_STACK0) gates fire as expected for
    self-attention.
    """
    assert 0 <= sp_byte0 <= 0xFF
    lo = sp_byte0 & 0xF
    hi = (sp_byte0 >> 4) & 0xF
    x = torch.zeros(1, 1, _DIM)
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.MARK_SP] = 1.0
    x[0, 0, _SetDim.H1 + _SP_I] = 1.0
    # The carry-forwarded SP byte 0 nibbles live in EMBED_LO/HI per the
    # L3 head 2 contract; the B7-2 spec reads those directly.
    x[0, 0, _SetDim.EMBED_LO + lo] = 1.0
    x[0, 0, _SetDim.EMBED_HI + hi] = 1.0
    return x


def test_l7_sp_byte0_is_f8_fires_only_when_sp_byte0_equals_0xf8(
    sp_byte0_is_f8_attn: PureAttention,
) -> None:
    """SP byte 0 == 0xF8 -> SP_BYTE0_IS_F8 ~ 1.0 (both nibble matches).

    With self-attention on a single row, softmax over one position is
    exactly 1.0; EMBED_LO+8 and EMBED_HI+15 each contribute 0.5 via
    the SP_BYTE0_IS_F8 O writes, summing to 1.0. This is the load-
    bearing observable for the downstream
    ``tail_sp_marker_byte0_f8_from_initial_stack_exact`` L10 consumer.
    """
    x = _sp_byte0_residual(0xF8)
    with torch.no_grad():
        y = sp_byte0_is_f8_attn(x)
    delta = y - x
    val = float(delta[0, 0, _SetDim.SP_BYTE0_IS_F8].item())
    assert val == pytest.approx(1.0, abs=1e-4), (
        f"L7 SP_BYTE0_IS_F8 producer: expected ~1.0 at SP byte 0=0xF8, "
        f"got Δ={val:.6f}. Check the V/O slots 6/7 in "
        f"_layer7_sp_byte0_is_f8_spec or the head-6 Q/K wiring from "
        f"layer7_memory_heads."
    )


@pytest.mark.parametrize(
    "sp_byte0",
    [0x00, 0x42, 0xFF, 0xE8, 0xF0, 0x18],
    ids=lambda b: f"byte=0x{b:02X}",
)
def test_l7_sp_byte0_is_f8_caps_half_matches_below_consumer_threshold(
    sp_byte0_is_f8_attn: PureAttention,
    sp_byte0: int,
) -> None:
    """SP byte 0 != 0xF8 -> SP_BYTE0_IS_F8 <= 0.5.

    Coverage matrix:
      0x00 / 0x42: neither nibble matches -> 0.0
      0xE8 / 0x18: lo nibble 8 matches, hi != 15 -> 0.5
      0xF0 / 0xFF: hi nibble 15 matches, lo != 8 -> 0.5

    The L10 consumer ``tail_sp_marker_byte0_f8_from_initial_stack_exact``
    gates on ``SP_BYTE0_IS_F8 +1e6`` against threshold ~10; the half-
    match cap is what lets the threshold cleanly discriminate the F8
    case. If both V writes ever scaled up to 1.0 (instead of 0.5 each)
    half-matches would also clear the threshold and falsely trigger
    the L10 rule.
    """
    x = _sp_byte0_residual(sp_byte0)
    with torch.no_grad():
        y = sp_byte0_is_f8_attn(x)
    delta = y - x
    val = float(delta[0, 0, _SetDim.SP_BYTE0_IS_F8].item())
    assert val <= 0.5 + 1e-4, (
        f"L7 SP_BYTE0_IS_F8 producer: expected <= 0.5 at SP byte 0="
        f"0x{sp_byte0:02X}, got Δ={val:.6f}. Half-match cap is broken; "
        f"L10 consumer threshold would lose F8 discrimination."
    )


def test_l7_sp_byte0_is_f8_clean_embed_context_full_vs_half(
    sp_byte0_is_f8_attn: PureAttention,
) -> None:
    """The full-match (0xF8) value must strictly dominate any half-match.

    This is the CLEAN_EMBED-context check the B7-2 spec hinges on: the
    consumer's threshold sits between the full-match (~1.0) and
    half-match (~0.5) outputs. We sample one full-match against a
    representative LO-half-match and HI-half-match and pin the
    inequality the consumer relies on.
    """

    def _eval(byte: int) -> float:
        x = _sp_byte0_residual(byte)
        with torch.no_grad():
            y = sp_byte0_is_f8_attn(x)
        return float((y - x)[0, 0, _SetDim.SP_BYTE0_IS_F8].item())

    full = _eval(0xF8)
    half_lo = _eval(0xE8)  # lo=8 matches, hi=14 misses
    half_hi = _eval(0xF0)  # hi=15 matches, lo=0 misses
    none = _eval(0x42)     # neither matches
    assert full > 0.75, (
        f"full match Δ={full} should clear 0.75 (production threshold lives "
        f"between half ~0.5 and full ~1.0)"
    )
    assert half_lo < 0.75 and half_hi < 0.75, (
        f"half matches must NOT clear 0.75; got lo={half_lo}, hi={half_hi}"
    )
    assert full > half_lo + 0.4 and full > half_hi + 0.4, (
        f"full match must dominate halves by >= 0.4; "
        f"full={full}, half_lo={half_lo}, half_hi={half_hi}"
    )
    assert none < 0.05, (
        f"no-match SP byte 0=0x42 should leave SP_BYTE0_IS_F8 near zero; "
        f"got Δ={none}"
    )
