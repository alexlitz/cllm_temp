"""Per-op audit for L9 ops.

Layer 9 ops (per the migration map):
  - ``layer9_alu`` -- FFN ADD/SUB hi nibble + bitwise byte 0 + marker
    suppression (combined bake; kind="block").
  - ``layer9_lev_addr_relay`` -- Attention head 0; prior BP byte 0 ->
    ADDR_B0 at the SP marker when OP_LEV.
  - ``layer9_lev_bp_to_pc_relay`` -- Attention head 1; same plumbing,
    fires at MARK_PC instead (LEV return path).
  - ``layer9_alibi_mem_attn`` -- ALiBi mem propagation head (enable=False
    default; proof-of-concept).
  - ``format_string_fetch_head`` -- L9 attn head 0 for convo-I/O
    (enable_conversational_io=False default).
  - ``layer9_marker_suppress`` -- topology anchor (bake is a no-op; actual
    work is chained from ``layer9_alu``).

Tests:
  1. ``assert_no_drift`` for the L9 ops that declare ``claims``.
  2. ``assert_fires_during_bake`` for the L9 ops that do real work in the
     default config; the gated ops (alibi_mem_attn, format_string_fetch_head,
     marker_suppress) get ``expect_inert=True`` smokes.
  3. Symbolic forward test that ``layer9_lev_addr_relay`` writes
     ADDR_B0 at the SP marker position from the prior BP byte 0's
     CLEAN_EMBED value (with OP_LEV held high).
  4. Symbolic forward unit-level test for the ADD hi nibble carry-in
     discrimination inside ``layer9_alu``.
"""

from __future__ import annotations

import torch

from c4_release.neural_vm.unified_compiler.ops.l9_ops import (
    make_format_string_fetch_head_op,
    make_layer9_alibi_mem_attn_op,
    make_layer9_alu_op,
    make_layer9_lev_addr_relay_op,
    make_layer9_lev_bp_to_pc_relay_op,
    make_layer9_marker_suppress_op,
)
from c4_release.tests._per_op_audit import (
    StubBlock,
    assert_fires_during_bake,
    assert_no_drift,
    compile_compact_layout,
)


# ---------------------------------------------------------------------------
# Mode A: drift checks for every L9 op
# ---------------------------------------------------------------------------


L9_OP_NAMES = (
    "layer9_alu",
    "layer9_lev_addr_relay",
    "layer9_lev_bp_to_pc_relay",
    "layer9_alibi_mem_attn",
    "format_string_fetch_head",
    "layer9_marker_suppress",
)


def test_l9_ops_have_no_declared_claim_drift():
    """Every L9 op with non-empty ``claims`` must write the cells it declares.

    Only the two LEV relay ops declare claims in the default config; the
    others are excluded from ``verify_claims_static`` automatically and pass
    via ``allow_missing=True``.
    """
    assert_no_drift(L9_OP_NAMES, allow_missing=True)


# ---------------------------------------------------------------------------
# Fires-during-bake smokes for every L9 op
# ---------------------------------------------------------------------------


def test_layer9_alu_bake_fires():
    layout = compile_compact_layout()
    changed = assert_fires_during_bake(
        make_layer9_alu_op(alu_mode="lookup"),
        layout.dim_positions,
        d_model=layout.d_model,
    )
    # The bake should populate FFN W_up / W_down for the ADD/LEA/ADJ/SUB/ENT
    # hi-nibble + bitwise byte 0 + marker-suppression cluster.
    assert any("W_up" in name or "W_down" in name for name in changed), (
        f"layer9_alu bake should write FFN W_up/W_down; got {changed}"
    )


def test_layer9_lev_addr_relay_bake_fires():
    layout = compile_compact_layout()
    changed = assert_fires_during_bake(
        make_layer9_lev_addr_relay_op(),
        layout.dim_positions,
        d_model=layout.d_model,
    )
    # The head bake should touch attention W_q / W_k / W_v / W_o (sent as
    # ``attn.<name>`` in the harness).
    attn_changed = [name for name in changed if name.startswith("attn.")]
    assert attn_changed, (
        f"layer9_lev_addr_relay should mutate attn weights; got {changed}"
    )


def test_layer9_lev_bp_to_pc_relay_bake_fires():
    layout = compile_compact_layout()
    changed = assert_fires_during_bake(
        make_layer9_lev_bp_to_pc_relay_op(),
        layout.dim_positions,
        d_model=layout.d_model,
    )
    attn_changed = [name for name in changed if name.startswith("attn.")]
    assert attn_changed, (
        f"layer9_lev_bp_to_pc_relay should mutate attn weights; got {changed}"
    )


def test_layer9_alibi_mem_attn_bake_inert_when_disabled():
    """Default ``enable=False`` -> bake is a documented no-op."""
    layout = compile_compact_layout()
    assert_fires_during_bake(
        make_layer9_alibi_mem_attn_op(enable=False),
        layout.dim_positions,
        d_model=layout.d_model,
        expect_inert=True,
    )


def test_format_string_fetch_head_bake_inert_when_disabled():
    """Default ``enable_conversational_io=False`` -> bake is a no-op."""
    layout = compile_compact_layout()
    assert_fires_during_bake(
        make_format_string_fetch_head_op(enable_conversational_io=False),
        layout.dim_positions,
        d_model=layout.d_model,
        expect_inert=True,
    )


def test_layer9_marker_suppress_topology_anchor_is_inert():
    """``layer9_marker_suppress`` exists only as a dep-graph anchor; the
    real bake is chained from ``layer9_alu``. Its standalone bake_fn is a
    documented no-op.
    """
    layout = compile_compact_layout()
    assert_fires_during_bake(
        make_layer9_marker_suppress_op(),
        layout.dim_positions,
        d_model=layout.d_model,
        expect_inert=True,
    )


# ---------------------------------------------------------------------------
# Symbolic forward: layer9_lev_addr_relay maps BP byte 0 -> ADDR_B0 at SP marker
# ---------------------------------------------------------------------------


def test_layer9_lev_addr_relay_symbolic_bp_to_addr_b0_at_sp_marker():
    """End-to-end head smoke: a position carrying a prior BP byte 0
    (with ``L1H1[BP_I]`` + ``BYTE_INDEX_0`` flags and a CLEAN_EMBED byte)
    should propagate that byte's CLEAN_EMBED into ``ADDR_B0_LO/HI`` at a
    later position that holds ``MARK_SP`` + ``OP_LEV``.

    The head spec uses Q at slot 0 = MARK_SP + OP_LEV - 2*CONST (fires only
    when both markers are held) and K at slot 0 = L1H1[BP_I] + BYTE_INDEX_0
    + GATE (fires only at the prior BP byte 0). V copies CLEAN_EMBED_LO/HI
    into head slots 1..32, and O routes them into ADDR_B0_LO/HI.
    """
    layout = compile_compact_layout()
    BD = layout.dim_positions
    stub = StubBlock(layout.d_model)
    with torch.no_grad():
        make_layer9_lev_addr_relay_op().bake_fn(stub, BD, 100.0)

    # Two-position sequence: pos 0 = prior BP byte 0; pos 1 = current SP
    # marker on a LEV step.
    seq_len = 2
    x = torch.zeros(1, seq_len, layout.d_model)
    # Bias term needed by the head's q-side -2L*CONST and k-side GATE bias.
    x[0, :, BD["CONST"]] = 1.0

    # Pos 0: prior-step BP byte 0. The head uses L1H1[BP_I=3] as the BP
    # discriminator and BYTE_INDEX_0 as the byte-0 discriminator.
    x[0, 0, BD["L1H1"] + 3] = 1.0
    x[0, 0, BD["BYTE_INDEX_0"]] = 1.0
    # The value to propagate: encode the byte 0x5A = 0101_1010 -> low=0xA,
    # high=0x5. CLEAN_EMBED is one-hot per nibble.
    x[0, 0, BD["CLEAN_EMBED_LO"] + 0xA] = 1.0
    x[0, 0, BD["CLEAN_EMBED_HI"] + 0x5] = 1.0

    # Pos 1: current step SP marker firing on OP_LEV. The L9 head's Q-side
    # threshold (-2L=-100 with L=50) bakes in the assumption that L6 relays
    # amplified OP_LEV to ~10 by the time it reaches L9 input -- see
    # ``_set_layer9_lev_addr_relay`` docstring "SP marker: Q[0] = L +
    # (10 * L/5) - 2L = L". We feed OP_LEV=10.0 directly to mirror that
    # amplified value.
    x[0, 1, BD["MARK_SP"]] = 1.0
    x[0, 1, BD["OP_LEV"]] = 10.0

    with torch.no_grad():
        y = stub.attn(x)

    # The attention block's forward returns ``x + attn_out`` (residual).
    # Subtracting the input gives only the head's contribution.
    delta = y - x
    out_lo = delta[0, 1, BD["ADDR_B0_LO"]:BD["ADDR_B0_LO"] + 16]
    out_hi = delta[0, 1, BD["ADDR_B0_HI"]:BD["ADDR_B0_HI"] + 16]

    # The argmax of the relayed low/high nibble must match the encoded
    # source byte (0x5A -> low=0xA, high=0x5). Magnitudes are softmax-
    # weighted but the peak slot is the load-bearing assertion.
    assert int(out_lo.argmax()) == 0xA, (
        f"expected ADDR_B0_LO argmax=0xA; out_lo={out_lo.tolist()}"
    )
    assert int(out_hi.argmax()) == 0x5, (
        f"expected ADDR_B0_HI argmax=0x5; out_hi={out_hi.tolist()}"
    )

    # Sanity: position 0 (the source BP byte) should NOT receive a write
    # into ADDR_B0 because its Q-side score does not fire at SP+OP_LEV.
    src_lo = delta[0, 0, BD["ADDR_B0_LO"]:BD["ADDR_B0_LO"] + 16]
    src_hi = delta[0, 0, BD["ADDR_B0_HI"]:BD["ADDR_B0_HI"] + 16]
    assert src_lo.abs().max().item() < 1e-3, (
        f"src position should not receive ADDR_B0_LO write; src_lo={src_lo}"
    )
    assert src_hi.abs().max().item() < 1e-3, (
        f"src position should not receive ADDR_B0_HI write; src_hi={src_hi}"
    )


# ---------------------------------------------------------------------------
# Symbolic forward: layer9_alu ADD hi nibble carry-in discrimination
# ---------------------------------------------------------------------------


def test_layer9_alu_add_hi_nibble_carry_in_discrimination():
    """The ADD hi-nibble units in ``layer9_alu`` come in two sets of 256:
    units 0..255 are the no-carry rows (carry_in=0) and units 256..511 are
    the with-carry rows (carry_in=1). Each row gates on a (a, b) pair.

    For the no-carry unit ``a=0, b=0`` (unit index 0): expected result is
    ``(0 + 0 + 0) % 16 = 0`` and the unit should fire IFF CARRY[0] is OFF.

    For the with-carry unit ``a=0, b=0`` (unit index 256): expected result
    is ``(0 + 0 + 1) % 16 = 1`` and the unit should fire IFF CARRY[0] is ON.

    We drive the FFN with MARK_AX + ALU_HI[0] + AX_CARRY_HI[0] + OP_ADD
    held one-hot, toggle CARRY[0], and assert the OUTPUT_HI argmax flips
    from 0 (no carry path) to 1 (carry path).
    """
    layout = compile_compact_layout()
    BD = layout.dim_positions
    # L9 ALU FFN cluster uses ~3405 units (per ``ffn_units_used``); 3500
    # leaves headroom over the cumulative ADD/LEA/ADJ/SUB/ENT + bitwise +
    # marker-suppress unit count.
    block = StubBlock(layout.d_model, ffn_hidden=3500)
    with torch.no_grad():
        make_layer9_alu_op(alu_mode="lookup").bake_fn(block, BD, 100.0)

    def _drive(carry_in: int) -> torch.Tensor:
        x = torch.zeros(1, 1, layout.d_model)
        x[0, 0, BD["MARK_AX"]] = 1.0
        x[0, 0, BD["ALU_HI"] + 0] = 1.0
        x[0, 0, BD["AX_CARRY_HI"] + 0] = 1.0
        x[0, 0, BD["OP_ADD"]] = 1.0
        if carry_in:
            x[0, 0, BD["CARRY"] + 0] = 1.0
        return x

    with torch.no_grad():
        y_no_carry = block.ffn(_drive(carry_in=0))
        y_carry = block.ffn(_drive(carry_in=1))

    out_no_carry = y_no_carry[0, 0, BD["OUTPUT_HI"]:BD["OUTPUT_HI"] + 16]
    out_carry = y_carry[0, 0, BD["OUTPUT_HI"]:BD["OUTPUT_HI"] + 16]

    # The discriminator: argmax of OUTPUT_HI must flip from 0 -> 1 when
    # CARRY[0] is toggled on, because the no-carry unit pumps OUTPUT_HI[0]
    # while the with-carry unit pumps OUTPUT_HI[1].
    assert int(out_no_carry.argmax()) == 0, (
        f"no-carry ADD a=0 b=0 should select OUTPUT_HI[0]; "
        f"out_no_carry={out_no_carry.tolist()}"
    )
    assert int(out_carry.argmax()) == 1, (
        f"with-carry ADD a=0 b=0 should select OUTPUT_HI[1]; "
        f"out_carry={out_carry.tolist()}"
    )
    # The active peak must dominate the off-peak rows by a comfortable
    # margin (otherwise we are looking at numerical noise).
    assert (out_no_carry[0] - out_no_carry[1]).item() > 0.5
    assert (out_carry[1] - out_carry[0]).item() > 0.5
