"""Per-op audit for L14 ops (MEM value generation + cleanup chain).

Complements ``test_l14_output_cleanup.py`` (cleanup-weight invariants)
with three orthogonal checks per L14 op:

1. **Drift** — bake the op twice on fresh stubs, assert bit-identical
   weights. Catches non-deterministic bakes.
2. **Fires-during-bake** — bake the op once, assert it wrote non-zero
   entries into its declared buffers. Catches no-op'd bake plumbing.
3. **Symbolic forward** — feed a hand-crafted residual through L14
   attention with a known source byte at the AX position and verify the
   bake routes the byte's nibbles to ``OUTPUT_LO``/``OUTPUT_HI`` at the
   MEM value byte query position.

The full symbolic forward (``test_l14_mem_generation_full_forward_...``)
is ``xfail`` because end-to-end MEM val byte generation depends on
upstream L1-L13 position flags (L1H4/L2H0/H0+MEM_I) that a 2-position
synthetic residual cannot reproduce in isolation. The V→O routing
invariant — the load-bearing part — is covered separately by
``test_l14_mem_generation_value_head_v_emits_known_byte_into_output``.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ops.l14_ops import (  # noqa: E402
    make_layer14_addr_key_neural_decode_op,
    make_layer14_alu_nocarry_ax_bytes_zero_op,
    make_layer14_clear_addr_key_pollution_op,
    make_layer14_clear_mem_marker_output_op,
    make_layer14_clear_output_corruption_op,
    make_layer14_jsr_ax_bytes_zero_op,
    make_layer14_lc_ax_bytes_zero_op,
    make_layer14_mem_generation_op,
    make_layer14_temp_clear_op,
)
# ``make_layer14_alu_high_byte_relay_op`` is intentionally excluded — it
# lives in l14_ops.py for historical reasons but actually targets
# ``layer_idx=15`` (head 8 of a resized L15 attention).
from neural_vm.vm_step import _SetDim  # noqa: E402

from tests._per_op_audit import (  # noqa: E402
    apply_attention,
    assert_no_drift,
    fires_during_bake,
    make_stub_block,
)


def _dim_positions() -> dict[str, int]:
    """Project ``_SetDim`` to the ``{name: int}`` map the bake helpers expect."""

    return {
        name: getattr(_SetDim, name)
        for name in dir(_SetDim)
        if not name.startswith("_") and isinstance(getattr(_SetDim, name), int)
    }


# ---------------------------------------------------------------------------
# Per-op drift + fires-during-bake checks
# ---------------------------------------------------------------------------


_L14_OP_FACTORIES = [
    ("mem_generation", make_layer14_mem_generation_op, {"attn.W_q", "attn.W_k", "attn.W_v", "attn.W_o"}),
    ("temp_clear", make_layer14_temp_clear_op, {"ffn.W_up", "ffn.W_down"}),
    ("clear_addr_key_pollution", make_layer14_clear_addr_key_pollution_op, {"ffn.W_up", "ffn.W_down"}),
    ("clear_output_corruption", make_layer14_clear_output_corruption_op, {"ffn.W_up", "ffn.W_down"}),
    ("clear_mem_marker_output", make_layer14_clear_mem_marker_output_op, {"ffn.W_up", "ffn.W_down"}),
    ("jsr_ax_bytes_zero", make_layer14_jsr_ax_bytes_zero_op, {"ffn.W_up", "ffn.W_down"}),
    ("lc_ax_bytes_zero", make_layer14_lc_ax_bytes_zero_op, {"ffn.W_up", "ffn.W_down"}),
    ("alu_nocarry_ax_bytes_zero", make_layer14_alu_nocarry_ax_bytes_zero_op, {"ffn.W_up", "ffn.W_down"}),
    # addr_key_neural_decode_op is the largest L14 op (1728 units when enabled)
    # so it gets the wider FFN stub.
    ("addr_key_neural_decode", lambda: make_layer14_addr_key_neural_decode_op(enable=True),
     {"ffn.W_up", "ffn.W_down"}),
]


@pytest.mark.parametrize(
    "label, factory, expected_buffers",
    _L14_OP_FACTORIES,
    ids=[label for label, *_ in _L14_OP_FACTORIES],
)
def test_l14_op_no_drift(label, factory, expected_buffers):
    """Bake each L14 op twice and assert byte-identical resulting weights."""

    del label, expected_buffers
    op = factory()
    assert_no_drift(op, _dim_positions(), S=100.0)


@pytest.mark.parametrize(
    "label, factory, expected_buffers",
    _L14_OP_FACTORIES,
    ids=[label for label, *_ in _L14_OP_FACTORIES],
)
def test_l14_op_fires_during_bake(label, factory, expected_buffers):
    """Each L14 op must actually write something to its declared buffer(s)."""

    del label
    op = factory()
    counts = fires_during_bake(op, _dim_positions(), S=100.0)
    nonzero = {name: count for name, count in counts.items() if count > 0}
    # At minimum every expected buffer must receive some writes.
    missing = [name for name in expected_buffers if counts.get(name, 0) == 0]
    assert not missing, (
        f"op did not write to expected buffers {missing}; observed "
        f"nonzero={nonzero}"
    )
    # Sanity: the total non-zero count must be positive (catches bake_fns
    # that silently early-return because of a flag).
    assert sum(nonzero.values()) > 0, (
        f"op produced no non-zero weights at all (flag-stuck bake?)"
    )


# ---------------------------------------------------------------------------
# Symbolic forward — L14 mem_generation MEM value byte routing
# ---------------------------------------------------------------------------


def _baked_mem_generation_attn():
    """Bake the L14 mem_generation op on a fresh stub block and return it."""

    block = make_stub_block()
    op = make_layer14_mem_generation_op()
    op.bake_fn(block.attn, _dim_positions(), 100.0)
    return block.attn


def test_l14_mem_generation_bake_sets_alibi_slopes():
    """The MEM generation bake bumps L14 ALiBi slopes to 5.0 across all heads."""

    attn = _baked_mem_generation_attn()
    assert attn.alibi_slopes[:8].tolist() == [5.0] * 8


def test_l14_mem_generation_value_head_v_copies_clean_embed():
    """Heads 4-7 V projection routes CLEAN_EMBED_LO/HI[k] → V slot 1+k/17+k.

    The overbroad_sp_suppression patch doubles the magnitude to 2.0 to
    overcome downstream defaults, so the assertion is ``>= 1.0``.
    """

    attn = _baked_mem_generation_attn()
    HD = attn.W_q.shape[0] // attn.num_heads
    for head in range(4, 8):
        base = head * HD
        for k in range(16):
            assert attn.W_v[base + 1 + k, _SetDim.CLEAN_EMBED_LO + k] >= 1.0
            assert attn.W_v[base + 17 + k, _SetDim.CLEAN_EMBED_HI + k] >= 1.0


def test_l14_mem_generation_value_head_o_writes_output():
    """Heads 4-7 O projection must route the V slots into OUTPUT_LO/HI[k]."""

    attn = _baked_mem_generation_attn()
    HD = attn.W_q.shape[0] // attn.num_heads
    for head in range(4, 8):
        base = head * HD
        for k in range(16):
            assert attn.W_o[_SetDim.OUTPUT_LO + k, base + 1 + k] == 1.0
            assert attn.W_o[_SetDim.OUTPUT_HI + k, base + 17 + k] == 1.0


def test_l14_mem_generation_addr_heads_keep_value_lanes_blocked():
    """Addr heads (0-3) keep ``MEM_VAL_B*`` query dims negative.

    Without this blocker, the addr heads fire at MEM value byte
    positions and overwrite the value heads' OUTPUT writes.
    """

    attn = _baked_mem_generation_attn()
    HD = attn.W_q.shape[0] // attn.num_heads
    for head in range(4):
        base = head * HD
        for dim in (
            _SetDim.MEM_VAL_B0,
            _SetDim.MEM_VAL_B1,
            _SetDim.MEM_VAL_B2,
            _SetDim.MEM_VAL_B3,
        ):
            assert attn.W_q[base + 38, dim] < 0.0


def test_l14_mem_generation_value_heads_block_addr_byte_positions():
    """Value heads (4-7) keep ``BYTE_INDEX_0..2`` query dims negative.

    Value heads emit MEM val byte 0 at the ``BYTE_INDEX_3`` query
    position (autoregressive shift from addr byte 3). At
    ``BYTE_INDEX_0``/``_1``/``_2`` the addr heads own the OUTPUT writes,
    so value heads must stay silent there.
    """

    attn = _baked_mem_generation_attn()
    HD = attn.W_q.shape[0] // attn.num_heads
    for head in range(4, 8):
        base = head * HD
        for dim in (
            _SetDim.BYTE_INDEX_0,
            _SetDim.BYTE_INDEX_1,
            _SetDim.BYTE_INDEX_2,
        ):
            assert attn.W_q[base + 38, dim] < 0.0


def _ax_source_row(byte_value: int) -> torch.Tensor:
    """A single residual row standing in for an AX byte-0 source token."""

    assert 0 <= byte_value <= 0xFF
    lo = byte_value & 0xF
    hi = (byte_value >> 4) & 0xF
    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.IS_BYTE] = 1.0
    row[_SetDim.H1 + 1] = 1.0  # H1[AX_I] — source side of value-head dim-1 K
    row[_SetDim.BYTE_INDEX_0] = 1.0
    row[_SetDim.CLEAN_EMBED_LO + lo] = 1.0
    row[_SetDim.CLEAN_EMBED_HI + hi] = 1.0
    return row


def _mem_val_b0_query_row() -> torch.Tensor:
    """A single residual row standing in for the MEM val byte-0 query position.

    The value heads' position gate (Q dim 33) fires at ``(H1+MEM_I) -
    (H0+MEM_I)``.  ``MEM_STORE`` is the gate that ``L7 head 7`` would
    have broadcast to MEM byte positions; we set it explicitly here.
    """

    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.MEM_STORE] = 2.0
    row[_SetDim.OP_PSH] = 1.0
    row[_SetDim.H1 + 4] = 1.0  # H1[MEM_I] — predicts val_b0
    return row


def test_l14_mem_generation_value_head_v_emits_known_byte_into_output():
    """V@O routes CLEAN_EMBED nibbles into OUTPUT for head 4 (val byte 0).

    This is the load-bearing routing invariant. The symbolic forward
    test layers attention selection on top; if this fails no chain of
    upstream evidence can rescue the MEM value byte.
    """

    attn = _baked_mem_generation_attn()
    HD = attn.W_q.shape[0] // attn.num_heads
    head = 4  # val byte 0
    base = head * HD

    byte_value = 0xAB
    source = _ax_source_row(byte_value)
    # V projection for this head: rows from W_v.t().
    v = source @ attn.W_v.t()  # [d_model]
    v_head = v[base : base + HD]
    # O projection sums v_head into the residual output dims.
    out = v_head @ attn.W_o[:, base : base + HD].t()
    # The V[0] cancel pushes OUTPUT_LO[0]/OUTPUT_HI[0] negative; the
    # value head V[1+k]/V[17+k] should write the source nibble pair.
    lo = byte_value & 0xF
    hi = (byte_value >> 4) & 0xF
    # CLEAN_EMBED gain through V is 2.0 (see overbroad_sp_suppression);
    # O passes that straight through to OUTPUT_LO/HI with weight 1.0.
    assert out[_SetDim.OUTPUT_LO + lo] >= 2.0, (
        f"V→O routing did not deposit CLEAN_EMBED_LO[{lo}] into OUTPUT_LO[{lo}]"
    )
    assert out[_SetDim.OUTPUT_HI + hi] >= 2.0, (
        f"V→O routing did not deposit CLEAN_EMBED_HI[{hi}] into OUTPUT_HI[{hi}]"
    )
    # And every *other* OUTPUT_LO nibble should not have been touched
    # (apart from the matched-cancel on slot 0, which is the V[0] route
    # and only ever subtracts).
    for k in range(16):
        if k == lo:
            continue
        # OUTPUT_LO[0] receives the V[0] cancel for nonzero source nibbles;
        # that's intentional, but the magnitude must stay small enough to
        # not beat the chosen nibble.
        assert out[_SetDim.OUTPUT_LO + k] < out[_SetDim.OUTPUT_LO + lo]
    for k in range(16):
        if k == hi:
            continue
        assert out[_SetDim.OUTPUT_HI + k] < out[_SetDim.OUTPUT_HI + hi]


@pytest.mark.xfail(
    reason=(
        "Full forward needs upstream L1-L13 position flags (L1H0..L2H0+MEM_I, "
        "H0+MEM_I) that a 2-position synthetic residual cannot reproduce. The "
        "V→O routing invariant is covered separately."
    ),
    strict=False,
)
def test_l14_mem_generation_full_forward_routes_ax_byte_to_mem_val_b0():
    """End-to-end synthetic forward: AX byte 0 → MEM val byte 0 OUTPUT."""

    attn = _baked_mem_generation_attn()
    byte_value = 0x2A

    rows = torch.zeros(2, 512)
    rows[0, :] = _ax_source_row(byte_value)
    rows[1, :] = _mem_val_b0_query_row()

    out = apply_attention(attn, rows, alibi_slopes=attn.alibi_slopes)
    output_lo = out[1, _SetDim.OUTPUT_LO : _SetDim.OUTPUT_LO + 16]
    output_hi = out[1, _SetDim.OUTPUT_HI : _SetDim.OUTPUT_HI + 16]
    lo = byte_value & 0xF
    hi = (byte_value >> 4) & 0xF
    assert int(output_lo.argmax()) == lo, (
        f"expected OUTPUT_LO argmax={lo}, got {int(output_lo.argmax())} "
        f"(values: {output_lo.tolist()})"
    )
    assert int(output_hi.argmax()) == hi, (
        f"expected OUTPUT_HI argmax={hi}, got {int(output_hi.argmax())} "
        f"(values: {output_hi.tolist()})"
    )


def test_l14_temp_clear_chain_advances_unit_counter():
    """L14 chain ops advance ``ffn._l14_unit_counter`` so they don't collide.

    A bake that forgets to update the counter would silently overwrite
    earlier chain ops' weights at the same unit indices.
    """

    block = make_stub_block()
    block.ffn._l14_unit_counter = 0

    chain_ops = [
        make_layer14_temp_clear_op(),
        make_layer14_clear_addr_key_pollution_op(),
        make_layer14_clear_output_corruption_op(),
        make_layer14_clear_mem_marker_output_op(),
    ]
    counters = []
    for op in chain_ops:
        op.bake_fn(block, _dim_positions(), 100.0)
        counters.append(block.ffn._l14_unit_counter)

    for i in range(1, len(counters)):
        assert counters[i] >= counters[i - 1], (
            f"chain op {chain_ops[i].name} did not advance counter past "
            f"{chain_ops[i - 1].name} (saw {counters[i]} vs {counters[i - 1]})"
        )
    # At least one op must have advanced the counter beyond zero.
    assert counters[-1] > 0, (
        f"counter stayed at 0 after the full chain; chain ops failed to "
        f"register their unit usage"
    )


def test_l14_chain_does_not_drift_across_full_bake():
    """Running the full L14 block-op chain twice produces identical weights."""

    def _bake_chain():
        block = make_stub_block()
        block.ffn._l14_unit_counter = 0
        ops = [
            make_layer14_temp_clear_op(),
            make_layer14_clear_addr_key_pollution_op(),
            make_layer14_clear_output_corruption_op(),
            make_layer14_clear_mem_marker_output_op(),
            make_layer14_jsr_ax_bytes_zero_op(),
            make_layer14_lc_ax_bytes_zero_op(),
            make_layer14_alu_nocarry_ax_bytes_zero_op(),
        ]
        for op in ops:
            op.bake_fn(block, _dim_positions(), 100.0)
        return block

    a = _bake_chain()
    b = _bake_chain()
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
        ta = getattr(a.ffn, name)
        tb = getattr(b.ffn, name)
        assert torch.equal(ta, tb), (
            f"L14 chain bake drifted in ffn.{name}: max |Δ|={float((ta - tb).abs().max())}"
        )
