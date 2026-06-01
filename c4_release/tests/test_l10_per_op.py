"""Per-op audit harness for Layer 10 itself (the ALU+attn passthrough block).

Layer 10 is the bitwise/DIV-MOD FFN plus an 8-head attention block that
relays operand bytes from the SP/BP/STACK0/AX rows into the byte-routing
slots downstream layers consume:

* ``layer10_carry_relay_bake`` -- attn head 0 carry relay; pulls
  CARRY[1]/CARRY[2] from MARK_AX rows into AX byte positions so the
  carry-propagation post-ops can read them. 2-cell claim grid.
* ``layer10_byte_passthrough_bake`` -- attn head 1 AX byte passthrough;
  routes CLEAN_EMBED_LO/HI into OUTPUT_LO/HI at AX byte positions when
  no upstream marker owns the value (IMM/LI/LC). 32-cell claim grid.
* ``layer10_sp_byte_passthrough_bake`` -- attn head 2 SP byte
  passthrough; same shape as head 1 but gated on SP markers and
  ENT/JSR. 32-cell claim grid.
* ``layer10_bp_byte_passthrough_bake`` -- attn head 7 BP upper-byte
  passthrough; gated on ENT/LEV. 32-cell claim grid.
* ``layer10_psh_stack0_passthrough_bake`` -- attn head 3 PSH STACK0
  passthrough; route CLEAN_EMBED to STACK0 marker rows on PSH steps.
  32-cell claim grid.
* ``layer10_stack0_byte_relay_bake`` -- attn heads 4/5/6 stack-memory
  byte relays into ALU at AX, plus the STACK0 upper-byte carry. 80-cell
  claim grid (heads 4/5 each carry 32 V slots, head 6 carries 32).
* ``layer10_alu`` -- block-pinned FFN baking AND/OR/XOR + DIV/MOD setup
  + 18-unit comparison-combine slice. Ships without per-cell claims
  today (the ~1846-unit cluster is sized via ``ffn_units_used`` rather
  than a per-cell map).
* The five topology-anchor ops (``layer10_carry_relay``,
  ``layer10_byte_passthrough``, ``layer10_sp_byte_passthrough``,
  ``layer10_psh_stack0_passthrough``, ``layer10_stack0_byte_relay``)
  all carry ``bake_fn=return None`` so they are dep-graph placeholders
  only. They ship without claims and are tracked as known-absent.
* ``l10_post_op_attach`` -- structural model bake at phase 10.7 that
  appends BinaryOpByteZeroing/AddSubBytePropagation/CarryPropagation/
  BitwiseBytePropagation post-ops to block.post_ops. Ships without
  claims (structural model, not a per-cell declarative op).

This file complements:
  - ``test_l10_post_ops_combined_per_op.py`` (the L17-pinned phase-10.5
    combined FFN).
  - ``test_addr_key_neural_decode.py`` and the rest of the L10
    integration sweep.

Patterns mirror ``test_l8_per_op.py`` / ``test_l15_per_op.py``:

  1. ``static_claims_report``-backed drift + fires checks for every
     L10 op with a populated claim grid (the 6 bake ops above).
  2. ``assert_op_absent`` watchdogs for every L10 op intentionally
     shipping with empty claims today (topology anchors + the L10
     ALU FFN + the post_op_attach structural bake).
  3. Symbolic forward for the L10 ALU FFN bitwise byte 0 path: drive
     a residual at MARK_AX with OP_AND / OP_OR / OP_XOR plus a pair
     of operand nibbles (ALU_LO + AX_CARRY_LO) and confirm OUTPUT_LO
     argmax matches the canonical truth table.
  4. Symbolic forward for the L10 byte-passthrough heads (head 1 AX,
     head 2 SP): confirm CLEAN_EMBED -> OUTPUT routing at the V/O
     weight level (the load-bearing routing invariant; full attention
     selection depends on upstream BYTE_INDEX flags that a synthetic
     residual cannot reproduce cleanly).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureAttention, PureFFN  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    make_layer10_alu_op,
    make_layer10_bp_byte_passthrough_bake_op,
    make_layer10_byte_passthrough_bake_op,
    make_layer10_carry_relay_bake_op,
    make_layer10_psh_stack0_passthrough_bake_op,
    make_layer10_sp_byte_passthrough_bake_op,
    make_layer10_stack0_byte_relay_bake_op,
)
from neural_vm.vm_step import _SetDim  # noqa: E402

from ._per_op_audit import (  # noqa: E402
    assert_no_drift,
    assert_op_absent,
    assert_op_fires,
)


# ---------------------------------------------------------------------------
# L10 op inventory. Kept in sync with c4_release/neural_vm/unified_compiler/
# ops/l10_ops.py and ops/all_core_ops.py registration order (phases 10.0
# through 10.7). ``verify_claims_static`` only inspects ops with non-empty
# claims, so the no-claims set must remain absent from the static report.
# ---------------------------------------------------------------------------


# Six bake ops ship populated ``claims`` grids: the five attn passthrough
# bakes + the stack0 byte relay bake. These all live under kind="block"
# layer_idx=10 and write to model.blocks[10].attn.
L10_OPS_WITH_CLAIMS = (
    "layer10_carry_relay_bake",
    "layer10_byte_passthrough_bake",
    "layer10_sp_byte_passthrough_bake",
    "layer10_bp_byte_passthrough_bake",
    "layer10_psh_stack0_passthrough_bake",
    "layer10_stack0_byte_relay_bake",
)


# Ops intentionally shipping with empty ``claims`` in the default build:
#   - 5 topology anchors: kind="attn" with bake_fn returning None. They
#     exist only to reserve dep-graph slots; the real bake is owned by
#     the corresponding ``*_bake`` op above.
#   - layer10_alu: 1846-unit FFN cluster sized via ffn_units_used; no
#     per-cell claim map authored yet.
#   - l10_post_op_attach: structural-model bake at phase 10.7 that
#     appends post-op modules onto block.post_ops. No claims by design.
L10_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "layer10_carry_relay",
    "layer10_byte_passthrough",
    "layer10_sp_byte_passthrough",
    "layer10_psh_stack0_passthrough",
    "layer10_stack0_byte_relay",
    "layer10_alu",
    "l10_post_op_attach",
)


# ---------------------------------------------------------------------------
# 1. Drift checks via ``static_claims_report``
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L10_OPS_WITH_CLAIMS)
def test_l10_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L10", op_name)


# ---------------------------------------------------------------------------
# 2. Fires-during-bake for every claim-bearing L10 op
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L10_OPS_WITH_CLAIMS)
def test_l10_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L10", op_name)


# ---------------------------------------------------------------------------
# 3. Known-absent ops (intentionally empty ``claims`` in default build)
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L10_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l10_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L10", op_name)


# ---------------------------------------------------------------------------
# 4. Direct bake fires-during-bake (catches no-op'd bake plumbing for
#    the topology-anchor ops too).
# ---------------------------------------------------------------------------


def _dim_positions() -> dict[str, int]:
    """Project _SetDim to the {name: int} map the bake helpers expect."""
    return {
        name: getattr(_SetDim, name)
        for name in dir(_SetDim)
        if not name.startswith("_") and isinstance(getattr(_SetDim, name), int)
    }


def _baked_l10_attn(
    op_factory,
    *,
    num_heads: int = 8,
    dim: int = 512,
) -> PureAttention:
    """Bake an L10 attn-passthrough op (kind="block") onto a fresh
    PureAttention via the same block-wrapper the production layout uses.
    """
    attn = PureAttention(dim=dim, num_heads=num_heads)

    class _Block:
        def __init__(self, attn):
            self.attn = attn

    block = _Block(attn)
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        op_factory().bake_fn(block, _dim_positions(), 100.0)
    return attn


@pytest.mark.parametrize(
    ("op_factory", "label"),
    [
        (make_layer10_carry_relay_bake_op, "carry_relay"),
        (make_layer10_byte_passthrough_bake_op, "ax_byte"),
        (make_layer10_sp_byte_passthrough_bake_op, "sp_byte"),
        (make_layer10_bp_byte_passthrough_bake_op, "bp_byte"),
        (make_layer10_psh_stack0_passthrough_bake_op, "psh_stack0"),
        (make_layer10_stack0_byte_relay_bake_op, "stack0_byte_relay"),
    ],
    ids=lambda val: val if isinstance(val, str) else "factory",
)
def test_l10_attn_bake_writes_at_least_one_nonzero_weight(op_factory, label):
    """Each L10 attn bake must mutate at least one attention weight.

    A silently-no-opped bake (e.g. an early return) would slip through
    the claim-grid drift check (which only catches missing-cell drift)
    but trigger a runtime regression at L10. This per-op smoke catches
    that explicitly.
    """
    del label
    attn = _baked_l10_attn(op_factory)
    total_nonzero = (
        int((attn.W_q != 0).sum().item())
        + int((attn.W_k != 0).sum().item())
        + int((attn.W_v != 0).sum().item())
        + int((attn.W_o != 0).sum().item())
    )
    assert total_nonzero > 0, (
        f"L10 attn op {op_factory.__name__} bake produced no non-zero "
        f"weights anywhere; bake_fn may be silently early-returning."
    )


# ---------------------------------------------------------------------------
# 5. Symbolic forward: L10 ALU bitwise byte 0 truth table (AND/OR/XOR).
# ---------------------------------------------------------------------------


def _build_l10_alu_ffn(d_model: int = 512, hidden_dim: int = 2048) -> PureFFN:
    """Build a PureFFN with only the L10 ALU lookup-mode units baked.

    The L10 ALU FFN reaches ~1846 units (per ``ffn_units_used``); a
    hidden width of 2048 leaves a comfortable margin without ballooning
    per-test allocations.
    """
    ffn = PureFFN(dim=d_model, hidden_dim=hidden_dim)

    class _Block:
        def __init__(self, ffn):
            self.ffn = ffn

    block = _Block(ffn)
    with torch.no_grad():
        ffn.W_up.data.zero_()
        ffn.b_up.data.zero_()
        ffn.W_gate.data.zero_()
        ffn.b_gate.data.zero_()
        ffn.W_down.data.zero_()
        make_layer10_alu_op().bake_fn(block, _dim_positions(), 100.0)
    return ffn


def _alu_input(*, op_dim: int, lhs: int, rhs: int) -> torch.Tensor:
    """One-row residual stream at the AX marker with ALU operands staged.

    L10's bitwise units read operand A from ``ALU_LO`` and operand B from
    ``AX_CARRY_LO`` -- the same convention L8's lookup ALU uses for the
    lo nibble. Only byte 0 is exercised here; bytes 1-3 live in other
    layers.
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, op_dim] = 1.0
    x[0, 0, _SetDim.ALU_LO + (lhs & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (rhs & 0xF)] = 1.0
    return x


@pytest.mark.parametrize(
    ("op_name", "op_dim", "lhs", "rhs", "expected_lo"),
    [
        ("AND", _SetDim.OP_AND, 0xC, 0xA, 0xC & 0xA),
        ("AND", _SetDim.OP_AND, 0xF, 0xF, 0xF),
        ("AND", _SetDim.OP_AND, 0x0, 0xF, 0x0),
        ("OR",  _SetDim.OP_OR,  0xC, 0xA, 0xC | 0xA),
        ("OR",  _SetDim.OP_OR,  0x0, 0x0, 0x0),
        ("OR",  _SetDim.OP_OR,  0x3, 0x5, 0x3 | 0x5),
        ("XOR", _SetDim.OP_XOR, 0xC, 0xA, 0xC ^ 0xA),
        ("XOR", _SetDim.OP_XOR, 0xF, 0xF, 0x0),
        ("XOR", _SetDim.OP_XOR, 0x5, 0xA, 0xF),
    ],
)
def test_l10_alu_bitwise_byte0_truth_table(
    op_name, op_dim, lhs, rhs, expected_lo
):
    """L10 ALU lookup bake produces canonical AND/OR/XOR truth-table outputs.

    The L10 ALU FFN allocates one cross-product unit per (a, b) pair per
    bitwise op (16 * 16 * 3 = 768 units total). Each unit gates on the
    appropriate OP_* opcode and writes the bitwise result nibble to
    OUTPUT_LO. If any of these units regresses, the entire L10 bitwise
    byte-0 path silently misroutes.

    Sibling regression to the L8 lookup-mode coverage in
    ``test_l8_per_op.py``: L8 owns the byte-0 ADD/SUB/CMP path; L10 owns
    the byte-0 bitwise path. Together they pin the byte-0 ALU semantics
    for the two layers that lower into ``_set_layer*_alu``.
    """
    ffn = _build_l10_alu_ffn()
    x = _alu_input(op_dim=op_dim, lhs=lhs, rhs=rhs)
    with torch.no_grad():
        y = ffn(x)
    output_lo = y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
    lo_argmax = int(output_lo.argmax().item())
    assert lo_argmax == expected_lo, (
        f"L10 ALU {op_name} byte 0: OUTPUT_LO argmax={lo_argmax} != "
        f"expected {expected_lo} for a=0x{lhs:X}, b=0x{rhs:X}. The "
        f"bitwise cross-product unit for this (a, b, op) tuple has "
        f"either not baked or its OUTPUT_LO write is being overwritten."
    )


# ---------------------------------------------------------------------------
# 6. Symbolic forward: L10 byte-passthrough V/O routing invariant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op_factory", "head_idx", "label"),
    [
        (make_layer10_byte_passthrough_bake_op, 1, "AX_head_1"),
        (make_layer10_sp_byte_passthrough_bake_op, 2, "SP_head_2"),
        (make_layer10_psh_stack0_passthrough_bake_op, 3, "STACK0_head_3"),
        (make_layer10_bp_byte_passthrough_bake_op, 7, "BP_head_7"),
    ],
)
def test_l10_byte_passthrough_v_o_routes_clean_embed_to_output(
    op_factory, head_idx, label
):
    """Per-head V/O routing: V[h*HD + k] reads CLEAN_EMBED_LO[k] and
    O[OUTPUT_LO+k] reads back from that V slot. Same for HI (offset +16).

    This is the load-bearing routing invariant for the L10 byte
    passthrough heads. The full attention selection depends on upstream
    BYTE_INDEX_0..3 flags that a synthetic 2-position residual cannot
    cleanly reproduce, but the V/O routing alone tells us the head will
    route the correct CLEAN_EMBED nibbles into OUTPUT once the attention
    softmax picks the right K row.

    Mirrors ``test_l15_per_op.py``'s
    ``test_l15_lookup_v_slots_route_clean_embed_to_output``.
    """
    del label
    attn = _baked_l10_attn(op_factory)
    HD = attn.W_q.shape[0] // attn.num_heads
    base = head_idx * HD
    # The byte-passthrough chain spec writes V slots 0..15 (lo) and 16..31
    # (hi) for each of the 16 CLEAN_EMBED nibbles. O writes OUTPUT_LO/HI[k]
    # from those V slots.
    for k in range(16):
        v_lo = attn.W_v.data[base + k, _SetDim.CLEAN_EMBED_LO + k].item()
        v_hi = attn.W_v.data[base + 16 + k, _SetDim.CLEAN_EMBED_HI + k].item()
        o_lo = attn.W_o.data[_SetDim.OUTPUT_LO + k, base + k].item()
        o_hi = attn.W_o.data[_SetDim.OUTPUT_HI + k, base + 16 + k].item()
        assert v_lo > 0.0, (
            f"V[h={head_idx} slot={k}, CLEAN_EMBED_LO+{k}] should be "
            f"positive; got {v_lo}"
        )
        assert v_hi > 0.0, (
            f"V[h={head_idx} slot={16 + k}, CLEAN_EMBED_HI+{k}] should "
            f"be positive; got {v_hi}"
        )
        assert o_lo > 0.0, (
            f"O[OUTPUT_LO+{k}, h={head_idx} slot={k}] should be "
            f"positive; got {o_lo}"
        )
        assert o_hi > 0.0, (
            f"O[OUTPUT_HI+{k}, h={head_idx} slot={16 + k}] should be "
            f"positive; got {o_hi}"
        )


# ---------------------------------------------------------------------------
# 7. Symbolic forward: L10 carry relay head 0 V/O routing.
# ---------------------------------------------------------------------------


def test_l10_carry_relay_head_v_o_routes_carry_to_carry():
    """L10 head 0 carry relay routes CARRY[1] / CARRY[2] across AX bytes.

    Head 0 reads V[1] from CARRY+1 and V[2] from CARRY+2 at MARK_AX
    source rows and writes O[CARRY+1] / O[CARRY+2] at AX byte query
    rows. The V/O routing here is what the carry-propagation post-ops
    downstream read. Without it the L10 ADD/SUB carry chain silently
    breaks.
    """
    attn = _baked_l10_attn(make_layer10_carry_relay_bake_op)
    HD = attn.W_q.shape[0] // attn.num_heads
    base = 0  # head 0

    # V[1] reads CARRY+1, V[2] reads CARRY+2.
    v_carry1 = attn.W_v.data[base + 1, _SetDim.CARRY + 1].item()
    v_carry2 = attn.W_v.data[base + 2, _SetDim.CARRY + 2].item()
    # O[CARRY+1] reads back from V slot 1, O[CARRY+2] from V slot 2.
    o_carry1 = attn.W_o.data[_SetDim.CARRY + 1, base + 1].item()
    o_carry2 = attn.W_o.data[_SetDim.CARRY + 2, base + 2].item()

    assert v_carry1 == 1.0, (
        f"V[h=0 slot=1, CARRY+1] should be 1.0 (the carry relay V row); "
        f"got {v_carry1}"
    )
    assert v_carry2 == 1.0, (
        f"V[h=0 slot=2, CARRY+2] should be 1.0; got {v_carry2}"
    )
    assert o_carry1 == 1.0, (
        f"O[CARRY+1, h=0 slot=1] should be 1.0 (the carry relay O row); "
        f"got {o_carry1}"
    )
    assert o_carry2 == 1.0, (
        f"O[CARRY+2, h=0 slot=2] should be 1.0; got {o_carry2}"
    )


# ---------------------------------------------------------------------------
# 8. Op-level metadata pin.
# ---------------------------------------------------------------------------


def test_l10_alu_op_metadata_pinned():
    """Pin the load-bearing Operation metadata for ``layer10_alu``.

    The phase=10.2 + kind="block" + layer_idx=10 + ffn_units_used=1846
    combination is what pins this op to ``model.blocks[10].ffn`` in the
    layout. Any drift silently relocates the op (Unit 9 finding).
    """
    op = make_layer10_alu_op()
    assert op.name == "layer10_alu"
    assert op.phase == 10.2
    assert op.kind == "block"
    assert op.layer_idx == 10
    assert op.ffn_units_used == 1846
    assert op.migrated is True
    assert "OUTPUT_LO" in op.writes
    assert "OUTPUT_HI" in op.writes
    assert "DIV_STAGING" in op.writes
