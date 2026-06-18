"""Per-op audit harness for Layer 3 (carry-forward attn + FFN).

Layer 3 owns the residual-stream book-keeping for register evolution:

* ``layer3_carry_forward_attn`` — 8 attention heads. Heads 0-3 use
  ``Primitives.carry_forward_attention`` to copy the previous step's
  PC / AX / SP / BP marker byte into the current step's marker
  ``EMBED_LO/HI`` (or ``AX_CARRY_LO/HI`` for head 1). Head 4 is the
  declarative STACK0 marker carry retirer. Heads 5-7 are declarative
  relays (AX_FULL, LEV BP->PC, PC byte1 -> TEMP). Has populated
  ``claims=`` (224 cells across V slots for heads 0-3, 5, 6, 7).
* ``layer3_ffn`` — block-pinned FFN that handles PC / SP / BP
  first-step defaults plus PC byte-0 increment plus the byte-1 carry
  repair (see ``_add_pc_byte1_output_rules``).
* ``_layer3_ffn_dep_anchor`` — no-op companion ``kind="ffn"`` op that
  reserves a dep-graph layer slot.
* ``layer3_convo_io_state_init`` — conversational-I/O state init FFN,
  gated by ``enable_conversational_io=True``. Default build does not
  flip the flag.

Only ``layer3_carry_forward_attn`` ships with non-empty ``claims`` in
the default build, so it is the sole op the static declarative verifier
will surface. The other three are tracked here as known-absent so a
future claim-map addition (or enable=True flip) trips a loud failure
forcing them into the drift-checked list.

See ``test_l0_marker_transitions.py`` for the shared fixture rationale
and ``test_l3_pc_byte1_carry.py`` for the focused FFN repair tests.
"""

from __future__ import annotations

import pytest
import torch

from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
from neural_vm.unified_compiler.ops.l3_ops import (
    make_carry_forward_attn_op,
    make_register_default_ffn_op,
)
from neural_vm.vm_step import _SetDim

from ._per_op_audit import assert_no_drift, assert_op_absent, assert_op_fires


# ---------------------------------------------------------------------------
# Declarative claim-verification gate
# ---------------------------------------------------------------------------

L3_OPS_WITH_CLAIMS = (
    "layer3_carry_forward_attn",
)

# Ops that intentionally ship with empty ``claims`` in the default
# build:
#
# * ``layer3_ffn`` — block bake; per-cell claims have not been
#   authored (the FFN's spec_section is BLOG_SPEC.md#registers but the
#   declarative cell map is still in flight).
# * ``_layer3_ffn_dep_anchor`` — kind="ffn" no-op companion whose only
#   job is to reserve a layer slot in the dep graph; bake is empty.
# * ``layer3_convo_io_state_init`` — gated by
#   ``enable_conversational_io`` (default False) — bake is a no-op so
#   no FFN units are touched and no claims are emitted.
#
# If any of these later gain a ``claims=`` set they will appear in the
# static report and the absent-check will fail, forcing migration into
# ``L3_OPS_WITH_CLAIMS`` above so they share the drift gate.
L3_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "layer3_ffn",
    "_layer3_ffn_dep_anchor",
    "layer3_convo_io_state_init",
)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L3_OPS_WITH_CLAIMS)
def test_l3_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L3", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L3_OPS_WITH_CLAIMS)
def test_l3_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L3", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L3_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l3_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L3", op_name)


# ---------------------------------------------------------------------------
# Symbolic-forward tests
#
# Local stub modules mirror the per-feature stubs used by
# ``test_l3_pc_byte1_carry.py`` and ``test_l14_output_cleanup.py``: a
# bare ``W_up/W_gate/W_down`` triple for the FFN and ``W_q/W_k/W_v/W_o``
# for the attention block. Both stubs are intentionally small so the
# symbolic forward is fast and isolated from compiler-side machinery.
# ---------------------------------------------------------------------------


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 200):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


class _StubAttn:
    def __init__(self, *, d_model: int = 512, num_heads: int = 8):
        self.num_heads = num_heads
        self.W_q = torch.zeros(d_model, d_model)
        self.W_k = torch.zeros(d_model, d_model)
        self.W_v = torch.zeros(d_model, d_model)
        self.W_o = torch.zeros(d_model, d_model)
        self.alibi_slopes = torch.zeros(num_heads)


class _StubBlock:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 200):
        self.ffn = _StubFFN(d_model=d_model, hidden_dim=hidden_dim)


def _apply_stub_ffn(ffn: _StubFFN, x: torch.Tensor) -> torch.Tensor:
    up = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up)
    gate = x @ ffn.W_gate.t() + ffn.b_gate
    return x + (up * gate) @ ffn.W_down.t()


def _multihead_attention_forward(
    attn: _StubAttn,
    x: torch.Tensor,
    head_idx: int,
) -> torch.Tensor:
    """Single-head softmax forward for one attention head.

    ``x`` is shape ``(seq, d_model)``. Returns shape ``(seq, d_model)``
    containing only this head's contribution (W_o restricted to the
    head's slot range).
    """
    d_model = x.shape[-1]
    HD = d_model // attn.num_heads
    base = head_idx * HD

    q_full = x @ attn.W_q.t()
    k_full = x @ attn.W_k.t()
    v_full = x @ attn.W_v.t()

    q = q_full[:, base : base + HD]
    k = k_full[:, base : base + HD]
    v = v_full[:, base : base + HD]

    scores = q @ k.t()
    attn_w = torch.softmax(scores, dim=-1)
    head_out = attn_w @ v  # (seq, HD)

    W_o_head = attn.W_o[:, base : base + HD]  # (d_model, HD)
    return head_out @ W_o_head.t()


# ---------------------------------------------------------------------------
# Symbolic forward: PC byte-0 increment (L3 FFN)
#
# At a step-0 PC marker (MARK_PC AND NOT HAS_SE), L3 FFN sets:
#   OUTPUT_LO[pc_lo] = +1     where (pc_lo, pc_hi) = nibbles of (PC_OFFSET+INSTR_WIDTH)
#   EMBED_LO[pc_lo]  = +1     (also written to EMBED for L4 relay)
#   OUTPUT_HI[pc_hi] = +1
#   EMBED_HI[pc_hi]  = +1
#
# At a step-1+ PC marker (MARK_PC AND HAS_SE), the same emission is
# cancelled by a follow-up unit (W_gate[MARK_PC]=1, W_up[HAS_SE]=S),
# so OUTPUT/EMBED at the default slot stays at zero (later layers /
# carry-forward attention own the next-step PC byte 0).
# ---------------------------------------------------------------------------


def _pc_marker_row(has_se: bool) -> torch.Tensor:
    row = torch.zeros(1, 1, 512)
    row[..., _SetDim.MARK_PC] = 1.0
    if has_se:
        row[..., _SetDim.HAS_SE] = 1.0
    return row


def test_l3_ffn_first_step_emits_pc_offset_plus_instr_width_at_pc_marker():
    """Symbolic forward: STEP 0 (NOT HAS_SE) MARK_PC -> PC = PC_OFFSET + INSTR_WIDTH.

    Verifies the FIRST-STEP default unit: at the PC marker with no
    prior step, OUTPUT_LO/HI carry the nibble decomposition of the
    PC after the synthetic first instruction. EMBED_LO/HI mirror this
    so L4 can relay the AX marker for L5 fetch.
    """
    block = _StubBlock()
    make_register_default_ffn_op().bake_fn(block, {}, 100.0)

    first_pc = PC_OFFSET + INSTR_WIDTH
    pc_lo = first_pc & 0xF
    pc_hi = (first_pc >> 4) & 0xF

    x = _pc_marker_row(has_se=False)
    y = _apply_stub_ffn(block.ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + pc_lo] > 0.9, (
        f"first-step PC marker should emit OUTPUT_LO[{pc_lo}]>0; "
        f"got {float(y[_SetDim.OUTPUT_LO + pc_lo]):.3f}"
    )
    assert y[_SetDim.OUTPUT_HI + pc_hi] > 0.9
    assert y[_SetDim.EMBED_LO + pc_lo] > 0.9, (
        "L4 PC relay reads from EMBED — first-step PC must also write "
        "EMBED_LO[pc_lo] so the AX marker's L4 relay can copy it."
    )
    assert y[_SetDim.EMBED_HI + pc_hi] > 0.9


def test_l3_ffn_subsequent_step_cancels_first_step_default_at_pc_marker():
    """Symbolic forward: STEP 1+ (HAS_SE) MARK_PC -> PC default cancelled.

    At step-1+ PC markers (HAS_SE=1), the carry-forward attention head
    plus L4 PC relay own the PC byte 0 value. L3 FFN's first-step
    default is cancelled by a follow-up unit gated on HAS_SE, leaving
    OUTPUT_LO/HI at the default slot near zero so it does not double-
    drive the byte.
    """
    block = _StubBlock()
    make_register_default_ffn_op().bake_fn(block, {}, 100.0)

    first_pc = PC_OFFSET + INSTR_WIDTH
    pc_lo = first_pc & 0xF
    pc_hi = (first_pc >> 4) & 0xF

    x_step0 = _pc_marker_row(has_se=False)
    x_step1 = _pc_marker_row(has_se=True)

    y0 = _apply_stub_ffn(block.ffn, x_step0)[0, 0]
    y1 = _apply_stub_ffn(block.ffn, x_step1)[0, 0]

    # Step-0 emits a strongly-positive value at the default slot.
    assert y0[_SetDim.OUTPUT_LO + pc_lo] > 0.9
    assert y0[_SetDim.EMBED_LO + pc_lo] > 0.9

    # Step-1+ leaves the default slot near zero (default + cancel sum
    # to ~0; SiLU nonlinearity introduces a tiny residual that is still
    # well below the step-0 magnitude).
    assert abs(float(y1[_SetDim.OUTPUT_LO + pc_lo])) < 0.1, (
        f"step-1+ PC default should be cancelled by HAS_SE-gated unit; "
        f"got {float(y1[_SetDim.OUTPUT_LO + pc_lo]):.4f}"
    )
    assert abs(float(y1[_SetDim.EMBED_LO + pc_lo])) < 0.1
    assert abs(float(y1[_SetDim.OUTPUT_HI + pc_hi])) < 0.1
    assert abs(float(y1[_SetDim.EMBED_HI + pc_hi])) < 0.1


# ---------------------------------------------------------------------------
# Symbolic forward: carry-forward PC head (L3 attn head 0)
#
# Carry-forward attention copies the previous step's MARK_PC token's
# EMBED_LO/HI nibbles into the current step's MARK_PC token's
# EMBED_LO/HI. The K pattern fires at the previous step's PC byte 0
# (L1H1[PC] AND NOT L1H0[PC]).
#
# To validate the propagation symbolically we build a 2-token sequence:
#   token 0: previous step, "PC byte 0" — L1H1[PC]=1, L1H0[PC]=0,
#            EMBED_LO[src_lo]=1, EMBED_HI[src_hi]=1
#   token 1: current step,  "PC marker" — MARK_PC=1
#
# Run a single-head softmax forward on head 0 and assert token 1's
# output writes EMBED_LO[src_lo] and EMBED_HI[src_hi] (carried from
# token 0). Token 0 must NOT spuriously copy anything (its Q is empty
# at MARK_PC, the carry-forward Q gate at slot 33 keeps it from firing).
# ---------------------------------------------------------------------------


def test_l3_carry_forward_attn_head0_propagates_prev_step_pc_embed():
    """Symbolic forward: previous PC byte 0 EMBED -> current PC marker EMBED.

    Head 0 (PC carry) is wired via
    ``Primitives.carry_forward_attention`` with src/out =
    EMBED_LO/EMBED_HI. With prior-step PC byte 0 carrying a known
    nibble pair, the current-step PC marker token should pick it up.
    """
    attn = _StubAttn()
    make_carry_forward_attn_op().bake_fn(attn, {}, 100.0)

    # Pick a non-zero nibble pair to make sure we are propagating
    # values rather than zeros.
    src_lo_nibble = 5
    src_hi_nibble = 11

    # Build a 2-token sequence (seq=2, d_model=512).
    x = torch.zeros(2, 512)
    # Token 0: previous step's PC byte 0 — the carry source.
    x[0, _SetDim.L1H1 + 0] = 1.0       # L1H1[PC_I=0]
    # L1H0[PC] stays 0 so K=L1H1*L - L1H0*L is maximally positive.
    x[0, _SetDim.EMBED_LO + src_lo_nibble] = 1.0
    x[0, _SetDim.EMBED_HI + src_hi_nibble] = 1.0
    x[0, _SetDim.CONST] = 1.0
    # Token 1: current step's PC marker — the carry destination.
    x[1, _SetDim.MARK_PC] = 1.0
    x[1, _SetDim.CONST] = 1.0

    head_out = _multihead_attention_forward(attn, x, head_idx=0)

    # The destination (token 1) should now carry the source nibble
    # contribution in EMBED_LO/EMBED_HI. With L=15 the softmax is
    # extremely sharp so the propagated value is ~1.0 modulo softmax-
    # over-2-tokens splitting.
    dst_embed_lo = float(head_out[1, _SetDim.EMBED_LO + src_lo_nibble])
    dst_embed_hi = float(head_out[1, _SetDim.EMBED_HI + src_hi_nibble])
    assert dst_embed_lo > 0.4, (
        f"head 0 should copy previous PC byte 0's EMBED_LO[{src_lo_nibble}] "
        f"into current PC marker; got {dst_embed_lo:.4f}"
    )
    assert dst_embed_hi > 0.4, (
        f"head 0 should copy previous PC byte 0's EMBED_HI[{src_hi_nibble}] "
        f"into current PC marker; got {dst_embed_hi:.4f}"
    )

    # Sanity check: the SOURCE slot is the only nibble that should be
    # strongly written. Other EMBED_LO slots must stay near zero.
    for k in range(16):
        if k == src_lo_nibble:
            continue
        val = float(head_out[1, _SetDim.EMBED_LO + k])
        assert abs(val) < 0.05, (
            f"head 0 leaked into EMBED_LO[{k}] (expected zero); "
            f"got {val:.4f}"
        )


def test_l3_carry_forward_attn_head1_propagates_prev_step_ax_to_ax_carry():
    """Symbolic forward: previous AX byte 0 EMBED -> current AX_CARRY band.

    Head 1 is the AX carry-forward head. Unlike head 0 it routes to
    AX_CARRY_LO/AX_CARRY_HI (so the AX marker's EMBED_LO/HI stays
    free for the current step's first-byte output). Source still
    reads EMBED_LO/HI from the previous AX byte 0 row.
    """
    attn = _StubAttn()
    make_carry_forward_attn_op().bake_fn(attn, {}, 100.0)

    src_lo_nibble = 3
    src_hi_nibble = 14

    AX_I = 1
    x = torch.zeros(2, 512)
    x[0, _SetDim.L1H1 + AX_I] = 1.0
    x[0, _SetDim.EMBED_LO + src_lo_nibble] = 1.0
    x[0, _SetDim.EMBED_HI + src_hi_nibble] = 1.0
    x[0, _SetDim.CONST] = 1.0
    x[1, _SetDim.MARK_AX] = 1.0
    x[1, _SetDim.CONST] = 1.0

    head_out = _multihead_attention_forward(attn, x, head_idx=1)

    dst_lo = float(head_out[1, _SetDim.AX_CARRY_LO + src_lo_nibble])
    dst_hi = float(head_out[1, _SetDim.AX_CARRY_HI + src_hi_nibble])
    assert dst_lo > 0.4, (
        f"head 1 should copy previous AX byte 0's EMBED_LO[{src_lo_nibble}] "
        f"into current AX marker's AX_CARRY_LO; got {dst_lo:.4f}"
    )
    assert dst_hi > 0.4, (
        f"head 1 should copy previous AX byte 0's EMBED_HI[{src_hi_nibble}] "
        f"into current AX marker's AX_CARRY_HI; got {dst_hi:.4f}"
    )

    # EMBED_LO must NOT receive this head's write (head 0 owns EMBED_LO
    # for PC carry; head 1 routes AX to a separate band).
    leak = float(head_out[1, _SetDim.EMBED_LO + src_lo_nibble])
    assert abs(leak) < 0.05, (
        f"head 1 should NOT write current AX marker's EMBED_LO; "
        f"got {leak:.4f}"
    )
