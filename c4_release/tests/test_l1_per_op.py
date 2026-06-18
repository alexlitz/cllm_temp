"""Per-op audit harness for Layer 1 (threshold attn + byte-index FFN).

Layer 1 is the first compute layer above L0's marker transitions. It
produces the residual-stream flags every downstream layer reads from:

* ``layer1_threshold_attn`` -- 6 attention heads:
    - Heads 0/1/2: fine threshold heads that read the marker count and
      produce ``L1H0`` / ``L1H1`` / ``L1H2`` (the "X markers within
      window" flags consumed by L2/L3/L4 carry-forward logic).
    - Head 3: ``HAS_SE`` global STEP_END existence detector.
    - Head 4: threshold 6.5 that produces ``L1H4`` (the "in BP area"
      flag at the 7th marker).
    - Head 5 (B7-1): ``IN_STEP_FRESH`` recency-to-SE decay using a
      positive ALiBi slope; emits ~1.0 at fresh STEP_END/MARK_CS and
      decays to ~0 within a ~35-token step window.
  Ships a populated 64-cell claim grid (V slots 1..7 for the threshold
  heads, plus the dedicated K/V/O cells for heads 3 and 5).
* ``layer1_ffn`` -- block-pinned FFN producing five canonical flags
  consumed everywhere downstream:
    - ``STACK0_BYTE0`` (unit 0): L1H4[BP_I] AND IS_BYTE AND NOT
      H1[BP_I] -- the "byte 0 of the STACK0 marker" flag.
    - ``BYTE_INDEX_0..3`` (units 1..4): IS_BYTE AND L1H{n}[any] AND
      NOT L1H{n-1}[any] -- the canonical "this is byte i of the
      current marker" flag the entire downstream byte-routing fabric
      reads from.
  Has a 5-cell ``claims=`` grid (one per output dim).

This harness mirrors the ``test_l8_per_op.py`` / ``test_l15_per_op.py``
patterns:

  1. ``static_claims_report``-backed drift checks for every L1 op (both
     ship populated claim grids in the default build).
  2. ``assert_op_fires`` smokes for every L1 op -- the bake must
     dispatch and emit at least one observable write.
  3. Symbolic forward for the L1 FFN ``STACK0_BYTE0`` flag: drive a
     residual with L1H4[BP_I] and IS_BYTE set, confirm STACK0_BYTE0
     fires; toggle H1[BP_I] on and confirm it is suppressed (the gate
     side of the unit).
  4. Symbolic forward for the L1 FFN ``BYTE_INDEX_0`` flag: drive a
     residual with L1H1 set at any marker slot, confirm BYTE_INDEX_0
     fires; toggle L1H0 (the blocker) on and confirm it is suppressed.
  5. Symbolic forward for the L1 threshold attn head 3 (HAS_SE): drive
     a 2-position residual with MARK_SE_ONLY at one position and assert
     HAS_SE is written at every position via the global softmax.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureAttention, PureFFN  # noqa: E402
from neural_vm.unified_compiler.ops.l1_ops import (  # noqa: E402
    make_threshold_ffn_op,
    make_threshold_attn_op,
)
from neural_vm.vm_step import _SetDim  # noqa: E402

from ._per_op_audit import (  # noqa: E402
    assert_no_drift,
    assert_op_fires,
)


# ---------------------------------------------------------------------------
# L1 op inventory. Both L1 ops ship a populated ``claims`` set in the
# default build, so the session-scoped ``static_claims_report`` fixture
# will produce a result entry for each.
# ---------------------------------------------------------------------------


L1_OPS_WITH_CLAIMS = (
    "layer1_threshold_attn",
    "layer1_ffn",
)


# ---------------------------------------------------------------------------
# 1. Drift checks via ``static_claims_report``
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L1_OPS_WITH_CLAIMS)
def test_l1_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L1", op_name)


# ---------------------------------------------------------------------------
# 2. Fires-during-bake for every claim-bearing L1 op
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L1_OPS_WITH_CLAIMS)
def test_l1_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L1", op_name)


# ---------------------------------------------------------------------------
# 3. Symbolic forward: L1 FFN STACK0_BYTE0 flag.
# ---------------------------------------------------------------------------
#
# Unit 0 of the L1 FFN is gated as:
#   gate = (1 - H1[BP_I])  (clamped via SwiGLU)
#   up   = S * L1H4[BP_I] + S * IS_BYTE - S * 1.5
# It writes ``2/S`` to STACK0_BYTE0 when both inputs are present AND H1[BP_I]
# is zero. At BP byte 0 the L1H4[BP_I] is high (since BP_I is the 4th
# marker / threshold 6.5) and H1[BP_I] is zero (BP byte 0 doesn't itself
# read H1[BP_I]).


_BP_I = 3
_SP_I = 2


def _build_layer1_ffn(d_model: int = 512, hidden_dim: int = 16) -> PureFFN:
    """Build a PureFFN with only the L1 FFN units baked.

    The L1 FFN occupies exactly 5 units (``ffn_units_used=5``); a hidden
    width of 16 gives generous slack for any future additions without
    forcing a multi-megabyte allocation per test.
    """
    ffn = PureFFN(dim=d_model, hidden_dim=hidden_dim)
    with torch.no_grad():
        ffn.W_up.data.zero_()
        ffn.b_up.data.zero_()
        ffn.W_gate.data.zero_()
        ffn.b_gate.data.zero_()
        ffn.W_down.data.zero_()
        # Build the dim_positions map _SetDim exposes, matching the
        # ``compile_compact_layout`` helper in ``_per_op_audit``.
        dim_positions = {
            name: getattr(_SetDim, name)
            for name in dir(_SetDim)
            if not name.startswith("_")
            and isinstance(getattr(_SetDim, name), int)
        }
        make_threshold_ffn_op().bake_fn(ffn, dim_positions, 100.0)
    return ffn


def test_l1_ffn_stack0_byte0_fires_at_bp_byte0_signature():
    """L1 FFN unit 0 emits STACK0_BYTE0 when L1H4[BP_I] AND IS_BYTE fire.

    This is the canonical "byte 0 of the STACK0 marker is here" flag that
    downstream L10/L15 stack pipelines read to discriminate STACK0 byte 0
    rows. If this regresses, every STACK0 byte 0 routing path silently
    misroutes.
    """
    ffn = _build_layer1_ffn()
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.L1H4 + _BP_I] = 1.0
    x[0, 0, _SetDim.IS_BYTE] = 1.0
    # H1[BP_I] stays 0 -> SwiGLU gate side stays at +1.0.

    with torch.no_grad():
        y = ffn(x)
    stack0_byte0 = float(y[0, 0, _SetDim.STACK0_BYTE0].item())
    assert stack0_byte0 > 0.1, (
        f"L1 FFN unit 0 should fire STACK0_BYTE0 at the BP byte 0 "
        f"signature; got STACK0_BYTE0={stack0_byte0:.4f}. The unit "
        f"writes 2/S * SiLU(up) * gate; with up=S*0.5 (after bias) and "
        f"gate=1.0 the SwiGLU output should be >0.1."
    )


def test_l1_ffn_stack0_byte0_suppressed_by_h1_bp_marker():
    """L1 FFN unit 0 gate is (1 - H1[BP_I]); H1[BP_I]=1 zeroes the gate.

    H1[BP_I]=1 happens on the BP marker token itself (not the byte 0
    position that follows). Without this suppression L1 would emit
    STACK0_BYTE0 at marker rows too and the byte-0 specificity would be
    lost.
    """
    ffn = _build_layer1_ffn()
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.L1H4 + _BP_I] = 1.0
    x[0, 0, _SetDim.IS_BYTE] = 1.0
    # H1[BP_I]=1 collapses the SwiGLU gate to 0.
    x[0, 0, _SetDim.H1 + _BP_I] = 1.0

    with torch.no_grad():
        y = ffn(x)
    stack0_byte0 = float(y[0, 0, _SetDim.STACK0_BYTE0].item())
    assert abs(stack0_byte0) < 0.05, (
        f"L1 FFN unit 0 should be gate-suppressed when H1[BP_I]=1; "
        f"got STACK0_BYTE0={stack0_byte0:.4f}. The gate side is "
        f"(1 - H1[BP_I]) so a marker row should produce ~0.0."
    )


# ---------------------------------------------------------------------------
# 4. Symbolic forward: L1 FFN BYTE_INDEX_0 flag.
# ---------------------------------------------------------------------------


def test_l1_ffn_byte_index_0_fires_when_l1h1_above_l1h0():
    """L1 FFN unit 1 emits BYTE_INDEX_0 when L1H1 is high AND L1H0 is low.

    The byte-index ladder is implemented as a difference: BYTE_INDEX_0
    fires at distance 1 from a marker (where L1H1 is high) but only when
    we are NOT also at a marker (L1H0 high). This is the canonical
    "byte 0 of any marker" flag the entire downstream byte-routing
    fabric depends on.
    """
    ffn = _build_layer1_ffn()
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.IS_BYTE] = 1.0
    # L1H1 at any marker slot fires the up side. We use slot 0 (PC marker).
    x[0, 0, _SetDim.L1H1 + 0] = 1.0
    # L1H0 stays 0 -> gate side stays at +1.0.

    with torch.no_grad():
        y = ffn(x)
    byte_index_0 = float(y[0, 0, _SetDim.BYTE_INDEX_0].item())
    assert byte_index_0 > 0.1, (
        f"L1 FFN unit 1 should fire BYTE_INDEX_0 when L1H1 is high "
        f"AND L1H0 is low; got BYTE_INDEX_0={byte_index_0:.4f}."
    )


def test_l1_ffn_byte_index_0_suppressed_by_l1h0_blocker():
    """L1 FFN unit 1 gate is (1 - L1H0); L1H0=1 zeroes the gate.

    Without this blocker, BYTE_INDEX_0 would also fire at marker rows
    (L1H0 high) and downstream byte-index discrimination would be
    broken.
    """
    ffn = _build_layer1_ffn()
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.IS_BYTE] = 1.0
    x[0, 0, _SetDim.L1H1 + 0] = 1.0
    # L1H0 at the same marker slot collapses the gate.
    x[0, 0, _SetDim.L1H0 + 0] = 1.0

    with torch.no_grad():
        y = ffn(x)
    byte_index_0 = float(y[0, 0, _SetDim.BYTE_INDEX_0].item())
    assert abs(byte_index_0) < 0.05, (
        f"L1 FFN unit 1 should be gate-suppressed when L1H0=1; got "
        f"BYTE_INDEX_0={byte_index_0:.4f}."
    )


# ---------------------------------------------------------------------------
# 5. Symbolic forward: L1 threshold attn head 3 (HAS_SE).
# ---------------------------------------------------------------------------


def _build_layer1_threshold_attn(
    *, num_heads: int = 8, dim: int = 512
) -> PureAttention:
    """PureAttention with the L1 threshold attn bake applied."""
    attn = PureAttention(dim=dim, num_heads=num_heads)
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.zero_()
        dim_positions = {
            name: getattr(_SetDim, name)
            for name in dir(_SetDim)
            if not name.startswith("_")
            and isinstance(getattr(_SetDim, name), int)
        }
        make_threshold_attn_op().bake_fn(attn, dim_positions, 100.0)
    return attn


def test_l1_threshold_attn_head3_has_se_writes_global_flag():
    """L1 attn head 3 propagates MARK_SE_ONLY to HAS_SE at every position.

    Head 3 uses a global softmax with Q[CONST]=10 over K[MARK_SE_ONLY]=10
    and V[MARK_SE_ONLY]=1.0 -> O[HAS_SE]=1.0. Once any position in the
    sequence carries MARK_SE_ONLY, every query position should receive
    a positive HAS_SE write.

    The L1 threshold attn bake also zeroes head 3's ALiBi slope so it
    can attend globally regardless of distance. If a regression flips
    head 3's slope back to the default ALIBI_S=10 the attention to
    distant SE tokens collapses and HAS_SE stays near zero at later
    positions.
    """
    attn = _build_layer1_threshold_attn()
    # Two-position sequence: pos 0 = SE marker; pos 1 = query.
    x = torch.zeros(1, 2, 512)
    x[0, :, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.MARK_SE_ONLY] = 1.0
    # pos 1 is the query position carrying no SE flag of its own; the
    # global softmax should still pull SE from pos 0 into HAS_SE.

    with torch.no_grad():
        y = attn(x)
    # The attention block returns ``x + attn_out`` (residual). Subtract
    # to isolate the head's contribution.
    delta = y - x
    has_se_at_query = float(delta[0, 1, _SetDim.HAS_SE].item())
    assert has_se_at_query > 0.1, (
        f"L1 head 3 should write HAS_SE>0.1 at the query position when "
        f"MARK_SE_ONLY appears anywhere in the sequence; got "
        f"HAS_SE={has_se_at_query:.4f}. Check that head 3's ALiBi slope "
        f"is zeroed (the bake sets alibi_slopes[3]=0.0)."
    )


def test_l1_threshold_attn_alibi_slopes_pinned():
    """The L1 threshold attn bake pins specific ALiBi slope values.

    Heads 0-2/4: ALIBI_S = 10.0 (the canonical threshold-head slope).
    Head 3:      0.0 (global STEP_END detection).
    Head 5:      0.5 (B7-1 IN_STEP_FRESH decay; matches L9 ALiBi
                  memory-lookup slope for ABI consistency).

    Pin these so a future drift can't silently break either the
    threshold head ladder or B7-1's recency decay.
    """
    attn = _build_layer1_threshold_attn()
    if not hasattr(attn, "alibi_slopes") or attn.alibi_slopes is None:
        pytest.skip("PureAttention build lacks ALiBi slopes on this branch")
    slopes = attn.alibi_slopes
    assert slopes[0].item() == 10.0
    assert slopes[1].item() == 10.0
    assert slopes[2].item() == 10.0
    assert slopes[3].item() == 0.0, (
        f"Head 3 (HAS_SE) ALiBi slope must be 0 for global SE detection; "
        f"got {slopes[3].item()}"
    )
    assert slopes[4].item() == 10.0
    assert slopes[5].item() == 0.5, (
        f"Head 5 (IN_STEP_FRESH) ALiBi slope must be 0.5 (B7-1); got "
        f"{slopes[5].item()}"
    )
