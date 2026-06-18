"""Byte-identity gate for the L0/L1/L2 attention-head pin drop.

Phase 7.B.2 attn relaxes the L0/L1/L2 attention-head allocators from
``pinned`` to ``dynamic_first_fit``. Because the declaration order in
each layer's layout table matches the legacy pinned positions and the
pools have no gaps, first-fit lands every head at exactly the same
``head_idx`` it held under the pinned regime. These tests bake a fresh
attention module twice -- once via the live (post-drop) op factory, and
once via a reference path that explicitly pins every head -- and assert
``W_q`` / ``W_k`` / ``W_v`` / ``W_o`` and ``alibi_slopes`` are
identical to the last byte.

The reference path bakes via the same primitives as the live path but
forces an explicit pinned-head schedule, so any structural drift in
:func:`Primitives.threshold_attention_head_specs` (or the new
spec-carried ``alibi_slope``) shows up as a tensor diff rather than
slipping through unnoticed.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.unified_compiler.ops.l0_ops import (  # noqa: E402
    _L0_ALIBI_S,
    _L0_HEAD_LAYOUT,
    _L0_OUT_BASE_NAMES,
    _L0_THRESHOLDS,
    make_layer0_threshold_attn_op,
)
from neural_vm.unified_compiler.ops.l1_ops import (  # noqa: E402
    _THRESHOLD_HEAD_LAYOUT,
    make_threshold_attn_op,
)
from neural_vm.unified_compiler.ops.l2_ops import (  # noqa: E402
    _L2_HEAD_LAYOUT,
    make_layer2_threshold_attn_op,
)
from neural_vm.unified_compiler.ops.shared import _as_setdim_proxy  # noqa: E402
from neural_vm.unified_compiler.primitives import (  # noqa: E402
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
    Primitives,
)
from neural_vm.vm_step import AutoregressiveAttention, _SetDim  # noqa: E402


def _new_attention(layer_idx: int) -> AutoregressiveAttention:
    """Mint a fresh AutoregressiveAttention sized to the production bake."""

    return AutoregressiveAttention(
        dim=512,
        num_heads=8,
        layer_idx=layer_idx,
        use_flash_attention=False,
    )


def _bake_via_op(make_op, layer_idx: int) -> AutoregressiveAttention:
    """Bake the live op's attention into a fresh AutoregressiveAttention.

    The L0 op dispatches as a ``kind="block"`` op, so its bake takes a
    ``block`` whose ``.attn`` is what gets written. L1/L2 threshold-attn
    ops dispatch as ``kind="attn"`` (write straight into the attn
    module). We accommodate both by wrapping the attn in a minimal stub
    block that exposes ``.attn``.
    """

    op = make_op()
    attn = _new_attention(layer_idx)

    class _StubBlock:
        pass

    stub = _StubBlock()
    stub.attn = attn

    # ``declarative_bake_fn`` is the canonical entry point; we feed it
    # whichever container its kind expects. ``kind="block"`` -> block,
    # ``kind="attn"`` -> attn directly.
    bake = op.declarative_bake_fn
    dim_positions = _SetDim_dim_positions()
    if op.kind == "block":
        bake(stub, dim_positions, S=100.0)
    elif op.kind == "attn":
        bake(attn, dim_positions, S=100.0)
    else:
        raise AssertionError(f"unexpected op kind {op.kind!r}")
    return attn


def _SetDim_dim_positions() -> dict[str, int]:
    """Build a ``dim_positions`` mapping mirroring ``_SetDim`` enums.

    The threshold ops resolve dim names via ``_as_setdim_proxy``; the
    proxy falls back to ``_SetDim.<NAME>`` when a name is not in the
    mapping, so we can pass an empty dict here without losing
    correctness. Using a dict (not a proxy) is enough -- the proxy
    receives it via ``_as_setdim_proxy(dim_positions)`` inside each bake.
    """

    return {}


# ---------------------------------------------------------------------------
# Reference baker: forces every head at an explicit pinned index using
# the same Primitives helpers the live op delegates to. Mirrors the
# behaviour of the pre-drop L0/L1/L2 bakes.
# ---------------------------------------------------------------------------
def _bake_l0_reference(attn: AutoregressiveAttention) -> None:
    proxy = _as_setdim_proxy(_SetDim_dim_positions())
    if attn.alibi_slopes is not None:
        attn.alibi_slopes.fill_(_L0_ALIBI_S)
    HD = attn.W_q.shape[0] // attn.num_heads
    out_bases = [getattr(proxy, name) for name in _L0_OUT_BASE_NAMES]
    # Reference: pin heads 0..7 explicitly (matches legacy pin layout).
    Primitives.generate_threshold_attention_heads(
        attn,
        list(_L0_THRESHOLDS),
        out_bases,
        _L0_ALIBI_S,
        HD,
        heads=list(range(len(_L0_HEAD_LAYOUT))),
        bd=proxy,
    )


def _bake_l1_reference(attn: AutoregressiveAttention) -> None:
    proxy = _as_setdim_proxy(_SetDim_dim_positions())
    ALIBI_S = 10.0
    IN_STEP_FRESH_ALIBI_S = 0.5
    # L1 reference: heads 0/1/2 (threshold), 3 (HAS_SE), 4 (L1H4), 5 (IN_STEP_FRESH)
    if attn.alibi_slopes is not None:
        attn.alibi_slopes.fill_(ALIBI_S)
        attn.alibi_slopes[3] = 0.0
        attn.alibi_slopes[5] = IN_STEP_FRESH_ALIBI_S
    HD = attn.W_q.shape[0] // attn.num_heads
    Primitives.generate_threshold_attention_heads(
        attn,
        [0.5, 1.5, 2.5],
        [proxy.L1H0, proxy.L1H1, proxy.L1H2],
        ALIBI_S,
        HD,
        heads=[0, 1, 2],
        bd=proxy,
    )
    Primitives.generate_attention_head(
        attn,
        DeclarativeAttentionHeadSpec(
            head_idx=3,
            q=(AP(0, proxy.CONST, 10.0),),
            k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
            v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
            o=(AO(proxy.HAS_SE, 1, 1.0),),
        ),
        HD,
    )
    Primitives.generate_threshold_attention_heads(
        attn,
        [6.5],
        [proxy.L1H4],
        ALIBI_S,
        HD,
        heads=[4],
        bd=proxy,
    )
    Primitives.generate_attention_head(
        attn,
        DeclarativeAttentionHeadSpec(
            head_idx=5,
            q=(AP(0, proxy.CONST, 10.0),),
            k=(
                AP(0, proxy.MARK_SE_ONLY, 10.0),
                AP(0, proxy.MARK_CS, 10.0),
            ),
            v=(
                AP(1, proxy.MARK_SE_ONLY, 1.0),
                AP(1, proxy.MARK_CS, 1.0),
            ),
            o=(AO(proxy.IN_STEP_FRESH, 1, 1.0),),
        ),
        HD,
    )


def _bake_l2_reference(attn: AutoregressiveAttention) -> None:
    proxy = _as_setdim_proxy(_SetDim_dim_positions())
    ALIBI_S = 10.0
    if attn.alibi_slopes is not None:
        attn.alibi_slopes.fill_(ALIBI_S)
    HD = attn.W_q.shape[0] // attn.num_heads
    # L2 reference: head 0 (threshold 5.5). Lookback head is conv-IO
    # gated; default build leaves it off, so head 1 stays zero.
    Primitives.generate_threshold_attention_heads(
        attn,
        [5.5],
        [proxy.L2H0],
        ALIBI_S,
        HD,
        heads=[0],
        bd=proxy,
    )


# ---------------------------------------------------------------------------
# Byte-identity assertions
# ---------------------------------------------------------------------------
def _assert_attn_equal(live: AutoregressiveAttention,
                       ref: AutoregressiveAttention,
                       label: str) -> None:
    for name in ("W_q", "W_k", "W_v", "W_o"):
        live_t = getattr(live, name)
        ref_t = getattr(ref, name)
        assert torch.equal(live_t, ref_t), (
            f"{label}: {name} byte-identity mismatch "
            f"(max abs diff={(live_t - ref_t).abs().max().item()})"
        )
    if live.alibi_slopes is not None and ref.alibi_slopes is not None:
        assert torch.equal(live.alibi_slopes, ref.alibi_slopes), (
            f"{label}: alibi_slopes byte-identity mismatch "
            f"(live={live.alibi_slopes.tolist()}, "
            f"ref={ref.alibi_slopes.tolist()})"
        )


def test_l0_threshold_attn_byte_identical_after_pin_drop():
    """L0 ``layer0_threshold_attn`` bakes byte-identically to the
    pinned reference after the Phase 7.B.2-attn pin drop. First-fit
    on a fresh 8-head pool with 8 declaration-order entries lands
    every head at its legacy index, so Q/K/V/O and alibi_slopes
    match exactly."""

    live = _bake_via_op(make_layer0_threshold_attn_op, layer_idx=0)
    ref = _new_attention(layer_idx=0)
    _bake_l0_reference(ref)
    _assert_attn_equal(live, ref, "L0 layer0_threshold_attn")


def test_l1_threshold_attn_byte_identical_after_pin_drop():
    """L1 ``layer1_threshold_attn`` bakes byte-identically to the
    pinned reference after the Phase 7.B.2-attn pin drop."""

    live = _bake_via_op(make_threshold_attn_op, layer_idx=1)
    ref = _new_attention(layer_idx=1)
    _bake_l1_reference(ref)
    _assert_attn_equal(live, ref, "L1 layer1_threshold_attn")


def test_l2_threshold_attn_byte_identical_after_pin_drop():
    """L2 ``layer2_threshold_attn`` bakes byte-identically to the
    pinned reference after the Phase 7.B.2-attn pin drop."""

    live = _bake_via_op(make_layer2_threshold_attn_op, layer_idx=2)
    ref = _new_attention(layer_idx=2)
    _bake_l2_reference(ref)
    _assert_attn_equal(live, ref, "L2 layer2_threshold_attn")


def test_l0_head_layout_has_no_pins():
    """Layout table for L0 attention heads must not carry ``head_idx``
    fields any more; first-fit resolves the head_idx at bake time.

    Each layout entry is a 1-tuple ``(op_name,)`` after the drop, not a
    2-tuple ``(op_name, head_idx)``.
    """

    for entry in _L0_HEAD_LAYOUT:
        assert len(entry) == 1, (
            f"L0 layout entry {entry!r} still carries a pinned head_idx; "
            f"expected a bare (op_name,) tuple."
        )


def test_l1_head_layout_has_no_pins():
    """Same constraint for L1 attention heads."""

    for entry in _THRESHOLD_HEAD_LAYOUT:
        assert len(entry) == 1, (
            f"L1 layout entry {entry!r} still carries a pinned head_idx; "
            f"expected a bare (op_name,) tuple."
        )


def test_l2_head_layout_has_no_pins():
    """Same constraint for L2 attention heads."""

    for entry in _L2_HEAD_LAYOUT:
        assert len(entry) == 1, (
            f"L2 layout entry {entry!r} still carries a pinned head_idx; "
            f"expected a bare (op_name,) tuple."
        )
