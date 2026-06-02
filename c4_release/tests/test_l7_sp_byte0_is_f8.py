"""Unit tests for the B7-2 SP_BYTE0_IS_F8 producer (L7 head 6 extension).

The op (``layer7_sp_byte0_is_f8``) writes 1.0 into the ``SP_BYTE0_IS_F8``
structural dim at every MARK_SP row whose carry-forwarded SP byte 0 == 0xF8
(i.e. EMBED_LO+8 AND EMBED_HI+15 both fire on the MARK_SP row, as L3 head
2's SP carry-forward leaves them).

Coverage:
  1. Static spec wiring -- V slots 6+7 read EMBED_LO+8 / EMBED_HI+15 and
     both write to ``SP_BYTE0_IS_F8`` with weight 0.5 each.
  2. Op registration -- the op lands in ``all_core_ops`` at L7 phase 7.6
     with the expected reads/writes.
  3. Synthetic forward -- driving a one-row sequence with carry-forwarded
     SP byte 0 set to various values, ``SP_BYTE0_IS_F8`` ~ 1.0 only when
     the byte is 0xF8; half-matches (only lo or only hi) cap at ~0.5.
  4. _SetDim layout -- ``SP_BYTE0_IS_F8`` aliases the dead H5+0 slot at 95.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.base_layers import PureAttention  # noqa: E402
from neural_vm.unified_compiler.ops.l7_ops import (  # noqa: E402
    _layer7_sp_byte0_is_f8_spec,
    make_layer7_sp_byte0_is_f8_op,
)
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Static spec wiring
# ---------------------------------------------------------------------------


def test_spec_writes_to_head_6_slot_6_and_7():
    spec = _layer7_sp_byte0_is_f8_spec(_SetDim)
    assert spec.head_idx == 6
    # Q and K are empty so we don't overwrite head 6's existing wiring
    # set by layer7_memory_heads (phase=7).
    assert spec.q == ()
    assert spec.k == ()
    # V slots 6 and 7 read the lo nibble = 8 and hi nibble = F columns.
    v_by_slot = {(w.slot, w.dim): w.weight for w in spec.v}
    assert v_by_slot[(6, _SetDim.EMBED_LO + 8)] == 1.0
    assert v_by_slot[(7, _SetDim.EMBED_HI + 15)] == 1.0
    # No other V writes -- we must not collide with slots 1..5 (head 6's
    # existing PSH/CMP relays).
    assert len(spec.v) == 2
    # Both O writes sum into SP_BYTE0_IS_F8 with weight 0.5.
    o_by_pair = {(w.out_dim, w.slot): w.weight for w in spec.o}
    assert o_by_pair[(_SetDim.SP_BYTE0_IS_F8, 6)] == 0.5
    assert o_by_pair[(_SetDim.SP_BYTE0_IS_F8, 7)] == 0.5
    assert len(spec.o) == 2


def test_setdim_sp_byte0_is_f8_is_slot_95():
    """SP_BYTE0_IS_F8 aliases the dead H5+0 slot (reclaimed per B6-K Section 5)."""
    assert _SetDim.SP_BYTE0_IS_F8 == 95
    assert _SetDim.SP_BYTE0_IS_F8 == _SetDim.H5  # aliased onto H5's base slot


# ---------------------------------------------------------------------------
# 2. Op registration
# ---------------------------------------------------------------------------


def test_op_registered_in_all_core_ops():
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    ops = all_core_ops()
    found = [op for op in ops if op.name == "layer7_sp_byte0_is_f8"]
    assert len(found) == 1, (
        f"Expected exactly 1 layer7_sp_byte0_is_f8 op, found {len(found)}"
    )
    op = found[0]
    assert op.layer_idx == 7
    assert op.kind == "block"
    assert op.phase == 7.6
    assert op.migrated is True
    assert op.declarative_authority == "spec_generated"
    assert op.writes == {"SP_BYTE0_IS_F8"}
    for required_read in ("MARK_SP", "EMBED_LO", "EMBED_HI"):
        assert required_read in op.reads, f"reads missing {required_read}"


def test_op_phase_after_layer7_memory_heads():
    """The new op must bake AFTER layer7_memory_heads so head 6's existing
    Q/K wiring is in place when our V/O slots fire."""
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    ops = {op.name: op for op in all_core_ops()}
    assert ops["layer7_sp_byte0_is_f8"].phase > ops["layer7_memory_heads"].phase


# ---------------------------------------------------------------------------
# 3. Synthetic forward -- head 6 with full Q/K (from memory_heads) +
#    our V/O slot addition produces SP_BYTE0_IS_F8 ~ 1.0 only on 0xF8.
# ---------------------------------------------------------------------------


def _build_baked_attn(num_heads: int = 8, dim: int = 512) -> PureAttention:
    """Build a PureAttention with both layer7_memory_heads head 6 wiring AND
    the layer7_sp_byte0_is_f8 V/O slot extension. We do NOT bake heads 0-5
    or 7 -- they would not contribute to slot 95 in this synthetic input."""
    from neural_vm.unified_compiler.ops.l7_ops import _layer7_memory_head_specs

    attn = PureAttention(dim=dim, num_heads=num_heads)
    HD = dim // num_heads
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        # Bake head 6 from layer7_memory_heads (Q/K + slots 1..5).
        all_specs = _layer7_memory_head_specs(_SetDim)
        head6_spec = next(s for s in all_specs if s.head_idx == 6)
        Primitives.generate_attention_head(attn, head6_spec, HD)
        # Bake our new SP_BYTE0_IS_F8 V/O slots on top.
        Primitives.generate_attention_head(
            attn, _layer7_sp_byte0_is_f8_spec(_SetDim), HD,
        )
    return attn


def _run_sp_marker_forward(sp_byte0: int, num_heads: int = 8, dim: int = 512):
    """Single MARK_SP token whose EMBED_LO/HI encode ``sp_byte0`` (the
    carry-forwarded SP byte 0, as L3 head 2 leaves it). Returns the
    SP_BYTE0_IS_F8 residual at that row."""
    attn = _build_baked_attn(num_heads=num_heads, dim=dim)
    x = torch.zeros(1, 1, dim)
    SP_I = 2
    # Mark the row as MARK_SP and set the upstream H1+SP / H4+BP / ... gates
    # that head 6's Q row requires to fire. Head 6 Q at slot 0 sums:
    #   +MARK_STACK0 +H4+BP -H1+BP +IS_BYTE +MARK_SP +H1+SP -H1+AX -H3+MEM
    # We set MARK_SP and H1+SP positive, leave the others at zero. The
    # subtractions (-H1+BP, -H1+AX, -H3+MEM) only matter if those dims fire.
    x[0, 0, _SetDim.MARK_SP] = 1.0
    x[0, 0, _SetDim.H1 + SP_I] = 1.0
    # K at slot 0 sums MARK_STACK0 + MARK_SP -- self-attention picks up
    # MARK_SP=1. Carry-forwarded SP byte 0 nibbles land in EMBED_LO/HI on
    # the MARK_SP row (L3 head 2 contract).
    lo = sp_byte0 & 0xF
    hi = (sp_byte0 >> 4) & 0xF
    x[0, 0, _SetDim.EMBED_LO + lo] = 1.0
    x[0, 0, _SetDim.EMBED_HI + hi] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    with torch.no_grad():
        y = attn(x)
    return float(y[0, 0, _SetDim.SP_BYTE0_IS_F8].item())


def test_sp_byte0_is_f8_fires_on_0xf8():
    """SP byte 0 == 0xF8 -> SP_BYTE0_IS_F8 ~ 1.0 (both nibble matches).

    Q.K dominance at MARK_SP self-attention with single-token sequence is
    perfect (softmax over one position == 1.0), so EMBED_LO+8 and
    EMBED_HI+15 each contribute 0.5 to SP_BYTE0_IS_F8 -> 1.0 total.
    """
    val = _run_sp_marker_forward(0xF8)
    assert val == pytest.approx(1.0, abs=1e-4), (
        f"Expected SP_BYTE0_IS_F8 ~ 1.0 for SP byte 0 = 0xF8, got {val}"
    )


@pytest.mark.parametrize("sp_byte0", [0x00, 0x42, 0xFF, 0xE8, 0xF0, 0x18])
def test_sp_byte0_is_f8_below_threshold_for_other_bytes(sp_byte0: int):
    """SP byte 0 != 0xF8 -> SP_BYTE0_IS_F8 <= 0.5 (at most one nibble match).

    0xE8: lo matches (8) but hi != 15 -> 0.5
    0xF0: hi matches (15) but lo != 8 -> 0.5
    0xFF: hi matches (15) but lo != 8 -> 0.5
    0x18: lo matches (8) but hi != 15 -> 0.5
    0x42, 0x00: neither matches -> 0.0
    """
    val = _run_sp_marker_forward(sp_byte0)
    assert val <= 0.5 + 1e-4, (
        f"Expected SP_BYTE0_IS_F8 <= 0.5 for SP byte 0 = 0x{sp_byte0:02X}, "
        f"got {val} (half-match cap broken: would let consumer threshold "
        f"fail to discriminate 0xF8 from other bytes)."
    )


def test_consumer_threshold_distinguishes_f8_from_half_match():
    """The L10 consumer compares ``SP_BYTE0_IS_F8 +1e6`` against threshold
    ~10. Empirically the F8 case produces ~1.0 and any half-match
    (e.g. 0xF0 or 0xE8) produces ~0.5; a threshold midway (e.g. 0.75)
    cleanly separates them."""
    full = _run_sp_marker_forward(0xF8)
    half_lo = _run_sp_marker_forward(0xE8)  # lo nibble 8 but hi != F
    half_hi = _run_sp_marker_forward(0xF0)  # hi nibble F but lo != 8
    assert full > 0.75, f"full match {full} should clear 0.75 threshold"
    assert half_lo < 0.75, f"half-lo match {half_lo} must NOT clear 0.75"
    assert half_hi < 0.75, f"half-hi match {half_hi} must NOT clear 0.75"


# ---------------------------------------------------------------------------
# 4. Compile-end-to-end smoke
# ---------------------------------------------------------------------------


def test_compile_full_vm_includes_sp_byte0_is_f8_dim():
    """compile_full_vm_dynamic allocates SP_BYTE0_IS_F8 and emits 0 STALENESS warnings."""
    import warnings

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model, layout = compile_full_vm_dynamic(disk_cache=False)

    assert "SP_BYTE0_IS_F8" in layout.dim_positions, (
        "compile_full_vm_dynamic did not allocate SP_BYTE0_IS_F8"
    )
    staleness = [str(w.message) for w in caught if "STALENESS" in str(w.message)]
    assert not staleness, f"compile_full_vm_dynamic emitted STALENESS warnings: {staleness}"
