"""V2 ADDR_KEY neural decode (BLOG_SPEC.md:830) — Phase 0 staging tests.

Verifies the new ``make_layer14_addr_key_neural_decode_op`` op
(see ``neural_vm/unified_compiler/ops/l14_ops.py``) at two levels:

1. **Registration / gate**: the op is in ``all_core_ops()``, lands at
   layer_idx=14 after ``compile_full_vm()``, and is a no-op when
   ``enable=False`` (so existing tests stay byte-identical).

2. **Bake parity**: with ``enable=True``, the FFN produces the same
   ADDR_KEY[lo, 16+hi, 32+top] one-hot encoding that
   ``NeuralVMEmbedding._inject_mem_metadata`` produces for the same
   inputs, across a range of (addr_b0, addr_b1, byte_off) values
   including the carry-overflow case.

The full integration (enabling the gate and deleting
``_inject_mem_metadata``) is deferred to a separate PR; see
``c4_release/docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md``.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_addr_key_decode_op_is_registered():
    """The new op is in all_core_ops() and lands at layer 14 after compile."""
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

    ops = all_core_ops()
    found = [op for op in ops if op.name == "layer14_addr_key_neural_decode"]
    assert len(found) == 1, (
        f"Expected exactly 1 layer14_addr_key_neural_decode op, "
        f"found {len(found)}"
    )
    op = found[0]
    assert op.layer_idx == 14
    assert op.migrated is True
    assert op.phase == 14.5
    assert op.kind == "block"
    # ADDR_KEY is the only write target.
    assert op.writes == {"ADDR_KEY"}

    # Verify it lands at L14 in the layout after compile.
    _, layout = compile_full_vm()
    block_ops_by_name = {bo.name: bo for bo in layout.block_ops}
    assert "layer14_addr_key_neural_decode" in block_ops_by_name


def test_addr_key_decode_op_is_noop_when_disabled():
    """With enable=False (default), the bake does not touch FFN unit counter.

    The bake is gated by an early return; it should leave the L14 FFN
    unit counter unchanged from what the other L14 cleanup ops produced.
    Verified indirectly: compile_full_vm must succeed and downstream tests
    must pass (the gate tests cover those).

    Here we verify by inspecting the compiled model that the FFN units
    used by the OTHER L14 cleanup ops are intact and no new units past
    that count have nonzero W_up rows referencing ADDR_B0_LO/HI (the
    new bake's distinctive input signature).
    """
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
    from neural_vm.vm_step import _SetDim

    model, layout = compile_full_vm()
    ffn14 = model.blocks[14].ffn
    # ADDR_B0_LO is at dim 12, ADDR_B0_HI is at dim 206 (= ADDR_KEY).
    # The new bake's units read 3-way AND across (MEM_VAL_B0 + ADDR_B0_LO +
    # ADDR_B0_HI). If the gate is off, no FFN unit should have nonzero W_up
    # at both ADDR_B0_LO[0] and ADDR_B0_HI[0] simultaneously.
    addr_b0_lo_col = ffn14.W_up.data[:, _SetDim.ADDR_B0_LO + 0]
    addr_b0_hi_col = ffn14.W_up.data[:, _SetDim.ADDR_B0_HI + 0]
    both_nonzero = (addr_b0_lo_col != 0) & (addr_b0_hi_col != 0)
    assert both_nonzero.sum().item() == 0, (
        f"Expected gate=False to leave no FFN units with both "
        f"ADDR_B0_LO and ADDR_B0_HI as up-reads; found "
        f"{both_nonzero.sum().item()} such units. The bake's "
        f"guard-clause early return must have leaked weights."
    )


def _compute_expected_addr_key_band(addr_b0, addr_b1, byte_off):
    """Reference: what ``_inject_mem_metadata`` would write into the
    48-dim ADDR_KEY band for a given (addr_b0, addr_b1, byte_off).

    Returns a [48]-tensor of one-hot values matching the Python injector's
    output at the corresponding val byte position.
    """
    # Build the 32-bit addr the same way _inject_mem_metadata does, but
    # only addr_b0 and addr_b1 contribute to the 12-bit ADDR_KEY.
    addr = addr_b0 | (addr_b1 << 8)
    byte_addr = addr + byte_off
    lo = byte_addr & 0xF
    hi = (byte_addr >> 4) & 0xF
    top = (byte_addr >> 8) & 0xF

    band = torch.zeros(48)
    band[lo] = 1.0
    band[16 + hi] = 1.0
    band[32 + top] = 1.0
    return band


def _build_residual_for_addr(addr_b0, addr_b1, byte_off, dim_positions, d_model):
    """Construct a synthetic residual [1, 1, d_model] tensor mimicking the
    state that L13's mem_addr_gather produces at a MEM val byte position.

    Sets:
      - MEM_VAL_B{byte_off} = 1.0 (gate)
      - ADDR_B0_LO[addr_b0 & 0xF] = 1.0
      - ADDR_B0_HI[(addr_b0 >> 4) & 0xF] = 1.0  (aliased with ADDR_KEY+0)
      - ADDR_B1_LO[addr_b1 & 0xF] = 1.0

    All other dims are zero.
    """
    x = torch.zeros(1, 1, d_model)
    MEM_VAL_BS = [
        dim_positions["MEM_VAL_B0"],
        dim_positions["MEM_VAL_B1"],
        dim_positions["MEM_VAL_B2"],
        dim_positions["MEM_VAL_B3"],
    ]
    x[0, 0, MEM_VAL_BS[byte_off]] = 1.0
    addr_b0_lo = addr_b0 & 0xF
    addr_b0_hi = (addr_b0 >> 4) & 0xF
    addr_b1_lo = addr_b1 & 0xF
    x[0, 0, dim_positions["ADDR_B0_LO"] + addr_b0_lo] = 1.0
    x[0, 0, dim_positions["ADDR_B0_HI"] + addr_b0_hi] = 1.0
    x[0, 0, dim_positions["ADDR_B1_LO"] + addr_b1_lo] = 1.0
    # CONST = 1 (some bakes rely on it; harmless if unused here).
    x[0, 0, dim_positions["CONST"]] = 1.0
    return x


@pytest.mark.parametrize(
    "addr_b0,addr_b1,byte_off",
    [
        # Common case: no carry.
        (0x00, 0x00, 0),
        (0x10, 0x00, 0),
        (0x37, 0x42, 0),
        (0x37, 0x42, 1),
        (0x37, 0x42, 2),
        (0x37, 0x42, 3),
        # Mid-range, no carry into byte 1.
        (0xAB, 0x05, 0),
        (0xAB, 0x05, 1),
        # Hi-nibble carry within byte 0 (lo + byte_off >= 16 but hi < 15):
        (0x1F, 0x07, 1),   # 0x1F + 1 = 0x20: lo=0, hi=2, top=0
        (0x2E, 0x07, 2),   # 0x2E + 2 = 0x30
        (0x3D, 0x07, 3),   # 0x3D + 3 = 0x40
        # High-byte carry (hi==15 AND lo+byte_off >= 16):
        (0xFF, 0x00, 1),   # 0xFF + 1 = 0x100: lo=0, hi=0, top=1
        (0xFF, 0x05, 1),   # 0xFF + 1 = 0x100, then + 0x500 = 0x600: top=6
        (0xFE, 0x00, 2),   # 0xFE + 2 = 0x100: lo=0, hi=0, top=1
        (0xFE, 0x07, 2),   # 0xFE + 2 = 0x100, + 0x700 = 0x800: top=8
        (0xFD, 0x00, 3),   # 0xFD + 3 = 0x100: top=1
    ],
)
def test_addr_key_decode_bake_matches_inject_mem_metadata(
    addr_b0, addr_b1, byte_off
):
    """Bake parity: enable=True FFN output equals _inject_mem_metadata.

    Builds an FFN with the new bake enabled, feeds it a synthetic residual
    matching what L13 produces at a MEM val byte position, and verifies the
    output ADDR_KEY band one-hots match the Python injector exactly.
    """
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
    from neural_vm.unified_compiler.ops.l14_ops import (
        _bake_addr_key_neural_decode,
    )
    from neural_vm.base_layers import PureFFN

    # Use the compiled model to get the canonical dim_positions; the bake
    # function reads them via _as_setdim_proxy.
    _, layout = compile_full_vm()
    dim_positions = layout.dim_positions
    d_model = layout.d_model

    # Build a fresh PureFFN with enough hidden capacity for the bake.
    # The bake consumes ~1184 units; allocate 2048 for headroom.
    HIDDEN = 2048
    ffn = PureFFN(dim=d_model, hidden_dim=HIDDEN)
    # Zero everything explicitly (PureFFN's _bake_weights is a no-op for the
    # base class, but be defensive).
    with torch.no_grad():
        ffn.W_up.data.zero_()
        ffn.W_gate.data.zero_()
        ffn.W_down.data.zero_()
        ffn.b_up.data.zero_()
        ffn.b_gate.data.zero_()
        ffn.b_down.data.zero_()
    # Bake. The bake function uses raw param indexing (ffn.W_up[i,j] = v)
    # which requires no_grad context for fresh nn.Parameters.
    S = 100.0
    with torch.no_grad():
        next_unit = _bake_addr_key_neural_decode(
            ffn, dim_positions, S, start_unit=0
        )
    assert next_unit <= HIDDEN, (
        f"Bake consumed {next_unit} units > headroom {HIDDEN}"
    )

    # Build synthetic residual at a MEM val byte position.
    x = _build_residual_for_addr(
        addr_b0, addr_b1, byte_off, dim_positions, d_model
    )

    # Forward through the FFN.
    with torch.no_grad():
        y = ffn(x)

    # The ADDR_KEY band in the output residual is y[..., ADDR_KEY:ADDR_KEY+48].
    addr_key_start = dim_positions["ADDR_KEY"]
    out_band = y[0, 0, addr_key_start:addr_key_start + 48]

    # The input residual already has ADDR_KEY[0..15] populated by
    # ADDR_B0_HI (= addr_b0_hi nibble) — that's the L13 contribution. The
    # baked FFN ADDs (residual stream) its own writes on top. Subtract the
    # input contribution to isolate the FFN's contribution.
    input_band = x[0, 0, addr_key_start:addr_key_start + 48].clone()
    ffn_contribution = out_band - input_band

    # Expected: _inject_mem_metadata's encoding.
    expected = _compute_expected_addr_key_band(addr_b0, addr_b1, byte_off)

    # The Python injector OVERWRITES (not adds) the band, while our FFN
    # adds on top of the L13 pollution. For the bake to be a drop-in
    # replacement, the L14 pollution-clear must zero the band BEFORE the
    # new bake fires. In isolation (this test), we just verify the FFN
    # produces the correct ADDITIVE contribution.
    #
    # Specifically, the FFN should produce:
    #   - +1.0 at lo, 16+hi, 32+top positions
    #   - 0.0 elsewhere
    # IF the input band is zero. In our synthetic input, the band has
    # exactly one hot bit at position addr_b0_hi (from ADDR_B0_HI), which
    # is what _set_layer13_mem_addr_gather produces. So the FFN output
    # band should equal expected + input_band, modulo silu rounding.

    expected_out = expected + input_band

    # Tolerance: silu is not exactly identity; allow small numerical
    # slack. The bake uses output scale 2.0/S where S=100, so each unit
    # contributes ~0.02 per silu activation. With the threshold gate
    # (silu(0.5*S) ≈ 0.5*S - 0), each fully-activated unit contributes
    # 2.0/S * S * 0.5 ≈ 1.0.
    tol = 0.15
    diff = (out_band - expected_out).abs()
    assert diff.max().item() < tol, (
        f"FFN output mismatch for "
        f"addr_b0=0x{addr_b0:02X} addr_b1=0x{addr_b1:02X} "
        f"byte_off={byte_off}: max diff {diff.max().item():.4f}, "
        f"\n  out_band={out_band.tolist()}, "
        f"\n  expected_out={expected_out.tolist()}"
    )
