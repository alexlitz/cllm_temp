import torch
from torch import nn

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
    declare_setdim_compat_dims,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import LayerCompiler
from c4_release.neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
from c4_release.neural_vm.unified_compiler.ops.l10_ops import (
    make_l10_post_op_attach_op,
    make_l10_post_ops_combined,
)


def _compact_layout():
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    compiler.add_op(make_l10_post_op_attach_op(alu_mode="efficient"))
    return compiler.compile()


def _decode_output_byte(x, dim_positions):
    lo_base = dim_positions["OUTPUT_LO"]
    hi_base = dim_positions["OUTPUT_HI"]
    lo = x[0, 0, lo_base:lo_base + 16]
    hi = x[0, 0, hi_base:hi_base + 16]
    return int(lo.argmax()) | (int(hi.argmax()) << 4)


class _DummyBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = PureFFN(d_model, 1)
        self.post_ops = nn.ModuleList()


def test_l10_attached_carry_cascade_does_not_rewrite_add_byte1_row():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    block = _DummyBlock(layout.d_model)
    make_l10_post_op_attach_op(alu_mode="efficient").bake_fn(
        block, dim_positions, 100.0,
    )

    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 1] = 1.0
    x[0, 0, dim_positions["BYTE_INDEX_0"]] = 0.970138
    x[0, 0, dim_positions["BYTE_INDEX_1"]] = 0.0132967
    x[0, 0, dim_positions["TEMP"] + 8] = 1.000002
    x[0, 0, dim_positions["CARRY"] + 1] = 2.0
    x[0, 0, dim_positions["OUTPUT_LO"] + 0] = -5.495
    x[0, 0, dim_positions["OUTPUT_LO"] + 1] = 8.435
    x[0, 0, dim_positions["OUTPUT_HI"] + 0] = 11.375
    x[0, 0, dim_positions["ALU_LO"] + 0] = -4.96
    x[0, 0, dim_positions["ALU_LO"] + 4] = 0.003

    # Start after the byte-0 carry stage: the ADD byte-1 row is already 0x01.
    # The later cascade carry, bitwise, and comparison post-ops must not turn
    # compact-layout byte-index leakage into another byte rewrite.
    for post_op in list(block.post_ops)[3:]:
        x = post_op(x)

    assert _decode_output_byte(x, dim_positions) == 0x01


def test_l10_combined_tail_blocks_wide_mul_temp_signature():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    ffn = PureFFN(layout.d_model, 2048)

    make_l10_post_ops_combined().bake_fn(ffn, dim_positions, 100.0)

    active_rows = ffn.W_down.abs().sum(dim=0) > 0
    assert active_rows.any()
    assert torch.all(
        ffn.W_up[active_rows, dim_positions["TEMP"] + 10] <= -100000.0
    )


def test_l10_combined_tail_blocks_add_row_with_large_output_residue():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    ffn = PureFFN(layout.d_model, 2048)

    make_l10_post_ops_combined().bake_fn(ffn, dim_positions, 100.0)

    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 1] = 1.0
    x[0, 0, dim_positions["BYTE_INDEX_0"]] = 0.970138
    x[0, 0, dim_positions["BYTE_INDEX_1"]] = 0.0132967
    x[0, 0, dim_positions["TEMP"] + 8] = 1.0
    x[0, 0, dim_positions["TEMP"] + 10] = 0.996
    x[0, 0, dim_positions["CARRY"] + 1] = 2.0
    x[0, 0, dim_positions["OUTPUT_LO"] + 0] = 12.84
    x[0, 0, dim_positions["OUTPUT_LO"] + 1] = -13.18
    x[0, 0, dim_positions["OUTPUT_LO"] + 2] = -75.5
    x[0, 0, dim_positions["OUTPUT_LO"] + 3] = 76.8
    x[0, 0, dim_positions["OUTPUT_HI"] + 0] = 251.0
    x[0, 0, dim_positions["OUTPUT_HI"] + 1] = 13.18

    out = ffn(x)

    output_dims = list(range(dim_positions["OUTPUT_LO"], dim_positions["OUTPUT_LO"] + 16))
    output_dims += list(range(dim_positions["OUTPUT_HI"], dim_positions["OUTPUT_HI"] + 16))
    assert torch.allclose(
        out[..., output_dims],
        x[..., output_dims],
        atol=1e-3,
    )


def test_l10_combined_tail_blocks_lea_relay_rows():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    ffn = PureFFN(layout.d_model, 2048)

    make_l10_post_ops_combined().bake_fn(ffn, dim_positions, 100.0)

    active_rows = ffn.W_down.abs().sum(dim=0) > 0
    assert active_rows.any()
    assert torch.all(
        ffn.W_up[active_rows, dim_positions["CMP"] + 7] <= -100000.0
    )


def test_l10_attached_addsub_ignores_pc_staging_temp10_without_addsub_relay():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    block = _DummyBlock(layout.d_model)
    make_l10_post_op_attach_op(alu_mode="efficient").bake_fn(
        block, dim_positions, 100.0,
    )

    addsub = list(block.post_ops)[1]
    active_rows = addsub.W_down.abs().sum(dim=0) > 0
    assert active_rows.any()
    assert torch.all(addsub.W_up[active_rows, dim_positions["TEMP"] + 10] == 0.0)

    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 1] = 1.0
    x[0, 0, dim_positions["BYTE_INDEX_0"]] = 1.0
    x[0, 0, dim_positions["TEMP"] + 10] = 1.0
    x[0, 0, dim_positions["OUTPUT_LO"] + 0] = 2.94
    x[0, 0, dim_positions["OUTPUT_HI"] + 0] = 2.94
    x[0, 0, dim_positions["ALU_LO"] + 1] = 6.0
    x[0, 0, dim_positions["ALU_HI"] + 0] = 6.0

    out = addsub(x)

    assert torch.allclose(out, x, atol=1e-6)

    x[0, 0, dim_positions["TEMP"] + 8] = 1.0
    out = addsub(x)

    assert _decode_output_byte(out, dim_positions) == 0x01


def test_l10_attached_addsub_tolerates_adjacent_byte_index_residue():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    block = _DummyBlock(layout.d_model)
    make_l10_post_op_attach_op(alu_mode="efficient").bake_fn(
        block, dim_positions, 100.0,
    )

    addsub = list(block.post_ops)[1]
    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 1] = 1.0
    x[0, 0, dim_positions["BYTE_INDEX_0"]] = 0.970138
    x[0, 0, dim_positions["BYTE_INDEX_1"]] = 0.0132967
    x[0, 0, dim_positions["TEMP"] + 8] = 1.0
    x[0, 0, dim_positions["OUTPUT_LO"] + 0] = 0.9403
    x[0, 0, dim_positions["OUTPUT_HI"] + 0] = 0.9403
    x[0, 0, dim_positions["ALU_LO"] + 2] = 6.0
    x[0, 0, dim_positions["ALU_HI"] + 0] = 6.24

    out = addsub(x)

    assert _decode_output_byte(out, dim_positions) == 0x02


def test_l10_attached_addsub_blocks_non_ax_h1_rows():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    block = _DummyBlock(layout.d_model)
    make_l10_post_op_attach_op(alu_mode="efficient").bake_fn(
        block, dim_positions, 100.0,
    )

    addsub = list(block.post_ops)[1]
    active_rows = addsub.W_down.abs().sum(dim=0) > 0
    assert active_rows.any()
    assert torch.all(
        addsub.W_up[active_rows, dim_positions["H1"] + 3] <= -1000000.0
    )

    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 3] = 1.0
    x[0, 0, dim_positions["BYTE_INDEX_1"]] = 1.0
    x[0, 0, dim_positions["CARRY"] + 2] = 3.0
    x[0, 0, dim_positions["CLEAN_EMBED_LO"] + 15] = 1.0
    x[0, 0, dim_positions["CLEAN_EMBED_HI"] + 15] = 1.0
    x[0, 0, dim_positions["OUTPUT_LO"] + 0] = 13.98
    x[0, 0, dim_positions["OUTPUT_HI"] + 0] = 13.98

    out = addsub(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_l10_attached_addsub_blocks_byte3_negative_residue_rows():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    block = _DummyBlock(layout.d_model)
    make_l10_post_op_attach_op(alu_mode="efficient").bake_fn(
        block, dim_positions, 100.0,
    )

    addsub = list(block.post_ops)[1]
    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 3] = 0.9867
    x[0, 0, dim_positions["BYTE_INDEX_0"]] = 0.0133
    x[0, 0, dim_positions["BYTE_INDEX_3"]] = 0.9701
    x[0, 0, dim_positions["TEMP"] + 8] = 1.0
    x[0, 0, dim_positions["OUTPUT_LO"]:dim_positions["OUTPUT_LO"] + 16] = -220.0
    x[0, 0, dim_positions["OUTPUT_HI"]:dim_positions["OUTPUT_HI"] + 16] = -220.0
    x[0, 0, dim_positions["ALU_LO"]:dim_positions["ALU_LO"] + 16] = -96.0
    x[0, 0, dim_positions["ALU_HI"]:dim_positions["ALU_HI"] + 16] = -96.0

    out = addsub(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_l10_attached_carry0_blocks_byte3_negative_residue_rows():
    layout = _compact_layout()
    dim_positions = layout.dim_positions
    block = _DummyBlock(layout.d_model)
    make_l10_post_op_attach_op(alu_mode="efficient").bake_fn(
        block, dim_positions, 100.0,
    )

    carry0 = list(block.post_ops)[2]
    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions["IS_BYTE"]] = 1.0
    x[0, 0, dim_positions["H1"] + 1] = 1.0
    x[0, 0, dim_positions["BYTE_INDEX_3"]] = 0.9701
    x[0, 0, dim_positions["CARRY"] + 1] = 2.0
    x[0, 0, dim_positions["OUTPUT_LO"]:dim_positions["OUTPUT_LO"] + 16] = -220.0
    x[0, 0, dim_positions["OUTPUT_HI"]:dim_positions["OUTPUT_HI"] + 16] = -220.0

    out = carry0(x)

    assert torch.allclose(
        out[0, 0, dim_positions["OUTPUT_LO"]:dim_positions["OUTPUT_LO"] + 16],
        x[0, 0, dim_positions["OUTPUT_LO"]:dim_positions["OUTPUT_LO"] + 16],
        atol=1e-6,
    )
    assert torch.allclose(
        out[0, 0, dim_positions["OUTPUT_HI"]:dim_positions["OUTPUT_HI"] + 16],
        x[0, 0, dim_positions["OUTPUT_HI"]:dim_positions["OUTPUT_HI"] + 16],
        atol=1e-6,
    )
