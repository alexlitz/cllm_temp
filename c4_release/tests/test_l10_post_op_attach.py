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
