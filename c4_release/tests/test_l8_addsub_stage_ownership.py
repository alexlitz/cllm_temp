import pytest
import torch

from c4_release.neural_vm.efficient_alu_addsub_split import (
    AddSub5StageBlock,
    make_addsub_stage_modules,
)
from c4_release.neural_vm.efficient_alu_neural import PureNeuralALU
from c4_release.neural_vm.vm_step import _SetDim as BD


def _bd_addsub_input(*, op_dim: int, lhs: int, rhs: int) -> torch.Tensor:
    x = torch.zeros(1, 3, 512)
    ax_pos = 1
    x[:, ax_pos, BD.MARK_AX] = 1.0
    x[:, ax_pos, op_dim] = 1.0

    x[:, ax_pos, BD.ALU_LO + (lhs & 0xF)] = 1.0
    x[:, ax_pos, BD.ALU_HI + ((lhs >> 4) & 0xF)] = 1.0
    x[:, ax_pos, BD.AX_CARRY_LO + (rhs & 0xF)] = 1.0
    x[:, ax_pos, BD.AX_CARRY_HI + ((rhs >> 4) & 0xF)] = 1.0
    return x


@pytest.mark.parametrize(
    ("op_dim", "lhs", "rhs"),
    [
        (BD.OP_ADD, 0x34, 0x12),
        (BD.OP_ADD, 0xFF, 0x01),
        (BD.OP_SUB, 0x34, 0x12),
        (BD.OP_SUB, 0x00, 0x01),
    ],
)
def test_l8_addsub_generated_stage_pipeline_matches_pure_neural_alu(
    op_dim,
    lhs,
    rhs,
):
    x = _bd_addsub_input(op_dim=op_dim, lhs=lhs, rhs=rhs)
    reference = PureNeuralALU(100.0, BD, operations="add_sub")
    staged = AddSub5StageBlock(100.0, BD)

    with torch.no_grad():
        expected = reference(x.clone())
        actual = staged(x.clone())

    assert torch.equal(actual, expected)


def test_l8_addsub_stage_factory_is_the_same_generated_pipeline():
    x = _bd_addsub_input(op_dim=BD.OP_ADD, lhs=0x7E, rhs=0x03)
    state, stage0, stage1, stage2, stage3, stage4 = make_addsub_stage_modules(
        BD,
        100.0,
    )
    staged = AddSub5StageBlock(100.0, BD)

    assert [type(stage).__name__ for stage in staged.stages] == [
        type(stage0).__name__,
        type(stage1).__name__,
        type(stage2).__name__,
        type(stage3).__name__,
        type(stage4).__name__,
    ]
    assert all(
        stage.state is state
        for stage in (stage0, stage1, stage2, stage3, stage4)
    )

    with torch.no_grad():
        manual = stage4(stage3(stage2(stage1(stage0(x.clone())))))
        sequential = staged(x.clone())

    assert torch.equal(manual, sequential)
