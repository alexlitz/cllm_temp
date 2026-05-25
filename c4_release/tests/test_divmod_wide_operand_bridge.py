"""Focused DIV/MOD bridge coverage for the BD <-> GenericE converter."""

import pytest
import torch

from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.common import GenericE
from neural_vm.efficient_alu_neural import (
    BDToGEConverter,
    GEToBDConverter,
)
from neural_vm.vm_step import _SetDim as BD


@pytest.mark.parametrize("op_dim", [BD.OP_DIV, BD.OP_MOD])
def test_divmod_bdtoge_uses_staged_stack_byte1(op_dim):
    ge = GenericE(NIBBLE)
    converter = BDToGEConverter(BD, ge)
    x = torch.zeros(1, 1, 512)

    # Stack operand A: 0x048a. Byte 0 is in ALU_*, byte 1 is staged in
    # AX_FULL_* by the wide-stack-byte relay.
    x[0, 0, BD.MARK_AX] = 1.0
    x[0, 0, op_dim] = 1.0
    x[0, 0, BD.ALU_LO + 0xA] = 1.0
    x[0, 0, BD.ALU_HI + 0x8] = 1.0
    x[0, 0, BD.AX_FULL_LO + 0x4] = 1.0
    x[0, 0, BD.AX_FULL_HI + 0x0] = 1.0

    # AX operand B: 0x25.
    x[0, 0, BD.AX_CARRY_LO + 0x5] = 1.0
    x[0, 0, BD.AX_CARRY_HI + 0x2] = 1.0

    out = converter(x)

    assert out[0, 0, 0, ge.NIB_A] == 0xA
    assert out[0, 0, 1, ge.NIB_A] == 0x8
    assert out[0, 0, 2, ge.NIB_A] == 0x4
    assert out[0, 0, 3, ge.NIB_A] == 0x0
    assert out[0, 0, 0, ge.NIB_B] == 0x5
    assert out[0, 0, 1, ge.NIB_B] == 0x2


def test_non_wide_bdtoge_does_not_read_stale_ax_full_byte1():
    ge = GenericE(NIBBLE)
    converter = BDToGEConverter(BD, ge)
    x = torch.zeros(1, 1, 512)

    x[0, 0, BD.MARK_AX] = 1.0
    x[0, 0, BD.OP_ADD] = 1.0
    x[0, 0, BD.ALU_LO + 0xA] = 1.0
    x[0, 0, BD.ALU_HI + 0x8] = 1.0
    x[0, 0, BD.AX_FULL_LO + 0x4] = 1.0
    x[0, 0, BD.AX_FULL_HI + 0x0] = 1.0

    out = converter(x)

    assert out[0, 0, 2, ge.NIB_A] == 0.0
    assert out[0, 0, 3, ge.NIB_A] == 0.0


@pytest.mark.parametrize("op_dim", [BD.OP_DIV, BD.OP_MOD])
def test_divmod_bdtoge_uses_prior_stack_byte1_fallback(op_dim):
    ge = GenericE(NIBBLE)
    converter = BDToGEConverter(BD, ge)
    x = torch.zeros(1, 2, 512)

    x[0, 0, BD.STACK0_BYTE1] = 1.0
    x[0, 0, BD.CLEAN_EMBED_LO + 0x4] = 1.0
    x[0, 0, BD.CLEAN_EMBED_HI + 0x0] = 1.0

    x[0, 1, BD.MARK_AX] = 1.0
    x[0, 1, op_dim] = 1.0
    x[0, 1, BD.ALU_LO + 0xA] = 1.0
    x[0, 1, BD.ALU_HI + 0x8] = 1.0
    x[0, 1, BD.AX_CARRY_LO + 0x5] = 1.0
    x[0, 1, BD.AX_CARRY_HI + 0x2] = 1.0

    out = converter(x)

    assert out[0, 1, 0, ge.NIB_A] == 0xA
    assert out[0, 1, 1, ge.NIB_A] == 0x8
    assert out[0, 1, 2, ge.NIB_A] == 0x4
    assert out[0, 1, 3, ge.NIB_A] == 0x0
    assert out[0, 1, 0, ge.NIB_B] == 0x5
    assert out[0, 1, 1, ge.NIB_B] == 0x2


@pytest.mark.parametrize("op_dim", [BD.OP_DIV, BD.OP_MOD])
def test_divmod_getobd_stages_result_byte1(op_dim):
    ge = GenericE(NIBBLE)
    converter = GEToBDConverter(BD, ge)
    x_ge = torch.zeros(1, 1, 8, ge.DIM)
    x_bd = torch.zeros(1, 1, 512)

    x_bd[0, 0, BD.MARK_AX] = 1.0
    x_bd[0, 0, op_dim] = 1.0
    x_ge[0, 0, 2, ge.RESULT] = 0x1
    x_ge[0, 0, 3, ge.RESULT] = 0x0

    out = converter(
        x_ge,
        x_bd,
        opcode_mask=torch.ones(1, 1),
        emit_carry=False,
    )

    assert out[0, 0, BD.AX_FULL_LO + 0x1] > 1.5
    assert out[0, 0, BD.AX_FULL_HI + 0x0] > 1.5
