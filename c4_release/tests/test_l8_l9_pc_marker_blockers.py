"""Regression tests for amplified L8/L9 ALU units at PC markers."""

import torch

from neural_vm.vm_step import (
    AddSubBytePropagationPostOp,
    _SetDim,
    _set_layer8_alu,
    _set_layer9_alu,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 4096):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _matching_units(ffn: _StubFFN, *, gate_dim: int, up_dims: tuple[int, ...], down_dim: int):
    mask = ffn.W_gate[:, gate_dim] != 0
    for dim in up_dims:
        mask &= ffn.W_up[:, dim] != 0
    mask &= ffn.W_down[down_dim, :] != 0
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def test_layer8_amplified_fetch_units_block_pc_marker():
    ffn = _StubFFN()
    _set_layer8_alu(ffn, 100.0, _SetDim)

    cases = (
        (
            _SetDim.OP_LEA,
            (_SetDim.ALU_LO + 0, _SetDim.FETCH_LO + 8),
            _SetDim.OUTPUT_LO + 8,
        ),
        (
            _SetDim.OP_ADJ,
            (_SetDim.ALU_LO + 0, _SetDim.FETCH_LO + 8),
            _SetDim.OUTPUT_LO + 8,
        ),
        (
            _SetDim.OP_ENT,
            (_SetDim.ALU_LO + 0, _SetDim.FETCH_LO + 8),
            _SetDim.OUTPUT_LO + 0,
        ),
        (
            _SetDim.OP_ENT,
            (_SetDim.ALU_LO + 0, _SetDim.FETCH_LO + 8),
            _SetDim.CARRY + 0,
        ),
    )
    for gate_dim, up_dims, down_dim in cases:
        units = _matching_units(ffn, gate_dim=gate_dim, up_dims=up_dims, down_dim=down_dim)
        assert units
        for unit in units:
            assert ffn.W_up[unit, _SetDim.MARK_PC] <= -100_000.0


def test_layer9_amplified_fetch_units_block_pc_marker():
    ffn = _StubFFN()
    _set_layer9_alu(ffn, 100.0, _SetDim)

    cases = (
        (
            _SetDim.OP_LEA,
            (_SetDim.ALU_HI + 0, _SetDim.FETCH_HI + 15),
            _SetDim.OUTPUT_HI + 15,
        ),
        (
            _SetDim.OP_ADJ,
            (_SetDim.ALU_HI + 0, _SetDim.FETCH_HI + 15),
            _SetDim.OUTPUT_HI + 15,
        ),
        (
            _SetDim.OP_ENT,
            (_SetDim.ALU_HI + 0, _SetDim.FETCH_HI + 15),
            _SetDim.OUTPUT_HI + 1,
        ),
    )
    for gate_dim, up_dims, down_dim in cases:
        units = _matching_units(ffn, gate_dim=gate_dim, up_dims=up_dims, down_dim=down_dim)
        assert units
        for unit in units:
            assert ffn.W_up[unit, _SetDim.MARK_PC] <= -100_000.0


def test_addsub_byte_postop_blocks_lea_relay_byte_lane():
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.IS_BYTE] = 1.0
    x[0, 0, _SetDim.H1 + 1] = 1.0
    x[0, 0, _SetDim.BYTE_INDEX_0] = 1.0
    x[0, 0, _SetDim.CMP + 7] = 1.0
    x[0, 0, _SetDim.TEMP + 8] = 1.0
    x[0, 0, _SetDim.OUTPUT_LO + 15] = 9.0
    x[0, 0, _SetDim.OUTPUT_HI + 15] = 9.0
    x[0, 0, _SetDim.ALU_LO + 15] = 6.0
    x[0, 0, _SetDim.ALU_HI + 15] = 6.0

    out = AddSubBytePropagationPostOp()(x)

    assert float(out[0, 0, _SetDim.OUTPUT_LO + 15]) == 9.0
    assert float(out[0, 0, _SetDim.OUTPUT_HI + 15]) == 9.0
    assert float(out[0, 0, _SetDim.OUTPUT_LO + 14]) == 0.0
    assert float(out[0, 0, _SetDim.OUTPUT_HI + 14]) == 0.0
