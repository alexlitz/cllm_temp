"""Focused L14 cleanup weight checks."""

import torch

from c4_release.neural_vm.setup_helpers import _set_layer14_clear_output_corruption
from c4_release.neural_vm.unified_compiler.ops.l14_ops import (
    _layer14_alu_high_byte_relay_spec,
)
from c4_release.neural_vm.vm_step import _SetDim


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 8):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def test_l14_stack0_zero_cleanup_suppresses_pop_group_rows():
    ffn = _StubFFN()

    end = _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    assert end >= 2
    assert ffn.W_up[0, _SetDim.CMP + 3] < -1000.0
    assert ffn.W_up[1, _SetDim.CMP + 3] < -1000.0


def test_l14_stack0_zero_cleanup_preserves_nonzero_nibble_support():
    ffn = _StubFFN()

    _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    assert ffn.W_up[0, _SetDim.OUTPUT_LO + 0] == 0.0
    assert torch.all(ffn.W_up[0, _SetDim.OUTPUT_LO + 1:_SetDim.OUTPUT_LO + 16] < 0.0)
    assert ffn.W_up[1, _SetDim.OUTPUT_HI + 0] == 0.0
    assert torch.all(ffn.W_up[1, _SetDim.OUTPUT_HI + 1:_SetDim.OUTPUT_HI + 16] < 0.0)


def test_l15_wide_alu_relay_blocks_add_sub_rows():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}
    k_terms = {(term.slot, term.dim, term.weight) for term in spec.k}

    assert (35, _SetDim.TEMP + 8, 10000.0) in q_terms
    assert (35, _SetDim.TEMP + 9, 10000.0) in q_terms
    assert (35, _SetDim.CONST, -20.0) in k_terms
