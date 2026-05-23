"""Parity tests for L15 FFN bands migrated to CompilerIR rules."""

import torch

from c4_release.neural_vm.unified_compiler.ops.l15_ops import (
    lower_l15_nibble_copy_ir,
    make_l15_nibble_copy_ir,
)
from c4_release.neural_vm.vm_step import _SetDim, _set_nibble_copy_ffn


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 128):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_layer15_nibble_copy_ir_matches_legacy_helper():
    actual = _StubFFN()
    expected = _StubFFN()

    end = lower_l15_nibble_copy_ir(actual, _SetDim, S=100.0)
    _set_nibble_copy_ffn(expected, 100.0, _SetDim)

    assert end == 42
    assert make_l15_nibble_copy_ir().required_ffn_units() == 42
    _assert_same_ffn(actual, expected)
