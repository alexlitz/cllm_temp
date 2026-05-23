"""Parity tests for L8 FFN bands migrated to declarative rules."""

import torch

from c4_release.neural_vm.unified_compiler.ops.l8_ops import (
    _layer8_multibyte_routing_rules,
    lower_layer8_multibyte_routing_ir,
)
from c4_release.neural_vm.vm_step import (
    _SetDim,
    _set_layer8_alu,
    _set_layer8_multibyte_routing,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 2200):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_layer8_multibyte_routing_ir_matches_legacy_helper():
    actual = _StubFFN()
    expected = _StubFFN()

    unit_start = _set_layer8_alu(actual, 100.0, _SetDim)
    end = lower_layer8_multibyte_routing_ir(
        actual,
        100.0,
        _SetDim,
        start_unit=unit_start,
    )
    _set_layer8_multibyte_routing(expected, 100.0, _SetDim)

    assert end == unit_start + 32
    assert len(_layer8_multibyte_routing_rules(100.0)) == 32
    _assert_same_ffn(actual, expected)
