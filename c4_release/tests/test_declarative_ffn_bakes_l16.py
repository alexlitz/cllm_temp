"""Parity tests for L16 FFN bands migrated to declarative rules."""

import torch

from c4_release.neural_vm.unified_compiler.ops.l16_ops import (
    _layer16_lev_routing_rules,
    lower_layer16_lev_routing_ir,
)
from c4_release.neural_vm.vm_step import _SetDim, _set_layer16_lev_routing


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 256):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def _assert_same_ffn_prefix(actual: _StubFFN, expected: _StubFFN, units: int):
    assert torch.equal(actual.W_up[:units], expected.W_up[:units]), "W_up"
    assert torch.equal(actual.b_up[:units], expected.b_up[:units]), "b_up"
    assert torch.equal(actual.W_gate[:units], expected.W_gate[:units]), "W_gate"
    assert torch.equal(actual.b_gate[:units], expected.b_gate[:units]), "b_gate"
    assert torch.equal(actual.W_down[:, :units], expected.W_down[:, :units]), "W_down"


def test_layer16_lev_routing_ir_matches_legacy_helper():
    actual = _StubFFN()
    expected = _StubFFN()

    end = lower_layer16_lev_routing_ir(actual, 100.0, _SetDim)
    legacy_end = _set_layer16_lev_routing(expected, 100.0, _SetDim)

    assert legacy_end == 121
    assert end == 189
    assert len(_layer16_lev_routing_rules(100.0)) == 189
    _assert_same_ffn_prefix(actual, expected, legacy_end)
    assert actual.W_down[:, legacy_end:end].abs().sum() > 0
