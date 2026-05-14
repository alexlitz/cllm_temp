"""Parity tests for declarative FFN bakes migrated in L1-L5 ops."""

import torch

from c4_release.neural_vm.setup_helpers import (
    _set_layer1_ffn,
    _set_layer2_mem_byte_flags,
)
from c4_release.neural_vm.vm_step import (
    _SetDim,
    _set_layer4_ffn,
    _set_opcode_decode_ffn,
)
from c4_release.neural_vm.unified_compiler.ops.l1_ops import _bake_layer1_ffn
from c4_release.neural_vm.unified_compiler.ops.l2_ops import (
    _bake_layer2_mem_byte_flags,
)
from c4_release.neural_vm.unified_compiler.ops.l4_ops import _bake_layer4_ffn
from c4_release.neural_vm.unified_compiler.ops.l5_ops import _bake_opcode_decode_ffn


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 1024):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_layer1_ffn_declarative_matches_legacy_helper():
    actual = _StubFFN(hidden_dim=8)
    expected = _StubFFN(hidden_dim=8)

    _bake_layer1_ffn(actual, 100.0, _SetDim)
    _set_layer1_ffn(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)


def test_layer2_mem_byte_flags_declarative_matches_legacy_helper():
    actual = _StubFFN(hidden_dim=16)
    expected = _StubFFN(hidden_dim=16)

    _bake_layer2_mem_byte_flags(actual, 100.0, _SetDim)
    _set_layer2_mem_byte_flags(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)


def test_layer4_ffn_declarative_matches_legacy_helper():
    actual = _StubFFN(hidden_dim=600)
    expected = _StubFFN(hidden_dim=600)

    _bake_layer4_ffn(actual, 100.0, _SetDim)
    _set_layer4_ffn(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)


def test_opcode_decode_ffn_declarative_matches_legacy_helper():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    _bake_opcode_decode_ffn(actual, 100.0, _SetDim)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)
