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
from c4_release.neural_vm.unified_compiler.ops.l5_ops import (
    _opcode_decode_all_step_pc_rules,
    _lower_l5_opcode_rules,
    _opcode_decode_first_step_rules,
    _opcode_decode_main_rules,
    _opcode_decode_temp_clear_rules,
)
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
)


_SILU_ONE_INPUT = 1.278464542761074


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


def _assert_same_ffn_units(
    actual: _StubFFN,
    expected: _StubFFN,
    start: int,
    end: int,
):
    assert torch.equal(actual.W_up[start:end], expected.W_up[start:end])
    assert torch.equal(actual.b_up[start:end], expected.b_up[start:end])
    assert torch.equal(actual.W_gate[start:end], expected.W_gate[start:end])
    assert torch.equal(actual.b_gate[start:end], expected.b_gate[start:end])
    assert torch.equal(actual.W_down[:, start:end], expected.W_down[:, start:end])


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


def test_opcode_decode_main_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_main_rules(100.0)
    end = _lower_l5_opcode_rules(actual, rules, _SetDim, unit=0, S=100.0)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 34
    _assert_same_ffn_units(actual, expected, 0, end)


def test_opcode_decode_first_step_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_first_step_rules(100.0)
    end = _lower_l5_opcode_rules(actual, rules, _SetDim, unit=34, S=100.0)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 52
    _assert_same_ffn_units(actual, expected, 34, end)


def test_opcode_decode_temp_clear_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_temp_clear_rules(100.0)
    end = _lower_l5_opcode_rules(actual, rules, _SetDim, unit=53, S=100.0)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 84
    assert not actual.W_up[52].any()
    assert not actual.b_up[52].any()
    assert not actual.W_gate[52].any()
    assert not actual.b_gate[52].any()
    assert not actual.W_down[:, 52].any()
    _assert_same_ffn_units(actual, expected, 53, end)


def test_opcode_decode_all_step_pc_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_all_step_pc_rules(100.0)
    end = _lower_l5_opcode_rules(actual, rules, _SetDim, unit=84, S=100.0)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 89
    _assert_same_ffn_units(actual, expected, 84, end)


def test_opcode_decode_temp_clear_ir_symbolic_matches_lowered():
    rule = _opcode_decode_temp_clear_rules(1.0)[0]
    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"MARK_PC": 0, "TEMP": 4},
        {
            "MARK_PC": 0.5 + _SILU_ONE_INPUT,
            "TEMP+1": 3.0,
        },
        S=1.0,
        atol=1e-5,
    )

    assert report.ok, report.format()
    assert report.symbolic_state["TEMP+1"] == -3.0
    assert abs(report.lowered_state["TEMP+1"] + 3.0) < 1e-5


def test_opcode_decode_all_step_pc_ir_symbolic_matches_lowered():
    rule = _opcode_decode_all_step_pc_rules(1.0)[0]
    report = compare_symbolic_to_lowered_ffn(
        rule,
        {
            "OPCODE_BYTE_LO": 0,
            "OPCODE_BYTE_HI": 16,
            "MARK_PC": 32,
            "OP_BZ": 40,
        },
        {
            "OPCODE_BYTE_LO+4": 1.0,
            "OPCODE_BYTE_HI+0": 1.0,
            "MARK_PC": 0.5 + _SILU_ONE_INPUT,
        },
        S=1.0,
        atol=1e-5,
    )

    assert report.ok, report.format()
    assert report.symbolic_state["OP_BZ+0"] == 10.0
    assert abs(report.lowered_state["OP_BZ+0"] - 10.0) < 1e-5
