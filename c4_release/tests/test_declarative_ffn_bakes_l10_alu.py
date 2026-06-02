"""Parity tests for L10 ALU FFN sub-stages migrated to declarative rules.

The ``_set_layer10_alu`` helper in ``vm_step`` writes 1846 hidden FFN
units across seven sub-stages (cmp_combine / bitwise OR / bitwise XOR /
bitwise AND / mul_lo / shl_shr_zero / ax_passthrough). Phase 6 Wave 4I
migrates each sub-stage to declarative ``FFNRule`` data; these tests
pin per-sub-stage byte-identity between the rules and the legacy
imperative helper so the migration stays load-bearing.
"""

import torch

from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.ops.l10_ops import (
    _L10_FFN_UNIT_LAYOUT_MAIN,
    _layer10_alu_bitwise_and_rules,
    _layer10_alu_bitwise_or_rules,
    _layer10_alu_bitwise_xor_rules,
    _layer10_alu_cmp_combine_rules,
    _layer10_alu_mul_lo_rules,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives
from c4_release.neural_vm.vm_step import _SetDim, _set_layer10_alu


_L10_HIDDEN_DIM = 2048  # comfortable margin above the 1846-unit footprint


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = _L10_HIDDEN_DIM):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn_units(
    actual: _StubFFN,
    expected: _StubFFN,
    start: int,
    end: int,
):
    assert torch.equal(actual.W_up[start:end], expected.W_up[start:end]), (
        f"W_up mismatch in units {start}..{end}"
    )
    assert torch.equal(actual.b_up[start:end], expected.b_up[start:end]), (
        f"b_up mismatch in units {start}..{end}"
    )
    assert torch.equal(actual.W_gate[start:end], expected.W_gate[start:end]), (
        f"W_gate mismatch in units {start}..{end}"
    )
    assert torch.equal(actual.b_gate[start:end], expected.b_gate[start:end]), (
        f"b_gate mismatch in units {start}..{end}"
    )
    assert torch.equal(
        actual.W_down[:, start:end], expected.W_down[:, start:end]
    ), f"W_down mismatch in units {start}..{end}"


def _lower_rules(ffn, rules, *, start_unit: int, S: float = 100.0) -> int:
    dim_positions = Primitives.dim_positions_from_bd(
        _SetDim, Primitives.ffn_rule_dim_names(rules)
    )
    return Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=start_unit, S=S
    )


def _layout_range(name: str) -> tuple[int, int]:
    for stage_name, start, n_units in _L10_FFN_UNIT_LAYOUT_MAIN:
        if stage_name == name:
            return start, start + n_units
    raise KeyError(f"unknown L10 ALU layout stage: {name}")


def _assert_substage_compare_clean(rules: tuple, *, S: float = 100.0) -> None:
    """``compare_symbolic_to_lowered_ffn`` must report no structural drift.

    The per-cell lowering check (W_up / b_up / W_gate / b_gate / W_down)
    is the byte-identity contract: zero structural failures means the
    rules lower to the same matrix the legacy helper writes. The
    ``weight_output_mismatch`` kind only fires when synthetic forward
    states collide across rules sharing condition dims; for the L10 ALU
    sub-stages those cross-rule states are pinned at the helper level
    via `_assert_same_ffn_units` further down, so it is safe to filter
    them out here.
    """

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    names = Primitives.ffn_rule_dim_names(ir.layer(0).ffn.rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    report = compare_symbolic_to_lowered_ffn(
        ir, dim_positions, S=S, atol=1e-4, rtol=1e-4,
    )
    structural = [
        issue for issue in report.issues
        if issue.kind in ("declaration_semantics", "lowering")
    ]
    assert not structural, (
        "compare_symbolic_to_lowered_ffn structural drift:\n"
        + "\n".join(f"  [{i.kind}] {i.message}" for i in structural)
    )


def test_layer10_alu_cmp_combine_rules_match_legacy_units():
    """Sub-stage 1 (units 0..17): EQ/NE/LT/GT/LE/GE comparison combine."""

    actual = _StubFFN()
    expected = _StubFFN()

    rules = _layer10_alu_cmp_combine_rules(100.0)
    start, end = _layout_range("layer10_alu.cmp_combine")
    next_unit = _lower_rules(actual, rules, start_unit=start, S=100.0)
    _set_layer10_alu(expected, 100.0, _SetDim)

    assert next_unit == end, (
        f"cmp_combine cursor drift: lowered to {next_unit}, expected {end}"
    )
    assert len(rules) == end - start, (
        f"cmp_combine rule count drift: {len(rules)} rules vs "
        f"{end - start} pinned units"
    )
    _assert_same_ffn_units(actual, expected, start, end)


def test_layer10_alu_cmp_combine_rules_compare_symbolic_to_lowered_ffn():
    """Structural ``compare_symbolic_to_lowered_ffn`` parity for cmp_combine."""

    _assert_substage_compare_clean(_layer10_alu_cmp_combine_rules(100.0))


def test_layer10_alu_bitwise_or_rules_match_legacy_units():
    """Sub-stage 2 (units 18..529): bitwise OR 3-way AND cross-product."""

    actual = _StubFFN()
    expected = _StubFFN()

    rules = _layer10_alu_bitwise_or_rules(100.0)
    start, end = _layout_range("layer10_alu.bitwise_or")
    next_unit = _lower_rules(actual, rules, start_unit=start, S=100.0)
    _set_layer10_alu(expected, 100.0, _SetDim)

    assert next_unit == end, (
        f"bitwise_or cursor drift: lowered to {next_unit}, expected {end}"
    )
    assert len(rules) == end - start, (
        f"bitwise_or rule count drift: {len(rules)} rules vs "
        f"{end - start} pinned units"
    )
    _assert_same_ffn_units(actual, expected, start, end)


def test_layer10_alu_bitwise_or_rules_compare_symbolic_to_lowered_ffn():
    """Structural ``compare_symbolic_to_lowered_ffn`` parity for bitwise_or."""

    _assert_substage_compare_clean(_layer10_alu_bitwise_or_rules(100.0))


def test_layer10_alu_bitwise_xor_rules_match_legacy_units():
    """Sub-stage 3 (units 530..1041): bitwise XOR 3-way AND cross-product."""

    actual = _StubFFN()
    expected = _StubFFN()

    rules = _layer10_alu_bitwise_xor_rules(100.0)
    start, end = _layout_range("layer10_alu.bitwise_xor")
    next_unit = _lower_rules(actual, rules, start_unit=start, S=100.0)
    _set_layer10_alu(expected, 100.0, _SetDim)

    assert next_unit == end, (
        f"bitwise_xor cursor drift: lowered to {next_unit}, expected {end}"
    )
    assert len(rules) == end - start, (
        f"bitwise_xor rule count drift: {len(rules)} rules vs "
        f"{end - start} pinned units"
    )
    _assert_same_ffn_units(actual, expected, start, end)


def test_layer10_alu_bitwise_xor_rules_compare_symbolic_to_lowered_ffn():
    """Structural ``compare_symbolic_to_lowered_ffn`` parity for bitwise_xor."""

    _assert_substage_compare_clean(_layer10_alu_bitwise_xor_rules(100.0))


def test_layer10_alu_bitwise_and_rules_match_legacy_units():
    """Sub-stage 4 (units 1042..1553): bitwise AND 3-way AND cross-product."""

    actual = _StubFFN()
    expected = _StubFFN()

    rules = _layer10_alu_bitwise_and_rules(100.0)
    start, end = _layout_range("layer10_alu.bitwise_and")
    next_unit = _lower_rules(actual, rules, start_unit=start, S=100.0)
    _set_layer10_alu(expected, 100.0, _SetDim)

    assert next_unit == end, (
        f"bitwise_and cursor drift: lowered to {next_unit}, expected {end}"
    )
    assert len(rules) == end - start, (
        f"bitwise_and rule count drift: {len(rules)} rules vs "
        f"{end - start} pinned units"
    )
    _assert_same_ffn_units(actual, expected, start, end)


def test_layer10_alu_bitwise_and_rules_compare_symbolic_to_lowered_ffn():
    """Structural ``compare_symbolic_to_lowered_ffn`` parity for bitwise_and."""

    _assert_substage_compare_clean(_layer10_alu_bitwise_and_rules(100.0))


def test_layer10_alu_mul_lo_rules_match_legacy_units():
    """Sub-stage 5 (units 1554..1809): MUL lo-nibble (a*b)%16 lookup."""

    actual = _StubFFN()
    expected = _StubFFN()

    rules = _layer10_alu_mul_lo_rules(100.0)
    start, end = _layout_range("layer10_alu.mul_lo")
    next_unit = _lower_rules(actual, rules, start_unit=start, S=100.0)
    _set_layer10_alu(expected, 100.0, _SetDim)

    assert next_unit == end, (
        f"mul_lo cursor drift: lowered to {next_unit}, expected {end}"
    )
    assert len(rules) == end - start, (
        f"mul_lo rule count drift: {len(rules)} rules vs "
        f"{end - start} pinned units"
    )
    _assert_same_ffn_units(actual, expected, start, end)


def test_layer10_alu_mul_lo_rules_compare_symbolic_to_lowered_ffn():
    """Structural ``compare_symbolic_to_lowered_ffn`` parity for mul_lo."""

    _assert_substage_compare_clean(_layer10_alu_mul_lo_rules(100.0))
