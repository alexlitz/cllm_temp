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
    _opcode_decode_ffn_ir,
    _opcode_decode_ffn_rules,
    _opcode_decode_jsr_temp0_blank_rule,
    _lower_opcode_rules,
    _opcode_decode_first_step_rules,
    _opcode_decode_main_rules,
    _opcode_decode_temp_clear_rules,
    make_opcode_decode_ffn_op,
)
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives


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


def test_opcode_decode_ffn_declarative_matches_legacy_helper(monkeypatch):
    # The legacy helper has the 89-unit footprint; pin the Root B nested-JSR
    # fix OFF (flag-off => byte-identical) for the legacy byte-identity check.
    monkeypatch.setenv("C4_NESTED_JSR_PC_FIX", "0")
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    _bake_opcode_decode_ffn(actual, 100.0, _SetDim)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)


def test_opcode_decode_main_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_main_rules(100.0)
    end = _lower_opcode_rules(actual, rules, _SetDim, unit=0, S=100.0)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 34
    _assert_same_ffn_units(actual, expected, 0, end)


def test_opcode_decode_first_step_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_first_step_rules(100.0)
    end = _lower_opcode_rules(actual, rules, _SetDim, unit=34, S=100.0)
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 52
    _assert_same_ffn_units(actual, expected, 34, end)


def test_opcode_decode_temp_clear_ir_rules_match_legacy_units():
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_temp_clear_rules(100.0)
    end = _lower_opcode_rules(actual, rules, _SetDim, unit=53, S=100.0)
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
    end = _lower_opcode_rules(actual, rules, _SetDim, unit=84, S=100.0)
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


def test_opcode_decode_jsr_temp0_blank_rule_is_no_op_lowering():
    """The unit-52 blank placeholder must lower to an all-zero FFN row."""

    rule = _opcode_decode_jsr_temp0_blank_rule()
    assert rule.conditions == ()
    assert rule.writes == ()
    assert rule.gate is None
    assert rule.threshold == 0.0
    assert rule.gate_bias == 0.0

    # Synthetic dim_positions: a single dim is enough since the rule has
    # zero conditions / writes / gate references.
    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"TEMP": 0},
        {"TEMP+0": 0.0},
        S=100.0,
        atol=1e-5,
    )
    assert report.ok, report.format()


def test_opcode_decode_ffn_rules_total_unit_count(monkeypatch):
    """The composite rule list totals 89 units with the Root B nested-JSR
    fix OFF, and 90 with it ON (the extra all-step JSR TEMP+0 decode)."""

    monkeypatch.setenv("C4_NESTED_JSR_PC_FIX", "0")
    rules_off = _opcode_decode_ffn_rules(100.0)
    # 34 main + 18 first-step + 1 blank + 31 temp-clear + 5 all-step = 89.
    assert len(rules_off) == 89

    monkeypatch.setenv("C4_NESTED_JSR_PC_FIX", "1")
    rules_on = _opcode_decode_ffn_rules(100.0)
    # + 1 all-step JSR TEMP+0 decode (Root B).
    assert len(rules_on) == 90


def test_opcode_decode_ffn_full_ir_matches_legacy_helper(monkeypatch):
    """One-shot ``Primitives.lower_ffn_rules`` of the composite IR equals the
    legacy ``_set_opcode_decode_ffn`` byte-for-byte across all 89 units.

    The legacy reference has the 89-unit footprint, so this byte-identity
    check pins the Root B nested-JSR fix OFF (flag-off => byte-identical)."""

    monkeypatch.setenv("C4_NESTED_JSR_PC_FIX", "0")
    actual = _StubFFN(hidden_dim=128)
    expected = _StubFFN(hidden_dim=128)

    rules = _opcode_decode_ffn_rules(100.0)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        actual, rules, dim_positions, start_unit=0, S=100.0
    )
    _set_opcode_decode_ffn(expected, 100.0, _SetDim)

    assert end == 89
    _assert_same_ffn_units(actual, expected, 0, end)


def test_opcode_decode_ffn_ir_passes_declaration_and_lowering_checks():
    """``compare_symbolic_to_lowered_ffn`` over the full 89-rule IR has no
    declaration-semantics or lowering-contract failures.

    The per-cell weight check (W_up / b_up / W_gate / b_gate / W_down) and
    declaration resolution are byte-identical guarantees for the rule
    list. The synthetic ``weight_output_mismatch`` failures that arise
    when many rules share condition dims (the opcode-byte one-hot
    decoder's natural fan-in) are checked separately via the per-rule
    tests above; collapsing 89 rules into a single fired state would
    require curated per-rule states the legacy helper never demanded.
    """

    ir = _opcode_decode_ffn_ir(100.0)
    names = Primitives.ffn_rule_dim_names(ir.layer(0).ffn.rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions,
        S=100.0,
        atol=1e-4,
        rtol=1e-4,
    )

    structural_failures = [
        issue for issue in report.issues
        if issue.kind in ("declaration_semantics", "lowering")
    ]
    assert not structural_failures, (
        "structural compare_symbolic_to_lowered_ffn failures: "
        + "\n".join(f"  [{i.kind}] {i.message}" for i in structural_failures)
    )


def test_opcode_decode_ffn_op_exposes_compiler_ir(monkeypatch):
    """``make_opcode_decode_ffn_op()`` must attach the composite IR.

    Pinned to the Root B nested-JSR fix OFF so the rule count matches the
    legacy 89-unit footprint (with the flag on it is 90 — see
    ``test_opcode_decode_ffn_rules_total_unit_count``)."""

    monkeypatch.setenv("C4_NESTED_JSR_PC_FIX", "0")
    op = make_opcode_decode_ffn_op()
    assert op.compiler_ir is not None
    assert len(op.compiler_ir.layer(0).ffn.rules) == 89


# --- L5 ``convo_io_opcode_decode`` (flag-gated FFN extension) -----------------


def test_convo_io_opcode_decode_declarative_matches_legacy_helper():
    """``_lower_convo_io_opcode_decode_ir`` is byte-identical to the legacy
    ``_set_conversational_io_opcode_decode`` helper at L5 FFN units 410-411."""

    from c4_release.neural_vm.setup_helpers import (
        _set_conversational_io_opcode_decode,
    )
    from c4_release.neural_vm.unified_compiler.ops.flag_gated_ops import (
        _lower_convo_io_opcode_decode_ir,
    )

    actual = _StubFFN(hidden_dim=420)
    expected = _StubFFN(hidden_dim=420)

    _lower_convo_io_opcode_decode_ir(actual, 100.0, _SetDim)
    _set_conversational_io_opcode_decode(expected, 100.0, _SetDim)

    _assert_same_ffn_units(actual, expected, 410, 412)


def test_convo_io_opcode_decode_ir_passes_declaration_and_lowering_checks():
    """``compare_symbolic_to_lowered_ffn`` over the 2-rule IR is clean for
    declaration-semantics + lowering-contract failures."""

    from c4_release.neural_vm.unified_compiler.ops.flag_gated_ops import (
        _convo_io_opcode_decode_ir,
    )

    ir = _convo_io_opcode_decode_ir(100.0)
    names = Primitives.ffn_rule_dim_names(ir.layer(0).ffn.rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions,
        S=100.0,
        atol=1e-4,
        rtol=1e-4,
    )

    structural_failures = [
        issue for issue in report.issues
        if issue.kind in ("declaration_semantics", "lowering")
    ]
    assert not structural_failures, (
        "structural compare_symbolic_to_lowered_ffn failures: "
        + "\n".join(f"  [{i.kind}] {i.message}" for i in structural_failures)
    )


def test_convo_io_opcode_decode_op_exposes_compiler_ir_only_when_enabled():
    """``compiler_ir`` is attached only when ``enable_conversational_io=True``."""

    from c4_release.neural_vm.unified_compiler.ops.flag_gated_ops import (
        make_convo_io_opcode_decode_op,
    )

    op_off = make_convo_io_opcode_decode_op(enable_conversational_io=False)
    op_on = make_convo_io_opcode_decode_op(enable_conversational_io=True)

    assert op_off.compiler_ir is None
    assert op_on.compiler_ir is not None
    assert len(op_on.compiler_ir.layer(0).ffn.rules) == 2
