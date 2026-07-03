"""Tests for ``DSLInterpreter`` — symbolic execution of Operations."""

from __future__ import annotations

import pytest

from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    cancel_residual_rule,
    multi_way_and_rule,
    one_hot_indicator_rule,
    step_function_rule,
)
from c4_release.neural_vm.verification.dsl_interpreter import (
    DSLInterpreter,
    InterpreterResult,
    InterpreterStep,
)
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    FFNRule,
)


# ===========================================================================
# Base: single-rule application
# ===========================================================================


def test_step_function_rule_fires_above_threshold():
    """``step_function_rule`` fires when ``input >= threshold`` and
    contributes ``write_value`` (scaled by 1/S in the rule) to the
    write_dim."""

    rule = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0",
        write_value=2.0, S=100.0,
    )
    interp = DSLInterpreter(initial_state={"X+0": 1.0})
    step = interp.apply_ffn_rules([rule])
    assert step.rules_fired == 1
    # write_value=2.0, S=100 → write_weight = 0.02; gate_bias=1.0
    # symbolic output: 1.0 * 0.02 = 0.02 added to OUT+0
    assert pytest.approx(interp.get("OUT+0"), abs=1e-9) == 0.02


def test_step_function_rule_blocks_below_threshold():
    rule = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0", write_value=2.0,
    )
    interp = DSLInterpreter(initial_state={"X+0": 0.0})
    step = interp.apply_ffn_rules([rule])
    assert step.rules_fired == 0
    assert interp.get("OUT+0") == 0.0


def test_one_hot_indicator_rule_fires_on_target_only():
    """Indicator at BAND+5 should fire iff BAND+5 is the active cell."""

    rule = one_hot_indicator_rule(
        band="BAND", value=5, write_dim="OUT+0", write_value=2.0,
    )
    # Active cell is 5 → should fire
    interp_hit = DSLInterpreter(initial_state={"BAND+5": 1.0})
    step_hit = interp_hit.apply_ffn_rules([rule])
    assert step_hit.rules_fired == 1
    assert interp_hit.get("OUT+0") > 0.0

    # Active cell is 3 → should not fire
    interp_miss = DSLInterpreter(initial_state={"BAND+3": 1.0})
    step_miss = interp_miss.apply_ffn_rules([rule])
    assert step_miss.rules_fired == 0
    assert interp_miss.get("OUT+0") == 0.0


def test_multi_way_and_rule_requires_all_conditions():
    """3-way AND fires only when all 3 conditions are set."""

    rule = multi_way_and_rule(
        conditions=(("A", 1.0), ("B", 1.0), ("C", 1.0)),
        writes=(("OUT+0", 0.02),),
    )
    # Default threshold = 2.5 → all three must fire
    for a, b, c in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        interp = DSLInterpreter(
            initial_state={"A+0": float(a), "B+0": float(b), "C+0": float(c)},
        )
        step = interp.apply_ffn_rules([rule])
        if a and b and c:
            assert step.rules_fired == 1, f"AND({a},{b},{c}) should fire"
        else:
            assert step.rules_fired == 0, f"AND({a},{b},{c}) should NOT fire"


def test_cancel_residual_rule_subtracts():
    rule = cancel_residual_rule(input_dim="X", write_value=2.0, S=100.0)
    interp = DSLInterpreter(initial_state={"X+0": 1.0})
    step = interp.apply_ffn_rules([rule])
    assert step.rules_fired == 1
    # X was 1.0; rule writes -2/S=-0.02 back to X → 1.0 + (-0.02) = 0.98
    assert pytest.approx(interp.get("X+0"), abs=1e-9) == 1.0 + (-0.02)


# ===========================================================================
# Multi-rule + tracing
# ===========================================================================


def test_run_sequence_accumulates_steps():
    """Apply two ops in sequence; verify writes accumulate across steps."""

    rule1 = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0", write_value=2.0,
    )
    rule2 = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0", write_value=3.0,
    )

    # Wrap in CompilerIR -> Operation-like stubs for run()
    ir1 = CompilerIR()
    ir1.layer(0).ffn.rules.append(rule1)
    ir2 = CompilerIR()
    ir2.layer(0).ffn.rules.append(rule2)

    class _Op:
        def __init__(self, name, ir):
            self.name = name
            self.kind = "ffn"
            self.compiler_ir = ir

    interp = DSLInterpreter(initial_state={"X+0": 1.0})
    result = interp.run([_Op("op1", ir1), _Op("op2", ir2)])
    assert len(result.steps) == 2
    # 0.02 + 0.03 = 0.05 cumulative
    assert pytest.approx(interp.get("OUT+0"), abs=1e-9) == 0.05


def test_writes_to_helper_tracks_per_op_contribution():
    """``InterpreterResult.writes_to`` finds all ops that wrote to a dim."""

    rule = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0", write_value=2.0,
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    class _Op:
        def __init__(self, name, ir):
            self.name = name
            self.kind = "ffn"
            self.compiler_ir = ir

    interp = DSLInterpreter(initial_state={"X+0": 1.0})
    result = interp.run([_Op("op_alpha", ir), _Op("op_beta", ir)])
    contributions = result.writes_to("OUT")
    op_names = [name for name, _ in contributions]
    assert "op_alpha" in op_names
    assert "op_beta" in op_names


def test_opaque_op_records_a_note():
    """An op with no IR + no factory should record a note without
    error."""

    class _Op:
        name = "opaque_block_op"
        kind = "block"
        layer_idx = 0
        # No compiler_ir, no compiler_ir_factory.

    interp = DSLInterpreter()
    result = interp.run([_Op()])
    assert len(result.steps) == 1
    assert result.steps[0].rules_fired == 0
    assert any("opaque" in note for note in result.steps[0].notes)


# ===========================================================================
# Interpreter vs CompilerIR.symbolic_ffn — agreement
# ===========================================================================


def test_interpreter_matches_compiler_ir_symbolic_ffn():
    """For pure FFN rules, the interpreter must produce the same state
    as ``CompilerIR.symbolic_ffn``."""

    rules = [
        step_function_rule(
            input_dim="X", threshold=0.5, write_dim="OUT+0",
            write_value=2.0,
        ),
        multi_way_and_rule(
            conditions=(("A", 1.0), ("B", 1.0)),
            threshold=1.5,
            writes=(("OUT+1", 0.05),),
        ),
        cancel_residual_rule(input_dim="X", write_value=1.0, S=100.0),
    ]
    initial_state = {"X+0": 1.0, "A+0": 1.0, "B+0": 1.0}

    # Reference: CompilerIR.symbolic_ffn
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ref_state = ir.symbolic_ffn(initial_state)

    # Interpreter
    interp = DSLInterpreter(initial_state=initial_state)
    interp.apply_ffn_rules(rules)

    # The two state dicts should agree on every key the rules touched
    for key, ref_value in ref_state.items():
        actual = interp.state.get(key, 0.0)
        assert pytest.approx(actual, abs=1e-9) == ref_value, (
            f"interpreter vs symbolic_ffn disagree on {key}: "
            f"interp={actual} ref={ref_value}"
        )


# ===========================================================================
# Convenience surface
# ===========================================================================


def test_get_set_reset():
    interp = DSLInterpreter()
    assert interp.get("X+0") == 0.0
    interp.set("X+0", 5.0)
    assert interp.get("X+0") == 5.0
    interp.reset({"Y+0": 1.0})
    assert interp.get("X+0") == 0.0
    assert interp.get("Y+0") == 1.0


def test_interpreter_step_dataclass_shape():
    """``InterpreterStep`` carries op_name, op_kind, layer_idx,
    rules_fired, writes, notes."""

    step = InterpreterStep(
        op_name="test", op_kind="ffn", layer_idx=3, rules_fired=2,
    )
    assert step.op_name == "test"
    assert step.op_kind == "ffn"
    assert step.layer_idx == 3
    assert step.rules_fired == 2
    assert step.writes == []
    assert step.notes == []


def test_interpreter_result_holds_state_and_steps():
    interp = DSLInterpreter(initial_state={"X+0": 1.0})
    result = InterpreterResult(state=interp.state)
    assert result.state == {"X+0": 1.0}
    assert result.steps == []
    assert result.writes_to("X") == []
