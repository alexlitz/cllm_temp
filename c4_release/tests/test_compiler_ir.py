import pytest
import torch

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    FFNRule,
    SymbolicDeclarativeRunner,
    SymbolicResidualState,
    compare_symbolic_to_lowered_ffn,
)


_SILU_ONE_INPUT = 1.278464542761074
_LOWERING_EQUIV_S = 100.0


def _active_ffn_threshold(score):
    return score - _SILU_ONE_INPUT / _LOWERING_EQUIV_S


from c4_release.neural_vm.unified_compiler.ops.l15_ops import (
    lower_l15_psh_stack_ir,
    make_l15_psh_stack_ir,
)
from c4_release.neural_vm.unified_compiler.ops.l10_ops import (
    _tail_bit32_result_correction_rules,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    Operation,
    dispatch_operation_bake,
    operation_supports_declarations_only,
)


def test_symbolic_ffn_constant_write_fires_when_threshold_met():
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.constant_write(
        name="psh_bp_byte2",
        conditions=(("PSH_AT_SP", 1.0), ("H1+3", 1.0), ("BYTE_INDEX_1", 1.0)),
        threshold=2.5,
        writes=(("OUTPUT_LO+1", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))

    out = ir.symbolic_ffn({
        "PSH_AT_SP+0": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_1+0": 1.0,
        "OUTPUT_LO+0": 4.0,
    })

    assert out["OUTPUT_LO+1"] == 4.0
    assert out["OUTPUT_LO+0"] == 0.0


def test_symbolic_ffn_constant_write_does_not_fire_below_threshold():
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.constant_write(
        conditions=(("PSH_AT_SP", 1.0), ("H1+3", 1.0)),
        threshold=1.5,
        writes=(("OUTPUT_LO+1", 4.0),),
    ))

    out = ir.symbolic_ffn({"PSH_AT_SP+0": 1.0})

    assert "OUTPUT_LO+1" not in out


def test_lower_ffn_matches_symbolic_positive_case():
    dim_positions = {
        "PSH_AT_SP": 0,
        "H1": 1,
        "BYTE_INDEX_1": 8,
        "OUTPUT_LO": 16,
    }
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.constant_write(
        conditions=(("PSH_AT_SP", 1.0), ("H1+3", 1.0), ("BYTE_INDEX_1", 1.0)),
        threshold=2.5,
        writes=(("OUTPUT_LO+1", 4.0 / 100.0),),
    ))

    ffn = PureFFN(dim=32, hidden_dim=ir.required_ffn_units())
    ir.lower_ffn(ffn, dim_positions, S=100.0)

    x = torch.zeros(1, 1, 32)
    x[..., dim_positions["PSH_AT_SP"]] = 1.0
    x[..., dim_positions["H1"] + 3] = 1.0
    x[..., dim_positions["BYTE_INDEX_1"]] = 1.0
    y = ffn(x)

    assert float(y[..., dim_positions["OUTPUT_LO"] + 1].item()) > 1.0


@pytest.mark.lowering
def test_lower_ffn_accumulates_duplicate_terms_like_symbolic_ffn():
    dim_positions = {
        "A": 0,
        "G": 1,
        "OUT": 2,
    }
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.gated_write(
        conditions=(("A", 0.25), ("A", 0.5)),
        threshold=0.5,
        gate="G",
        gate_terms=(("G", 2.0),),
        writes=(("OUT", 1.0), ("OUT", 2.0)),
    ))

    ffn = PureFFN(dim=8, hidden_dim=ir.required_ffn_units())
    ir.lower_ffn(ffn, dim_positions, S=10.0)

    assert float(ffn.W_up[0, dim_positions["A"]].item()) == pytest.approx(7.5)
    assert float(ffn.W_gate[0, dim_positions["G"]].item()) == pytest.approx(3.0)
    assert float(ffn.W_down[dim_positions["OUT"], 0].item()) == pytest.approx(3.0)


def test_compare_symbolic_to_lowered_ffn_psh_like_rule_matches():
    dim_positions = {
        "OP_PSH": 0,
        "MARK_SP": 1,
        "BYTE_INDEX_0": 2,
        "STACK0_LO": 8,
        "OUTPUT_LO": 16,
    }
    rule = FFNRule.constant_write(
        name="psh_stack0_byte0_to_output",
        conditions=(
            ("OP_PSH", 1.0),
            ("MARK_SP", 1.0),
            ("BYTE_INDEX_0", 1.0),
        ),
        threshold=2.0,
        writes=(("OUTPUT_LO+0", 4.0), ("OUTPUT_LO+1", -4.0)),
    )
    state = {
        "OP_PSH": 1.0,
        "MARK_SP": 1.0,
        "BYTE_INDEX_0": _SILU_ONE_INPUT,
        "OUTPUT_LO+1": 4.0,
    }

    report = compare_symbolic_to_lowered_ffn(
        rule,
        dim_positions,
        state,
        S=1.0,
        atol=1e-5,
    )

    assert report.ok, report.format()
    assert report.symbolic_state["OUTPUT_LO+0"] == 4.0
    assert abs(report.lowered_state["OUTPUT_LO+0"] - 4.0) < 1e-5
    assert abs(report.lowered_state["OUTPUT_LO+1"]) < 1e-5


def test_compare_symbolic_to_lowered_ffn_can_build_synthetic_state():
    rule = FFNRule.constant_write(
        conditions=(("OP_PSH", 1.0), ("MARK_SP", 1.0)),
        threshold=1.5,
        writes=(("OUT", 2.0),),
    )

    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"OP_PSH": 0, "MARK_SP": 1, "OUT": 4},
    )

    assert report.ok, report.format()
    assert abs(report.lowered_state["OUT+0"] - 2.0) < 3e-5


def test_compare_symbolic_to_lowered_ffn_gated_canceling_rules_match():
    dim_positions = {
        "OP_PSH": 0,
        "MARK_SP": 1,
        "SRC_LO": 8,
        "OUTPUT_LO": 16,
    }
    threshold = 2.0 - _SILU_ONE_INPUT
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.gated_write(
        name="psh_gated_add",
        conditions=(("OP_PSH", 1.0), ("MARK_SP", 1.0)),
        threshold=threshold,
        gate="SRC_LO",
        gate_weight=2.0,
        writes=(("OUTPUT_LO", 3.0),),
    ))
    ir.layer(0).ffn.append(FFNRule.gated_write(
        name="psh_gated_cancel",
        conditions=(("OP_PSH", 1.0), ("MARK_SP", 1.0)),
        threshold=threshold,
        gate="SRC_LO",
        gate_weight=2.0,
        writes=(("OUTPUT_LO", -3.0),),
    ))

    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions,
        {
            "OP_PSH": 1.0,
            "MARK_SP": 1.0,
            "SRC_LO": 0.5,
            "OUTPUT_LO": 7.0,
        },
        S=1.0,
        atol=1e-5,
    )

    assert report.ok, report.format()
    assert abs(report.lowered_state["OUTPUT_LO+0"] - 7.0) < 1e-5


def test_compare_symbolic_to_lowered_ffn_multi_gate_rules_match():
    dim_positions = {
        "COND": 0,
        "OLD": 1,
        "NEW": 2,
        "OUT": 3,
    }
    threshold = 1.0 - _SILU_ONE_INPUT
    rule = FFNRule.gated_write(
        name="multi_gate_cancel_and_write",
        conditions=(("COND", 1.0),),
        threshold=threshold,
        gate_terms=(("OLD", -1.0), ("NEW", 1.0)),
        writes=(("OUT", 2.0),),
    )

    report = compare_symbolic_to_lowered_ffn(
        rule,
        dim_positions,
        {"COND": 1.0, "OLD": 3.0, "NEW": 8.0},
        S=1.0,
        atol=1e-5,
    )

    assert report.ok, report.format()
    assert abs(report.lowered_state["OUT+0"] - 10.0) < 1e-5


@pytest.mark.lowering
@pytest.mark.parametrize(
    ("rule", "dim_positions", "state", "expected"),
    [
        pytest.param(
            FFNRule.constant_write(
                name="duplicate_condition_dims",
                conditions=(("COND", 0.25), ("COND", 0.75)),
                threshold=_active_ffn_threshold(1.0),
                writes=(("OUT", 2.5),),
            ),
            {"COND": 0, "OUT": 4},
            {"COND": 1.0},
            {"OUT+0": 2.5},
            id="duplicate-condition-dims",
        ),
        pytest.param(
            FFNRule.gated_write(
                name="duplicate_gate_dims",
                conditions=(("COND", 1.0),),
                threshold=_active_ffn_threshold(1.0),
                gate="GATE",
                gate_weight=0.5,
                gate_terms=(("GATE", 1.5),),
                writes=(("OUT", 2.0),),
            ),
            {"COND": 0, "GATE": 1, "OUT": 4},
            {"COND": 1.0, "GATE": 3.0},
            {"OUT+0": 12.0},
            id="duplicate-gate-dims",
        ),
        pytest.param(
            FFNRule.constant_write(
                name="duplicate_writes",
                conditions=(("COND", 1.0),),
                threshold=_active_ffn_threshold(1.0),
                writes=(("OUT", 3.0), ("OUT", -0.75), ("OUT+0", 0.25)),
            ),
            {"COND": 0, "OUT": 4},
            {"COND": 1.0},
            {"OUT+0": 2.5},
            id="duplicate-writes",
        ),
        pytest.param(
            FFNRule.constant_write(
                name="negative_blocker_allows",
                conditions=(("COND", 1.0), ("BLOCK", -1.0)),
                threshold=_active_ffn_threshold(1.0),
                writes=(("OUT", 1.5),),
            ),
            {"COND": 0, "BLOCK": 1, "OUT": 4},
            {"COND": 1.0, "BLOCK": 0.0},
            {"OUT+0": 1.5},
            id="negative-blocker-allows",
        ),
        pytest.param(
            FFNRule.constant_write(
                name="negative_blocker_blocks",
                conditions=(("COND", 1.0), ("BLOCK", -1.0)),
                threshold=_active_ffn_threshold(1.0),
                writes=(("OUT", 1.5),),
            ),
            {"COND": 0, "BLOCK": 1, "OUT": 4},
            {"COND": 1.0, "BLOCK": 1.0},
            {"OUT+0": 0.0},
            id="negative-blocker-blocks",
        ),
        pytest.param(
            FFNRule.gated_write(
                name="gated_write_terms",
                conditions=(("COND", 1.0),),
                threshold=_active_ffn_threshold(1.0),
                gate="SRC",
                gate_weight=1.25,
                gate_terms=(("OFFSET", -0.5),),
                gate_bias=0.75,
                writes=(("OUT", -2.0),),
            ),
            {"COND": 0, "SRC": 1, "OFFSET": 2, "OUT": 4},
            {"COND": 1.0, "SRC": 4.0, "OFFSET": 2.0},
            {"OUT+0": -9.5},
            id="gated-write",
        ),
    ],
)
def test_compare_symbolic_to_lowered_ffn_equivalence_cases(
    rule, dim_positions, state, expected
):
    report = compare_symbolic_to_lowered_ffn(
        rule,
        dim_positions,
        state,
        S=_LOWERING_EQUIV_S,
        atol=3e-5,
        rtol=3e-5,
    )

    assert report.ok, report.format()
    for key, value in expected.items():
        assert report.symbolic_state.get(key, 0.0) == pytest.approx(value)
        assert report.lowered_state[key] == pytest.approx(
            value,
            abs=3e-5,
            rel=3e-5,
        )


def test_compare_symbolic_to_lowered_ffn_reports_declaration_semantics():
    rule = FFNRule.constant_write(
        conditions=(("OP_PSH", 1.0),),
        threshold=0.5,
        writes=(("MISSING_OUTPUT", 1.0),),
    )

    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"OP_PSH": 0},
        {"OP_PSH": 1.0},
    )

    assert not report.ok
    assert report.primary_failure_kind == "declaration_semantics"
    assert "undeclared dim 'MISSING_OUTPUT'" in report.format()


def test_compare_symbolic_to_lowered_ffn_reports_lowering_mismatch():
    dim_positions = {"OP_PSH": 0, "OUT": 4}
    rule = FFNRule.constant_write(
        name="bad_lowering_probe",
        conditions=(("OP_PSH", 1.0),),
        threshold=1.0 - _SILU_ONE_INPUT,
        writes=(("OUT", 2.0),),
    )
    ffn = PureFFN(dim=8, hidden_dim=1)
    ir = CompilerIR()
    ir.layer(0).ffn.append(rule)
    ir.lower_ffn(ffn, dim_positions, S=1.0)
    ffn.W_down.data[dim_positions["OUT"], 0] = 3.0

    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions,
        {"OP_PSH": 1.0},
        ffn=ffn,
        lower=False,
        S=1.0,
    )

    assert not report.ok
    assert report.primary_failure_kind == "lowering"
    assert "W_down" in report.format()


def test_compare_symbolic_to_lowered_ffn_reports_weight_output_mismatch():
    dim_positions = {"OP_PSH": 0, "OUT": 4}
    rule = FFNRule.constant_write(
        name="swiglu_scale_probe",
        conditions=(("OP_PSH", 1.0),),
        threshold=0.0,
        writes=(("OUT", 2.0),),
    )

    report = compare_symbolic_to_lowered_ffn(
        rule,
        dim_positions,
        {"OP_PSH": 1.0},
        S=10.0,
    )

    assert not report.ok
    assert report.primary_failure_kind == "weight_output_mismatch"
    assert "symbolic expected" in report.format()


def _psh_state(marker_index, byte_index_name):
    return {
        "PSH_AT_SP+0": 1.0,
        f"H1+{marker_index}": 1.0,
        "IS_BYTE+0": 1.0,
        f"{byte_index_name}+0": 1.0,
        "OUTPUT_LO+0": 4.0,
        "OUTPUT_HI_THIS_STEP+0": 4.0,
    }


def test_l15_psh_stack_ir_symbolic_sp_byte_outputs():
    ir = make_l15_psh_stack_ir()

    sp_byte0 = ir.symbolic_ffn(_psh_state(2, "BYTE_INDEX_0"))
    assert sp_byte0["OUTPUT_LO+15"] == 4.0
    assert sp_byte0["OUTPUT_HI_THIS_STEP+15"] == 4.0
    assert sp_byte0["OUTPUT_LO+0"] == 0.0
    assert sp_byte0["OUTPUT_HI_THIS_STEP+0"] == 0.0

    sp_byte1 = ir.symbolic_ffn(_psh_state(2, "BYTE_INDEX_1"))
    assert sp_byte1["OUTPUT_LO+0"] == 6.0
    assert sp_byte1["OUTPUT_HI_THIS_STEP+0"] == 6.0

    sp_byte2 = ir.symbolic_ffn(_psh_state(2, "BYTE_INDEX_2"))
    assert sp_byte2["OUTPUT_LO+0"] == 6.0
    assert sp_byte2["OUTPUT_HI_THIS_STEP+0"] == 6.0


def test_l15_psh_stack_ir_blocks_stack0_byte_residue():
    ir = make_l15_psh_stack_ir()

    stack0_byte0 = _psh_state(2, "BYTE_INDEX_0")
    stack0_byte0["PSH_AT_SP+0"] = 2.0
    stack0_byte0["H1+2"] = 0.0
    stack0_byte0["H1+10"] = 1.0
    stack0_byte0["H4+3"] = 1.0

    out = ir.symbolic_ffn(stack0_byte0)

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI_THIS_STEP+15", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 4.0
    assert out["OUTPUT_HI_THIS_STEP+0"] == 4.0


def test_l15_psh_stack_ir_symbolic_bp_byte2_preserved():
    ir = make_l15_psh_stack_ir()

    bp_byte1 = ir.symbolic_ffn(_psh_state(3, "BYTE_INDEX_1"))
    assert bp_byte1["OUTPUT_LO+1"] == 4.0
    assert bp_byte1["OUTPUT_HI_THIS_STEP+0"] == 8.0
    assert bp_byte1["OUTPUT_LO+0"] == 0.0


def test_l15_psh_stack_ir_lowers_legacy_weight_scale():
    dim_positions = {
        "PSH_AT_SP": 0,
        "H1": 1,
        "IS_BYTE": 8,
        "BYTE_INDEX_0": 9,
        "BYTE_INDEX_1": 10,
        "BYTE_INDEX_2": 11,
        "H4": 12,
        "OUTPUT_LO": 16,
        "OUTPUT_HI": 32,
        "OUTPUT_HI_THIS_STEP": 32,
    }
    ffn = PureFFN(dim=48, hidden_dim=8)

    next_unit = lower_l15_psh_stack_ir(
        ffn,
        dim_positions,
        start_unit=0,
        S=100.0,
    )

    assert next_unit == 8
    assert ffn.W_down.data[
        dim_positions["OUTPUT_LO"] + 15, 0
    ].item() == pytest.approx(0.04)
    assert ffn.W_down.data[
        dim_positions["OUTPUT_LO"] + 0, 0
    ].item() == pytest.approx(-0.04)


def test_symbolic_ffn_positions_runs_rules_on_each_position():
    dim_positions = {"COND": 0, "SRC": 1, "OUT": 2}
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.gated_write(
        conditions=(("COND", 1.0),),
        threshold=0.5,
        gate="SRC",
        writes=(("OUT", 2.0),),
    ))
    state = SymbolicResidualState([
        {},
        {0: 1.0, 1: 3.0},
        {0: 0.0, 1: 9.0},
    ])

    out = ir.symbolic_ffn_positions(state, dim_positions)

    assert out.get(1, 2) == 6.0
    assert out.get(2, 2) == 0.0


def test_symbolic_residual_state_from_named_positions_resolves_dim_refs():
    state = SymbolicResidualState.from_named_positions(
        [{"OUTPUT_LO+3": 1.5, "COND": 1.0}],
        {"COND": 0, "OUTPUT_LO": 16},
    )

    assert state.get(0, 0) == 1.0
    assert state.get(0, 19) == 1.5


def test_symbolic_layout_runner_reports_unsupported_ops():
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.constant_write(
        conditions=(("COND", 1.0),),
        threshold=0.5,
        writes=(("OUT", 1.0),),
    ))
    supported = Operation(
        name="supported",
        reads={"COND"},
        writes={"OUT"},
        kind="ffn",
        bake_fn=lambda *args, **kwargs: None,
        compiler_ir=ir,
    )
    unsupported = Operation(
        name="opaque",
        reads=set(),
        writes=set(),
        kind="ffn",
        bake_fn=lambda *args, **kwargs: None,
    )

    class _Layout:
        dim_positions = {"COND": 0, "OUT": 1}
        ops_per_layer = [[supported, unsupported]]
        block_ops = []
        model_ops = []

    report = SymbolicDeclarativeRunner(head_dim=1).run_layout(
        _Layout(),
        SymbolicResidualState([{0: 1.0}]),
    )

    assert not report.ok
    assert report.executed_ops == ["supported"]
    assert report.unsupported_ops == ["opaque"]
    assert report.final_state.get(0, 1) == 1.0


def test_operation_owned_ir_dispatches_in_declarations_only_mode():
    dim_positions = {"COND": 0, "OUT": 4}
    ir = CompilerIR()
    ir.layer(0).ffn.append(FFNRule.constant_write(
        conditions=(("COND", 1.0),),
        threshold=0.5,
        writes=(("OUT+1", 4.0 / 100.0),),
    ))
    op = Operation(
        name="ir_backed_test_op",
        reads={"COND"},
        writes={"OUT"},
        kind="ffn",
        bake_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("imperative bake must not run")
        ),
        compiler_ir=ir,
        ffn_units_used=ir.required_ffn_units(),
    )
    ffn = PureFFN(dim=8, hidden_dim=1)

    assert operation_supports_declarations_only(op)
    dispatch_operation_bake(
        op, ffn, dim_positions, 100.0, declarations_only=True,
    )

    x = torch.zeros(1, 1, 8)
    x[..., dim_positions["COND"]] = 1.0
    y = ffn(x)
    assert float(y[..., dim_positions["OUT"] + 1].item()) > 1.0


def test_tail_bit32_symbolic_mul_byte1_blocks_add_carry_correction():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_tail_bit32_result_correction_rules())

    state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,  # wide-MUL byte-1 signature
        "CARRY+1": 1.0,  # stale ADD carry residue must not be authoritative
        "ALU_LO+0": 1.0,
        "OUTPUT_LO+0": 43.0,
        "OUTPUT_HI+0": 43.0,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+0"] > 1000.0
    assert out["OUTPUT_LO+1"] < -1000.0
    assert out["OUTPUT_HI+0"] > 1000.0


def test_tail_bit32_symbolic_mul_byte1_preserves_nonzero_high_byte():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_tail_bit32_result_correction_rules())

    state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "CARRY+1": 1.0,
        "ALU_LO+1": 1.0,
        "OUTPUT_LO+1": 43.0,
        "OUTPUT_HI+0": 43.0,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+1"] > 1000.0
    assert out["OUTPUT_LO+2"] < -1000.0
    assert out["OUTPUT_HI+0"] > 1000.0


def test_tail_bit32_symbolic_gt_false_blocks_stale_tail_rules():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_tail_bit32_result_correction_rules())

    state = {
        "MARK_AX": 1.0,
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_GT": 5.0,
        "CMP+0": 1.0,
        "CMP+3": 55.0,
        "CARRY+1": 250000.0,
        "OUTPUT_LO+15": 1.0,
        "OUTPUT_HI+15": 1.0,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+0"] >= 1000.0
    assert out["OUTPUT_HI+0"] >= 1000.0
    assert out["OUTPUT_LO+15"] <= -999.0
    assert out["OUTPUT_HI+15"] <= -999.0


def test_tail_bit32_symbolic_lt_false_blocks_wide_mul_preserve():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_tail_bit32_result_correction_rules())

    state = {
        "MARK_AX": 1.0,
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_LT": 5.0,
        "TEMP+10": 1.0,
        "CMP+0": 154.0,
        "OUTPUT_LO+2": 1.0,
        "OUTPUT_HI+0": 1.0,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+0"] >= 1000.0
    assert out["OUTPUT_LO+2"] <= -999.0
    assert out["OUTPUT_HI+0"] >= 1000.0
