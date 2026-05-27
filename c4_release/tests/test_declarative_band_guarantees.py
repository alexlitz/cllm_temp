import pytest

from neural_vm.unified_compiler.band_guarantees import (
    OneHotBandGuarantee,
    ScalarValueGuarantee,
    expected_byte_guarantee_rules,
    expected_nibble_guarantee_rules,
    one_hot_band_guarantee_rules,
    scalar_byte_guarantee_rules,
    scalar_nibble_guarantee_rules,
    scalar_value_guarantee_rules,
)
from neural_vm.unified_compiler.ir import CompilerIR


def _apply_rules(rules, state):
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return ir.symbolic_ffn(state)


def _guarantee(**overrides):
    kwargs = {
        "band_base": "OUTPUT_LO",
        "expected_index": 0xA,
        "activation_conditions": (("OP_PRTF", 1.0),),
        "read_conditions": (("IS_BYTE", 1.0), ("BYTE_INDEX_0", 1.0)),
        "min_margin": 0.75,
        "inactive_value": 0.0,
        "active_value": 1.0,
        "name": "output_lo_expected_a",
    }
    kwargs.update(overrides)
    return OneHotBandGuarantee(**kwargs)


def test_one_hot_guarantee_fires_when_activation_and_read_conditions_are_met():
    rules = _guarantee().to_ffn_rules()
    state = {
        **{
            f"OUTPUT_LO+{lane}": 0.25 + lane * 0.01
            for lane in range(16)
        },
        "OP_PRTF": 1.0,
        "IS_BYTE": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OUTPUT_LO+10": -0.2,
        "UNRELATED": 13.0,
    }

    out = _apply_rules(rules, state)

    assert out["OUTPUT_LO+10"] == pytest.approx(1.0)
    for lane in range(16):
        if lane == 10:
            continue
        assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
    assert out["OUTPUT_LO+10"] - max(
        out.get(f"OUTPUT_LO+{lane}", 0.0)
        for lane in range(16)
        if lane != 10
    ) >= 0.75
    assert out["UNRELATED"] == 13.0


def test_one_hot_guarantee_does_not_fire_when_activation_condition_is_missing():
    rules = _guarantee().to_ffn_rules()
    state = {
        "IS_BYTE": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OUTPUT_LO+7": 0.4,
        "OUTPUT_LO+10": -0.2,
        "UNRELATED": 13.0,
    }

    out = _apply_rules(rules, state)

    assert out == state


def test_one_hot_guarantee_does_not_fire_when_read_condition_is_missing():
    rules = _guarantee().to_ffn_rules()
    state = {
        "OP_PRTF": 1.0,
        "IS_BYTE": 1.0,
        "OUTPUT_LO+7": 0.4,
        "OUTPUT_LO+10": -0.2,
        "UNRELATED": 13.0,
    }

    out = _apply_rules(rules, state)

    assert out == state


def test_one_hot_guarantee_preserves_unrelated_band_state():
    rules = _guarantee(width=4, expected_index=2).to_ffn_rules()
    state = {
        "OP_PRTF": 1.0,
        "IS_BYTE": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OUTPUT_LO+0": 0.2,
        "OUTPUT_LO+1": 0.9,
        "OUTPUT_LO+2": 0.1,
        "OUTPUT_LO+3": -0.3,
        "OUTPUT_HI+1": 5.0,
        "TEMP+4": -7.0,
    }

    out = _apply_rules(rules, state)

    assert [out[f"OUTPUT_LO+{idx}"] for idx in range(4)] == pytest.approx([
        0.0,
        0.0,
        1.0,
        0.0,
    ])
    assert out["OUTPUT_HI+1"] == 5.0
    assert out["TEMP+4"] == -7.0


def test_one_hot_guarantee_emits_finite_bounded_lane_corrections():
    spec = _guarantee(max_abs_weight=4.0)
    rules = spec.to_ffn_rules()

    assert len(rules) == 16
    for lane, rule in enumerate(rules):
        assert rule.name == f"output_lo_expected_a.lane_{lane}"
        assert rule.threshold == pytest.approx(2.5)
        assert rule.gate is None
        assert rule.gate_terms[0].dim.key() == f"OUTPUT_LO+{lane}"
        assert rule.gate_terms[0].weight == pytest.approx(-1.0)
        assert rule.gate_bias == pytest.approx(1.0 if lane == 10 else 0.0)
        assert rule.writes[0].dim.key() == f"OUTPUT_LO+{lane}"
        assert rule.writes[0].weight == pytest.approx(1.0)
        assert all(abs(term.weight) <= 4.0 for term in rule.conditions)
        assert abs(rule.threshold) <= 4.0
        assert abs(rule.gate_bias) <= 4.0
        assert all(abs(term.weight) <= 4.0 for term in rule.gate_terms)
        assert all(abs(write.weight) <= 4.0 for write in rule.writes)


def test_function_helpers_support_nibble_and_general_band_contracts():
    nibble_rules = expected_nibble_guarantee_rules(
        band_base="OUTPUT_HI",
        expected_nibble=0x3,
        activation_conditions=(("OP_LI", 1.0),),
        read_conditions=(("IS_BYTE", 1.0),),
        min_margin=0.5,
    )
    general_rules = one_hot_band_guarantee_rules(
        band_base="MODE",
        expected_index=1,
        activation_conditions=(("COND", 1.0),),
        width=3,
        min_margin=2.0,
        inactive_value=-1.0,
    )

    assert len(nibble_rules) == 16
    assert nibble_rules[3].gate_bias == pytest.approx(0.5)
    assert len(general_rules) == 3
    assert general_rules[1].gate_bias == pytest.approx(1.0)
    assert general_rules[0].gate_bias == pytest.approx(-1.0)


def test_expected_byte_helper_enforces_low_and_high_nibble_bands():
    rules = expected_byte_guarantee_rules(
        expected_byte=0x2A,
        activation_conditions=(("IS_BYTE", 1.0), ("OP_ADD", 1.0)),
        condition_threshold=1.5,
        inactive_value=-3.0,
        active_value=3.0,
        min_margin=6.0,
        max_abs_weight=6.0,
        name="ax_byte",
    )

    assert len(rules) == 32
    out = _apply_rules(rules, {
        "IS_BYTE": 1.0,
        "OP_ADD": 1.0,
        "OUTPUT_LO+0": 4.0,
        "OUTPUT_LO+10": -1.0,
        "OUTPUT_HI+1": 9.0,
        "OUTPUT_HI+2": -2.0,
    })

    assert out["OUTPUT_LO+10"] == pytest.approx(3.0)
    assert out["OUTPUT_HI+2"] == pytest.approx(3.0)
    for lane in range(16):
        if lane != 10:
            assert out[f"OUTPUT_LO+{lane}"] == pytest.approx(-3.0)
        if lane != 2:
            assert out[f"OUTPUT_HI+{lane}"] == pytest.approx(-3.0)


def test_scalar_value_guarantee_corrects_drift_to_exact_target():
    rules = scalar_value_guarantee_rules(
        value_dim="AX_BYTE1_LO_VALUE+0",
        expected_value=2.0,
        activation_conditions=(("IS_BYTE", 1.0), ("OP_ADD", 1.0)),
        read_conditions=(("BYTE_INDEX_1", 1.0),),
        condition_threshold=2.5,
        name="ax_byte1_lo_exact_2",
    )

    out = _apply_rules(rules, {
        "IS_BYTE": 1.0,
        "OP_ADD": 1.0,
        "BYTE_INDEX_1": 1.0,
        "AX_BYTE1_LO_VALUE+0": 1.73,
        "UNRELATED": 9.0,
    })

    assert out["AX_BYTE1_LO_VALUE+0"] == pytest.approx(2.0)
    assert out["UNRELATED"] == 9.0


def test_scalar_value_guarantee_does_not_fire_without_proof_conditions():
    rules = scalar_value_guarantee_rules(
        value_dim="AX_BYTE1_LO_VALUE+0",
        expected_value=2.0,
        activation_conditions=(("IS_BYTE", 1.0), ("OP_ADD", 1.0)),
        read_conditions=(("BYTE_INDEX_1", 1.0),),
        condition_threshold=2.5,
    )

    state = {
        "IS_BYTE": 1.0,
        "OP_ADD": 1.0,
        "AX_BYTE1_LO_VALUE+0": 1.73,
    }
    assert _apply_rules(rules, state) == state


def test_scalar_nibble_and_byte_helpers_validate_ranges_and_correct_values():
    nibble_rules = scalar_nibble_guarantee_rules(
        value_dim="SP_LO_VALUE+0",
        expected_nibble=0xE,
        activation_conditions=(("MARK_SP", 1.0),),
        max_abs_weight=16.0,
    )
    byte_rules = scalar_byte_guarantee_rules(
        low_value_dim="OUTPUT_LOW_VALUE+0",
        high_value_dim="OUTPUT_HIGH_VALUE+0",
        expected_byte=0x2A,
        activation_conditions=(("OP_ADD", 1.0),),
        name="output_byte_exact",
    )

    out = _apply_rules(nibble_rules + byte_rules, {
        "MARK_SP": 1.0,
        "OP_ADD": 1.0,
        "SP_LO_VALUE+0": 13.6,
        "OUTPUT_LOW_VALUE+0": 9.1,
        "OUTPUT_HIGH_VALUE+0": 1.8,
    })

    assert out["SP_LO_VALUE+0"] == pytest.approx(14.0)
    assert out["OUTPUT_LOW_VALUE+0"] == pytest.approx(10.0)
    assert out["OUTPUT_HIGH_VALUE+0"] == pytest.approx(2.0)
    assert byte_rules[0].name == "output_byte_exact.lo"
    assert byte_rules[1].name == "output_byte_exact.hi"


def test_scalar_guarantees_refuse_invalid_contracts():
    with pytest.raises(ValueError, match="expected_nibble"):
        scalar_nibble_guarantee_rules(
            value_dim="NIB",
            expected_nibble=16,
            activation_conditions=(("COND", 1.0),),
        )
    with pytest.raises(ValueError, match="expected_byte"):
        scalar_byte_guarantee_rules(
            low_value_dim="LO",
            high_value_dim="HI",
            expected_byte=256,
            activation_conditions=(("COND", 1.0),),
        )
    with pytest.raises(ValueError, match="exceeds max_abs_weight"):
        ScalarValueGuarantee(
            value_dim="NIB",
            expected_value=32.0,
            activation_conditions=(("COND", 1.0),),
            max_abs_weight=16.0,
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"width": 0}, "width must be positive"),
        ({"expected_index": -1}, "expected_index must be within band width"),
        ({"expected_index": 16}, "expected_index must be within band width"),
        (
            {"activation_conditions": (), "read_conditions": ()},
            "at least one activation or read condition is required",
        ),
        (
            {"active_value": 0.25, "inactive_value": 0.0, "min_margin": 0.75},
            "active_value - inactive_value must satisfy min_margin",
        ),
        (
            {"activation_conditions": (("OP_PRTF", 5.0),), "max_abs_weight": 4.0},
            "exceeds max_abs_weight",
        ),
    ],
)
def test_one_hot_guarantee_refuses_invalid_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _guarantee(**overrides)
