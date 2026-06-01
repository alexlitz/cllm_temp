"""S-5: tests for FFNRule.dominates_at field threading."""
import pytest
from neural_vm.unified_compiler.ir import FFNRule


def test_default_dominates_at_is_none():
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
    )
    assert r.dominates_at is None


def test_constant_write_accepts_dominates_at():
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
        dominates_at={"Y": "mark == SP"},
    )
    assert r.dominates_at == {"Y": "mark == SP"}


def test_gated_write_accepts_dominates_at():
    r = FFNRule.gated_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
        gate="G",
        dominates_at={"Y": "mark == AX AND step_is_fresh"},
    )
    assert r.dominates_at == {"Y": "mark == AX AND step_is_fresh"}


def test_dominates_at_for_per_dim():
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0), ("Z", 1.0)),
        scope="mark == SP",
        dominates_at={"Y": "mark == SP AND has_se"},
    )
    # Y has explicit dominates_at; Z falls back to rule.scope
    assert r.dominates_at_for("Y") == "mark == SP AND has_se"
    assert r.dominates_at_for("Z") == "mark == SP"


def test_dominates_at_for_with_no_scope_no_dominates_at():
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
    )
    assert r.dominates_at_for("Y") is None


def test_frozen_dominates_at_immutable():
    """FFNRule is frozen — can't reassign dominates_at after construction."""
    r = FFNRule.constant_write(
        conditions=(("X", 1.0),),
        threshold=0.5,
        writes=(("Y", 1.0),),
        dominates_at={"Y": "mark == SP"},
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.dominates_at = {"Y": "mark == AX"}


def test_exact_output_byte_rules_threads_dominates_at():
    """If exact_output_byte_rules helper exists, dominates_at should
    thread through to generated rules."""
    try:
        from neural_vm.unified_compiler.ops.l10_ops import exact_output_byte_rules
    except ImportError:
        pytest.skip("exact_output_byte_rules not importable")
    rules = list(exact_output_byte_rules(
        name="test",
        expected_byte=0xF8,
        conditions=(("MARK_SP", 10.0),),
        threshold=10.0,
        max_abs_weight=1e6,
        scope="mark == SP",
        dominates_at={"OUTPUT_LO": "mark == SP AND step_index == 0",
                       "OUTPUT_HI": "mark == SP AND step_index == 0"},
    ))
    assert len(rules) > 0
    for r in rules:
        assert r.dominates_at is not None
        # Each generated rule should have inherited the dict
        assert "OUTPUT_LO" in r.dominates_at or "OUTPUT_HI" in r.dominates_at
