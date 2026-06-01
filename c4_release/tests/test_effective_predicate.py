"""Tests for F-5 effective-predicate inference."""
import pytest
from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule
from neural_vm.unified_compiler.predicates import parse, entails
from neural_vm.unified_compiler.effective_predicate import effective_predicate


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP marker", semantics="mark == SP")
    r.alloc("MARK_AX", 1, 1, "AX marker", semantics="mark == AX")
    r.alloc("CMP_JSR", 2, 1, "CMP+4 flag", semantics="opcode_at_AX == JSR")
    r.alloc("IN_STEP_FRESH", 3, 1, "Step is fresh", semantics="in_step_fresh")
    r.alloc("HAS_SE", 4, 1, "Step end", semantics="has_se")
    return r


def test_single_positive_meets_threshold(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("MARK_AX", 1.0),),
        name="test_single",
    )
    pred = effective_predicate(rule, reg)
    # Effective: positions where mark == SP
    assert entails(pred, parse("mark == SP"))


def test_two_positives_must_combine(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 3.0), ("CMP_JSR", 3.0)),
        threshold=5.0,
        writes=(("MARK_AX", 1.0),),
        name="test_two",
    )
    pred = effective_predicate(rule, reg)
    # Either alone is insufficient; only the combination fires:
    assert entails(pred, parse("mark == SP AND opcode_at_AX == JSR"))


def test_hard_blocker_excludes_positions(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0), ("HAS_SE", -1e9)),
        threshold=5.0,
        writes=(("MARK_AX", 1.0),),
        name="test_blocker",
    )
    pred = effective_predicate(rule, reg)
    # Should require mark==SP AND NOT has_se
    assert entails(pred, parse("mark == SP AND NOT has_se"))
    # And should NOT entail "mark == SP" without the NOT-has_se part
    # (positions with mark==SP and has_se are NOT in the effective firing set)


def test_soft_blocker_ignored_in_over_approximation(reg):
    """Soft blockers (|w| < 1e6) tighten the firing region but the
    over-approximation should still admit those positions."""
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0), ("HAS_SE", -10.0)),
        threshold=5.0,
        writes=(("MARK_AX", 1.0),),
        name="test_soft",
    )
    pred = effective_predicate(rule, reg)
    # Soft blocker ignored: just mark == SP
    assert entails(pred, parse("mark == SP"))


def test_e5_case(reg):
    """The headline E5 case: a rule with CMP+4 as positive evidence
    has effective scope = positions where opcode_at_AX == JSR (ANY
    JSR), not 'step-0 only'."""
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0), ("CMP_JSR", 0.5)),
        threshold=10.04,
        writes=(("MARK_AX", 1.0),),
        name="e5_case",
    )
    pred = effective_predicate(rule, reg)
    # Effective predicate admits positions with mark == SP AND opcode_at_AX == JSR
    # (regardless of step_index)
    assert entails(pred, parse("mark == SP AND opcode_at_AX == JSR"))
    # And does NOT entail "step_index == 0" — i.e., the rule fires on
    # mid-program JSR rows, not just step-0 bootstrap.
    assert not entails(pred, parse("step_index == 0"))
