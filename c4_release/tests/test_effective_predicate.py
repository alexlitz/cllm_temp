"""Tests for F-5 effective-predicate inference."""
import pytest
from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule
from neural_vm.unified_compiler.predicates import parse, entails, satisfiable
from neural_vm.verification.effective_predicate import effective_predicate


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP marker", semantics="mark == SP")
    r.alloc("MARK_AX", 1, 1, "AX marker", semantics="mark == AX")
    r.alloc("CMP_JSR", 2, 1, "CMP+4 flag", semantics="opcode_at_AX == JSR")
    r.alloc("IN_STEP_FRESH", 3, 1, "Step is fresh", semantics="in_step_fresh")
    r.alloc("HAS_SE", 4, 1, "Step end", semantics="has_se")
    r.alloc("MARK_MEM", 5, 1, "MEM marker", semantics="mark == MEM")
    r.alloc("MARK_STACK0", 6, 1, "STACK0 marker", semantics="mark == STACK0")
    r.alloc(
        "MEM_STORE", 7, 1,
        "Store op active at MEM",
        semantics="mark == MEM AND opcode_in_step in {SI, SC, PSH}",
    )
    r.alloc("OUT", 8, 1, "Output dim", semantics="mark == STACK0")
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


def test_gate_semantics_included(reg):
    """F-5-gate-extension: gated_write rules use gate= to FORCE firing at
    specific positions regardless of conditions. The gate's semantics
    must be ANDed positively into the effective predicate, otherwise
    rules like L10's tail_stack0_store_loaded_byte_* (gate=MARK_STACK0,
    conditions include MEM_STORE + MARK_MEM-as-blocker) collapse to a
    contradiction sentinel."""
    rule = FFNRule.gated_write(
        conditions=(("MEM_STORE", 1.0), ("MARK_MEM", -1e9)),
        threshold=0.5,
        gate="MARK_STACK0",  # The gate makes this STACK0-only
        writes=(("OUT", 1.0),),
        name="gated",
    )
    pred = effective_predicate(rule, reg)
    # Should NOT be a contradiction — the gate's mark==STACK0 dominates.
    assert satisfiable(pred)
    # Should entail mark == STACK0
    assert entails(pred, parse("mark == STACK0"))
