"""S-10: assert_rule_strength_dominance helper tests."""
import pytest
from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule, FFNOp
from neural_vm.unified_compiler.backbone_bounds import BackboneBounds
from tests._per_op_audit import assert_rule_strength_dominance


class _FakeOp:
    def __init__(self, rules, name="fake"):
        self.compiler_ir = FFNOp(rules=list(rules))
        self.name = name


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP", semantics="mark == SP")
    r.alloc("MARK_AX", 1, 1, "AX", semantics="mark == AX")
    r.alloc("OUT_LO", 16, 16, "out", semantics="is_byte OR NOT is_byte")
    return r


def test_no_rules_passes(reg):
    op = _FakeOp([])
    assert_rule_strength_dominance(op, reg, bounds=BackboneBounds.empty())


def test_no_dominates_at_optin_skipped(reg):
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="r",
        scope="mark == SP",
        # no dominates_at
    )
    op = _FakeOp([r])
    assert_rule_strength_dominance(op, reg, bounds=BackboneBounds.empty())


def test_solo_rule_with_dominates_at_passes(reg):
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="solo",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )
    op = _FakeOp([r])
    assert_rule_strength_dominance(op, reg, bounds=BackboneBounds.empty())


def test_overpowered_rule_raises(reg):
    weak = FFNRule.constant_write(
        conditions=(("MARK_SP", 20.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="weak",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )
    strong = FFNRule.constant_write(
        conditions=(("MARK_SP", 1000.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="strong",
        scope="mark == SP",
    )
    op = _FakeOp([weak, strong])
    with pytest.raises(AssertionError, match="strength_violation"):
        assert_rule_strength_dominance(op, reg, bounds=BackboneBounds.empty())


def test_require_dominates_flags_missing(reg):
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="r",
        scope="mark == SP",
        # no dominates_at
    )
    op = _FakeOp([r])
    with pytest.raises(AssertionError, match="no_dominates_at"):
        assert_rule_strength_dominance(
            op, reg, bounds=BackboneBounds.empty(), require_dominates=True,
        )
