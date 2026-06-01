"""F-12: assert_rule_scopes_satisfied helper tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.dim_registry import DimRegistry  # noqa: E402
from neural_vm.unified_compiler.ir import FFNRule, FFNOp  # noqa: E402

from tests._per_op_audit import assert_rule_scopes_satisfied  # noqa: E402


class _FakeOp:
    def __init__(self, rules, name="fake_op"):
        self.compiler_ir = FFNOp(rules=list(rules))
        self.name = name


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP", semantics="mark == SP")
    r.alloc("CMP_JSR", 1, 1, "CMP+4", semantics="opcode_at_AX == JSR")
    r.alloc("OUT", 2, 1, "out", semantics="mark == AX")
    return r


def test_no_rules_passes(reg):
    op = _FakeOp([])
    assert_rule_scopes_satisfied(op, reg)


def test_rule_without_scope_passes_in_optin_mode(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="no_scope_rule",
    )
    op = _FakeOp([rule])
    assert_rule_scopes_satisfied(op, reg)  # opt-in: silently skipped


def test_rule_without_scope_fails_in_require_mode(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="no_scope_rule",
    )
    op = _FakeOp([rule])
    with pytest.raises(AssertionError, match="no_scope"):
        assert_rule_scopes_satisfied(op, reg, require_scope=True)


def test_rule_with_matching_scope_passes(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="good",
        scope="mark == SP",
    )
    op = _FakeOp([rule])
    assert_rule_scopes_satisfied(op, reg)


def test_e5_case_fails(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0), ("CMP_JSR", 0.5)),
        threshold=10.04,
        writes=(("OUT", 1.0),),
        name="e5_case",
        scope="mark == SP AND opcode_at_AX == JSR AND step_index == 0",
    )
    op = _FakeOp([rule])
    with pytest.raises(AssertionError, match="scope_violation"):
        assert_rule_scopes_satisfied(op, reg)
