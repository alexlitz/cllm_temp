"""F-7: verify_rule_scopes tests."""
import pytest
from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule, FFNOp
from neural_vm.unified_compiler.decl_verifier import verify_rule_scopes


class _FakeOp:
    """Minimal stand-in for Operation."""
    def __init__(self, rules):
        self.compiler_ir = FFNOp(rules=list(rules))


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP", semantics="mark == SP")
    r.alloc("CMP_JSR", 1, 1, "CMP+4", semantics="opcode_at_AX == JSR")
    r.alloc("IN_STEP_FRESH", 2, 1, "fresh", semantics="in_step_fresh")
    r.alloc("HAS_SE", 3, 1, "se", semantics="has_se")
    r.alloc("OUT", 4, 1, "out", semantics="mark == AX")
    return r


def test_rule_without_scope_is_skipped(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="no_scope",
    )
    issues = verify_rule_scopes(_FakeOp([rule]), reg)
    assert issues == []


def test_rule_with_matching_scope_passes(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="matching",
        scope="mark == SP",
    )
    issues = verify_rule_scopes(_FakeOp([rule]), reg)
    assert issues == [], f"unexpected issues: {issues}"


def test_rule_with_overly_tight_scope_fails(reg):
    """The E5 case: rule's conditions admit ANY JSR row, but its scope
    claims step-0 only. Verifier rejects."""
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0), ("CMP_JSR", 0.5)),
        threshold=10.04,
        writes=(("OUT", 1.0),),
        name="e5_case",
        scope="mark == SP AND opcode_at_AX == JSR AND step_index == 0",
    )
    issues = verify_rule_scopes(_FakeOp([rule]), reg)
    assert any(i['kind'] == 'scope_violation' for i in issues), (
        f"expected scope_violation, got {issues}"
    )


def test_rule_with_loose_scope_passes(reg):
    """Effective predicate is a SUBSET of scope — that's fine
    (effective ⊨ scope)."""
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0), ("HAS_SE", -1e9)),  # mark==SP AND NOT has_se
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="loose",
        scope="mark == SP",  # superset of effective — fine
    )
    issues = verify_rule_scopes(_FakeOp([rule]), reg)
    assert issues == [], f"unexpected: {issues}"


def test_require_scope_flag_flags_missing(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="missing",
    )
    issues = verify_rule_scopes(_FakeOp([rule]), reg, require_scope=True)
    assert any(i['kind'] == 'no_scope' for i in issues)


def test_unparseable_scope_reported(reg):
    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT", 1.0),),
        name="bad_scope",
        scope="this is not valid DSL syntax!!!",
    )
    issues = verify_rule_scopes(_FakeOp([rule]), reg)
    assert any(i['kind'] == 'scope_parse_error' for i in issues)
