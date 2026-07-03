"""S-6: strength verifier tests."""
import pytest
from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule, FFNOp
from neural_vm.verification.decl_verifier import verify_rule_strength


class _FakeOp:
    def __init__(self, name, rules):
        self.name = name
        self.compiler_ir = FFNOp(rules=list(rules))


@pytest.fixture
def reg():
    r = DimRegistry(d_model=64)
    r.alloc("MARK_SP", 0, 1, "SP", semantics="mark == SP")
    r.alloc("MARK_AX", 1, 1, "AX", semantics="mark == AX")
    r.alloc("OUT_LO", 16, 16, "out lo", semantics="is_byte OR NOT is_byte")
    r.alloc("OUT_HI", 32, 16, "out hi", semantics="is_byte OR NOT is_byte")
    return r


def test_rule_without_dominates_at_skipped(reg):
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="no_dom",
        # no dominates_at and no scope
    )
    issues = verify_rule_strength(_FakeOp("op", [r]), reg)
    assert issues == []


def test_solo_rule_passes_when_alone(reg):
    """Rule with dominates_at, no competitors → passes."""
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="solo",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )
    issues = verify_rule_strength(_FakeOp("op", [r]), reg)
    # my_contrib = 1 * (100-10) = 90; no competitors; backbone=0; margin=1
    # required = 0 + 0 + 1 = 1; 90 > 1 → passes
    assert issues == []


def test_overpowered_competitor_flagged(reg):
    """Rule's contribution is less than a competitor's. Should flag."""
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
        # no dominates_at — just a competitor
    )
    issues = verify_rule_strength(_FakeOp("op", [weak, strong]), reg)

    # weak contrib: 1 * (20-10) = 10
    # strong contrib: 1 * (1000-10) = 990
    # weak claims dominance but only contributes 10 vs 990
    sv = [i for i in issues if i['kind'] == 'strength_violation']
    assert len(sv) == 1
    assert sv[0]['rule'] == 'weak'
    assert sv[0]['my_contribution'] < sv[0]['competing_max']


def test_non_overlapping_scopes_no_competition(reg):
    """Two rules at same dim but DIFFERENT scopes → no competition."""
    sp_writer = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="sp",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )
    ax_writer = FFNRule.constant_write(
        conditions=(("MARK_AX", 1000.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="ax",
        scope="mark == AX",
        # competes only at mark==AX, which doesn't overlap sp's dominates_at
    )
    issues = verify_rule_strength(_FakeOp("op", [sp_writer, ax_writer]), reg)
    # No strength_violation because effective scope of ax_writer is mark==AX
    # which doesn't overlap mark==SP
    sv = [i for i in issues if i['kind'] == 'strength_violation']
    assert sv == []


def test_backbone_bound_can_cause_violation(reg):
    """Rule alone but backbone bound > rule's contribution."""
    r = FFNRule.constant_write(
        conditions=(("MARK_SP", 20.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="r",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )

    def bounds(output_dim, offset, scope_str):
        return 50.0  # backbone produces +50

    # my_contrib = 10; required = 0 + 50 + 1 = 51; 10 < 51 → violation
    issues = verify_rule_strength(_FakeOp("op", [r]), reg, backbone_bounds=bounds)
    sv = [i for i in issues if i['kind'] == 'strength_violation']
    assert len(sv) == 1
    assert sv[0]['backbone_max'] == 50.0


def test_e5_class_strength_violation(reg):
    """The if_var case: rule has correct scope but loses to attention writer."""
    override = FFNRule.constant_write(
        conditions=(("MARK_SP", 50.0),),
        threshold=10.0,
        writes=(("OUT_LO+15", 0.5),),  # weak override
        name="weak_override",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )
    attn_proxy = FFNRule.constant_write(
        conditions=(("MARK_SP", 800.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),  # wrong byte at LO+0
        name="attn_writer",
        scope="mark == SP",
    )
    # override writes OUT_LO+15 (correct nibble); attn_writer writes
    # OUT_LO+0 (wrong nibble). But both are OUT_LO — same dim_name
    # different offsets — NOT competing in our V1 (we key by name+offset).
    issues = verify_rule_strength(_FakeOp("op", [override, attn_proxy]), reg)

    # Override at LO+15 vs no-one at LO+15 → passes
    sv = [i for i in issues if i['kind'] == 'strength_violation']
    assert sv == [], f"unexpected violations: {sv}"


def test_cross_op_competition(reg):
    """Rule in op1 competes against rule in op2 -- should flag if op2's
    rule is stronger and overlaps op1's dominance scope."""
    weak = FFNRule.constant_write(
        conditions=(("MARK_SP", 20.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="weak_in_op1",
        scope="mark == SP",
        dominates_at={"OUT_LO": "mark == SP"},
    )
    strong = FFNRule.constant_write(
        conditions=(("MARK_SP", 1000.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 1.0),),
        name="strong_in_op2",
        scope="mark == SP",
    )
    op1 = _FakeOp("op1", [weak])
    op2 = _FakeOp("op2", [strong])

    # Solo verification -- weak passes (no in-op competitors)
    issues = verify_rule_strength(op1, reg)
    sv = [i for i in issues if i['kind'] == 'strength_violation']
    assert sv == [], "expected no violations in solo verification"

    # Cross-op verification -- weak should be flagged
    issues = verify_rule_strength(op1, reg, ops_for_competition=[op2])
    sv = [i for i in issues if i['kind'] == 'strength_violation']
    assert len(sv) == 1
    assert sv[0]['rule'] == 'weak_in_op1'
    assert sv[0]['top_competitor'] == 'strong_in_op2'
