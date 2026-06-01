"""S-2: tests for contribution_algebra."""
import pytest
from neural_vm.unified_compiler.ir import FFNRule
from neural_vm.unified_compiler.contribution_algebra import (
    max_contribution,
    signed_contribution_bound,
    write_dims,
    is_dominant_writer,
)


def _rule(name, conds, threshold, writes):
    return FFNRule.constant_write(
        name=name,
        conditions=conds,
        threshold=threshold,
        writes=writes,
    )


def test_single_positive_simple():
    r = _rule(
        "simple",
        conds=(("A", 10.0),),
        threshold=5.0,
        writes=(("OUT", 100.0),),
    )
    # max_activation = max(0, 10 - 5) = 5; contribution = 100 * 5 = 500
    assert max_contribution(r, "OUT") == 500.0


def test_subthreshold_yields_zero():
    r = _rule(
        "weak",
        conds=(("A", 3.0),),
        threshold=10.0,
        writes=(("OUT", 100.0),),
    )
    # positive_sum=3, threshold=10 -> max_act = max(0, -7) = 0
    assert max_contribution(r, "OUT") == 0.0


def test_negatives_ignored_in_positive_sum():
    r = _rule(
        "with_negative",
        conds=(("A", 10.0), ("B", -5.0)),
        threshold=3.0,
        writes=(("OUT", 1.0),),
    )
    # positive_sum=10, max_act = 10 - 3 = 7
    assert max_contribution(r, "OUT") == 7.0


def test_unwritten_dim_returns_zero():
    r = _rule(
        "r",
        conds=(("A", 10.0),),
        threshold=5.0,
        writes=(("OUT", 100.0),),
    )
    assert max_contribution(r, "OTHER_DIM") == 0.0


def test_write_dims_lists_writes():
    r = _rule(
        "r",
        conds=(("A", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO", 50.0), ("OUT_HI+3", 30.0)),
    )
    wd = write_dims(r)
    assert ("OUT_LO", 0, 50.0) in wd
    assert ("OUT_HI", 3, 30.0) in wd


def test_dominance_simple():
    strong = _rule("strong", conds=(("A", 100.0),), threshold=10.0, writes=(("OUT", 1.0),))
    weak = _rule("weak", conds=(("A", 20.0),), threshold=10.0, writes=(("OUT", 1.0),))
    # strong: max_act = 90, contribution = 90
    # weak: max_act = 10, contribution = 10
    dominant, margin = is_dominant_writer(strong, "OUT", [weak])
    assert dominant
    assert margin > 70


def test_dominance_fails_when_weaker():
    strong_other = _rule("strong_other", conds=(("A", 100.0),), threshold=10.0, writes=(("OUT", 1.0),))
    me = _rule("me", conds=(("A", 20.0),), threshold=10.0, writes=(("OUT", 1.0),))
    dominant, margin = is_dominant_writer(me, "OUT", [strong_other])
    assert not dominant


def test_dominance_with_backbone():
    me = _rule("me", conds=(("A", 100.0),), threshold=10.0, writes=(("OUT", 1.0),))
    # contribution = 90; backbone bound = 50 -> dominant if 90 >= 50 + 1
    dominant, _ = is_dominant_writer(me, "OUT", [], backbone_bound=50.0, margin=1.0)
    assert dominant
    # contribution = 90; backbone bound = 100 -> not dominant
    dominant, _ = is_dominant_writer(me, "OUT", [], backbone_bound=100.0, margin=1.0)
    assert not dominant


def test_suppressor_dominates_weaker_suppressor():
    """Two suppressors at same dim: stronger (more negative) suppressor dominates."""
    strong_supp = _rule(
        "strong_supp",
        conds=(("A", 100.0),),
        threshold=10.0,
        writes=(("OUT", -1000.0),),
    )
    weak_supp = _rule(
        "weak_supp",
        conds=(("A", 20.0),),
        threshold=10.0,
        writes=(("OUT", -500.0),),
    )
    # strong: magnitude 1000 * 90 = 90000
    # weak: magnitude 500 * 10 = 5000
    dominant, _ = is_dominant_writer(strong_supp, "OUT", [weak_supp])
    assert dominant
    dominant, _ = is_dominant_writer(weak_supp, "OUT", [strong_supp])
    assert not dominant


def test_suppressor_doesnt_compete_with_override():
    """A negative suppressor and a positive override don't compete —
    they have different goals."""
    override = _rule(
        "override",
        conds=(("A", 100.0),),
        threshold=10.0,
        writes=(("OUT", 1.0),),
    )
    supp = _rule(
        "supp",
        conds=(("A", 100.0),),
        threshold=10.0,
        writes=(("OUT", -1000.0),),
    )
    # override doesn't compete with the suppressor
    dominant, _ = is_dominant_writer(override, "OUT", [supp])
    assert dominant  # solo positive
    dominant, _ = is_dominant_writer(supp, "OUT", [override])
    assert dominant  # solo negative


def test_signed_contribution_bound_positive_and_negative():
    pos = _rule(
        "pos",
        conds=(("A", 100.0),),
        threshold=10.0,
        writes=(("OUT", 2.0),),
    )
    neg = _rule(
        "neg",
        conds=(("A", 100.0),),
        threshold=10.0,
        writes=(("OUT", -3.0),),
    )
    mag, sign = signed_contribution_bound(pos, "OUT")
    assert mag == 180.0
    assert sign == "positive"
    mag, sign = signed_contribution_bound(neg, "OUT")
    assert mag == 270.0
    assert sign == "negative"


def test_e5_style_case():
    """A rule with weak positive evidence vs a much-stronger competing rule."""
    weak = _rule(
        "weak_override",
        conds=(("MARK_BP", 10.0), ("H1+3", 0.01)),
        threshold=10.0,
        writes=(("OUTPUT_LO+15", 0.5),),
    )
    strong_attn = _rule(
        "attention_writer",
        conds=(("ANY", 800.0),),
        threshold=10.0,
        writes=(("OUTPUT_LO+0", 1.0),),
    )
    # Note: they write to DIFFERENT offsets within OUTPUT_LO; check
    # what max_contribution returns when offset specified:
    weak_c = max_contribution(weak, "OUTPUT_LO", output_offset=15)
    strong_c = max_contribution(strong_attn, "OUTPUT_LO", output_offset=0)
    assert weak_c < strong_c  # qualitative: weak << strong
