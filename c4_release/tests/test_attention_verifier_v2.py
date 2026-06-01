"""V2 tests for ``attention_verifier.verify_attention_head``.

Covers the Q-side scope-overlap competition filter added in V2:

* Two heads with disjoint Q scopes: V2 must not flag an
  ``attention_strength_violation`` (since the heads can never fire at
  the same query position), even when their O magnitudes overlap and V1
  would flag.
* Two heads with overlapping Q scopes: V1-style strength check still
  fires (V2 is a strict refinement of V1, not a behaviour swap).
* Heads with no declared Q projection: V2 must degrade to V1 behaviour
  (treat as wildcard / always overlapping) so we never regress on
  legacy heads that have not declared a Q projection.
* The ``effective_attention_q_scope`` helper extracts positively-
  weighted Q dim names, identically to ``effective_attention_scope``
  but reading from the Q projection.
* The ``q_scope_filter=False`` kwarg restores V1 behaviour for callers
  who explicitly want the unfiltered V1 strength check.
* Entries returned by ``build_attention_writer_index`` carry the new
  ``q_effective_scope`` field.
"""
import warnings

import pytest

# Silence the dim_registry deprecation about semantics='' in fixtures.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.attention_verifier import (
    build_attention_writer_index,
    effective_attention_q_scope,
    effective_attention_scope,
    verify_attention_head,
)
from neural_vm.unified_compiler.ir import (
    AttentionHeadIR,
    AttentionOp,
)
from neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reg() -> DimRegistry:
    """Tiny registry with the dims used by the synthetic heads below."""
    r = DimRegistry(d_model=128)
    # Q-side markers
    r.alloc("Q_MARK_A", 0, 1, "q == A", semantics="is_byte OR NOT is_byte")
    r.alloc("Q_MARK_B", 1, 1, "q == B", semantics="is_byte OR NOT is_byte")
    r.alloc("Q_SHARED", 2, 1, "q shared", semantics="is_byte OR NOT is_byte")
    # K-side markers
    r.alloc("K_MARK_S", 3, 1, "k == S", semantics="is_byte OR NOT is_byte")
    r.alloc("K_MARK_T", 4, 1, "k == T", semantics="is_byte OR NOT is_byte")
    # Source dims for V projections.
    r.alloc("EMBED_LO", 16, 16, "embed lo", semantics="is_byte OR NOT is_byte")
    r.alloc("EMBED_HI", 32, 16, "embed hi", semantics="is_byte OR NOT is_byte")
    # Output dims for O projections (the contested residual column).
    r.alloc("OUT_LO", 48, 16, "output lo", semantics="is_byte OR NOT is_byte")
    return r


class _FakeOp:
    """An Operation stub carrying an ``AttentionOp`` as its compiler_ir."""

    def __init__(self, name: str, heads):
        self.name = name
        attn = AttentionOp()
        for head in heads:
            attn.rules.append(head)
        self.compiler_ir = attn


def _make_head(
    reg,
    *,
    name,
    head_idx,
    q_dim,
    k_dim,
    v_weight=1.0,
    o_weight=1.0,
    slot=1,
    out_name="OUT_LO",
) -> AttentionHeadIR:
    """Build a tiny head with a single Q-marker, K-marker, V and O write.

    Both heads in the V2 tests write OUT_LO+0 to ensure they collide in
    the writer index; Q dim controls Q-scope overlap.
    """
    spec = DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(AP(0, reg.slots[q_dim].start, 10.0),),
        k=(AP(0, reg.slots[k_dim].start, 10.0),),
        v=(AP(slot, reg.slots["EMBED_LO"].start, v_weight),),
        o=(AO(reg.slots[out_name].start, slot, o_weight),),
    )
    return AttentionHeadIR(spec=spec, name=name)


def _make_head_no_q(
    reg,
    *,
    name,
    head_idx,
    k_dim,
    v_weight=1.0,
    o_weight=1.0,
    slot=1,
    out_name="OUT_LO",
) -> AttentionHeadIR:
    """Build a head with an empty Q projection (V1-style legacy head)."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(),    # no Q writes -> empty Q scope -> V2 treats as wildcard
        k=(AP(0, reg.slots[k_dim].start, 10.0),),
        v=(AP(slot, reg.slots["EMBED_LO"].start, v_weight),),
        o=(AO(reg.slots[out_name].start, slot, o_weight),),
    )
    return AttentionHeadIR(spec=spec, name=name)


# ---------------------------------------------------------------------------
# effective_attention_q_scope unit checks
# ---------------------------------------------------------------------------


def test_effective_attention_q_scope_positive_dims(reg):
    """Positive Q writes contribute their dim names; sorted/deduped."""
    head = _make_head(reg, name="h", head_idx=0, q_dim="Q_MARK_A", k_dim="K_MARK_S")
    assert effective_attention_q_scope(head, reg) == ("Q_MARK_A",)


def test_effective_attention_q_scope_negative_weight_dropped(reg):
    """Negative Q writes are not counted as active query markers."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(
            AP(0, reg.slots["Q_MARK_A"].start, 10.0),
            AP(0, reg.slots["Q_MARK_B"].start, -10.0),
        ),
        k=(AP(0, reg.slots["K_MARK_S"].start, 10.0),),
        v=(AP(1, reg.slots["EMBED_LO"].start, 1.0),),
        o=(AO(reg.slots["OUT_LO"].start, 1, 1.0),),
    )
    head = AttentionHeadIR(spec=spec, name="signed")
    # Only the positively-weighted Q dim shows up.
    assert effective_attention_q_scope(head, reg) == ("Q_MARK_A",)


def test_effective_attention_q_scope_empty_when_no_q_writes(reg):
    """Heads with no Q projection report empty Q scope (wildcard)."""
    head = _make_head_no_q(reg, name="legacy", head_idx=0, k_dim="K_MARK_S")
    assert effective_attention_q_scope(head, reg) == ()


# ---------------------------------------------------------------------------
# Writer index carries q_effective_scope
# ---------------------------------------------------------------------------


def test_writer_index_records_q_scope(reg):
    """``AttentionWriterEntry.q_effective_scope`` is populated from the Q proj."""
    head_a = _make_head(
        reg, name="ha", head_idx=0, q_dim="Q_MARK_A", k_dim="K_MARK_S",
    )
    head_b = _make_head(
        reg, name="hb", head_idx=1, q_dim="Q_MARK_B", k_dim="K_MARK_T",
    )
    index = build_attention_writer_index(
        [_FakeOp("op", [head_a, head_b])], reg,
    )
    entries = index.get(("OUT_LO", 0), [])
    by_name = {e.head_name: e for e in entries}
    assert by_name["ha"].q_effective_scope == ("Q_MARK_A",)
    assert by_name["hb"].q_effective_scope == ("Q_MARK_B",)


# ---------------------------------------------------------------------------
# Q-overlap filter behaviour
# ---------------------------------------------------------------------------


def test_disjoint_q_scopes_suppresses_violation(reg):
    """V2: two heads with disjoint Q scopes don't flag strength violations.

    Without the V2 filter, the weak head would lose an attention_strength
    check to the strong head writing the same residual dim. With V2, the
    Q scopes don't overlap (Q_MARK_A vs Q_MARK_B), so the heads can't
    fire at the same query position and the violation is suppressed.

    Note: the "weak" head still needs my_magnitude > margin so the
    no-competitor case (V2 path) doesn't flag a self-margin violation.
    """
    weak = _make_head(
        reg, name="weak_a", head_idx=0,
        q_dim="Q_MARK_A", k_dim="K_MARK_S",
        # large enough to clear the margin alone, but a fraction of strong
        v_weight=2.0, o_weight=2.0,
    )
    strong = _make_head(
        reg, name="strong_b", head_idx=1,
        q_dim="Q_MARK_B", k_dim="K_MARK_T",
        v_weight=10.0, o_weight=10.0,
    )
    other_op = _FakeOp("strong_op", [strong])

    issues_v2 = verify_attention_head(
        weak, reg, ops_for_competition=[other_op],
    )
    sv_v2 = [i for i in issues_v2 if i["kind"] == "attention_strength_violation"]
    assert sv_v2 == [], (
        "V2 must not flag disjoint-Q-scope heads as competitors; "
        f"got {sv_v2}"
    )

    # Sanity: with q_scope_filter=False, the V1 behaviour is restored
    # and the strength violation reappears, confirming the suppression
    # was solely due to the V2 Q-overlap filter (not e.g. a magnitude
    # bug introduced by the V2 refactor).
    issues_v1 = verify_attention_head(
        weak, reg, ops_for_competition=[other_op], q_scope_filter=False,
    )
    sv_v1 = [i for i in issues_v1 if i["kind"] == "attention_strength_violation"]
    assert len(sv_v1) == 1
    assert sv_v1[0]["top_competitor"] == "strong_b"


def test_overlapping_q_scopes_still_flag_violation(reg):
    """V2: heads sharing a Q dim still get the V1-style strength check.

    V2 is a strict refinement of V1: it removes only the false positives
    where Q scopes are provably disjoint. When two heads have overlapping
    Q scopes (here, both pos-write Q_SHARED), the strength check fires
    exactly as in V1.
    """
    weak = _make_head(
        reg, name="weak_shared", head_idx=0,
        q_dim="Q_SHARED", k_dim="K_MARK_S",
        v_weight=0.1, o_weight=0.1,
    )
    strong = _make_head(
        reg, name="strong_shared", head_idx=1,
        q_dim="Q_SHARED", k_dim="K_MARK_T",
        v_weight=10.0, o_weight=10.0,
    )
    other_op = _FakeOp("strong_op", [strong])

    issues = verify_attention_head(
        weak, reg, ops_for_competition=[other_op],
    )
    sv = [i for i in issues if i["kind"] == "attention_strength_violation"]
    assert len(sv) == 1
    assert sv[0]["top_competitor"] == "strong_shared"
    # The V2 issue carries q-scope context for triage.
    assert sv[0]["my_q_scope"] == ["Q_SHARED"]
    assert sv[0]["competitor_q_scope"] == ["Q_SHARED"]


def test_partial_q_scope_overlap_flags_violation(reg):
    """V2: scopes overlap as soon as they share *any* Q dim.

    Weak head queries from {Q_MARK_A, Q_SHARED}; strong head queries from
    {Q_MARK_B, Q_SHARED}. Their intersection is {Q_SHARED} so the heads
    can fire at the same query position and competition is real.
    """
    weak_spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(
            AP(0, reg.slots["Q_MARK_A"].start, 10.0),
            AP(0, reg.slots["Q_SHARED"].start, 10.0),
        ),
        k=(AP(0, reg.slots["K_MARK_S"].start, 10.0),),
        v=(AP(1, reg.slots["EMBED_LO"].start, 0.1),),
        o=(AO(reg.slots["OUT_LO"].start, 1, 0.1),),
    )
    strong_spec = DeclarativeAttentionHeadSpec(
        head_idx=1,
        q=(
            AP(0, reg.slots["Q_MARK_B"].start, 10.0),
            AP(0, reg.slots["Q_SHARED"].start, 10.0),
        ),
        k=(AP(0, reg.slots["K_MARK_T"].start, 10.0),),
        v=(AP(1, reg.slots["EMBED_LO"].start, 10.0),),
        o=(AO(reg.slots["OUT_LO"].start, 1, 10.0),),
    )
    weak = AttentionHeadIR(spec=weak_spec, name="weak_partial")
    strong = AttentionHeadIR(spec=strong_spec, name="strong_partial")

    issues = verify_attention_head(
        weak, reg, ops_for_competition=[_FakeOp("op", [strong])],
    )
    sv = [i for i in issues if i["kind"] == "attention_strength_violation"]
    assert len(sv) == 1
    assert sv[0]["top_competitor"] == "strong_partial"


# ---------------------------------------------------------------------------
# Empty-Q-scope wildcard / V1 fallback
# ---------------------------------------------------------------------------


def test_legacy_head_with_no_q_scope_falls_back_to_v1(reg):
    """V2: heads with empty Q scope are treated as wildcards.

    A head with no declared Q projection (legacy / threshold style) has
    no Q-side filter information. V2 conservatively treats this as
    "could fire at any query position" so the V1 strength check still
    fires, preserving V1 behaviour for legacy heads.
    """
    weak = _make_head_no_q(
        reg, name="legacy_weak", head_idx=0, k_dim="K_MARK_S",
        v_weight=0.1, o_weight=0.1,
    )
    strong = _make_head_no_q(
        reg, name="legacy_strong", head_idx=1, k_dim="K_MARK_T",
        v_weight=10.0, o_weight=10.0,
    )
    issues = verify_attention_head(
        weak, reg, ops_for_competition=[_FakeOp("op", [strong])],
    )
    sv = [i for i in issues if i["kind"] == "attention_strength_violation"]
    assert len(sv) == 1
    assert sv[0]["top_competitor"] == "legacy_strong"


def test_mixed_q_scope_one_side_empty_treated_as_overlap(reg):
    """V2: a head with empty Q scope vs a head with a declared Q scope
    is treated as overlapping (the empty side is a wildcard).

    This protects against false-negative suppression when one head has
    not yet declared its Q projection but the competitor has.
    """
    weak_legacy = _make_head_no_q(
        reg, name="weak_legacy", head_idx=0, k_dim="K_MARK_S",
        v_weight=0.1, o_weight=0.1,
    )
    strong_declared = _make_head(
        reg, name="strong_declared", head_idx=1,
        q_dim="Q_MARK_A", k_dim="K_MARK_T",
        v_weight=10.0, o_weight=10.0,
    )
    issues = verify_attention_head(
        weak_legacy, reg,
        ops_for_competition=[_FakeOp("op", [strong_declared])],
    )
    sv = [i for i in issues if i["kind"] == "attention_strength_violation"]
    assert len(sv) == 1
    assert sv[0]["top_competitor"] == "strong_declared"


# ---------------------------------------------------------------------------
# u32 invariant: the V2 issue dict has no new floats that violate u32
# ---------------------------------------------------------------------------


def test_v2_issue_fields_serializable(reg):
    """V2 issues are simple dict[str, Any] entries — no NumPy/u32 leaks."""
    weak = _make_head(
        reg, name="weak", head_idx=0, q_dim="Q_SHARED", k_dim="K_MARK_S",
        v_weight=0.1, o_weight=0.1,
    )
    strong = _make_head(
        reg, name="strong", head_idx=1, q_dim="Q_SHARED", k_dim="K_MARK_T",
        v_weight=10.0, o_weight=10.0,
    )
    issues = verify_attention_head(
        weak, reg, ops_for_competition=[_FakeOp("op", [strong])],
    )
    for issue in issues:
        if issue["kind"] != "attention_strength_violation":
            continue
        # No numpy types should sneak in.
        for k, v in issue.items():
            assert type(v).__module__ in ("builtins",), (
                f"non-builtin type in issue field {k!r}: {type(v)}"
            )
