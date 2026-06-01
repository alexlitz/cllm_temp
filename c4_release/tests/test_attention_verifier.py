"""V1 tests for ``attention_verifier.verify_attention_head``.

Mirrors the shape of ``test_strength_verifier.py``: tests build small
synthetic ``AttentionHeadIR``s and pass them through the verifier against
a tiny ``DimRegistry`` to assert the issue-list shape.

These are smoke tests for the verifier scaffolding, not the underlying
heads. See the module docstring for V1 vs V2 caveats.
"""
import warnings

import pytest

# Silence the dim_registry deprecation about semantics='' in fixtures.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.attention_verifier import (
    build_attention_writer_index,
    effective_attention_scope,
    head_write_magnitude,
    verify_attention_head,
)
from neural_vm.unified_compiler.ir import (
    AttentionHeadIR,
    AttentionOp,
    CompilerIR,
    FFNOp,
    FFNRule,
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
    # Marker / condition dims (used by Q and K projections in the
    # synthetic heads).
    r.alloc("MARK_SP", 0, 1, "mark == SP", semantics="mark == SP")
    r.alloc("MARK_AX", 1, 1, "mark == AX", semantics="mark == AX")
    r.alloc("CONST", 2, 1, "const 1", semantics="is_byte OR NOT is_byte")
    r.alloc("IS_MARK", 3, 1, "any mark", semantics="is_byte OR NOT is_byte")
    # Source dims for V projections.
    r.alloc("EMBED_LO", 16, 16, "embed lo", semantics="is_byte OR NOT is_byte")
    r.alloc("EMBED_HI", 32, 16, "embed hi", semantics="is_byte OR NOT is_byte")
    # Output dims for O projections.
    r.alloc("OUT_LO", 48, 16, "output lo", semantics="is_byte OR NOT is_byte")
    r.alloc("OUT_HI", 64, 16, "output hi", semantics="is_byte OR NOT is_byte")
    r.alloc("SP_BYTE0_IS_F8", 80, 1, "sp byte0 == 0xF8",
            semantics="is_byte OR NOT is_byte")
    return r


class _FakeOp:
    """An Operation stub carrying an ``AttentionOp`` as its compiler_ir."""

    def __init__(self, name: str, heads):
        self.name = name
        attn = AttentionOp()
        for head in heads:
            if isinstance(head, AttentionHeadIR):
                attn.rules.append(head)
            else:
                attn.add_head(head)
        self.compiler_ir = attn


class _FakeFFNOp:
    """An Operation stub carrying an ``FFNOp`` for cross-modality tests."""

    def __init__(self, name: str, rules):
        self.name = name
        self.compiler_ir = FFNOp(rules=list(rules))


def _strong_head(reg, *, name, head_idx, out_name, slot=1,
                 q_dim="CONST", k_dim="MARK_SP",
                 v_dim="EMBED_LO", k_weight=10.0, v_weight=1.0,
                 o_weight=1.0, scope=None) -> AttentionHeadIR:
    """Build a tiny declarative head: Q[CONST], K[k_dim], V[v_dim] -> O[out_name+0]."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(AP(0, reg.slots[q_dim].start, 10.0),),
        k=(AP(0, reg.slots[k_dim].start, k_weight),),
        v=(AP(slot, reg.slots[v_dim].start, v_weight),),
        o=(AO(reg.slots[out_name].start, slot, o_weight),),
    )
    metadata = {"scope": scope} if scope is not None else {}
    return AttentionHeadIR(spec=spec, name=name, metadata=metadata)


# ---------------------------------------------------------------------------
# Test cases (V1)
# ---------------------------------------------------------------------------


def test_solo_head_passes(reg):
    """Head alone, no competition -> verifier emits no strength issues."""
    head = _strong_head(reg, name="solo", head_idx=0, out_name="OUT_LO")
    issues = verify_attention_head(
        head, reg, ops_for_competition=[_FakeOp("solo_op", [head])],
    )
    # The solo head has magnitude > 0 and no competitor -> no strength
    # violations. Empty value path / unresolved should also be clean.
    assert not any(
        i["kind"]
        in (
            "attention_strength_violation",
            "cross_modality_strength_violation",
            "empty_value_path",
            "unresolved_output_dim",
        )
        for i in issues
    ), f"unexpected issues: {issues}"


def test_overpowered_competitor_flagged(reg):
    """Weak head loses to a strong attention competitor at the same dim."""
    weak = _strong_head(
        reg, name="weak", head_idx=0, out_name="OUT_LO",
        v_weight=0.1, o_weight=0.1,
    )
    strong = _strong_head(
        reg, name="strong", head_idx=1, out_name="OUT_LO",
        v_weight=10.0, o_weight=1.0,
    )
    other_op = _FakeOp("strong_op", [strong])
    issues = verify_attention_head(
        weak, reg, ops_for_competition=[other_op],
    )
    sv = [i for i in issues if i["kind"] == "attention_strength_violation"]
    assert len(sv) == 1
    assert sv[0]["head"] == "weak"
    assert sv[0]["top_competitor"] == "strong"
    assert sv[0]["my_magnitude"] < sv[0]["competing_max"]


def test_non_overlapping_scopes_attn_index_separate(reg):
    """Two heads at the same dim but different K-side dims => different
    effective scopes; both surface in the index but each is its own
    competitor only by raw magnitude (V1 conservative bound)."""
    sp_writer = _strong_head(
        reg, name="sp", head_idx=0, out_name="OUT_LO",
        k_dim="MARK_SP",
    )
    ax_writer = _strong_head(
        reg, name="ax", head_idx=1, out_name="OUT_LO",
        k_dim="MARK_AX",
    )
    # effective_attention_scope is purely K-side dim names.
    assert effective_attention_scope(sp_writer, reg) == ("MARK_SP",)
    assert effective_attention_scope(ax_writer, reg) == ("MARK_AX",)
    # Build the writer index and confirm both heads land at OUT_LO+0.
    index = build_attention_writer_index(
        [_FakeOp("ops", [sp_writer, ax_writer])], reg,
    )
    entries = index.get(("OUT_LO", 0), [])
    names = sorted(e.head_name for e in entries)
    assert names == ["ax", "sp"]


def test_ffn_cross_modality_competitor_flagged(reg):
    """Attention head loses to an FFN rule at the same output dim."""
    head = _strong_head(
        reg, name="weak_attn", head_idx=0, out_name="OUT_LO",
        v_weight=0.1, o_weight=0.1,
    )
    strong_ffn = FFNRule.constant_write(
        conditions=(("MARK_SP", 100.0),),
        threshold=10.0,
        writes=(("OUT_LO+0", 5.0),),
        name="strong_ffn_writer",
        scope="mark == SP",
    )
    ffn_op = _FakeFFNOp("ffn_op", [strong_ffn])
    issues = verify_attention_head(
        head, reg, ops_for_competition=[ffn_op],
    )
    cv = [
        i for i in issues
        if i["kind"] == "cross_modality_strength_violation"
    ]
    assert len(cv) == 1
    assert cv[0]["top_competitor_kind"] == "ffn_rule"
    assert cv[0]["my_magnitude"] < cv[0]["competing_max"]


def test_empty_value_path_flagged(reg):
    """O write references a slot the V projection never feeds."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, reg.slots["CONST"].start, 10.0),),
        k=(AP(0, reg.slots["MARK_SP"].start, 10.0),),
        v=(AP(1, reg.slots["EMBED_LO"].start, 1.0),),    # slot 1
        o=(AO(reg.slots["OUT_LO"].start, 99, 1.0),),     # slot 99: empty!
    )
    head = AttentionHeadIR(spec=spec, name="orphan_o")
    issues = verify_attention_head(head, reg)
    kinds = [i["kind"] for i in issues]
    assert "empty_value_path" in kinds


def test_require_scope_surfaces_missing_declaration(reg):
    """``require_scope=True`` flags heads with no ``metadata['scope']``."""
    head = _strong_head(
        reg, name="undeclared", head_idx=0, out_name="OUT_LO",
        # no scope=...
    )
    issues = verify_attention_head(head, reg, require_scope=True)
    assert any(i["kind"] == "no_scope_declared" for i in issues)


def test_unresolved_output_dim_flagged(reg):
    """O write targets an integer column outside every registry slot."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, reg.slots["CONST"].start, 10.0),),
        k=(AP(0, reg.slots["MARK_SP"].start, 10.0),),
        v=(AP(1, reg.slots["EMBED_LO"].start, 1.0),),
        o=(AO(1000, 1, 1.0),),    # dim 1000 -- well outside d_model=128.
    )
    head = AttentionHeadIR(spec=spec, name="oob")
    issues = verify_attention_head(head, reg)
    assert any(i["kind"] == "unresolved_output_dim" for i in issues)


def test_head_write_magnitude_v1_bound(reg):
    """V1 magnitude = |o_weight| * sum(|v_weight| at matching slot)."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, reg.slots["CONST"].start, 10.0),),
        k=(AP(0, reg.slots["MARK_SP"].start, 10.0),),
        v=(
            AP(1, reg.slots["EMBED_LO"].start, 0.5),
            AP(1, reg.slots["EMBED_HI"].start, 0.5),
        ),
        o=(AO(reg.slots["OUT_LO"].start, 1, 2.0),),
    )
    head = AttentionHeadIR(spec=spec, name="bound")
    # V1 bound: |2.0| * (|0.5| + |0.5|) = 2.0
    assert head_write_magnitude(head, "OUT_LO", 0, reg) == pytest.approx(2.0)
    # Other dim with no O write -> 0.
    assert head_write_magnitude(head, "OUT_HI", 0, reg) == 0.0
