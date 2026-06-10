"""Tests for the AttentionIntent IR primitive.

``AttentionIntent`` is a verifier-friendly Q/K declaration that replaces
hand-tuned ``AP(slot, dim, weight)`` writes for the binary-address-match
+ anchor pattern (L7/L13/L15 lookups). The framework synthesizes the
raw projection writes and the verifier runs synthetic Q/K positions to
assert the intended K row dominates every plausible competitor.

These tests cover:

1. Basic synthesis: an intent without binary-match fields produces a
   well-formed ``DeclarativeAttentionHeadSpec`` and verifies cleanly.
2. Binary-match synthesis: bit count and Q/K dim mismatches raise
   ``ValueError`` at synthesis time.
3. K-selection bug detection: a deliberately mis-anchored intent
   surfaces verifier errors that name the failing competitor.
4. Negative anchors: a K predicate with negative anchors causes the
   verifier to also test a K row that hits the negative anchors.
5. POC migration: the L15 ``memory_lookup`` head 0 binary-address
   block (slots 4..27) synthesized via ``AttentionIntent`` is
   byte-identical with the existing imperative writes.
"""

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from neural_vm.unified_compiler.primitives import (  # noqa: E402
    AP,
    AO,
    AttentionIntent,
    DeclarativeAttentionHeadSpec,
    PositionPredicate,
    Primitives,
    verify_attention_intent,
)
from neural_vm.vm_step import _SetDim as BD  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Basic synthesis
# ---------------------------------------------------------------------------


def test_attention_intent_synthesizes_anchor_only_head_and_verifies():
    """An intent without binary-match fields synthesizes a well-formed
    spec where the Q-anchor block at slot 0 carries CONST + positive
    anchors, the K-anchor block at slot 1 carries CONST + positive
    anchors, and the verifier reports zero errors."""

    intent = AttentionIntent(
        name="anchor_only",
        q_at=PositionPredicate(
            label="Q at AX marker with LI",
            positive_anchors=(BD.MARK_AX, BD.OP_LI_RELAY),
        ),
        k_at=PositionPredicate(
            label="K at MEM store row",
            positive_anchors=(BD.MEM_STORE,),
        ),
        v_pulls=tuple(
            (BD.CLEAN_EMBED_LO + k, 32 + k) for k in range(16)
        ),
        o_writes_to=tuple(
            (BD.OUTPUT_LO + k, 32 + k, 1.0) for k in range(16)
        ),
        head_idx=3,
    )

    spec = intent.to_spec()

    assert isinstance(spec, DeclarativeAttentionHeadSpec)
    assert spec.head_idx == 3

    # Q slot 0 anchor block: CONST + 2 positive anchors.
    q_slot0 = sorted([w for w in spec.q if w.slot == 0], key=lambda w: w.dim)
    dims = [w.dim for w in q_slot0]
    assert BD.CONST in dims
    assert BD.MARK_AX in dims
    assert BD.OP_LI_RELAY in dims

    # K slot 1 anchor block: CONST + MEM_STORE.
    k_slot1 = sorted([w for w in spec.k if w.slot == 1], key=lambda w: w.dim)
    dims = [w.dim for w in k_slot1]
    assert BD.MEM_STORE in dims
    assert BD.CONST in dims

    # V/O writes mirror declared mappings.
    assert len(spec.v) == 16
    assert len(spec.o) == 16

    errors = verify_attention_intent(intent)
    assert errors == [], f"Expected zero errors, got: {errors}"


def test_attention_intent_v_pulls_and_o_writes_to_pass_through():
    """V slot and O slot mappings flow from intent to spec unchanged."""

    intent = AttentionIntent(
        name="vo_passthrough",
        q_at=PositionPredicate(
            label="Q", positive_anchors=(BD.MARK_AX,),
        ),
        k_at=PositionPredicate(
            label="K", positive_anchors=(BD.MEM_STORE,),
        ),
        v_pulls=((BD.CLEAN_EMBED_LO, 40), (BD.CLEAN_EMBED_HI, 41)),
        o_writes_to=(
            (BD.OUTPUT_LO, 40, 1.5),
            (BD.OUTPUT_HI, 41, 2.0),
        ),
    )

    spec = intent.to_spec()

    v_by_dim = {w.dim: (w.slot, w.weight) for w in spec.v}
    assert v_by_dim[BD.CLEAN_EMBED_LO] == (40, 1.0)
    assert v_by_dim[BD.CLEAN_EMBED_HI] == (41, 1.0)

    o_by_out = {w.out_dim: (w.slot, w.weight) for w in spec.o}
    assert o_by_out[BD.OUTPUT_LO] == (40, 1.5)
    assert o_by_out[BD.OUTPUT_HI] == (41, 2.0)


# ---------------------------------------------------------------------------
# 2. Binary-match synthesis / validation
# ---------------------------------------------------------------------------


def test_attention_intent_binary_match_field_count_mismatch_raises():
    """Q and K predicates must declare the same number of binary-match
    fields. Mismatch raises ``ValueError`` at synthesis time."""

    intent = AttentionIntent(
        name="bad_field_count",
        q_at=PositionPredicate(
            label="Q",
            binary_match_fields=(
                ("a", (BD.ADDR_B0_LO,), (BD.ADDR_B0_LO,), 4),
            ),
        ),
        k_at=PositionPredicate(
            label="K",
            binary_match_fields=(),
        ),
    )

    with pytest.raises(ValueError, match="binary_match_fields"):
        intent.synthesize_writes()


def test_attention_intent_binary_match_bit_mismatch_raises():
    """Q and K binary-match bit counts must agree."""

    intent = AttentionIntent(
        name="bad_bits",
        q_at=PositionPredicate(
            label="Q",
            binary_match_fields=(
                ("a", (BD.ADDR_B0_LO,), (BD.ADDR_B0_LO,), 4),
            ),
        ),
        k_at=PositionPredicate(
            label="K",
            binary_match_fields=(
                ("a", (BD.ADDR_B0_LO,), (BD.ADDR_B0_LO,), 3),
            ),
        ),
    )

    with pytest.raises(ValueError, match="bit count"):
        intent.synthesize_writes()


# ---------------------------------------------------------------------------
# 3. K-selection bug detection
# ---------------------------------------------------------------------------


def test_verify_attention_intent_catches_missing_k_anchor():
    """An intent whose K predicate declares no anchors (or address
    match) is structurally ambiguous — the intended K row scores the
    same as a bare K row. The verifier surfaces this."""

    intent = AttentionIntent(
        name="missing_k_anchor",
        q_at=PositionPredicate(
            label="Q", positive_anchors=(BD.OP_LI_RELAY,),
        ),
        k_at=PositionPredicate(label="K"),
    )

    errors = verify_attention_intent(intent)
    assert errors, "Expected K-selection bug to be reported"
    joined = " ".join(errors)
    assert "K-selection bug" in joined
    assert "missing_k_anchor" in joined


def test_verify_attention_intent_passes_with_binary_address_match():
    """A well-formed binary-address-match intent (Q and K share the
    same 24-bit address field; K has a MEM_STORE anchor) verifies
    cleanly: the intended K row dominates wrong-address, no-anchor,
    and bare K candidates."""

    nibble_bases = (
        BD.ADDR_B0_LO, BD.ADDR_B0_HI,
        BD.ADDR_B1_LO, BD.ADDR_B1_HI,
        BD.ADDR_B2_LO, BD.ADDR_B2_HI,
    )
    intent = AttentionIntent(
        name="li_to_mem_val_b0",
        q_at=PositionPredicate(
            label="LI at AX marker",
            positive_anchors=(BD.OP_LI_RELAY, BD.MARK_AX),
            binary_match_fields=(
                ("addr", nibble_bases, nibble_bases, 4),
            ),
        ),
        k_at=PositionPredicate(
            label="MEM_STORE at val byte 0",
            positive_anchors=(BD.MEM_STORE, BD.L2H0 + 4),
            binary_match_fields=(
                ("addr", nibble_bases, nibble_bases, 4),
            ),
        ),
        v_pulls=tuple(
            [(BD.CLEAN_EMBED_LO + k, 32 + k) for k in range(16)]
            + [(BD.CLEAN_EMBED_HI + k, 48 + k) for k in range(16)]
        ),
        o_writes_to=tuple(
            [(BD.OUTPUT_LO + k, 32 + k, 1.0) for k in range(16)]
            + [(BD.OUTPUT_HI + k, 48 + k, 1.0) for k in range(16)]
        ),
        head_idx=0,
    )

    errors = verify_attention_intent(intent)
    assert errors == [], (
        f"Expected the well-formed intent to verify cleanly, got: {errors}"
    )


# ---------------------------------------------------------------------------
# 4. Negative anchors
# ---------------------------------------------------------------------------


def test_attention_intent_negative_anchor_kills_competitor_row():
    """When the K predicate carries a negative anchor (e.g. H1+MEM_I —
    "must not be on a register byte-1 position"), the verifier checks
    a K row that hits the negative anchor and asserts it loses to the
    intended row."""

    intent = AttentionIntent(
        name="neg_anchor_check",
        q_at=PositionPredicate(
            label="Q", positive_anchors=(BD.OP_LI_RELAY,),
        ),
        k_at=PositionPredicate(
            label="K",
            positive_anchors=(BD.MEM_STORE,),
            negative_anchors=(BD.H1 + 4,),  # H1+MEM_I
        ),
    )

    errors = verify_attention_intent(intent)
    assert errors == [], (
        f"Negative anchor should kill the competitor row, got: {errors}"
    )


# ---------------------------------------------------------------------------
# 5. POC migration: L15 memory_lookup head 0 binary-address block
# ---------------------------------------------------------------------------


def _l15_head0_intent_binary_block() -> AttentionIntent:
    """L15 head 0 binary-address block expressed as an AttentionIntent.

    The intent's binary_match block covers slots 4..27 of the legacy
    spec; this lets us byte-identity-check the migration without
    reproducing the dozens of ad-hoc CONST/blocker writes that live in
    the non-binary slots.
    """

    nibble_bases = (
        BD.ADDR_B0_LO, BD.ADDR_B0_HI,
        BD.ADDR_B1_LO, BD.ADDR_B1_HI,
        BD.ADDR_B2_LO, BD.ADDR_B2_HI,
    )
    return AttentionIntent(
        name="layer15_memory_lookup.li_lc_stack0_h0.binary_block",
        # Anchors empty here so the synthesizer emits ONLY the
        # binary-match block (slots 4..27). The full L15 head spec
        # composes this block with hand-authored anchor/blocker writes
        # at slots 0..3, 28..33.
        q_at=PositionPredicate(
            label="Q address nibbles",
            binary_match_fields=(
                ("addr", nibble_bases, nibble_bases, 4),
            ),
        ),
        k_at=PositionPredicate(
            label="K address nibbles",
            binary_match_fields=(
                ("addr", nibble_bases, nibble_bases, 4),
            ),
        ),
        head_idx=0,
        bias_slot=0,
        match_slot_base=4,
        match_scale=10.0,
    )


def test_l15_head0_intent_binary_block_byte_identical_to_legacy_spec():
    """The binary-address block synthesized by ``AttentionIntent``
    must produce the same ``(slot, dim, weight)`` tuples at slots
    4..27 as the legacy imperative spec from
    ``_layer15_memory_lookup_heads_0_3_specs``."""

    from neural_vm.unified_compiler.ops.l15_ops import (
        _layer15_memory_lookup_heads_0_3_specs,
    )

    intent = _l15_head0_intent_binary_block()
    intent_spec = intent.to_spec()

    legacy = _layer15_memory_lookup_heads_0_3_specs(BD)[0]

    def block_writes(writes, lo=4, hi=27):
        return {
            (w.slot, w.dim): w.weight
            for w in writes
            if lo <= w.slot <= hi
        }

    intent_q = block_writes(intent_spec.q)
    intent_k = block_writes(intent_spec.k)
    legacy_q = block_writes(legacy.q)
    legacy_k = block_writes(legacy.k)

    assert intent_q == legacy_q, (
        f"Q binary block differs: intent has "
        f"{len(intent_q)} writes, legacy {len(legacy_q)}; "
        f"diff sample: "
        f"{set(intent_q.items()) ^ set(legacy_q.items())}"
    )
    assert intent_k == legacy_k, (
        f"K binary block differs: intent has "
        f"{len(intent_k)} writes, legacy {len(legacy_k)}"
    )
    # Sanity: the binary block should be 24 slots * 16 nibbles = 384
    # writes per side (4 bits * 2 nibbles * 3 address bytes * 16
    # nibble values).
    assert len(intent_q) == 24 * 16
    assert len(intent_k) == 24 * 16


def test_declarative_spec_intent_field_defaults_to_none_for_backcompat():
    """Existing call sites that construct ``DeclarativeAttentionHeadSpec``
    without ``intent=`` must continue to work unchanged."""

    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, BD.CONST, 1.0),),
        k=(AP(0, BD.CONST, 1.0),),
        v=(),
        o=(),
    )
    assert spec.intent is None


def test_generate_attention_head_runs_intent_check_at_bake_time():
    """When ``DeclarativeAttentionHeadSpec.intent`` is set, the bake
    helper runs ``verify_attention_intent`` and raises if the head
    does not select the intended K row."""

    from neural_vm.vm_step import AutoregressiveAttention

    d_model = 512
    num_heads = 8
    HD = d_model // num_heads
    attn = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=0,
        use_flash_attention=False,
    )

    # Build a broken intent (no K anchors → bare K wins). Hand-author
    # spec writes that match the broken structure so the verifier
    # sees them at bake time.
    broken_intent = AttentionIntent(
        name="broken_no_k_anchor",
        q_at=PositionPredicate(label="Q", positive_anchors=(BD.OP_LI_RELAY,)),
        k_at=PositionPredicate(label="K"),
        head_idx=0,
    )
    broken_spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=broken_intent.to_spec().q,
        k=broken_intent.to_spec().k,
        v=(),
        o=(),
        intent=broken_intent,
    )

    with pytest.raises(ValueError, match="AttentionIntent verification failed"):
        with torch.no_grad():
            Primitives.generate_attention_head(attn, broken_spec, HD)


def test_l15_head0_intent_block_lowered_attention_matches_legacy():
    """End-to-end: lower the intent-derived block writes and the
    legacy binary-block writes to a Q/K matrix and assert byte-
    identical entries at the head-0 row block."""

    from neural_vm.vm_step import AutoregressiveAttention

    from neural_vm.unified_compiler.ops.l15_ops import (
        _layer15_memory_lookup_heads_0_3_specs,
    )

    d_model = 512
    num_heads = 8
    HD = d_model // num_heads

    # Build two minimal specs that only carry slots 4..27. Use the
    # AttentionIntent for one, hand-extract the same block from the
    # legacy spec for the other.
    intent_spec = _l15_head0_intent_binary_block().to_spec()

    legacy = _layer15_memory_lookup_heads_0_3_specs(BD)[0]
    legacy_q_block = tuple(
        w for w in legacy.q if 4 <= w.slot <= 27
    )
    legacy_k_block = tuple(
        w for w in legacy.k if 4 <= w.slot <= 27
    )
    legacy_block_spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=legacy_q_block,
        k=legacy_k_block,
        v=(),
        o=(),
    )

    intent_attn = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=15,
        use_flash_attention=False,
    )
    legacy_attn = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=15,
        use_flash_attention=False,
    )

    with torch.no_grad():
        Primitives.generate_attention_head(intent_attn, intent_spec, HD)
        Primitives.generate_attention_head(
            legacy_attn, legacy_block_spec, HD,
        )

    # Compare the head-0 row block of W_q and W_k.
    base = 0  # head_idx 0
    block_q_a = intent_attn.W_q.data[base:base + HD, :]
    block_q_b = legacy_attn.W_q.data[base:base + HD, :]
    block_k_a = intent_attn.W_k.data[base:base + HD, :]
    block_k_b = legacy_attn.W_k.data[base:base + HD, :]

    assert torch.equal(block_q_a, block_q_b), (
        "Intent-lowered Q block does not byte-match legacy block"
    )
    assert torch.equal(block_k_a, block_k_b), (
        "Intent-lowered K block does not byte-match legacy block"
    )
