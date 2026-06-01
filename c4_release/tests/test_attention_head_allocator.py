"""
Tests for :mod:`neural_vm.attention_head_allocator`.

Scaffolding-only coverage — no production op has been migrated yet, so
there is no byte-identity test (cf. ``test_dim_allocator.py`` which
checks ``build_default_registry_dynamic`` against the static registry).
The follow-up migration wave will add that gate per op family.

What's covered here:

1. **Pinned-collision behaviour.** Two ops pinning the same
   ``(layer_idx, head_idx)`` is a hard error; the error message names
   both ops so the operator can locate the conflict.
2. **Unpinned first-fit.** Auto-placed heads pick the lowest free index
   in the requested layer, skipping pinned heads.
3. **Per-layer isolation.** Layer N's allocations never see layer M's
   heads — an op pinned at ``head_idx=0`` in layer 4 does not block a
   pin at ``head_idx=0`` in layer 5.
4. **Oversize / OOB rejection.** Pinning beyond ``layer_max_heads`` is a
   clean error; filling a layer and asking for another auto head also
   errors cleanly.
5. **Duplicate op_name rejection.** Two ops with the same name is a
   hard error regardless of which layer or head they target.
"""

from __future__ import annotations

import pytest

from neural_vm.attention_head_allocator import (
    AttentionHeadAllocator,
    AttentionHeadAllocatorError,
    DEFAULT_LAYER_MAX_HEADS,
)


# ---------------------------------------------------------------------------
# 1. Pinned-collision behaviour
# ---------------------------------------------------------------------------
def test_pinned_allocation_respects_collision():
    """Pinning two ops at the same (layer, head) is a hard error.

    The error message must name BOTH the new op and the prior op it
    collided with so the operator can locate the conflict without
    grepping the bake.
    """
    a = AttentionHeadAllocator(layer_max_heads=8)
    a.alloc("first_op", layer_idx=4, pin=2)

    with pytest.raises(AttentionHeadAllocatorError) as excinfo:
        a.alloc("second_op", layer_idx=4, pin=2)
    msg = str(excinfo.value)
    assert "second_op" in msg, f"error should name the new op: {msg!r}"
    assert "first_op" in msg, f"error should name the colliding prior op: {msg!r}"
    # The error must say *which* layer and head collided so the operator
    # doesn't have to cross-reference the allocator state.
    assert "layer 4" in msg, f"error should name the layer: {msg!r}"
    assert "head_idx=2" in msg, f"error should name the head_idx: {msg!r}"


def test_pinned_oob_rejected():
    """Pinning at or beyond layer_max_heads is a clean error."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    with pytest.raises(AttentionHeadAllocatorError) as excinfo:
        a.alloc("oob_op", layer_idx=0, pin=8)
    assert "exceeds layer_max_heads" in str(excinfo.value)


def test_pin_negative_rejected():
    """Negative pin is a usage error, not silently coerced."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    with pytest.raises(AttentionHeadAllocatorError):
        a.alloc("neg_op", layer_idx=0, pin=-1)


def test_negative_layer_rejected():
    """Negative layer_idx is rejected on both alloc and free_heads."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    with pytest.raises(AttentionHeadAllocatorError):
        a.alloc("op", layer_idx=-1, pin=0)
    with pytest.raises(AttentionHeadAllocatorError):
        a.free_heads(-1)


def test_duplicate_op_name_rejected():
    """Two ops with the same name is a hard error even across layers."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    a.alloc("dup_op", layer_idx=0, pin=0)
    # Same name, different layer — still a duplicate.
    with pytest.raises(AttentionHeadAllocatorError):
        a.alloc("dup_op", layer_idx=1, pin=0)


# ---------------------------------------------------------------------------
# 2. Unpinned first-fit
# ---------------------------------------------------------------------------
def test_unpinned_allocation_fills_gaps():
    """First-fit must pick the LOWEST free head_idx in the target layer.

    Layout under test (layer 3, layer_max_heads=8):
        head 0  PINNED (op_a)
        head 1  FREE   ← first-fit target
        head 2  PINNED (op_b)
        head 3+ FREE
    Allocating an unpinned op into layer 3 must land at head_idx=1.
    """
    a = AttentionHeadAllocator(layer_max_heads=8)
    a.alloc("op_a", layer_idx=3, pin=0)
    a.alloc("op_b", layer_idx=3, pin=2)

    picked = a.alloc("op_auto", layer_idx=3)
    assert picked == 1, f"first-fit should pick lowest free (1), got {picked}"

    # The next unpinned alloc should take head_idx=3 (next-lowest free).
    nxt = a.alloc("op_auto2", layer_idx=3)
    assert nxt == 3, f"next first-fit head is 3, got {nxt}"

    # free_heads now reports the remaining unclaimed indices.
    assert a.free_heads(3) == [4, 5, 6, 7]


def test_unpinned_records_pinned_flag_false():
    """Auto-placed allocations must be marked pinned=False so the
    migration tooling can distinguish legacy hand-picked heads from
    fresh auto-allocated ones at a glance.
    """
    a = AttentionHeadAllocator(layer_max_heads=8)
    a.alloc("legacy", layer_idx=0, pin=4)
    a.alloc("fresh", layer_idx=0)  # auto
    recs = {r.op_name: r for r in a.heads()}
    assert recs["legacy"].pinned is True
    assert recs["fresh"].pinned is False


# ---------------------------------------------------------------------------
# 3. Per-layer isolation
# ---------------------------------------------------------------------------
def test_layers_are_independent():
    """Pin at (layer=4, head=0) must not block pin at (layer=5, head=0).

    Layers maintain independent head pools; an op in layer 4 has zero
    interaction with an op in layer 5 even if they share a head_idx.
    """
    a = AttentionHeadAllocator(layer_max_heads=8)
    a.alloc("l4_h0", layer_idx=4, pin=0)
    # Same head index, different layer — must succeed.
    a.alloc("l5_h0", layer_idx=5, pin=0)

    # Auto allocation in layer 4 must skip the pinned head 0 and land
    # at head 1; in layer 5 it must skip pinned head 0 and also land at 1
    # — neither sees the other layer's claims.
    p4 = a.alloc("l4_auto", layer_idx=4)
    p5 = a.alloc("l5_auto", layer_idx=5)
    assert p4 == 1, f"layer-4 first-fit should pick 1, got {p4}"
    assert p5 == 1, f"layer-5 first-fit should pick 1, got {p5}"


# ---------------------------------------------------------------------------
# 4. Oversize / OOB rejection
# ---------------------------------------------------------------------------
def test_unpinned_full_layer_rejected():
    """When every head in a layer is claimed, an auto alloc errors cleanly."""
    a = AttentionHeadAllocator(layer_max_heads=4)
    for h in range(4):
        a.alloc(f"fill_{h}", layer_idx=0, pin=h)
    with pytest.raises(AttentionHeadAllocatorError) as excinfo:
        a.alloc("overflow", layer_idx=0)
    msg = str(excinfo.value)
    assert "no free head" in msg, msg
    assert "layer 0" in msg, msg


def test_invalid_layer_max_heads_rejected():
    """layer_max_heads <= 0 is caught up front."""
    with pytest.raises(AttentionHeadAllocatorError):
        AttentionHeadAllocator(layer_max_heads=0)
    with pytest.raises(AttentionHeadAllocatorError):
        AttentionHeadAllocator(layer_max_heads=-1)


def test_invalid_op_name_rejected():
    """Empty or non-string op_name is caught up front."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    with pytest.raises(AttentionHeadAllocatorError):
        a.alloc("", layer_idx=0, pin=0)
    with pytest.raises(AttentionHeadAllocatorError):
        a.alloc(None, layer_idx=0, pin=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Inspection helpers
# ---------------------------------------------------------------------------
def test_default_layer_max_heads_matches_production():
    """The bare constructor must default to the production bake's 8 heads."""
    a = AttentionHeadAllocator()
    assert a.layer_max_heads == DEFAULT_LAYER_MAX_HEADS == 8


def test_free_heads_initial_state():
    """An untouched layer reports every head as free."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    assert a.free_heads(0) == [0, 1, 2, 3, 4, 5, 6, 7]
    # Touching layer 0 does not allocate anything in layer 1.
    assert a.free_heads(1) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_heads_returns_insertion_order():
    """heads() reproduces allocations in insertion order for deterministic replay."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    a.alloc("third", layer_idx=2, pin=5)
    a.alloc("first", layer_idx=0)
    a.alloc("second", layer_idx=1, pin=3)
    names = [r.op_name for r in a.heads()]
    assert names == ["third", "first", "second"]
