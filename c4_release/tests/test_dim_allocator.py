"""
Tests for :mod:`neural_vm.dim_allocator` and
:mod:`neural_vm.dim_registry_dynamic`.

Two layers of coverage:

1. **Byte-identity:** the dynamic registry (every dim pinned at its
   existing start) reproduces the static :func:`build_default_registry`
   slot for slot. This is the gate that lets a follow-up commit migrate
   the static registry to the allocator without moving trained weights.
2. **Allocator behaviour:** synthetic cases for pinned collision,
   unpinned first-fit, and oversize rejection — small enough to read
   without a full bake.
"""

from __future__ import annotations

import warnings

import pytest

from neural_vm.dim_allocator import Allocator, AllocatorError
from neural_vm.dim_registry import build_default_registry
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic


# ---------------------------------------------------------------------------
# 1. Byte-identity vs static registry
# ---------------------------------------------------------------------------
def test_allocator_byte_identical_to_static():
    """Every dim in build_default_registry() must appear at the SAME
    (start, size, semantics) in build_default_registry_dynamic().

    Description text is left out of the comparison: a few dims have
    long, hand-written descriptions and the dynamic mirror condenses
    whitespace; the (start, size, semantics) triple is what the bake
    and the predicate verifier actually consume.
    """
    # Both registries warn on DimSlots that omit semantics= during F-3
    # rollout; silence them so the test signal is just the byte-identity
    # diff below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        static = build_default_registry()
        dynamic = build_default_registry_dynamic()

    static_keys = set(static.slots)
    dynamic_keys = set(dynamic.slots)
    missing = static_keys - dynamic_keys
    extra = dynamic_keys - static_keys
    assert not missing, f"dynamic registry missing slots: {sorted(missing)}"
    assert not extra, f"dynamic registry has unexpected slots: {sorted(extra)}"

    diffs = []
    for name in sorted(static_keys):
        s = static.slots[name]
        d = dynamic.slots[name]
        if (s.start, s.size, s.semantics) != (d.start, d.size, d.semantics):
            diffs.append(
                f"{name}: static=({s.start},{s.size},{s.semantics!r}) "
                f"dynamic=({d.start},{d.size},{d.semantics!r})"
            )
    assert not diffs, "byte-identity mismatch:\n" + "\n".join(diffs)

    assert static.d_model == dynamic.d_model, (
        f"d_model mismatch: static={static.d_model} dynamic={dynamic.d_model}"
    )


# ---------------------------------------------------------------------------
# 2. Pinned-collision behaviour
# ---------------------------------------------------------------------------
def test_pinned_allocation_respects_collision():
    """Pinning two dims at the same start is a hard error by default.

    The error message must name BOTH the new dim and the prior dim it
    collided with so the operator can locate the conflict without
    grepping the registry.
    """
    a = Allocator(d_model=32)
    a.alloc("FIRST", 4, pin=0, semantics="is_byte")

    with pytest.raises(AllocatorError) as excinfo:
        a.alloc("SECOND", 4, pin=0, semantics="is_byte")
    msg = str(excinfo.value)
    assert "SECOND" in msg, f"error should name the new dim: {msg!r}"
    assert "FIRST" in msg, f"error should name the colliding prior dim: {msg!r}"

    # Partial overlap (FIRST at [0,4), THIRD at [2,6)) is also a collision.
    with pytest.raises(AllocatorError):
        a.alloc("THIRD", 4, pin=2, semantics="is_byte")

    # allow_overlap=True bypasses the check — used for intentional aliases.
    aliased = a.alloc(
        "FIRST_ALIAS", 4, pin=0, semantics="is_byte",
        allow_overlap=True,
    )
    assert aliased.start == 0
    assert aliased.overlap is True


def test_pinned_oob_rejected():
    """Pinning beyond d_model is a clean error (not a silent truncation)."""
    a = Allocator(d_model=16)
    with pytest.raises(AllocatorError) as excinfo:
        a.alloc("OOB", 4, pin=14, semantics="is_byte")
    assert "exceeds d_model" in str(excinfo.value)


def test_duplicate_name_rejected():
    """Two dims with the same name is a hard error even if positions differ."""
    a = Allocator(d_model=32)
    a.alloc("X", 4, pin=0, semantics="is_byte")
    with pytest.raises(AllocatorError):
        a.alloc("X", 4, pin=10, semantics="is_byte")


# ---------------------------------------------------------------------------
# 3. Unpinned first-fit
# ---------------------------------------------------------------------------
def test_unpinned_allocation_fills_gaps():
    """First-fit must pick the LOWEST-address free gap large enough.

    Layout under test:
        [0,10)  HEAD (pinned)
        [10,20) FREE  ← first-fit target
        [20,30) MID  (pinned)
        [30,40) FREE
    Allocating an unpinned size-10 dim must land at 10.
    """
    a = Allocator(d_model=40)
    a.alloc("HEAD", 10, pin=0,  semantics="is_byte")
    a.alloc("MID",  10, pin=20, semantics="is_byte")

    picked = a.alloc("AUTO", 10, semantics="is_byte")
    assert picked.start == 10, (
        f"first-fit should pick lowest gap (10), got {picked.start}"
    )
    assert picked.pinned is False
    assert picked.overlap is False

    # The next unpinned alloc should land in the remaining [30, 40) gap.
    nxt = a.alloc("AUTO2", 5, semantics="is_byte")
    assert nxt.start == 30, f"next first-fit gap is 30, got {nxt.start}"

    # And free_pool now reports only the trailing [35, 40) sliver.
    assert a.free_pool() == [(35, 5)]


def test_unpinned_skips_gap_too_small():
    """First-fit skips gaps smaller than the request size."""
    a = Allocator(d_model=20)
    a.alloc("A", 5, pin=0,  semantics="is_byte")   # [0,5)
    a.alloc("B", 2, pin=7,  semantics="is_byte")   # [7,9), leaving [5,7) gap
    a.alloc("C", 5, pin=12, semantics="is_byte")   # [12,17)
    # [5,7) is too small for size=3; [9,12) is exactly 3 → pick it.
    picked = a.alloc("AUTO", 3, semantics="is_byte")
    assert picked.start == 9, (
        f"first-fit should skip [5,7) (too small) and pick [9,12); got {picked.start}"
    )


def test_unpinned_disallows_allow_overlap():
    """allow_overlap=True with no pin is a usage error; first-fit can
    never produce an overlap so accepting the flag would silently
    misrepresent intent."""
    a = Allocator(d_model=32)
    a.alloc("X", 4, pin=0, semantics="is_byte")
    with pytest.raises(AllocatorError):
        a.alloc("Y", 4, semantics="is_byte", allow_overlap=True)


# ---------------------------------------------------------------------------
# 4. Oversize rejection
# ---------------------------------------------------------------------------
def test_allocator_oversize_rejection():
    """Allocating past the pool width is a clean error.

    Two flavours are checked:

    * a pin= that exceeds d_model (caught even when the pool is empty), and
    * an unpinned alloc that doesn't fit in any remaining gap.
    """
    # Pinned oversize.
    a = Allocator(d_model=64)
    with pytest.raises(AllocatorError) as excinfo:
        a.alloc("HUGE", 100, pin=0, semantics="is_byte")
    assert "exceeds d_model" in str(excinfo.value), str(excinfo.value)

    # Unpinned oversize: fill most of the pool, then ask for more than fits.
    b = Allocator(d_model=16)
    b.alloc("FILL", 14, pin=0, semantics="is_byte")  # leaves only [14,16)
    with pytest.raises(AllocatorError) as excinfo:
        b.alloc("WONTFIT", 4, semantics="is_byte")
    assert "no free gap" in str(excinfo.value), str(excinfo.value)


def test_invalid_size_rejected():
    """Sizes <= 0 or non-int are caught up front."""
    a = Allocator(d_model=16)
    with pytest.raises(AllocatorError):
        a.alloc("ZERO", 0, pin=0, semantics="is_byte")
    with pytest.raises(AllocatorError):
        a.alloc("NEG", -1, pin=0, semantics="is_byte")


# ---------------------------------------------------------------------------
# 5. free_pool / to_registry sanity
# ---------------------------------------------------------------------------
def test_free_pool_initial_state():
    """An empty allocator should report the entire pool as one gap."""
    a = Allocator(d_model=64)
    assert a.free_pool() == [(0, 64)]


def test_to_registry_roundtrip():
    """to_registry() returns a DimRegistry with the same slots."""
    a = Allocator(d_model=32)
    a.alloc("A", 4, pin=0, semantics="is_byte")
    a.alloc("B", 4, semantics="is_byte")  # auto-placed
    a.alloc("A_ALIAS", 4, pin=0, semantics="is_byte", allow_overlap=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        reg = a.to_registry()
    assert reg.d_model == 32
    assert reg.slots["A"].start == 0
    assert reg.slots["A"].size == 4
    assert reg.slots["B"].start == 4  # first-fit after A
    assert reg.slots["A_ALIAS"].start == 0  # alias of A
