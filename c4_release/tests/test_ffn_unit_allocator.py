"""
Tests for :mod:`neural_vm.ffn_unit_allocator`.

Scaffolding-stage coverage: no production op has been migrated to the
allocator yet, so there is no byte-identity gate (the dim-allocator
test suite covers that pattern). What we DO want pinned down before
any migration starts:

* **Pinned-collision behaviour** — pinning two ranges that overlap is a
  hard error by default; ``allow_overlap=True`` is the explicit alias
  path.
* **Unpinned first-fit** — auto-placed ranges take the lowest-address
  free gap large enough; gaps too small are skipped.
* **Oversize/duplicate/invalid-input rejection** — every malformed call
  produces a :class:`FFNUnitAllocatorError` instead of silently
  truncating or shadowing a prior range.
* **(start, end) return contract** — the public API hands back a
  half-open range that callers can splat into existing ``start_unit,
  next_unit = ...`` patterns from the current ``_lXX_unit_counter``
  convention.
"""

from __future__ import annotations

import pytest

from neural_vm.ffn_unit_allocator import (
    DEFAULT_LAYER_MAX_UNITS,
    AllocatedUnitRange,
    FFNUnitAllocator,
    FFNUnitAllocatorError,
)


# ---------------------------------------------------------------------------
# 1. Construction / defaults
# ---------------------------------------------------------------------------
def test_default_layer_max_units_is_4096():
    """The default pool matches the brief: 4096 units per layer, which
    comfortably covers every existing per-layer FFN in the bake."""
    assert DEFAULT_LAYER_MAX_UNITS == 4096
    a = FFNUnitAllocator()
    assert a.layer_max_units == 4096


def test_custom_layer_max_units():
    """Callers can shrink the pool for synthetic tests / wider models."""
    a = FFNUnitAllocator(layer_max_units=256)
    assert a.layer_max_units == 256


def test_non_positive_layer_max_units_rejected():
    """``layer_max_units <= 0`` is a clean error, not a silent zero-pool."""
    with pytest.raises(FFNUnitAllocatorError):
        FFNUnitAllocator(layer_max_units=0)
    with pytest.raises(FFNUnitAllocatorError):
        FFNUnitAllocator(layer_max_units=-1)


# ---------------------------------------------------------------------------
# 2. Pinned allocation
# ---------------------------------------------------------------------------
def test_pinned_alloc_returns_start_end_tuple():
    """The (start, end) return matches the half-open convention used by
    today's ``start_unit, next_unit = ...`` ``_lXX_unit_counter`` code."""
    a = FFNUnitAllocator(layer_max_units=512)
    start, end = a.alloc("L14_PC_PLUS_1", 64, pin=192)
    assert (start, end) == (192, 256)


def test_pinned_alloc_records_metadata():
    """Insertion-ordered ``ranges()`` carries pin/overlap flags so a
    future to_registry helper can replay them deterministically."""
    a = FFNUnitAllocator(layer_max_units=512)
    a.alloc("PINNED", 32, pin=100)
    rng = a.ranges()[0]
    assert isinstance(rng, AllocatedUnitRange)
    assert rng.op_name == "PINNED"
    assert rng.start == 100
    assert rng.n_units == 32
    assert rng.end == 132
    assert rng.pinned is True
    assert rng.overlap is False


def test_pinned_allocation_respects_collision():
    """Pinning two ranges at the same start is a hard error by default.

    The error message must name BOTH the new op and the prior op it
    collided with so a migration agent can locate the conflict without
    grepping the per-layer ops file.
    """
    a = FFNUnitAllocator(layer_max_units=512)
    a.alloc("FIRST", 64, pin=192)

    with pytest.raises(FFNUnitAllocatorError) as excinfo:
        a.alloc("SECOND", 64, pin=192)
    msg = str(excinfo.value)
    assert "SECOND" in msg, f"error should name the new op: {msg!r}"
    assert "FIRST" in msg, f"error should name the colliding prior op: {msg!r}"

    # Partial overlap is also a collision.
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("THIRD", 64, pin=224)  # overlaps [192, 256) on [224, 256)


def test_pinned_allocation_allows_intentional_alias():
    """``allow_overlap=True`` is the explicit opt-in for intentional
    aliases (e.g. a debug op that reads from the same hidden units as
    its primary)."""
    a = FFNUnitAllocator(layer_max_units=512)
    a.alloc("PRIMARY", 64, pin=192)
    start, end = a.alloc("ALIAS", 64, pin=192, allow_overlap=True)
    assert (start, end) == (192, 256)
    alias_rng = a._by_name["ALIAS"]  # type: ignore[attr-defined]
    assert alias_rng.overlap is True
    # The alias must NOT push subsequent auto-placed ranges higher —
    # the underlying 192..256 units belong to PRIMARY only.
    auto_start, auto_end = a.alloc("AUTO", 64)
    assert auto_start == 0  # first-fit picks [0, 64), well below PRIMARY


def test_pinned_oob_rejected():
    """Pinning past layer_max_units is a clean error (no silent truncation)."""
    a = FFNUnitAllocator(layer_max_units=256)
    with pytest.raises(FFNUnitAllocatorError) as excinfo:
        a.alloc("OOB", 64, pin=224)  # [224, 288) > 256
    assert "exceeds layer_max_units" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Unpinned first-fit
# ---------------------------------------------------------------------------
def test_unpinned_allocation_fills_gaps():
    """First-fit picks the LOWEST-address free gap large enough.

    Layout under test:
        [0,   64)  HEAD (pinned)
        [64, 128)  FREE  ← first-fit target
        [128,192)  MID  (pinned)
        [192,256)  FREE
    Allocating an unpinned size-64 range must land at 64.
    """
    a = FFNUnitAllocator(layer_max_units=256)
    a.alloc("HEAD", 64, pin=0)
    a.alloc("MID", 64, pin=128)

    start, end = a.alloc("AUTO", 64)
    assert (start, end) == (64, 128), (
        f"first-fit should pick lowest gap (64), got ({start}, {end})"
    )
    auto_rng = a._by_name["AUTO"]  # type: ignore[attr-defined]
    assert auto_rng.pinned is False
    assert auto_rng.overlap is False

    # The next unpinned alloc lands in the remaining [192, 256) gap.
    nxt_start, nxt_end = a.alloc("AUTO2", 32)
    assert (nxt_start, nxt_end) == (192, 224)

    # And free_pool reports only the trailing [224, 256) sliver.
    assert a.free_pool() == [(224, 32)]


def test_unpinned_skips_gap_too_small():
    """First-fit skips gaps smaller than the requested n_units."""
    a = FFNUnitAllocator(layer_max_units=128)
    a.alloc("A", 32, pin=0)    # [0, 32)
    a.alloc("B", 16, pin=48)   # [48, 64), leaving [32, 48) gap (size 16)
    a.alloc("C", 32, pin=80)   # [80, 112)
    # [32, 48) is size 16, too small for 24. Next gap [64, 80) is size 16,
    # also too small. [112, 128) is size 16, also too small. Should fail.
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("WONTFIT", 24)
    # But a size-16 request fits the FIRST size-16 gap at [32, 48).
    start, end = a.alloc("AUTO", 16)
    assert (start, end) == (32, 48)


def test_unpinned_disallows_allow_overlap():
    """``allow_overlap=True`` with no pin= is a usage error: first-fit can
    never produce an overlap so accepting the flag would silently
    misrepresent intent."""
    a = FFNUnitAllocator(layer_max_units=128)
    a.alloc("X", 32, pin=0)
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("Y", 32, allow_overlap=True)


# ---------------------------------------------------------------------------
# 4. Oversize / invalid-input rejection
# ---------------------------------------------------------------------------
def test_allocator_oversize_rejection():
    """Allocating past the pool width is a clean error.

    Two flavours: a pin= past the end, and an unpinned alloc whose
    request exceeds any remaining gap.
    """
    a = FFNUnitAllocator(layer_max_units=256)
    with pytest.raises(FFNUnitAllocatorError) as excinfo:
        a.alloc("HUGE", 512, pin=0)
    assert "exceeds layer_max_units" in str(excinfo.value)

    b = FFNUnitAllocator(layer_max_units=64)
    b.alloc("FILL", 60, pin=0)  # leaves only [60, 64), size 4
    with pytest.raises(FFNUnitAllocatorError) as excinfo:
        b.alloc("WONTFIT", 16)
    assert "no free gap" in str(excinfo.value)


def test_invalid_n_units_rejected():
    """``n_units <= 0`` (or non-int) is caught up front."""
    a = FFNUnitAllocator(layer_max_units=128)
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("ZERO", 0, pin=0)
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("NEG", -1, pin=0)


def test_invalid_pin_rejected():
    """Negative ``pin=`` is caught before any range bookkeeping runs."""
    a = FFNUnitAllocator(layer_max_units=128)
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("BAD", 4, pin=-1)


def test_duplicate_op_name_rejected():
    """Two allocations with the same op_name is a hard error even at
    different positions — name uniqueness is a precondition for a
    future to_registry helper."""
    a = FFNUnitAllocator(layer_max_units=128)
    a.alloc("X", 4, pin=0)
    with pytest.raises(FFNUnitAllocatorError):
        a.alloc("X", 4, pin=64)


# ---------------------------------------------------------------------------
# 5. free_pool / ranges sanity
# ---------------------------------------------------------------------------
def test_free_pool_initial_state():
    """An empty allocator reports the whole pool as one contiguous gap."""
    a = FFNUnitAllocator(layer_max_units=256)
    assert a.free_pool() == [(0, 256)]


def test_free_pool_ignores_aliases():
    """Aliases (allow_overlap=True) do not consume free units — they're
    bookkeeping, not bytes. ``free_pool`` reflects this so callers can
    plan auto-placement without double-counting."""
    a = FFNUnitAllocator(layer_max_units=64)
    a.alloc("P", 16, pin=0)
    a.alloc("P_ALIAS", 16, pin=0, allow_overlap=True)
    # Only [0, 16) is consumed; [16, 64) is one big free gap.
    assert a.free_pool() == [(16, 48)]


def test_ranges_returns_copy_in_insertion_order():
    """``ranges()`` is a defensive copy in insertion order so callers
    can iterate without mutating the allocator's internal list."""
    a = FFNUnitAllocator(layer_max_units=128)
    a.alloc("A", 16, pin=0)
    a.alloc("B", 16)  # auto-placed → [16, 32)
    a.alloc("C", 16, pin=64)
    rs = a.ranges()
    assert [r.op_name for r in rs] == ["A", "B", "C"]
    # Mutating the returned list does not affect the allocator.
    rs.clear()
    assert [r.op_name for r in a.ranges()] == ["A", "B", "C"]


def test_pin_zero_is_honoured():
    """``pin=0`` is a real pin, not falsy: the brief's PC+1 chain starts
    at 192 today but future ops may legitimately want unit 0."""
    a = FFNUnitAllocator(layer_max_units=64)
    start, end = a.alloc("AT_ZERO", 8, pin=0)
    assert (start, end) == (0, 8)
    assert a._by_name["AT_ZERO"].pinned is True  # type: ignore[attr-defined]
