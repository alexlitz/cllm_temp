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
# 1b. (category, role) bindings all resolve in the BUILT layout
# ---------------------------------------------------------------------------
def test_verify_categories_resolve_in_built_layout():
    """Every ``(category, role)`` in the default registry's category index
    must resolve to a slot NAME that is present in the BUILT layout's
    ``dim_positions``.

    This is the permanent guard for the Phase 7.E.0 Bug-A class: the static
    registry pins an alias NAME (e.g. ``POST_PRTF_PC_LO``, ``SP_OLD_LO``,
    ``ADJ_CARRY``) that the dim-liveness allocator MERGES AWAY in the built
    layout — so a rule that ``dim_ref``'d that ``(category, role)`` would
    ``KeyError`` at lowering even though the static registry "knows" the
    name. The widen repack moves dims; probing the static registry reads the
    wrong cell. This test codifies "every category we expose must survive
    into the build", catching both the original Bug-A bindings AND any
    future ``register_band_category`` (Bug-B) tag whose name is dropped.

    Slow (compiles the full VM on CPU, ``disk_cache=False`` so the env is
    observed fresh); kept here next to the byte-identity gate because it is
    the same static-vs-built consistency contract.
    """
    verify_categories_resolve_in_built_layout()


def verify_categories_resolve_in_built_layout():
    """Assert every default-registry ``(category, role)`` resolves to a NAME
    present in ``compile_full_vm_dynamic(disk_cache=False)[1].dim_positions``.

    Importable as a standalone audit (callers other than pytest can invoke
    it directly). Raises ``AssertionError`` listing each ``(category, role)``
    whose bound NAME is absent from the built layout.
    """
    # Local imports keep the fast allocator unit tests above free of the
    # heavy compiler import when this audit is deselected.
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        reg = build_default_registry()
        _model, layout = compile_full_vm_dynamic(disk_cache=False)

    built_names = set(layout.dim_positions)

    # ``reg._category_index`` maps every registered (category, role) -> the
    # slot NAME it resolves to (both the primary tags AND the alias= entries
    # for the re-pointed PC/SP keys, AND the register_band_category tags).
    unresolved = [
        (cat, role, name)
        for (cat, role), name in reg._category_index.items()
        if name not in built_names
    ]

    assert not unresolved, (
        "category/role bindings resolve to a NAME absent from the BUILT "
        "layout (would KeyError at lowering — the Phase 7.E.0 Bug-A class):\n"
        + "\n".join(
            f"  ({cat!r}, {role!r}) -> {name!r}"
            for cat, role, name in sorted(unresolved)
        )
        + "\nRe-point the binding onto a surviving alias NAME in "
        "_register_default_categories (or register_band_category), or drop "
        "it if no honest target exists."
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
