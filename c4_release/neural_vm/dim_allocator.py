"""
Dynamic dim-slot allocator for the Autoregressive Neural VM.

Replaces the hand-picked ``start`` argument on every
``DimRegistry.alloc()`` call with a first-fit allocator that can either:

  * Honor an explicit ``pin=<start>`` (preserves the existing layout —
    trained weights stay valid because nothing moves), or
  * Pick the first free slot of the requested size automatically
    (unpinned allocation; used by new dims that don't care where they
    land).

This module is **parallel** to :mod:`neural_vm.dim_registry` and is not
yet wired into the production bake. The companion
:mod:`neural_vm.dim_registry_dynamic` re-encodes every existing
:func:`build_default_registry` call with ``pin=`` so the dynamic output is
byte-identical to the static one. Once the static registry stabilises a
follow-up commit can flip the default ``build`` path to the dynamic
allocator and start dropping pins on a per-op basis.

Design choices:

* **First-fit** allocation. Pinned dims claim their range first; auto
  dims then sweep ``[0, d_model)`` and take the lowest-address free gap
  large enough to hold the request. Predictable, deterministic, and easy
  to reason about when migrating an op family at a time.
* **Aliases are explicit.** The existing static registry allocates many
  intentionally overlapping aliases (``AX_FULL_LO`` and
  ``FORMAT_PTR_LO`` both at slot 471, etc.). Pinned allocations error on
  *unintended* collisions by default; intentional overlap requires
  ``allow_overlap=True``. The allocator never inserts an alias on the
  unpinned path — an auto-placed dim always finds a never-claimed gap.
* **u32-everywhere.** Every start/size is a plain Python ``int`` (which
  is unbounded but always non-negative here). No floats, no 16-bit
  intermediates — matches the project-wide u32 invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Default pool width — matches DimRegistry(d_model=512) used by the
# production bake. Wider models can pass d_model= to Allocator(...) to
# get a bigger pool.
DEFAULT_POOL_WIDTH = 512


@dataclass
class AllocatedSlot:
    """A single allocation record produced by :class:`Allocator`.

    Mirrors :class:`neural_vm.dim_registry.DimSlot` so :meth:`Allocator.to_registry`
    can copy fields one-for-one.
    """

    name: str
    start: int
    size: int
    description: str = ""
    semantics: Optional[str] = None
    group: Optional[str] = None
    pinned: bool = False
    # Whether this allocation was permitted to overlap an earlier one.
    # Tracked so :meth:`Allocator.free_pool` correctly accounts for the
    # underlying byte coverage (aliases do not consume new bytes).
    overlap: bool = False

    @property
    def end(self) -> int:
        return self.start + self.size


class AllocatorError(ValueError):
    """Raised by :class:`Allocator` for collisions and oversize requests.

    Subclassing ``ValueError`` keeps the failure mode compatible with
    callers that ``except ValueError:`` around the static
    :meth:`DimRegistry.alloc` (it raises ``ValueError`` for the same
    family of mistakes).
    """


class Allocator:
    """First-fit slot allocator with an explicit ``pin=`` field.

    Parameters
    ----------
    d_model:
        Width of the slot pool. Defaults to :data:`DEFAULT_POOL_WIDTH`
        (512), matching the production bake.

    Notes
    -----
    The allocator tracks each :class:`AllocatedSlot` in insertion order
    so :meth:`to_registry` can replay them deterministically. Internally
    it also maintains a byte-level ``_claimed`` set for O(size) collision
    checks; this keeps :meth:`alloc` fast even when the registry grows
    past ~100 dims.
    """

    def __init__(self, d_model: int = DEFAULT_POOL_WIDTH):
        if d_model <= 0:
            raise AllocatorError(f"d_model must be positive (got {d_model})")
        self.d_model: int = int(d_model)
        # Insertion-ordered list; used as the source of truth when
        # rebuilding a DimRegistry from this allocator.
        self._slots: List[AllocatedSlot] = []
        # Fast lookup by name (used for duplicate-name checks).
        self._by_name: Dict[str, AllocatedSlot] = {}
        # Set of byte indices currently claimed by a non-overlap
        # allocation. Aliases (allow_overlap=True) do NOT add to this
        # set, so subsequent unpinned allocations will not "see" them as
        # blocked — matches the static registry's behaviour where aliases
        # share underlying bytes with their parent slot.
        self._claimed: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def alloc(
        self,
        name: str,
        size: int,
        *,
        pin: Optional[int] = None,
        semantics: Optional[str] = None,
        group: Optional[str] = None,
        description: str = "",
        allow_overlap: bool = False,
    ) -> AllocatedSlot:
        """Allocate ``size`` consecutive slots for ``name``.

        If ``pin`` is provided the allocation lands at exactly that
        start; collisions raise :class:`AllocatorError` unless
        ``allow_overlap=True`` (used for intentional aliases that share
        bytes with a prior slot).

        If ``pin`` is ``None`` the allocator scans ``[0, d_model)`` and
        returns the first free gap large enough to satisfy ``size``.
        Auto-placed slots never overlap with any prior slot.
        """
        # ---- shared input validation ----
        if not isinstance(size, int) or size <= 0:
            raise AllocatorError(
                f"alloc({name!r}): size must be a positive int (got {size!r})"
            )
        if name in self._by_name:
            raise AllocatorError(f"alloc({name!r}): duplicate slot name")

        if pin is not None:
            # ---- pinned path ----
            if not isinstance(pin, int) or pin < 0:
                raise AllocatorError(
                    f"alloc({name!r}): pin must be a non-negative int "
                    f"(got {pin!r})"
                )
            if pin + size > self.d_model:
                raise AllocatorError(
                    f"alloc({name!r}): pinned range [{pin}, {pin + size}) "
                    f"exceeds d_model={self.d_model}"
                )
            overlap_hit = self._collision(pin, size)
            if overlap_hit is not None and not allow_overlap:
                other = overlap_hit
                raise AllocatorError(
                    f"alloc({name!r}): pinned range "
                    f"[{pin}, {pin + size}) collides with "
                    f"{other.name!r} at [{other.start}, {other.end}); "
                    f"pass allow_overlap=True if the alias is intentional"
                )
            start = pin
            pinned = True
            overlap = allow_overlap and (overlap_hit is not None)
        else:
            # ---- auto-placed path ----
            if allow_overlap:
                # Auto-placed allocations never overlap by construction;
                # accepting allow_overlap here would be confusing.
                raise AllocatorError(
                    f"alloc({name!r}): allow_overlap requires pin= "
                    f"(auto placement never overlaps)"
                )
            start = self._find_free(size)
            if start is None:
                raise AllocatorError(
                    f"alloc({name!r}): no free gap of size {size} in "
                    f"pool of width {self.d_model} "
                    f"(used={self._used_bytes()}, "
                    f"free_pool={self.free_pool()})"
                )
            pinned = False
            overlap = False

        slot = AllocatedSlot(
            name=name,
            start=start,
            size=size,
            description=description,
            semantics=semantics,
            group=group,
            pinned=pinned,
            overlap=overlap,
        )
        self._slots.append(slot)
        self._by_name[name] = slot
        if not overlap:
            # Only "real" (non-alias) bytes count toward the claimed
            # footprint so aliases don't push subsequent auto-placed
            # dims into a higher gap.
            for d in range(start, start + size):
                self._claimed.add(d)
        return slot

    def free_pool(self) -> List[Tuple[int, int]]:
        """Return list of ``(start, length)`` free gaps, sorted by start.

        A "free" byte is one not held by any non-overlap allocation;
        aliases (``allow_overlap=True``) do not consume free bytes.
        """
        gaps: List[Tuple[int, int]] = []
        d = 0
        while d < self.d_model:
            if d in self._claimed:
                d += 1
                continue
            run_start = d
            while d < self.d_model and d not in self._claimed:
                d += 1
            gaps.append((run_start, d - run_start))
        return gaps

    def slots(self) -> List[AllocatedSlot]:
        """Return a shallow copy of all allocations in insertion order."""
        return list(self._slots)

    def to_registry(self):
        """Materialise a :class:`DimRegistry` from current allocations.

        Imported lazily so this module stays importable in environments
        that don't ship the full ``neural_vm`` package (handy for the
        unit tests, which only need the allocator).
        """
        # Local import keeps the test surface narrow and avoids a cycle
        # if DimRegistry ever wants to reach back into the allocator.
        from neural_vm.dim_registry import DimRegistry

        reg = DimRegistry(d_model=self.d_model)
        for s in self._slots:
            # DimRegistry.alloc() warns when semantics is None — that's
            # the static registry's behaviour and we want to inherit it,
            # so we pass through whatever the caller provided.
            reg.alloc(s.name, s.start, s.size, s.description, semantics=s.semantics)
        return reg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collision(self, start: int, size: int) -> Optional[AllocatedSlot]:
        """Return the FIRST previously-allocated slot that overlaps
        ``[start, start+size)``, or ``None`` if the range is free.

        Iteration order matches insertion order so error messages name
        the earliest conflicting slot — easiest to recognise in a stack
        trace.
        """
        end = start + size
        for s in self._slots:
            if s.overlap:
                # Aliases don't own their bytes; subsequent pins land on
                # the parent, not on the alias.
                continue
            if start < s.end and s.start < end:
                return s
        return None

    def _find_free(self, size: int) -> Optional[int]:
        """Return the start of the first free gap of width ``size``.

        Linear sweep over ``[0, d_model)``; for the 512-wide pool the
        cost is negligible compared to the ~100 alloc calls per bake.
        """
        run_start: Optional[int] = None
        for d in range(self.d_model):
            if d in self._claimed:
                run_start = None
                continue
            if run_start is None:
                run_start = d
            if d - run_start + 1 >= size:
                return run_start
        return None

    def _used_bytes(self) -> int:
        return len(self._claimed)


__all__ = [
    "AllocatedSlot",
    "Allocator",
    "AllocatorError",
    "DEFAULT_POOL_WIDTH",
]
