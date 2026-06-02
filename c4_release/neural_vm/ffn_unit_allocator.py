"""
Per-layer FFN unit allocator for the Autoregressive Neural VM.

Parallel to :mod:`neural_vm.dim_allocator` but for the *FFN hidden-unit*
axis instead of the residual-stream dim axis. Each transformer block's
SwiGLU FFN has a 1-D bank of hidden units; ops within that block claim
disjoint runs of units (e.g. L14's PC+1 chain at unit 192, the PC+3
chain at 320). Today those start indices are hand-picked author-chosen
offsets sprinkled across the ``unified_compiler/ops/lX_ops.py`` files,
which makes inserting a new op family painful — every neighbour has to
shift, and the unit_counter convention (``ffn._l14_unit_counter``) only
gives sequential placement, never a pinned-and-checked layout.

This module is **scaffolding only**: it does not yet replace any
production allocation. Once landed, ops can declare "I need N units"
and the allocator picks; existing ops keep their unit indices via
``pin=`` to preserve trained weights byte-identically. Migration is a
follow-up wave (one op family at a time, same playbook as the dim
allocator).

Design choices mirror :class:`neural_vm.dim_allocator.Allocator`:

* **First-fit** over ``[0, layer_max_units)``. Pinned ranges claim
  first; auto-placed ranges take the lowest free gap large enough to
  hold the request. Predictable and trivial to reason about during
  migration.
* **Aliases require opt-in.** Pinned ranges that overlap a prior
  allocation raise :class:`FFNUnitAllocatorError` unless
  ``allow_overlap=True``. Auto-placed allocations never overlap.
* **u32-everywhere.** Every start/n_units/end is a plain Python ``int``.
  No floats, no width-bounded intermediates — matches the project-wide
  u32 invariant.

Scope deliberately omits the ``to_registry()`` round-trip that
:mod:`dim_allocator` carries, because there is no FFN-unit equivalent
of :class:`DimRegistry` yet — the migration path will likely grow one
in a follow-up commit, at which point this module gets the same
``to_registry`` helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


# Strategy modes for :class:`FFNUnitAllocator`. ``"pinned"`` (default)
# preserves the historical byte-identity contract: ``pin=`` is mandatory
# for migrated ops and a collision is a hard error. ``"dynamic_first_fit"``
# treats ``pin=`` as a HINT — if the pinned range is free the allocator
# honors it (so already-trained weights stay in place when possible);
# otherwise it falls back to a first-fit gap. Used by Phase 7.B to drop
# pins corpus-wide once the allocator self-fits the layout.
AllocStrategy = Literal["pinned", "dynamic_first_fit"]


# Default per-layer FFN unit budget. Wider FFNs can pass
# layer_max_units= to FFNUnitAllocator(...) to widen the pool.
DEFAULT_LAYER_MAX_UNITS = 4096


@dataclass
class AllocatedUnitRange:
    """A single unit-range allocation produced by :class:`FFNUnitAllocator`.

    Mirrors :class:`neural_vm.dim_allocator.AllocatedSlot` on the FFN
    hidden-unit axis. The ``op_name`` is the human-readable owner (the
    op factory that requested the range), used for error messages and
    duplicate-name checks.
    """

    op_name: str
    start: int
    n_units: int
    pinned: bool = False
    # Whether this allocation was permitted to overlap an earlier one.
    # Aliases (``overlap=True``) do not consume free units, so subsequent
    # auto-placed ranges do not "see" them as blocked.
    overlap: bool = False

    @property
    def end(self) -> int:
        return self.start + self.n_units


class FFNUnitAllocatorError(ValueError):
    """Raised by :class:`FFNUnitAllocator` for collisions and oversize requests.

    Subclassing ``ValueError`` keeps the failure mode consistent with
    :class:`neural_vm.dim_allocator.AllocatorError` so callers can
    handle both with a single ``except ValueError:``.
    """


class FFNUnitAllocator:
    """First-fit FFN unit allocator with an explicit ``pin=`` field.

    Parameters
    ----------
    layer_max_units:
        Width of the per-layer unit pool. Defaults to
        :data:`DEFAULT_LAYER_MAX_UNITS` (4096), which comfortably covers
        every existing per-layer FFN in the production bake.

    Notes
    -----
    The allocator tracks each :class:`AllocatedUnitRange` in insertion
    order so a future ``to_registry``-style helper can replay them
    deterministically. Internally it maintains a unit-level
    ``_claimed`` set for O(n_units) collision checks; this keeps
    :meth:`alloc` fast even as a layer accumulates dozens of op
    families.
    """

    def __init__(
        self,
        layer_max_units: int = DEFAULT_LAYER_MAX_UNITS,
        *,
        strategy: AllocStrategy = "pinned",
    ):
        if layer_max_units <= 0:
            raise FFNUnitAllocatorError(
                f"layer_max_units must be positive (got {layer_max_units})"
            )
        if strategy not in ("pinned", "dynamic_first_fit"):
            raise FFNUnitAllocatorError(
                f"strategy must be 'pinned' or 'dynamic_first_fit' "
                f"(got {strategy!r})"
            )
        self.layer_max_units: int = int(layer_max_units)
        self._strategy: AllocStrategy = strategy
        # Insertion-ordered list; the source of truth for any future
        # registry-rebuild path.
        self._ranges: List[AllocatedUnitRange] = []
        # Fast lookup by op_name for duplicate-name checks.
        self._by_name: Dict[str, AllocatedUnitRange] = {}
        # Set of unit indices held by a non-overlap allocation. Aliases
        # (allow_overlap=True) do NOT add to this set, matching the
        # dim_allocator semantics.
        self._claimed: set[int] = set()
        # Migration audit: every (op_name, pin) where the caller provided
        # a pin hint that the allocator could NOT honor under
        # ``"dynamic_first_fit"`` (collision) and had to first-fit
        # elsewhere. Empty in ``"pinned"`` mode because a colliding pin
        # is a hard error there.
        self._pin_collisions: List[Tuple[str, int, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def alloc(
        self,
        op_name: str,
        n_units: int,
        *,
        pin: Optional[int] = None,
        allow_overlap: bool = False,
    ) -> Tuple[int, int]:
        """Allocate ``n_units`` consecutive FFN units for ``op_name``.

        Returns
        -------
        (start, end):
            Half-open range ``[start, end)`` of unit indices owned by
            ``op_name``. ``end == start + n_units``.

        If ``pin`` is provided the allocation lands at exactly that
        start; collisions raise :class:`FFNUnitAllocatorError` unless
        ``allow_overlap=True``.

        If ``pin`` is ``None`` the allocator scans
        ``[0, layer_max_units)`` and returns the first free gap large
        enough to satisfy ``n_units``. Auto-placed ranges never overlap.
        """
        # ---- shared input validation ----
        if not isinstance(n_units, int) or n_units <= 0:
            raise FFNUnitAllocatorError(
                f"alloc({op_name!r}): n_units must be a positive int "
                f"(got {n_units!r})"
            )
        if op_name in self._by_name:
            raise FFNUnitAllocatorError(
                f"alloc({op_name!r}): duplicate op_name"
            )

        if pin is not None:
            # ---- pinned path ----
            if not isinstance(pin, int) or pin < 0:
                raise FFNUnitAllocatorError(
                    f"alloc({op_name!r}): pin must be a non-negative int "
                    f"(got {pin!r})"
                )
            if pin + n_units > self.layer_max_units:
                raise FFNUnitAllocatorError(
                    f"alloc({op_name!r}): pinned range "
                    f"[{pin}, {pin + n_units}) exceeds "
                    f"layer_max_units={self.layer_max_units}"
                )
            overlap_hit = self._collision(pin, n_units)
            if overlap_hit is not None and not allow_overlap:
                # In ``"dynamic_first_fit"`` mode the pin is a HINT —
                # a collision triggers a first-fit fallback instead of
                # erroring. ``allow_overlap=True`` still wins for
                # explicit aliases so byte-identity migration keeps
                # working alongside the relaxed pinning.
                if self._strategy == "dynamic_first_fit":
                    found = self._find_free(n_units)
                    if found is None:
                        raise FFNUnitAllocatorError(
                            f"alloc({op_name!r}): pin hint {pin} collides "
                            f"with {overlap_hit.op_name!r} and no free "
                            f"gap of size {n_units} remains in pool of "
                            f"width {self.layer_max_units} "
                            f"(used={self._used_units()}, "
                            f"free_pool={self.free_pool()})"
                        )
                    self._pin_collisions.append((op_name, pin, found))
                    start = found
                    pinned = False
                    overlap = False
                else:
                    other = overlap_hit
                    raise FFNUnitAllocatorError(
                        f"alloc({op_name!r}): pinned range "
                        f"[{pin}, {pin + n_units}) collides with "
                        f"{other.op_name!r} at [{other.start}, {other.end}); "
                        f"pass allow_overlap=True if the alias is intentional"
                    )
            else:
                start = pin
                pinned = True
                overlap = allow_overlap and (overlap_hit is not None)
        else:
            # ---- auto-placed path ----
            if allow_overlap:
                # Auto placement never produces an overlap; accepting
                # the flag here would silently misrepresent intent.
                raise FFNUnitAllocatorError(
                    f"alloc({op_name!r}): allow_overlap requires pin= "
                    f"(auto placement never overlaps)"
                )
            found = self._find_free(n_units)
            if found is None:
                raise FFNUnitAllocatorError(
                    f"alloc({op_name!r}): no free gap of size {n_units} "
                    f"in pool of width {self.layer_max_units} "
                    f"(used={self._used_units()}, "
                    f"free_pool={self.free_pool()})"
                )
            start = found
            pinned = False
            overlap = False

        rng = AllocatedUnitRange(
            op_name=op_name,
            start=start,
            n_units=n_units,
            pinned=pinned,
            overlap=overlap,
        )
        self._ranges.append(rng)
        self._by_name[op_name] = rng
        if not overlap:
            # Only "real" (non-alias) units count toward the claimed
            # footprint so aliases don't push subsequent auto-placed
            # ranges into a higher gap.
            for u in range(start, start + n_units):
                self._claimed.add(u)
        return start, start + n_units

    def free_pool(self) -> List[Tuple[int, int]]:
        """Return list of ``(start, length)`` free gaps, sorted by start.

        A "free" unit is one not held by any non-overlap allocation;
        aliases (``allow_overlap=True``) do not consume free units.
        """
        gaps: List[Tuple[int, int]] = []
        u = 0
        while u < self.layer_max_units:
            if u in self._claimed:
                u += 1
                continue
            run_start = u
            while u < self.layer_max_units and u not in self._claimed:
                u += 1
            gaps.append((run_start, u - run_start))
        return gaps

    def ranges(self) -> List[AllocatedUnitRange]:
        """Return a shallow copy of all allocations in insertion order."""
        return list(self._ranges)

    @property
    def strategy(self) -> AllocStrategy:
        """Current allocation strategy. Read-only; use
        :meth:`set_strategy` to switch modes mid-stream."""
        return self._strategy

    def set_strategy(self, strategy: AllocStrategy) -> None:
        """Switch allocation mode in place.

        Used by the Phase 7.B compile pass to flip a freshly-constructed
        allocator from the default ``"pinned"`` (byte-identity) mode into
        ``"dynamic_first_fit"`` (pins-are-hints) before any allocations
        happen. Mid-stream switching is permitted — already-recorded
        ranges keep their starts; only subsequent :meth:`alloc` calls
        observe the new mode.
        """
        if strategy not in ("pinned", "dynamic_first_fit"):
            raise FFNUnitAllocatorError(
                f"set_strategy(): strategy must be 'pinned' or "
                f"'dynamic_first_fit' (got {strategy!r})"
            )
        self._strategy = strategy

    def evict_pin_hints(self) -> List[Tuple[str, int]]:
        """Return ``(op_name, pin)`` for every pinned range still in use.

        Migration-auditing helper: walk every allocation that landed at
        its caller-provided ``pin=`` (i.e. ``pinned=True``) and report
        the pair. The Phase 7.B compile pass uses this output to track
        which op families are still relying on hard-coded offsets after
        the strategy switch — anything that turns up here in
        ``"dynamic_first_fit"`` mode is a candidate for pin removal.
        """
        return [(r.op_name, r.start) for r in self._ranges if r.pinned]

    def pin_collisions(self) -> List[Tuple[str, int, int]]:
        """Return ``(op_name, requested_pin, actual_start)`` for every
        pin hint that could NOT be honored under ``"dynamic_first_fit"``.

        Empty list in ``"pinned"`` mode (a colliding pin is a hard error
        there). Used by the Phase 7.B reporter to surface which op
        families need their pins relaxed before the allocator can fold
        them into a smaller layout.
        """
        return list(self._pin_collisions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collision(
        self, start: int, n_units: int
    ) -> Optional[AllocatedUnitRange]:
        """Return the FIRST previously-allocated range that overlaps
        ``[start, start+n_units)``, or ``None`` if the range is free.

        Iteration order matches insertion order so error messages name
        the earliest conflicting range — easiest to recognise in a
        stack trace.
        """
        end = start + n_units
        for r in self._ranges:
            if r.overlap:
                # Aliases don't own their units; subsequent pins land on
                # the parent, not on the alias.
                continue
            if start < r.end and r.start < end:
                return r
        return None

    def _find_free(self, n_units: int) -> Optional[int]:
        """Return the start of the first free gap of width ``n_units``.

        Linear sweep over ``[0, layer_max_units)``; for a 4096-wide pool
        the cost is negligible compared to the per-bake alloc volume.
        """
        run_start: Optional[int] = None
        for u in range(self.layer_max_units):
            if u in self._claimed:
                run_start = None
                continue
            if run_start is None:
                run_start = u
            if u - run_start + 1 >= n_units:
                return run_start
        return None

    def _used_units(self) -> int:
        return len(self._claimed)


__all__ = [
    "AllocStrategy",
    "AllocatedUnitRange",
    "DEFAULT_LAYER_MAX_UNITS",
    "FFNUnitAllocator",
    "FFNUnitAllocatorError",
]
