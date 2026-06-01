"""
Per-layer attention-head allocator for the Autoregressive Neural VM.

Parallel to :mod:`neural_vm.dim_allocator` (slot pool, ``Allocator``) but
narrower: each transformer layer has a fixed number of attention heads
(typically 8) and every op that needs a head must claim a ``head_idx`` in
``[0, layer_max_heads)``. Today those ``head_idx`` values are hand-coded
in every ``AttentionHeadIR`` construction, which makes adding a new head
fragile — the author has to remember which slots are already taken in
that layer.

:class:`AttentionHeadAllocator` mirrors the slot allocator's contract:

  * ``pin=<idx>`` claims a specific head (used by existing ops during a
    migration so the bake stays byte-identical), or
  * ``pin=None`` first-fits the lowest free head in the requested layer.

This module is scaffolding only. Nothing in the production bake is wired
to it yet — a follow-up wave will migrate ``AttentionHeadIR``
constructions one op family at a time, pinning each at its current
``head_idx`` so the trained weights stay valid.

Design choices (kept parallel to :class:`neural_vm.dim_allocator.Allocator`):

* **Per-layer first-fit.** Each layer has its own ``[0, layer_max_heads)``
  pool; allocations never cross layers. Pinned heads claim first, then
  unpinned allocations sweep upward from index 0 and take the lowest
  free slot.
* **No overlap, ever.** Unlike slot dims, attention heads cannot be
  aliased — two ops writing the same ``head_idx`` in the same layer
  would clobber each other's logits. Any collision (pinned or auto) is
  a hard :class:`AttentionHeadAllocatorError`.
* **u32-everywhere.** ``layer_idx`` and ``head_idx`` are plain Python
  ``int``; never negative, never fractional. Matches the project-wide
  u32 invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# Default heads-per-layer — matches the production bake's attention
# config (8 heads per transformer layer). Wider configs can pass
# ``layer_max_heads=`` to the constructor.
DEFAULT_LAYER_MAX_HEADS = 8


@dataclass
class AllocatedHead:
    """A single attention-head allocation produced by
    :class:`AttentionHeadAllocator`.

    Captures the op that claimed the head, which layer it lives in, the
    chosen ``head_idx`` within that layer, and whether the allocation
    was pinned. Pin-vs-auto is tracked so the migration tooling can tell
    at a glance which ops still hold legacy hand-picked indices.
    """

    op_name: str
    layer_idx: int
    head_idx: int
    pinned: bool = False


class AttentionHeadAllocatorError(ValueError):
    """Raised by :class:`AttentionHeadAllocator` for invalid requests.

    Subclasses ``ValueError`` so callers that already ``except
    ValueError:`` around hand-coded ``head_idx`` validation pick up the
    new failure mode for free.
    """


class AttentionHeadAllocator:
    """First-fit attention-head allocator with an explicit ``pin=`` field.

    Parameters
    ----------
    layer_max_heads:
        Number of attention heads available per transformer layer.
        Defaults to :data:`DEFAULT_LAYER_MAX_HEADS` (8), matching the
        production bake's attention config.

    Notes
    -----
    The allocator tracks each :class:`AllocatedHead` in insertion order
    (for deterministic replay during a bake) and maintains a per-layer
    ``set[int]`` of claimed head indices for O(1) collision checks.
    Layers are created lazily on first reference so a 32-layer model
    that only touches a handful of layers doesn't pay for empty bookkeeping.
    """

    def __init__(self, layer_max_heads: int = DEFAULT_LAYER_MAX_HEADS):
        if not isinstance(layer_max_heads, int) or layer_max_heads <= 0:
            raise AttentionHeadAllocatorError(
                f"layer_max_heads must be a positive int "
                f"(got {layer_max_heads!r})"
            )
        self.layer_max_heads: int = int(layer_max_heads)
        # Insertion-ordered list; used as the source of truth when
        # replaying allocations during a bake.
        self._heads: List[AllocatedHead] = []
        # Fast duplicate-name lookup. Two ops sharing the same name would
        # be ambiguous to the verifier, so we reject up front.
        self._by_name: Dict[str, AllocatedHead] = {}
        # Per-layer set of claimed head indices for O(1) collision
        # checks. Lazily populated on first alloc into a layer.
        self._claimed: Dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def alloc(
        self,
        op_name: str,
        layer_idx: int,
        *,
        pin: Optional[int] = None,
    ) -> int:
        """Allocate an attention head in ``layer_idx`` for ``op_name``.

        If ``pin`` is provided the allocation lands at exactly that
        ``head_idx``; a collision with a prior head in the same layer
        raises :class:`AttentionHeadAllocatorError`.

        If ``pin`` is ``None`` the allocator scans
        ``[0, layer_max_heads)`` in the target layer and returns the
        lowest free index.

        Returns
        -------
        int
            The chosen ``head_idx`` in ``[0, layer_max_heads)``.
        """
        # ---- shared input validation ----
        if not isinstance(op_name, str) or not op_name:
            raise AttentionHeadAllocatorError(
                f"alloc(): op_name must be a non-empty str (got {op_name!r})"
            )
        if op_name in self._by_name:
            raise AttentionHeadAllocatorError(
                f"alloc({op_name!r}): duplicate op_name"
            )
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise AttentionHeadAllocatorError(
                f"alloc({op_name!r}): layer_idx must be a non-negative int "
                f"(got {layer_idx!r})"
            )

        claimed = self._claimed.setdefault(layer_idx, set())

        if pin is not None:
            # ---- pinned path ----
            if not isinstance(pin, int) or pin < 0:
                raise AttentionHeadAllocatorError(
                    f"alloc({op_name!r}): pin must be a non-negative int "
                    f"(got {pin!r})"
                )
            if pin >= self.layer_max_heads:
                raise AttentionHeadAllocatorError(
                    f"alloc({op_name!r}): pinned head_idx={pin} "
                    f"exceeds layer_max_heads={self.layer_max_heads}"
                )
            if pin in claimed:
                other = self._holder(layer_idx, pin)
                raise AttentionHeadAllocatorError(
                    f"alloc({op_name!r}): pinned head_idx={pin} in "
                    f"layer {layer_idx} collides with {other!r}; "
                    f"attention heads cannot be aliased"
                )
            head_idx = pin
            pinned = True
        else:
            # ---- auto-placed path ----
            head_idx = self._find_free(claimed)
            if head_idx is None:
                raise AttentionHeadAllocatorError(
                    f"alloc({op_name!r}): no free head in layer {layer_idx} "
                    f"(layer_max_heads={self.layer_max_heads}, "
                    f"used={sorted(claimed)})"
                )
            pinned = False

        rec = AllocatedHead(
            op_name=op_name,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pinned=pinned,
        )
        self._heads.append(rec)
        self._by_name[op_name] = rec
        claimed.add(head_idx)
        return head_idx

    def heads(self) -> List[AllocatedHead]:
        """Return a shallow copy of all allocations in insertion order."""
        return list(self._heads)

    def free_heads(self, layer_idx: int) -> List[int]:
        """Return the sorted list of unclaimed head indices in
        ``layer_idx``. Cheap inspection helper for migration tooling.
        """
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise AttentionHeadAllocatorError(
                f"free_heads(): layer_idx must be a non-negative int "
                f"(got {layer_idx!r})"
            )
        claimed = self._claimed.get(layer_idx, set())
        return [h for h in range(self.layer_max_heads) if h not in claimed]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_free(self, claimed: set[int]) -> Optional[int]:
        """Return the lowest unclaimed head index, or ``None`` if the
        layer is full.

        Linear sweep over ``[0, layer_max_heads)`` — at 8 heads this is
        trivially cheap and keeps allocation order deterministic.
        """
        for h in range(self.layer_max_heads):
            if h not in claimed:
                return h
        return None

    def _holder(self, layer_idx: int, head_idx: int) -> str:
        """Return the op_name that previously claimed
        ``(layer_idx, head_idx)``. Used for collision error messages so
        the operator can locate the conflict without grepping the bake.
        """
        for rec in self._heads:
            if rec.layer_idx == layer_idx and rec.head_idx == head_idx:
                return rec.op_name
        # Should be unreachable — _claimed and _heads are kept in sync.
        return "<unknown>"


__all__ = [
    "AllocatedHead",
    "AttentionHeadAllocator",
    "AttentionHeadAllocatorError",
    "DEFAULT_LAYER_MAX_HEADS",
]
