"""Slot-conflict registry — Phase 1 of the memory cluster fix plan.

This module owns the compile-time detection of silent slot overwrites
that broke V2/V3/V4 of the memory cluster fix. The premise of those
attempts was the bake dispatcher: two ops would each call
``block.ffn = X`` or ``Primitives.generate_attention_head(..., head_idx=k)``
at the same layer, and the second silently overwrote the first. The
dynamic verifier could not catch this because both bakes "succeeded".

Slot IDs (tuples):

    ("ffn",)                    — whole block.ffn module replacement.
    ("attn",)                   — whole block.attn module replacement.
    ("attn", "head", k)         — attention head k.
    ("attn", "Wq")              — whole-W_q claim (non-head-aligned).
    ("attn", "Wk") / ("attn", "Wv") / ("attn", "Wo")
    ("post_ops", i)             — explicit post_op slot at index i.
    ("post_ops_append",)        — append-only post_ops claim.
    ("ffn_units", start, end)   — per-unit-range FFN claim. ``end`` is
                                  EXCLUSIVE. Overlaps with another
                                  ``("ffn_units", ...)`` at the same
                                  layer are conflicts; non-overlap is
                                  fine.

Ops opt out of conflict detection for specific slot KINDS (the first
element of the slot tuple) by passing ``slot_share=("ffn_units",)`` on
the Operation. The whitelist applies per (layer, kind): once any
participant at that layer declares slot-share for the kind, conflicts
of that kind at that layer are suppressed.

The registry never changes runtime behaviour — it only refuses to
compile when an unauthorized conflict is detected.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_SLOT_KINDS = frozenset({
    "ffn",
    "attn",
    "attn_head",
    "attn_matrix",
    "post_ops",
    "post_ops_append",
    "ffn_units",
})


def _slot_kind(slot_id: Tuple[Any, ...]) -> str:
    """Return the canonical slot-kind token from a slot-id tuple.

    Conventions (matching the slot-id schema in the module docstring)::

        ("ffn",)              -> "ffn"
        ("attn",)             -> "attn"
        ("attn", "head", k)   -> "attn_head"
        ("attn", "Wq" / ...)  -> "attn_matrix"
        ("post_ops", i)       -> "post_ops"
        ("post_ops_append",)  -> "post_ops_append"
        ("ffn_units", s, e)   -> "ffn_units"
    """
    if not slot_id:
        return ""
    head = slot_id[0]
    if head == "attn" and len(slot_id) >= 2:
        if slot_id[1] == "head":
            return "attn_head"
        if slot_id[1] in ("Wq", "Wk", "Wv", "Wo"):
            return "attn_matrix"
    if head in ALLOWED_SLOT_KINDS:
        return head
    return str(head)


@dataclass(frozen=True)
class SlotClaim:
    """One ``(op_name, layer_idx, slot_id, op_kind)`` claim record."""

    op_name: str
    layer_idx: int
    slot_id: Tuple[Any, ...]
    op_kind: str
    slot_share: Tuple[str, ...] = ()


def _claims_conflict(a: SlotClaim, b: SlotClaim) -> bool:
    """Return True if two claims at the same layer conflict.

    For non-range slot ids, equality of the slot tuple is the conflict
    condition. For ``("ffn_units", start, end)`` claims, conflicts arise
    only on range overlap (``end`` exclusive).
    """
    sa, sb = a.slot_id, b.slot_id
    if not sa or not sb or sa[0] != sb[0]:
        return False
    if sa[0] == "ffn_units":
        try:
            s1, e1 = int(sa[1]), int(sa[2])
            s2, e2 = int(sb[1]), int(sb[2])
        except (IndexError, ValueError, TypeError):
            return sa == sb
        return s1 < e2 and s2 < e1
    return sa == sb


class SlotConflictError(RuntimeError):
    """Raised by :meth:`SlotRegistry.raise_on_conflict` when conflicts exist.

    Carries the list of ``(claim_a, claim_b)`` pairs so callers can
    inspect / format them.
    """

    def __init__(self, conflicts: List[Tuple[SlotClaim, SlotClaim]]):
        self.conflicts = list(conflicts)
        lines = ["Slot-conflict registry detected unauthorized conflicts:"]
        for a, b in self.conflicts:
            lines.append(
                f"  layer={a.layer_idx} slot={a.slot_id!r}: "
                f"{a.op_name!r} (kind={a.op_kind!r}) vs "
                f"{b.op_name!r} (kind={b.op_kind!r})"
            )
        lines.append(
            "Resolve by (a) routing one op to a different slot, or "
            "(b) adding slot_share=(<kind>,) on the Operation when the "
            "shared claim is legitimate (see "
            "docs/SLOT_REGISTRY_AUDIT_2026_06_05.md)."
        )
        super().__init__("\n".join(lines))


def _validate_slot_share(op_name: str, slot_share: Tuple[str, ...]) -> None:
    """Reject malformed ``slot_share`` tuples with a clear error.

    A typo like ``"fnn_units"`` would silently disable nothing — fail
    loudly at registration instead.
    """
    for kind in slot_share:
        if kind not in ALLOWED_SLOT_KINDS:
            raise ValueError(
                f"SlotRegistry.claim({op_name!r}): slot_share entry "
                f"{kind!r} is not a valid slot kind. Allowed: "
                f"{sorted(ALLOWED_SLOT_KINDS)}"
            )


class SlotRegistry:
    """Records ``(layer_idx, slot_id) -> [SlotClaim, ...]`` and detects conflicts.

    Usage::

        registry = SlotRegistry()
        registry.claim("op_a", layer_idx=13, slot_id=("ffn",), op_kind="block")
        registry.claim("op_b", layer_idx=13, slot_id=("ffn",), op_kind="block")
        registry.raise_on_conflict()  # raises SlotConflictError

    Per-claim opt-out::

        registry.claim(
            "op_c", layer_idx=10, slot_id=("ffn_units", 0, 64),
            op_kind="ffn", slot_share=("ffn_units",),
        )
        # Subsequent overlapping ffn_units claims at layer 10 do not
        # conflict with op_c (the share-kind rule is symmetric).
    """

    def __init__(self) -> None:
        self._by_layer: Dict[int, List[SlotClaim]] = defaultdict(list)
        # (layer, kind) pairs where at least one participant opted into
        # slot-share. Conflicts of that kind at that layer are suppressed.
        self._shared_kinds: Set[Tuple[int, str]] = set()

    def claim(
        self,
        op_name: str,
        layer_idx: int,
        slot_id: Tuple[Any, ...],
        op_kind: str,
        slot_share: Tuple[str, ...] = (),
    ) -> None:
        """Record one slot claim. Use :meth:`raise_on_conflict` to surface conflicts."""
        _validate_slot_share(op_name, slot_share)
        claim = SlotClaim(
            op_name=op_name,
            layer_idx=int(layer_idx),
            slot_id=tuple(slot_id),
            op_kind=str(op_kind),
            slot_share=tuple(slot_share),
        )
        # Record opt-out BEFORE appending so registration order doesn't
        # matter — a later op that hasn't claimed yet can still suppress
        # an already-registered peer's conflict via all_conflicts().
        kind = _slot_kind(claim.slot_id)
        if kind in claim.slot_share:
            self._shared_kinds.add((claim.layer_idx, kind))
        self._by_layer[claim.layer_idx].append(claim)

    def _is_suppressed(
        self,
        layer: int,
        a: SlotClaim,
        b: SlotClaim,
        kind_a: str,
        kind_b: str,
    ) -> bool:
        """Return True if this conflict is opted-out by either party."""
        return (
            (layer, kind_a) in self._shared_kinds
            or (layer, kind_b) in self._shared_kinds
            or kind_a in a.slot_share
            or kind_b in b.slot_share
        )

    def all_conflicts(self) -> List[Tuple[SlotClaim, SlotClaim]]:
        """Return every unauthorized conflict pair, deterministically ordered."""
        out: List[Tuple[SlotClaim, SlotClaim]] = []
        for layer in sorted(self._by_layer):
            claims = self._by_layer[layer]
            for i, a in enumerate(claims):
                kind_a = _slot_kind(a.slot_id)
                for b in claims[i + 1:]:
                    if a.op_name == b.op_name and a.slot_id == b.slot_id:
                        continue
                    if not _claims_conflict(a, b):
                        continue
                    kind_b = _slot_kind(b.slot_id)
                    if self._is_suppressed(layer, a, b, kind_a, kind_b):
                        continue
                    out.append((a, b))
        return out

    def raise_on_conflict(self) -> None:
        """Raise :class:`SlotConflictError` when any conflicts exist."""
        conflicts = self.all_conflicts()
        if conflicts:
            raise SlotConflictError(conflicts)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_layer.values())

    def all_claims(self) -> List[SlotClaim]:
        """Return every recorded claim in (layer asc, registration) order."""
        return [c for layer in sorted(self._by_layer) for c in self._by_layer[layer]]


# ---------------------------------------------------------------------------
# Slot-id derivation from an Operation
# ---------------------------------------------------------------------------

_SENTINEL_KEY = "__module_replacement"

# Mapping from the ``<target>`` token inside a ``produces`` sentinel
# value (e.g. the ``"ffn"`` in ``"L13.ffn[ALUShiftComposite]"``) to the
# slot id it represents. ``post_ops`` collapses to append-grain because
# every historical sentinel author uses ``.append`` or ``.insert(0)`` —
# a future audit could tighten this to ``("post_ops", i)``.
_SENTINEL_TARGET_TO_SLOT_ID: Dict[str, Tuple[Any, ...]] = {
    "ffn": ("ffn",),
    "attn": ("attn",),
    "post_ops": ("post_ops_append",),
}


def _parse_produces_sentinel(sentinel_value: str) -> Optional[Tuple[Any, ...]]:
    """Decode ``produces["__module_replacement"]`` strings to a slot id.

    Examples::

        "L13.ffn[ALUShiftComposite]"          -> ("ffn",)
        "L10.post_ops[FlattenedDivMod]"       -> ("post_ops_append",)
        "L15.attn[resize num_heads]"          -> ("attn",)
    """
    if not isinstance(sentinel_value, str):
        return None
    _, _, payload = sentinel_value.partition(".")
    if not payload:
        return None
    target = payload.split("[", 1)[0].strip()
    return _SENTINEL_TARGET_TO_SLOT_ID.get(target)


def derive_slot_ids_for_op(op) -> List[Tuple[Any, ...]]:
    """Derive zero-or-more slot ids the op claims at its target layer.

    Topology-anchor ops (whose bake is a no-op by contract) contribute
    nothing. Otherwise the derivation walks four signals in order:

      1. ``produces["__module_replacement"]`` sentinel.
      2. ``compiler_ir`` attention specs — one ``("attn", "head", k)``
         per declared head.
      3. ``ffn_units_used`` annotation (FFN/block ops only) →
         ``("ffn_units", 0, ffn_units_used)``.
      4. Coarse ``kind`` fallback (only when no signal above fired).
    """
    if getattr(op, "declarative_authority", None) == "topology_anchor":
        return []

    ids: List[Tuple[Any, ...]] = []

    produces = getattr(op, "produces", None) or {}
    sentinel_value = produces.get(_SENTINEL_KEY)
    if sentinel_value is not None:
        parsed = _parse_produces_sentinel(sentinel_value)
        if parsed is not None:
            ids.append(parsed)

    ir = getattr(op, "compiler_ir", None)
    for layer_spec in getattr(ir, "layers", None) or ():
        attention = getattr(layer_spec, "attention", None)
        for head_ir in getattr(attention, "rules", None) or ():
            spec = getattr(head_ir, "spec", None)
            head_idx = (
                getattr(spec, "head_idx", None) if spec is not None
                else getattr(head_ir, "head_idx", None)
            )
            if head_idx is not None:
                ids.append(("attn", "head", int(head_idx)))

    ffn_units_used = getattr(op, "ffn_units_used", None)
    kind = getattr(op, "kind", None)
    if ffn_units_used and kind in ("ffn", "block"):
        # The range is coarse ``[0, ffn_units_used)`` — ops that pack
        # tail-correction families into a non-zero start unit cannot be
        # expressed precisely here. The legitimate-sharing case is
        # covered by ``slot_share=("ffn_units",)``.
        ids.append(("ffn_units", 0, int(ffn_units_used)))

    if not ids:
        if kind == "ffn":
            ids.append(("ffn",))
        elif kind == "attn":
            ids.append(("attn",))

    return ids


__all__ = [
    "ALLOWED_SLOT_KINDS",
    "SlotClaim",
    "SlotConflictError",
    "SlotRegistry",
    "derive_slot_ids_for_op",
]
