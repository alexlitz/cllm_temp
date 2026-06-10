"""Dim-aliasing verifier — catch rules that read aliased dims without
disambiguating which alias is live at the firing position.

Background
----------

The residual stream packs multiple *semantically distinct* dims onto the
same byte range when the build registry knows their gating predicates are
disjoint. The textbook example is slot 12:

* ``ADDR_B0_LO`` (slot 12, size 16) — semantics ``mark == MEM``.
  One-hot of the gathered MEM-address byte-0 low nibble.
* ``OPCODE_BYTE_LO`` (slot 12, size 16) — semantics
  ``mark == MEM OR (is_byte AND byte_index == 0)``.
  One-hot of the opcode byte's low nibble; written at the AX/BYTE0 row
  for dispatch.

At a ``mark == MEM`` position the slot carries the **address** byte. At
a ``is_byte AND byte_index == 0`` position it can carry the **opcode**
byte. The two semantics overlap on ``mark == MEM`` only because the
opcode pathway is *not* supposed to fire at MEM rows — but the registry
itself has no way to enforce that.

If a rule's ``conditions`` (or ``gate_terms``) reference
``OPCODE_BYTE_LO+i`` and the rule's effective firing predicate is
*compatible* with ``mark == MEM`` (i.e. could fire at a MEM-marker
row), it will silently read whichever value happens to be in the slot
at that position — almost always the *address* byte, not the opcode.
This class of bug is invisible to the existing scope verifier because
the semantics string of the aliased dim doesn't constrain firing
position — it just labels *one* of the two roles the byte range can
play.

This module enumerates every set of aliased dims sharing a byte range,
then checks every FFNRule that reads an aliased dim: the rule's
effective predicate (what positions it can fire at) must NOT overlap
the semantics of any *other* alias of that byte range. If it does, the
rule reads garbage at the overlapping positions.

Public API
----------

* :func:`enumerate_dim_aliases(registry)` — return a list of
  :class:`AliasGroup` records, one per set of >=2 slots that share at
  least one byte position.
* :func:`verify_dim_aliases(op, registry)` — run the verifier against
  one Operation. Returns a list of issue dicts (one per offending
  ``(rule, read_dim, conflicting_alias)`` triple).
* :func:`verify_dim_aliases_for_ops(ops, registry)` — convenience
  wrapper that iterates over a list of ops.
* :func:`format_violations(issues)` — human-readable summary string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.decl_verifier import _collect_ffn_rules_from_op
from neural_vm.unified_compiler.effective_predicate import effective_predicate
from neural_vm.unified_compiler.ir import FFNRule
from neural_vm.unified_compiler.predicates import (
    Predicate,
    entails,
    overlaps,
    parse,
    satisfiable,
)


__all__ = [
    "AliasGroup",
    "AliasViolation",
    "enumerate_dim_aliases",
    "verify_dim_aliases",
    "verify_dim_aliases_for_ops",
    "format_violations",
]


# ---------------------------------------------------------------------------
# Alias enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasGroup:
    """One set of >=2 slots that overlap on at least one byte position.

    Attributes
    ----------
    start, end
        Byte range ``[start, end)`` shared by every member of an exact
        group, or the run of consecutive shared bytes for a partial
        overlap.
    members
        Tuple of slot names that overlap on this range. Sorted by
        name for determinism.
    """

    start: int
    end: int
    members: Tuple[str, ...]

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class AliasViolation:
    """One rule that reads an aliased dim at positions where the alias
    could carry a different semantic value."""

    op_name: Optional[str]
    rule_name: Optional[str]
    read_dim: str          # slot name being read
    read_offset: int       # offset within the slot
    conflicting_alias: str # the other slot name whose semantics overlaps
    conflicting_semantics: Optional[str]
    effective_predicate: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": "dim_alias_read_without_disambiguation",
            "op": self.op_name,
            "rule": self.rule_name,
            "read_dim": f"{self.read_dim}+{self.read_offset}",
            "conflicting_alias": self.conflicting_alias,
            "conflicting_semantics": self.conflicting_semantics,
            "effective_predicate": self.effective_predicate,
        }


def enumerate_dim_aliases(registry: DimRegistry) -> List[AliasGroup]:
    """Return one :class:`AliasGroup` per set of overlapping slots.

    Two slots ``a`` and ``b`` overlap iff their byte ranges intersect.
    The result groups slots by EXACT ``(start, size)`` (production
    aliases all share an exact range), then adds partial-overlap runs
    for any other overlapping pair.

    Sorted by ``(start, end, members)`` for determinism.
    """
    by_range: Dict[Tuple[int, int], List[str]] = {}
    for name, slot in registry.slots.items():
        by_range.setdefault((slot.start, slot.size), []).append(name)

    groups: List[AliasGroup] = []
    seen_exact: set = set()
    for (start, size), names in by_range.items():
        if len(names) < 2:
            continue
        key = (start, start + size)
        seen_exact.add(key)
        groups.append(
            AliasGroup(
                start=start,
                end=start + size,
                members=tuple(sorted(names)),
            )
        )

    # Partial-overlap runs: collapse consecutive bytes with identical
    # multi-owner sets. Skip runs that exactly coincide with an
    # exact-range group already emitted above.
    byte_owners: Dict[int, List[str]] = {}
    for name, slot in registry.slots.items():
        for b in range(slot.start, slot.start + slot.size):
            byte_owners.setdefault(b, []).append(name)

    partial: Dict[Tuple[str, ...], Tuple[int, int]] = {}
    current_members: Optional[Tuple[str, ...]] = None
    run_start: Optional[int] = None
    last_byte: Optional[int] = None
    for b in sorted(byte_owners.keys()):
        members = tuple(sorted(byte_owners[b]))
        if len(members) < 2:
            members = ()
        if members != current_members or (
            last_byte is not None and b != last_byte + 1
        ):
            if current_members and run_start is not None:
                _emit_partial(
                    partial,
                    current_members,
                    run_start,
                    last_byte + 1,  # type: ignore[arg-type]
                    seen_exact,
                )
            current_members = members
            run_start = b if members else None
        last_byte = b
    if current_members and run_start is not None:
        _emit_partial(
            partial,
            current_members,
            run_start,
            last_byte + 1,  # type: ignore[arg-type]
            seen_exact,
        )

    for members, (s, e) in partial.items():
        groups.append(AliasGroup(start=s, end=e, members=members))

    groups.sort(key=lambda g: (g.start, g.end, g.members))
    return groups


def _emit_partial(
    out: Dict[Tuple[str, ...], Tuple[int, int]],
    members: Tuple[str, ...],
    start: int,
    end: int,
    skip_exact: set,
) -> None:
    if (start, end) in skip_exact:
        return
    cur = out.get(members)
    if cur is None or (end - start) > (cur[1] - cur[0]):
        out[members] = (start, end)


# ---------------------------------------------------------------------------
# Per-op verification
# ---------------------------------------------------------------------------


def _is_colocated_subbank(
    read_dim: str, sibling: str, registry: DimRegistry,
) -> bool:
    """Heuristic: a sibling is a "sub-bank parent" of the read dim if
    they have the SAME slot range (start + size) AND the read dim's
    semantics is a STRICT subset of the sibling's. The ``OPCODE_FLAGS``
    -> ``OP_LEV`` case: same slot range, OP_LEV's semantics
    ``mark == AX AND opcode_at_AX == LEV`` is strictly tighter than the
    parent's ``mark == AX``. Both reference the same byte; the child is
    just an indicator subset of the parent.

    This is a conservative filter — it suppresses parent/child false
    positives without dropping the OPCODE_BYTE_LO/ADDR_B0_LO case where
    one semantics is a subset of the other but the slots are TRULY
    aliased (different writers, different content). The distinguishing
    test is "same writer" which we can't infer from the registry alone,
    so we use the proxy "same slot range" — production registry's
    parent/child families all match this; production true-aliasing pairs
    have either different slot ranges or non-subset semantics.
    """
    slot_a = registry.slots.get(read_dim)
    slot_b = registry.slots.get(sibling)
    if slot_a is None or slot_b is None:
        return False
    sem_a = slot_a.semantics
    sem_b = slot_b.semantics
    if sem_a is None or sem_b is None:
        return False
    # Parent/child sub-bank: one slot is FULLY CONTAINED in the other
    # AND their semantics are in a subset relation. ``OP_LEV`` (1 wide
    # at byte 270) sits inside ``OPCODE_FLAGS`` (34 wide at 262..295)
    # and OP_LEV's semantics is a strict subset of OPCODE_FLAGS'. The
    # true-aliasing OPCODE_BYTE_LO/ADDR_B0_LO pair has the SAME slot
    # range (both 12..28), not a strict containment, so this filter
    # leaves them flagged.
    a_in_b = (
        slot_b.start <= slot_a.start
        and slot_a.start + slot_a.size <= slot_b.start + slot_b.size
        and (slot_b.size > slot_a.size or slot_b.start != slot_a.start)
    )
    b_in_a = (
        slot_a.start <= slot_b.start
        and slot_b.start + slot_b.size <= slot_a.start + slot_a.size
        and (slot_a.size > slot_b.size or slot_a.start != slot_b.start)
    )
    if not (a_in_b or b_in_a):
        return False
    try:
        p_a = parse(sem_a)
        p_b = parse(sem_b)
    except Exception:
        return False
    try:
        return entails(p_a, p_b) or entails(p_b, p_a)
    except Exception:
        return False


def verify_dim_aliases(
    op,
    registry: DimRegistry,
    *,
    alias_index: Optional[Dict[str, List[str]]] = None,
    skip_colocated_subbank: bool = True,
) -> List[AliasViolation]:
    """Walk an op's FFNRules and report reads of aliased dims whose
    effective firing predicate is COMPATIBLE with another alias's
    semantics (i.e. they could fire at a position where the slot
    carries the *other* alias's value).

    The verifier reports a violation per ``(rule, read_dim,
    conflicting_alias)`` triple. A rule reading a dim ``D`` at offset
    ``k`` from a slot shared with ``D'`` is OK iff the rule's effective
    predicate is DISJOINT from ``D'``'s semantics — every position
    where the rule fires is a position where ``D'`` does not write the
    slot, so the read is unambiguous.

    Parameters
    ----------
    op
        An Operation with a ``compiler_ir`` (any of the shapes accepted
        by :func:`decl_verifier._collect_ffn_rules_from_op`).
    registry
        The :class:`DimRegistry` whose aliasing structure to check
        against.
    alias_index
        Optional pre-computed mapping ``slot_name -> [other_slot_name,
        ...]`` of every other slot whose byte range OVERLAPS this slot.
        Pass to amortise across many calls. See :func:`_build_alias_index`.

    Returns
    -------
    List[AliasViolation]
    """
    if alias_index is None:
        alias_index = _build_alias_index(registry)

    op_name = getattr(op, "name", None)
    rules = _collect_ffn_rules_from_op(op)

    violations: List[AliasViolation] = []
    for rule in rules:
        eff = _safe_effective_predicate(rule, registry)
        if eff is None:
            continue
        # If the effective predicate is unsatisfiable, the rule never
        # fires; skip it.
        try:
            if not satisfiable(eff):
                continue
        except Exception:
            continue

        read_dims = _collect_read_dim_refs(rule)
        for dim_name, dim_offset in read_dims:
            siblings = alias_index.get(dim_name, [])
            if not siblings:
                continue
            for sibling in siblings:
                sibling_sem = _slot_semantics(registry, sibling)
                if sibling_sem is None:
                    continue
                try:
                    sib_pred = parse(sibling_sem)
                except Exception:
                    continue
                try:
                    if not overlaps(eff, sib_pred):
                        continue
                except Exception:
                    continue
                if skip_colocated_subbank and _is_colocated_subbank(
                    dim_name, sibling, registry,
                ):
                    continue
                violations.append(
                    AliasViolation(
                        op_name=op_name,
                        rule_name=getattr(rule, "name", None),
                        read_dim=dim_name,
                        read_offset=dim_offset,
                        conflicting_alias=sibling,
                        conflicting_semantics=sibling_sem,
                        effective_predicate=str(eff),
                    )
                )
    return violations


def verify_dim_aliases_for_ops(
    ops: Sequence,
    registry: DimRegistry,
    *,
    skip_colocated_subbank: bool = True,
) -> List[AliasViolation]:
    """Run :func:`verify_dim_aliases` over an iterable of ops, sharing
    the alias index across calls.

    De-duplicates violations on ``(op_name, rule_name, read_dim,
    read_offset, conflicting_alias)`` so the same rule appearing in
    multiple ops does not flood the report.
    """
    alias_index = _build_alias_index(registry)
    seen: set = set()
    out: List[AliasViolation] = []
    for op in ops:
        for v in verify_dim_aliases(
            op, registry, alias_index=alias_index,
            skip_colocated_subbank=skip_colocated_subbank,
        ):
            key = (
                v.op_name,
                v.rule_name,
                v.read_dim,
                v.read_offset,
                v.conflicting_alias,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
    return out


def format_violations(violations: Iterable[AliasViolation]) -> str:
    """Return a human-readable multi-line summary of the violations."""
    by_op: Dict[Optional[str], List[AliasViolation]] = {}
    for v in violations:
        by_op.setdefault(v.op_name, []).append(v)
    if not by_op:
        return "no dim-alias violations"
    lines: List[str] = []
    for op_name in sorted(by_op.keys(), key=lambda x: (x or "")):
        lines.append(f"== {op_name or '<unknown op>'} ==")
        for v in by_op[op_name]:
            lines.append(
                f"  rule={v.rule_name!r} reads {v.read_dim}+{v.read_offset} "
                f"which aliases {v.conflicting_alias} "
                f"({v.conflicting_semantics!r}); "
                f"effective={v.effective_predicate!r}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_alias_index(registry: DimRegistry) -> Dict[str, List[str]]:
    """Return ``slot_name -> [sibling_slot_name, ...]`` for every slot
    that overlaps at least one other slot."""
    out: Dict[str, List[str]] = {}
    slots = list(registry.slots.values())
    for i, a in enumerate(slots):
        siblings: List[str] = []
        for j, b in enumerate(slots):
            if i == j:
                continue
            if a.overlaps(b):
                siblings.append(b.name)
        if siblings:
            out[a.name] = sorted(siblings)
    return out


def _collect_read_dim_refs(rule: FFNRule) -> List[Tuple[str, int]]:
    """Return ``[(dim_name, offset), ...]`` for every dim the rule
    reads (conditions, gate_terms, and gate)."""
    seen: set = set()
    out: List[Tuple[str, int]] = []
    for term in rule.conditions:
        key = (term.dim.name, term.dim.offset)
        if key not in seen:
            seen.add(key)
            out.append(key)
    for term in rule.gate_terms:
        key = (term.dim.name, term.dim.offset)
        if key not in seen:
            seen.add(key)
            out.append(key)
    if rule.gate is not None:
        key = (rule.gate.name, rule.gate.offset)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _safe_effective_predicate(
    rule: FFNRule,
    registry: DimRegistry,
) -> Optional[Predicate]:
    """Best-effort wrapper around :func:`effective_predicate`.

    Returns ``None`` if the predicate cannot be inferred (missing
    semantics, parse errors, etc.).
    """
    try:
        return effective_predicate(rule, registry)
    except Exception:
        return None


def _slot_semantics(registry: DimRegistry, name: str) -> Optional[str]:
    """Return the semantics string for ``name`` or ``None`` if the slot
    has no declared semantics or is not in the registry."""
    try:
        return registry.semantics(name)
    except KeyError:
        return None
