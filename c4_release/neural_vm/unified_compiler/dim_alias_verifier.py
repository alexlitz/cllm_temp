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
    And,
    Atom,
    OpcodeAtAxEq,
    OpcodeAtAxIn,
    OpcodeInStepIn,
    Or,
    Predicate,
    entails,
    is_tautology,
    overlaps,
    parse,
    satisfiable,
    strictly_refines,
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
    they share a byte range AND one's semantics is a strict refinement
    of the other's (extra atoms beyond a verbatim disjunct copy).

    Two cases are recognised as parent/child sub-banks:

    * **Strict containment**: ``OP_LEV`` (1 wide at byte 270) sits
      inside ``OPCODE_FLAGS`` (34 wide at 262..295) and OP_LEV's
      semantics ``mark == AX AND opcode_at_AX == LEV`` strictly
      refines ``mark == AX`` by adding the ``opcode_at_AX`` atom.
    * **Same-extent refinement** (Improvement B, 2026-06-10):
      ``OPCODE_BASE`` and ``OP_LEA`` occupy the same 1-wide slot at
      byte 262. OP_LEA's semantics
      ``mark == AX AND opcode_at_AX == LEA`` strictly refines
      OPCODE_BASE's ``mark == AX OR (is_byte AND byte_index == 0)``
      by ADDING the ``opcode_at_AX == LEA`` atom.

      The discriminator is :func:`predicates.strictly_refines`: the
      child must add at least one atom not present in any parent
      disjunct, so the child is a true narrowing of the parent (the
      same physical signal, refined). The genuine alias pair
      ``OPCODE_BYTE_LO`` vs ``ADDR_B0_LO`` does NOT qualify: ADDR_B0_LO
      ``mark == MEM`` equals one disjunct of OPCODE_BYTE_LO
      ``mark == MEM OR (is_byte AND byte_index == 0)`` verbatim — no
      extra atoms, so it remains flagged as a true alias.

    This is a conservative filter — it suppresses parent/child false
    positives without dropping the OPCODE_BYTE_LO/ADDR_B0_LO case.
    """
    slot_a = registry.slots.get(read_dim)
    slot_b = registry.slots.get(sibling)
    if slot_a is None or slot_b is None:
        return False
    sem_a = slot_a.semantics
    sem_b = slot_b.semantics
    if sem_a is None or sem_b is None:
        return False
    # The slots must share at least one byte position — either strict
    # containment (one fully inside the other, possibly equal extents)
    # OR same range entirely. We require A ⊆ B or B ⊆ A (range
    # containment, equal allowed).
    a_in_b = (
        slot_b.start <= slot_a.start
        and slot_a.start + slot_a.size <= slot_b.start + slot_b.size
    )
    b_in_a = (
        slot_a.start <= slot_b.start
        and slot_b.start + slot_b.size <= slot_a.start + slot_a.size
    )
    if not (a_in_b or b_in_a):
        return False
    try:
        p_a = parse(sem_a)
        p_b = parse(sem_b)
    except Exception:
        return False
    # Whichever slot is the smaller range plays the "child" role for
    # the semantic discriminator; if extents are equal we try both
    # directions.
    try:
        if a_in_b and not b_in_a:
            # A strictly inside B's range -> A is the child candidate.
            return strictly_refines(p_a, p_b)
        if b_in_a and not a_in_b:
            return strictly_refines(p_b, p_a)
        # Equal extents: either could be the refinement of the other.
        return strictly_refines(p_a, p_b) or strictly_refines(p_b, p_a)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Improvement C (2026-06-10): opcode_in_step disjointness
# ---------------------------------------------------------------------------
#
# Many residual alias pairs are time-shared by VM step phase: the same slot
# carries different semantic values depending on which opcode is running in
# the current step. Examples (from `docs/DIM_ALIAS_TRIAGE_2026_06_10.md`):
#
# * AX_CARRY_LO/HI ↔ POST_PRTF_SP_LO/HI — MUL scratch (compute) vs
#   POST-PRTF SP save (IO phase). Disjoint by opcode_in_step.
# * DIV_STAGING ↔ FETCH_LO/HI — DIV/MOD scratch vs instruction-byte fetch.
# * MUL_ACCUM ↔ FETCH_LO — MUL scratch vs fetch.
# * IO_OUTPUT_COUNT ↔ PSH_AT_SP — IO state vs PSH stack write.
# * MEM_STORE ↔ IMM_STAGING — SI/SC/PSH store wire vs PC fetch.
#
# Neither slot's semantics encodes the opcode constraint at the registry
# level (POST_PRTF_SP_LO and AX_CARRY_LO have IDENTICAL semantics strings);
# the disambiguation lives in the per-rule conditions (positive OP_<X>
# references) and in design-time knowledge of which opcodes touch which
# scratch slot.
#
# This refinement encodes both sides:
#
# 1. **Rule opcode set** — derived from positive OP_<X> conditions /
#    gate_terms / gate on the rule. ``OP_ADD+0`` as a positive condition
#    implies the rule fires only when ``opcode_in_step == ADD`` (since
#    OP_ADD's semantics pins ``opcode_at_AX == ADD`` at the AX marker
#    position, and ``opcode_at_AX`` and ``opcode_in_step`` agree on
#    non-IO opcodes at AX rows).
# 2. **Slot opcode owner set** — derived from the sibling slot's semantics
#    (via ``opcode_at_AX in/==`` or ``opcode_in_step in`` atoms in the DNF)
#    OR from the static ``_SLOT_OPCODE_OWNERS`` table for slots whose
#    semantics is opcode-agnostic by historical convention (POST_PRTF_*,
#    MUL_ACCUM, DIV_STAGING, FETCH_*, IMM_STAGING, MEM_STORE, MEM_VAL_B*,
#    PSH_AT_SP, IO_OUTPUT_COUNT, etc.).
#
# When BOTH sets are non-empty and DISJOINT, the rule and the conflicting
# alias cannot both be live at the same step — emit ``disjoint=true`` and
# skip the violation. Conservative: empty / unknown sets yield NO
# suppression (we keep the violation).

# Static map: slot name → set of opcodes that "own" the slot's value in
# any given step. Derived from the slot descriptions in
# ``neural_vm/dim_registry.py`` (search for "OP_<X>" / "PRTF" / "READ"
# / per-opcode staging slot in the docstrings). Conservative:
#
# * Only include slots whose lifetime is genuinely opcode-scoped (the
#   slot is dead under other opcodes).
# * Slots that are written every step (FETCH_LO/HI, IMM_STAGING) get
#   the FULL opcode set — they're owned by every opcode's fetch phase
#   and cannot suppress anything via disjointness (the rule's set is a
#   subset of the full set, never disjoint).
# * Slots whose opcode-ownership is already encoded in their semantics
#   (e.g. ``SP_OLD_LO`` semantics ends ``opcode_in_step in {ADJ}``) are
#   handled by the semantics extractor and don't need a table entry.
#
# Adding a slot here is the CHEAPEST cleanup. Removing a slot is safe
# (loses suppression power, never introduces false negatives).
# Static set: ALU-family opcodes that write/read AX_CARRY_*, ALU_*, CMP
# during the compute phase (the FIXME in dim_registry.py for AX_CARRY_*/
# ALU_*/CARRY confirms this set). Used as the opcode-owner set for
# AX_CARRY_*, ALU_*, CARRY, CMP* below — these slots are dead under
# non-ALU/non-CMP opcodes (notably the PRTF/READ IO opcodes whose tool-
# call returns alias them via POST_PRTF_SP_*/POST_PRTF_PC_*).
_ALU_OPCODES: frozenset[str] = frozenset({
    "ADD", "SUB", "MUL", "DIV", "MOD",
    "OR", "XOR", "AND",
    "SHL", "SHR",
})
_CMP_OPCODES: frozenset[str] = frozenset({
    "EQ", "NE", "LT", "GT", "LE", "GE",
})
_ALU_CMP_OPCODES: frozenset[str] = _ALU_OPCODES | _CMP_OPCODES


_SLOT_OPCODE_OWNERS: Dict[str, frozenset[str]] = {
    # POST_PRTF_* aliases on AX_FULL_*/AX_CARRY_* — only written when the
    # PRTF tool-call has just returned (post-return SP/PC save).
    "POST_PRTF_PC_LO": frozenset({"PRTF"}),
    "POST_PRTF_PC_HI": frozenset({"PRTF"}),
    "POST_PRTF_SP_LO": frozenset({"PRTF"}),
    "POST_PRTF_SP_HI": frozenset({"PRTF"}),
    # FORMAT_PTR_* — format-string pointer for PRTF (alias of AX_FULL_*).
    "FORMAT_PTR_LO": frozenset({"PRTF"}),
    "FORMAT_PTR_HI": frozenset({"PRTF"}),
    "FORMAT_PTR_LO_PIN": frozenset({"PRTF"}),
    "FORMAT_PTR_HI_PIN": frozenset({"PRTF"}),
    # MUL/DIV staging — alias of FETCH_LO/HI. Only used during MUL/DIV/MOD
    # compute phases.
    "MUL_ACCUM": frozenset({"MUL"}),
    "DIV_STAGING": frozenset({"DIV", "MOD"}),
    # MEM_STORE — set at MEM positions only during SI/SC/PSH stores.
    "MEM_STORE": frozenset({"SI", "SC", "PSH"}),
    # MEM_VAL_B0..3 — predicted memory value at LI/LC load.
    "MEM_VAL_B0": frozenset({"LI", "LC"}),
    "MEM_VAL_B1": frozenset({"LI", "LC"}),
    "MEM_VAL_B2": frozenset({"LI", "LC"}),
    "MEM_VAL_B3": frozenset({"LI", "LC"}),
    # PSH_AT_SP — PSH opcode flag relayed to SP/STACK0 positions.
    "PSH_AT_SP": frozenset({"PSH"}),
    # IO_OUTPUT_COUNT — counts output bytes remaining for PRTF/READ.
    "IO_OUTPUT_COUNT": frozenset({"PRTF", "READ"}),
    # IO_IS_PRTF / IO_IS_READ — opcode detection flags (alias of MEM_VAL_B3
    # / OP_LI_RELAY positions in the compact layout).
    "IO_IS_PRTF": frozenset({"PRTF"}),
    "IO_IS_READ": frozenset({"READ"}),
    # OP_LI_RELAY / OP_LC_RELAY — already encoded via opcode_in_step in
    # semantics; include here for symmetry.
    "OP_LI_RELAY": frozenset({"LI"}),
    "OP_LC_RELAY": frozenset({"LC"}),
    # CMP_GROUP — set at AX when any comparison opcode active.
    "CMP_GROUP": _CMP_OPCODES,
    # ADJ staging — already opcode_in_step-encoded; include for symmetry.
    "SP_OLD_LO": frozenset({"ADJ"}),
    "SP_OLD_HI": frozenset({"ADJ"}),
    "ADJ_CARRY": frozenset({"ADJ"}),
    # IO state-machine bookkeeping bits (alias MEM_VAL_B*/AX_FULL_HI tail);
    # these latch on the PRTF/READ thinking-loop transitions only.
    "LAST_WAS_IO_STATE_EMIT_BYTE": frozenset({"PRTF", "READ"}),
    "LAST_WAS_IO_STATE_EMIT_THINKING": frozenset({"PRTF", "READ"}),
    "LAST_WAS_THINKING_START": frozenset({"PRTF", "READ"}),
    "LAST_WAS_THINKING_END": frozenset({"PRTF", "READ"}),
    # ALU result + carry-forward staging — written at AX byte positions
    # by the ALU compute phase (ADD/SUB/MUL/DIV/MOD/OR/XOR/AND/SHL/SHR)
    # and read by CMP for the per-nibble comparison cascade. Dead under
    # non-ALU/non-CMP opcodes (notably PRTF/READ whose IO write-back
    # phase aliases AX_CARRY_* via POST_PRTF_SP_*). See FIXME notes on
    # ALU_LO/HI, AX_CARRY_LO/HI, CARRY in ``dim_registry.py``.
    "AX_CARRY_LO": _ALU_CMP_OPCODES,
    "AX_CARRY_HI": _ALU_CMP_OPCODES,
    "ALU_LO": _ALU_CMP_OPCODES,
    "ALU_HI": _ALU_CMP_OPCODES,
    "CARRY": _ALU_OPCODES,
    # CMP cascade (LT/EQ/GT/ZERO) — written at AX by the CMP family.
    # The L6 delayed-JMP cancel band reads ``CMP+0`` (positive) along
    # with MARK_PC at a PC marker row one step after the CMP fires; the
    # opcode pinned by the positive ``CMP+0`` reference is therefore the
    # CMP-family writer that latched the cascade byte. The cancel is
    # never fired under PRTF — the POST_PRTF_SP_* alias suppression
    # holds via disjointness with the CMP set.
    "CMP": _CMP_OPCODES,
}


# ---------------------------------------------------------------------------
# Improvement D (2026-06-10): phase_in_step disjointness
# ---------------------------------------------------------------------------
#
# Several alias pairs are NOT disambiguable via opcode_in_step (the previous
# atom) because at least one side is opcode-universal (every opcode performs
# a fetch). The textbook example is ``DIV_STAGING ↔ FETCH_LO/HI``:
#
# * DIV_STAGING owner set = {DIV, MOD} (EXEC phase, DIV/MOD only).
# * FETCH_LO/HI owner set = ALL opcodes (every opcode fetches its
#   instruction byte during the FETCH phase).
#
# Opcode-set intersection is non-empty (DIV/MOD ⊂ all), so opcode_in_step
# disjointness CAN'T fire. But the two slots are LIVE in different VM-step
# PHASES:
#
#   step phases (ordered): FETCH -> DECODE -> EXEC -> WRITEBACK
#
# A rule that fires during FETCH (gates/conditions a FETCH-phase slot like
# FETCH_HI/IMM_STAGING) cannot collide with an EXEC-phase scratch slot like
# DIV_STAGING / MUL_ACCUM. Phases are mutually disjoint by VM construction.
#
# This atom mirrors opcode_in_step exactly:
#
# 1. **Rule phase set** — derived from positive references in
#    ``conditions`` / ``gate_terms`` / ``gate`` to slots whose static phase
#    owner is known. ``FETCH_HI+N`` as a positive condition implies the
#    rule fires during the FETCH phase.
# 2. **Slot phase owner set** — taken from the static
#    ``_SLOT_PHASE_OWNERS`` table below (slots whose lifetime is genuinely
#    phase-scoped by VM-step construction).
#
# When BOTH sets are non-empty and DISJOINT, the rule cannot fire in any
# step-phase that touches the sibling's value — skip the violation.
# Conservative: empty / unknown sets yield NO suppression.

# Phase names used by the atom. Closed set; alphabetical for determinism.
# FETCH        — instruction-byte fetch (PC/byte_index=0..3 rows)
# DECODE       — opcode dispatch / immediate staging assembly
# EXEC         — opcode-specific compute (ALU scratch, address calc)
# WRITEBACK    — MEM stores, IO state-machine latches
_PHASE_NAMES = frozenset({"FETCH", "DECODE", "EXEC", "WRITEBACK"})

# Static map: slot name → set of phases that "own" the slot's value in any
# given step. Conservative — only include slots whose lifetime is genuinely
# phase-scoped. A slot omitted from this table contributes ``None`` (unknown)
# to the rule's phase set and never participates in suppression.
#
# Notes
# -----
# * FETCH_LO/HI / IMM_STAGING are fetched/staged during FETCH; they are
#   read by DECODE-phase dispatch rules so we include DECODE as well (a
#   rule that gates on FETCH_HI may fire in either FETCH or DECODE).
# * MEM_STORE is the SI/SC/PSH store wire — strictly WRITEBACK.
# * DIV_STAGING / MUL_ACCUM are ALU compute scratch — strictly EXEC.
# * IO_OUTPUT_COUNT and the IO state-machine latches fire on the
#   PRTF/READ tool-call return path — strictly WRITEBACK.
#
# Adding a slot here is the CHEAPEST cleanup. Removing a slot is safe
# (loses suppression power, never introduces false negatives).
_SLOT_PHASE_OWNERS: Dict[str, frozenset[str]] = {
    # FETCH-phase slots: instruction-byte fetch and immediate staging.
    # Conservative: a rule whose positive condition is a FETCH_* / IMM_*
    # reference fires during the FETCH/DECODE window; assigning {FETCH,
    # DECODE} keeps the rule phase set disjoint from EXEC/WRITEBACK
    # without artificially constraining DECODE-phase consumers.
    "FETCH_LO": frozenset({"FETCH", "DECODE"}),
    "FETCH_HI": frozenset({"FETCH", "DECODE"}),
    "IMM_STAGING": frozenset({"FETCH", "DECODE"}),
    # EXEC-phase scratch: ALU compute.
    "DIV_STAGING": frozenset({"EXEC"}),
    "MUL_ACCUM": frozenset({"EXEC"}),
    "ADJ_CARRY": frozenset({"EXEC"}),
    # WRITEBACK-phase slots: MEM stores and IO state-machine latches.
    "MEM_STORE": frozenset({"WRITEBACK"}),
    "IO_OUTPUT_COUNT": frozenset({"WRITEBACK"}),
}


# ---------------------------------------------------------------------------
# Improvement I (2026-06-10, scattered-sweep): "displaced ambient slot"
# ---------------------------------------------------------------------------
#
# Some "ambient" slots (AX_FULL_LO/HI, FORMAT_PTR_LO/HI, POST_PRTF_PC_LO/HI)
# carry the AX register's bytes by default but are *displaced* under specific
# opcodes by a different slot at the same byte range. The textbook case is
# AX_FULL_LO/HI bytes 471..503 during PRTF/READ sub-phases: at those rows
# the bytes carry FORMAT_PTR_* / POST_PRTF_PC_* values instead of AX. So a
# rule whose opcode set is contained in the displacer's opcode set ({PRTF,
# READ}) is reading the displacer's content, not AX_FULL's. The sibling
# AX_FULL_* is therefore NOT the actual content at the rule's firing
# position; the alias is design-time time-shared, not a real read bug.
#
# Map: ``slot_name -> set of opcodes that displace this slot``.
# When evaluating an alias check where the SIBLING is in this table and the
# rule's opcode set is a subset of the displacing-opcode set, suppress the
# violation: the sibling's value is not actually present at the rule's
# firing position.
_SLOT_DISPLACED_BY: Dict[str, frozenset[str]] = {
    # AX_FULL_{LO,HI} bytes 471..503 are displaced by FORMAT_PTR_* /
    # POST_PRTF_PC_* during PRTF state-machine sub-phases. At PRTF rows
    # the byte carries the displacer's value, not the AX register value.
    "AX_FULL_LO": frozenset({"PRTF", "READ"}),
    "AX_FULL_HI": frozenset({"PRTF", "READ"}),
    # FORMAT_PTR_{LO,HI} is displaced by POST_PRTF_PC_* in the post-
    # return sub-phase (same range, different PRTF sub-phase).
    "FORMAT_PTR_LO": frozenset({"PRTF", "READ"}),
    "FORMAT_PTR_HI": frozenset({"PRTF", "READ"}),
    # POST_PRTF_PC_{LO,HI} is displaced by FORMAT_PTR_* in the format-
    # string-fetch sub-phase. Symmetric with FORMAT_PTR_* above.
    "POST_PRTF_PC_LO": frozenset({"PRTF", "READ"}),
    "POST_PRTF_PC_HI": frozenset({"PRTF", "READ"}),
}


def _walk_atoms(p: Predicate, out: List[Atom]) -> None:
    """Append every leaf Atom node reachable from ``p`` into ``out``."""
    if isinstance(p, Atom):
        out.append(p)
        return
    if isinstance(p, And) or isinstance(p, Or):
        for child in p.children:
            _walk_atoms(child, out)
        return
    # Not: recurse into the child (we only need positive atom mentions
    # for opcode extraction; negated opcode atoms don't pin a single
    # opcode and are skipped by the caller).
    inner = getattr(p, "child", None)
    if inner is not None:
        _walk_atoms(inner, out)


def _opcodes_from_semantics(
    sem: Optional[str],
    cache: Dict[str, Optional[frozenset[str]]],
) -> Optional[frozenset[str]]:
    """Extract the opcode-in-step set from a slot's parsed semantics.

    Walks the semantics AST looking for positive ``opcode_at_AX == X``,
    ``opcode_at_AX in {...}``, and ``opcode_in_step in {...}`` atoms.
    If multiple appear, returns their UNION (the slot is alive under any
    of those opcodes; the disjointness check below stays conservative).

    Returns:
      * ``None`` if no opcode atom is found (semantics is opcode-agnostic
        — the caller should fall back to the static owner table).
      * A frozenset of opcode names (e.g. ``{"ADD"}``, ``{"DIV", "MOD"}``)
        when at least one opcode atom is present.

    Cached per-``sem`` string so the production registry's ~200 slots
    are parsed once. Parse errors degrade to ``None``.
    """
    if sem is None:
        return None
    if sem in cache:
        return cache[sem]
    try:
        pred = parse(sem)
    except Exception:
        cache[sem] = None
        return None
    atoms: List[Atom] = []
    _walk_atoms(pred, atoms)
    found: set[str] = set()
    saw_opcode_atom = False
    for a in atoms:
        if isinstance(a, OpcodeAtAxEq):
            found.add(a.opcode)
            saw_opcode_atom = True
        elif isinstance(a, OpcodeAtAxIn):
            found.update(a.opcodes)
            saw_opcode_atom = True
        elif isinstance(a, OpcodeInStepIn):
            found.update(a.opcodes)
            saw_opcode_atom = True
    result = frozenset(found) if saw_opcode_atom else None
    cache[sem] = result
    return result


def _slot_opcode_in_step_set(
    registry: DimRegistry,
    name: str,
    semantics_cache: Dict[str, Optional[frozenset[str]]],
) -> Optional[frozenset[str]]:
    """Best-effort opcode set for slot ``name``.

    Tries the static ``_SLOT_OPCODE_OWNERS`` table first (covers slots
    whose semantics is opcode-agnostic by historical convention), then
    falls back to extracting opcode atoms from the slot's semantics.

    Returns ``None`` when nothing is known — the disjointness check
    skips the suppression and the violation stands.
    """
    owners = _SLOT_OPCODE_OWNERS.get(name)
    if owners:
        return owners
    sem = _slot_semantics(registry, name)
    return _opcodes_from_semantics(sem, semantics_cache)


# Known opcode mnemonics (mirrors ``_OPCODE_ROLES`` in dim_registry.py).
# Kept inline so the verifier doesn't reach into the registry-builder's
# private symbol just for a literal fallback list. Adding an opcode here
# is safe — the only effect is that a positive ``OP_<X>`` reference with
# an opcode-agnostic semantics string will resolve to the matching
# opcode role.
_ALL_OPCODE_ROLES: frozenset[str] = frozenset({
    "LEA", "IMM", "JMP", "JSR", "BZ", "BNZ", "ENT", "ADJ", "LEV",
    "LI", "LC", "SI", "SC", "PSH", "OR", "XOR", "AND",
    "EQ", "NE", "LT", "GT", "LE", "GE",
    "SHL", "SHR", "ADD", "SUB", "MUL", "DIV", "MOD",
    "EXIT", "NOP", "PUTCHAR", "GETCHAR",
    # IO tool-call opcodes used by POST_PRTF_*/IO_*/MEM_VAL_* owners.
    "PRTF", "READ",
})


def _op_x_opcode_from_dim(
    dim_name: str,
    registry: DimRegistry,
    semantics_cache: Dict[str, Optional[frozenset[str]]],
) -> Optional[frozenset[str]]:
    """Recover the opcode role for an ``OP_<X>`` dim reference.

    Tries the slot's semantics first (the standard
    ``mark == AX AND opcode_at_AX == X`` form parses to ``{X}``); when
    the registry uses the DSL-keyword fallback (``OR``/``AND``/``NOT``
    cannot appear as opcode atoms in the predicate grammar so OP_OR/
    OP_AND/OP_NOT carry a position-only semantics string), recovers the
    opcode role from the dim-name suffix.

    Returns ``None`` if the dim is not named ``OP_<role>`` for a known
    role.
    """
    if not dim_name.startswith("OP_"):
        return None
    sem = _slot_semantics(registry, dim_name)
    opcodes = _opcodes_from_semantics(sem, semantics_cache)
    if opcodes:
        return opcodes
    suffix = dim_name[3:]
    if suffix in _ALL_OPCODE_ROLES:
        return frozenset({suffix})
    return None


def _rule_opcode_in_step_set(
    rule: FFNRule,
    registry: DimRegistry,
    semantics_cache: Dict[str, Optional[frozenset[str]]],
) -> Optional[frozenset[str]]:
    """Derive the rule's opcode-in-step constraint from its positive
    condition / gate_term / gate references.

    The intuition: a rule that lists ``("OP_ADD+0", 1.0)`` as a positive
    condition only fires when ``opcode_at_AX == ADD`` (the registry
    semantics of OP_ADD); since each step has a single active opcode at
    the AX marker, this pins ``opcode_in_step == ADD``.

    Two-phase derivation:

      * **Phase 1 (OP_<X> UNION)** — collect opcode roles named by any
        positive ``OP_<X>`` reference. Multiple positive ``OP_<X>``
        references widen the set (a rule that lists both ``OP_ADD`` and
        ``OP_SUB`` is design-time alive under either). This matches the
        original (pre-Improvement D) behavior. Recovery covers both the
        standard ``opcode_at_AX == X`` semantics form AND the DSL-keyword
        literal fallback (OP_OR/OP_AND/OP_NOT — Improvement D).

      * **Phase 2 (non-OP intersection)** — only runs when Phase 1
        produced an empty set (the rule carries NO positive OP_<X>
        reference). Walks the same positive condition / gate_term / gate
        list and INTERSECTS the opcode-owner sets pulled from
        ``_SLOT_OPCODE_OWNERS`` / slot semantics. INTERSECTION is the
        correct combinator for AND'd conditions: a rule that needs BOTH
        ``LAST_WAS_THINKING_START`` (owners ``{PRTF, READ}``) AND a
        ``POST_PRTF_SP_LO`` gate (owners ``{PRTF}``) fires only when the
        active opcode is in ``{PRTF, READ} ∩ {PRTF} = {PRTF}``. Examples
        recovered by Phase 2 (Improvement E):
          - L9 CMP rules gated on ``CMP_GROUP`` recover ``{CMP_OPCODES}``.
          - convo-IO PC/SP latch rules recover ``{PRTF}``.
          - L6 delayed-JMP cancel rules recover ``{CMP_OPCODES}``.

    Phase 2 ONLY runs when Phase 1 yielded nothing. This preserves the
    original behavior for rules that already had OP_<X> refs: their
    opcode set is unchanged, so the historical DIV/MUL_ACCUM/FETCH_*
    suppression matches stay stable. The intersection is over-cautious
    when a positive non-OP reference is a weak booster (not necessary
    for firing), but the failure mode is "we don't narrow enough" which
    means MORE violations stand — conservative.

    Positive references are collected from:
      * ``rule.conditions`` with positive weight,
      * ``rule.gate_terms`` with positive weight,
      * ``rule.gate`` (always positive when present).

    Returns ``None`` when neither phase yields any opcode — the rule's
    opcode context is unknown and disjointness can't be asserted.
    """
    # Collect positive references once (shared by both phases).
    positive_refs: List[str] = []
    for term in rule.conditions:
        if term.weight > 0:
            positive_refs.append(term.dim.name)
    for term in rule.gate_terms:
        if term.weight > 0:
            positive_refs.append(term.dim.name)
    if rule.gate is not None:
        positive_refs.append(rule.gate.name)

    # --- Phase 1: OP_<X> UNION. ----------------------------------------
    op_x_set: set[str] = set()
    for dim_name in positive_refs:
        opcodes = _op_x_opcode_from_dim(dim_name, registry, semantics_cache)
        if opcodes:
            op_x_set.update(opcodes)
    if op_x_set:
        return frozenset(op_x_set)

    # --- Phase 2: non-OP positive INTERSECTION. -----------------------
    # Walk the same positive refs; intersect opcode-owner sets pulled
    # from _SLOT_OPCODE_OWNERS or slot semantics. Intersection assumes
    # the rule's threshold needs every positive contributor active in
    # the same step — an over-cautious model for rules whose weights
    # encode an OR'd group, but never widens the rule's set so disjoint-
    # ness suppressions never fire spuriously.
    intersect: Optional[set[str]] = None
    for dim_name in positive_refs:
        # Skip OP_<X> — already handled in Phase 1 and gave nothing.
        if dim_name.startswith("OP_"):
            continue
        opcodes = _slot_opcode_in_step_set(
            registry, dim_name, semantics_cache,
        )
        if not opcodes:
            continue
        if intersect is None:
            intersect = set(opcodes)
        else:
            intersect &= opcodes
        if not intersect:
            # Empty intersection means the rule's positive conditions
            # demand mutually exclusive opcodes — i.e. it cannot fire
            # under any single opcode. Treat as "unknown" rather than
            # claim a vacuously-disjoint set; the conservative choice
            # keeps the violation visible.
            return None
    if intersect:
        return frozenset(intersect)
    return None


def _slot_phase_in_step_set(
    name: str,
) -> Optional[frozenset[str]]:
    """Best-effort phase set for slot ``name``.

    Only consults the static ``_SLOT_PHASE_OWNERS`` table — there is no
    phase atom in the predicate DSL, so semantics-based extraction is not
    available. Returns ``None`` when the slot is not phase-scoped (the
    disjointness check skips the suppression and the violation stands).
    """
    return _SLOT_PHASE_OWNERS.get(name)


def _rule_phase_in_step_set(rule: FFNRule) -> Optional[frozenset[str]]:
    """Derive the rule's phase-in-step constraint from its positive
    references to phase-scoped slots.

    The intuition mirrors :func:`_rule_opcode_in_step_set`: a rule that
    lists ``("FETCH_HI+5", 1.0)`` as a positive condition only fires
    during the FETCH-phase window in which FETCH_HI is live, so the
    rule's phase set is FETCH_HI's owner set ``{FETCH, DECODE}``.
    Multiple positive references intersect (the rule fires only when
    every cited slot is live), so the rule phase set is the
    intersection of each referenced slot's phase owners.

    Positive references are collected from:
      * ``rule.conditions`` with positive weight,
      * ``rule.gate_terms`` with positive weight,
      * ``rule.gate`` (always positive when present).

    Returns ``None`` if no positive reference targets a phase-owned slot
    — the rule's phase context is unknown and disjointness can't be
    asserted. Returns the intersection of owner sets otherwise (empty
    intersection means the rule cites mutually-exclusive phase slots,
    which already entails ``satisfiable`` would be False; treated as
    unknown for safety).
    """
    sets: List[frozenset[str]] = []

    def _add_from_dim_name(dim_name: str) -> None:
        owners = _SLOT_PHASE_OWNERS.get(dim_name)
        if owners:
            sets.append(owners)

    for term in rule.conditions:
        if term.weight > 0:
            _add_from_dim_name(term.dim.name)
    for term in rule.gate_terms:
        if term.weight > 0:
            _add_from_dim_name(term.dim.name)
    if rule.gate is not None:
        _add_from_dim_name(rule.gate.name)

    if not sets:
        return None
    # Intersection across all cited phase-scoped references.
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    if not out:
        return None
    return frozenset(out)


def _slot_semantics_is_tautology(
    registry: DimRegistry,
    name: str,
    cache: Dict[str, bool],
) -> bool:
    """Return True iff ``registry``'s slot ``name`` carries a
    tautological semantics string (always true, e.g. the production
    umbrella ``is_byte OR NOT is_byte`` declared on TEMP and friends).

    Cached per-name across the verifier run so repeated lookups for
    the same alias parent (TEMP shows up as a sibling of ~30 slots) are
    O(1). Parse / solver errors degrade to False (treat as
    non-tautological) so a malformed semantics does NOT silently
    suppress a real alias.
    """
    cached = cache.get(name)
    if cached is not None:
        return cached
    sem = _slot_semantics(registry, name)
    if sem is None:
        cache[name] = False
        return False
    try:
        cache[name] = is_tautology(parse(sem))
    except Exception:
        cache[name] = False
    return cache[name]


def verify_dim_aliases(
    op,
    registry: DimRegistry,
    *,
    alias_index: Optional[Dict[str, List[str]]] = None,
    skip_colocated_subbank: bool = True,
    skip_tautological_siblings: bool = True,
    skip_opcode_in_step_disjoint: bool = True,
    skip_phase_in_step_disjoint: bool = True,
    tautology_cache: Optional[Dict[str, bool]] = None,
    opcode_semantics_cache: Optional[
        Dict[str, Optional[frozenset[str]]]
    ] = None,
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
    if tautology_cache is None:
        tautology_cache = {}
    if opcode_semantics_cache is None:
        opcode_semantics_cache = {}

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

        # Improvement C (2026-06-10): derive the rule's opcode_in_step
        # constraint set from positive OP_<X> condition/gate references.
        # Cached per (rule_id) for the inner sibling loop. ``None`` =
        # unknown; never used to suppress.
        rule_opcode_set: Optional[frozenset[str]] = None
        if skip_opcode_in_step_disjoint:
            rule_opcode_set = _rule_opcode_in_step_set(
                rule, registry, opcode_semantics_cache,
            )

        # Improvement D (2026-06-10): derive the rule's phase_in_step
        # constraint set from positive references to phase-scoped slots
        # (FETCH_LO/HI, IMM_STAGING, DIV_STAGING, MUL_ACCUM, MEM_STORE,
        # IO_OUTPUT_COUNT, ADJ_CARRY). Cached per (rule_id) for the inner
        # sibling loop. ``None`` = unknown; never used to suppress.
        rule_phase_set: Optional[frozenset[str]] = None
        if skip_phase_in_step_disjoint:
            rule_phase_set = _rule_phase_in_step_set(rule)

        read_dims = _collect_read_dim_refs(rule)
        for dim_name, dim_offset in read_dims:
            siblings = alias_index.get(dim_name, [])
            if not siblings:
                continue
            # Improvement F (2026-06-10, scattered-sweep): physical-byte
            # filter. The alias index is slot-level (FETCH_HI [436..452)
            # overlaps IMM_STAGING [448..464) on bytes 448..452 alone),
            # but a rule reading ``FETCH_HI+0`` lands at physical byte
            # 436, which is OUTSIDE IMM_STAGING's range — no possible
            # alias collision. Skip any sibling whose byte range does
            # not include the rule's physical read byte.
            read_slot = registry.slots.get(dim_name)
            read_byte = (
                read_slot.start + dim_offset if read_slot is not None
                else None
            )
            # Improvement A (2026-06-10): skip the entire alias check
            # when EITHER the slot being read or its sibling carries a
            # tautological semantics (e.g. TEMP's umbrella
            # ``is_byte OR NOT is_byte``). A tautology declares the
            # slot "ambient" — no positional constraint is asserted at
            # the registry level, so every overlap report is structural
            # noise rather than a real read bug.
            read_dim_taut = skip_tautological_siblings and \
                _slot_semantics_is_tautology(
                    registry, dim_name, tautology_cache,
                )
            for sibling in siblings:
                if read_dim_taut:
                    continue
                if skip_tautological_siblings and \
                        _slot_semantics_is_tautology(
                            registry, sibling, tautology_cache,
                        ):
                    continue
                # Improvement F: per-byte alias filter. A sibling that
                # does not cover the rule's actual physical read byte
                # cannot supply the aliased value at the firing position.
                if read_byte is not None:
                    sib_slot = registry.slots.get(sibling)
                    if sib_slot is not None and not (
                        sib_slot.start <= read_byte
                        < sib_slot.start + sib_slot.size
                    ):
                        continue
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
                # Improvement C (2026-06-10): opcode_in_step disjointness.
                # If the rule's opcode set (derived from positive OP_<X>
                # conditions/gate) and the sibling slot's opcode owner
                # set (from semantics or the static table) are BOTH
                # non-empty and DISJOINT, the rule cannot fire in any
                # step that touches the sibling's value — the alias is
                # design-time time-shared, not a real bug.
                if (
                    skip_opcode_in_step_disjoint
                    and rule_opcode_set is not None
                    and rule_opcode_set
                ):
                    sibling_opcode_set = _slot_opcode_in_step_set(
                        registry, sibling, opcode_semantics_cache,
                    )
                    if (
                        sibling_opcode_set is not None
                        and sibling_opcode_set
                        and rule_opcode_set.isdisjoint(sibling_opcode_set)
                    ):
                        continue
                # Improvement D (2026-06-10): phase_in_step disjointness.
                # If the rule's phase set (derived from positive
                # references to phase-scoped slots) and the sibling slot's
                # phase owner set (from the static table) are BOTH
                # non-empty and DISJOINT, the rule fires in a step phase
                # that does not touch the sibling's value — the alias is
                # design-time phase-shared, not a real bug. Catches the
                # DIV_STAGING ↔ FETCH_LO/HI family where opcode_in_step
                # cannot fire (FETCH is opcode-universal).
                if (
                    skip_phase_in_step_disjoint
                    and rule_phase_set is not None
                    and rule_phase_set
                ):
                    sibling_phase_set = _slot_phase_in_step_set(sibling)
                    if (
                        sibling_phase_set is not None
                        and sibling_phase_set
                        and rule_phase_set.isdisjoint(sibling_phase_set)
                    ):
                        continue
                # Improvement I (2026-06-10, scattered-sweep): "displaced
                # ambient slot". If the SIBLING is in the
                # ``_SLOT_DISPLACED_BY`` table and the rule's opcode set
                # is contained in the displacing-opcode set, the
                # sibling's content is overlaid by a different slot at
                # the rule's firing position — the named sibling is NOT
                # the actual byte content here. Suppress.
                if (
                    skip_opcode_in_step_disjoint
                    and rule_opcode_set is not None
                    and rule_opcode_set
                ):
                    displacer = _SLOT_DISPLACED_BY.get(sibling)
                    if (
                        displacer is not None
                        and rule_opcode_set.issubset(displacer)
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
    skip_tautological_siblings: bool = True,
    skip_opcode_in_step_disjoint: bool = True,
    skip_phase_in_step_disjoint: bool = True,
) -> List[AliasViolation]:
    """Run :func:`verify_dim_aliases` over an iterable of ops, sharing
    the alias index across calls.

    De-duplicates violations on ``(op_name, rule_name, read_dim,
    read_offset, conflicting_alias)`` so the same rule appearing in
    multiple ops does not flood the report.
    """
    alias_index = _build_alias_index(registry)
    tautology_cache: Dict[str, bool] = {}
    opcode_semantics_cache: Dict[str, Optional[frozenset[str]]] = {}
    seen: set = set()
    out: List[AliasViolation] = []
    for op in ops:
        for v in verify_dim_aliases(
            op, registry, alias_index=alias_index,
            skip_colocated_subbank=skip_colocated_subbank,
            skip_tautological_siblings=skip_tautological_siblings,
            skip_opcode_in_step_disjoint=skip_opcode_in_step_disjoint,
            skip_phase_in_step_disjoint=skip_phase_in_step_disjoint,
            tautology_cache=tautology_cache,
            opcode_semantics_cache=opcode_semantics_cache,
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
    *reads as a positive activation signal* (conditions, gate_terms,
    and gate).

    Improvement G (2026-06-10, scattered-sweep): NEGATIVE-weight
    condition / gate_term references are blockers, not value reads —
    they can only push the sum BELOW the threshold (suppressing
    firing), never above it. If the slot's value at the rule's firing
    position is actually the aliased sibling's, a negative-weight read
    still produces a suppressive contribution (or zero), so the rule
    cannot mis-fire on the wrong value. Skip them.

    Example: ``layer14_clear_addr_key_pollution`` lists
    ``("MEM_VAL_B3", -100.0)`` as a blocker — at PRTF AX rows where the
    slot carries ``IO_IS_PRTF`` instead, the rule still conservatively
    under-fires (no incorrect activation).
    """
    seen: set = set()
    out: List[Tuple[str, int]] = []
    for term in rule.conditions:
        if term.weight <= 0:
            continue
        key = (term.dim.name, term.dim.offset)
        if key not in seen:
            seen.add(key)
            out.append(key)
    for term in rule.gate_terms:
        if term.weight <= 0:
            continue
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
