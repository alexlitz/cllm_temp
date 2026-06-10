"""STEP_END migration scaffolding (Wave B helper).

Once the Wave A ``step_end_operand_relay`` head broadcasts
``OP_<NAME>`` / ``AX_CARRY_*`` / ``ALU_*`` / ``CMP`` from ``MARK_AX``
into ``MARK_SE`` (see ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md``
section 4), the ~15-20 compute rules listed in Wave B can be migrated
from MARK_AX gating to MARK_SE gating. The semantics are unchanged:
the same SwiGLU AND fires, just one row later, after every byte and
inter-cycle relay has settled.

This module provides three helpers so each migration is a 5-line
commit instead of a hand-rewritten rule:

* :func:`migrate_rule_to_step_end` -- copy an :class:`FFNRule` with
  every ``MARK_AX`` condition (and any condition listed in
  ``relayed_dims``) rewritten to its ``MARK_SE_ONLY`` analogue.
* :func:`migrate_attention_head_to_step_end` -- same idea for a
  :class:`DeclarativeAttentionHeadSpec`: rewrite Q/K projection
  writes that key on ``MARK_AX`` to key on ``MARK_SE_ONLY``.
* :func:`assert_migration_safe` -- quick structural check that the
  rule's writes do NOT target byte-emission slots (``OUTPUT_LO`` /
  ``OUTPUT_HI``) or per-row marker dims that are valid only at
  ``MARK_AX``. Raises :class:`MigrationSafetyError` if the rule
  cannot be safely migrated.

The helpers are pure: they operate on declarative IR dataclasses
and never touch model weights. The migration template
(``docs/STEP_END_MIGRATION_TEMPLATE.md``) documents the surrounding
recipe.

See ``tests/test_step_end_migration.py`` for executable examples.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Optional, Tuple

from .ir import ConditionTerm, DimRef, FFNRule, WriteTerm
from .primitives import (
    AttentionProjectionWrite,
    DeclarativeAttentionHeadSpec,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Source marker that ``MARK_AX``-gated compute fires under today.
MARK_AX_NAME: str = "MARK_AX"

#: Destination marker after migration. ``MARK_SE_ONLY`` is the canonical
#: STEP_END-only dim (set at row 34 of every 35-token VM step, NOT at
#: ``HAS_SE`` rows broadcast across the whole step). See
#: ``dim_registry_dynamic.py`` for the pinned slot.
MARK_SE_NAME: str = "MARK_SE_ONLY"

#: Output-slot prefixes that MUST land at byte rows (1-4, 6-9, 11-14,
#: ...). A rule writing into one of these cannot be migrated to
#: STEP_END (its byte would never reach the per-byte residual cell).
#: See ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md`` section 4
#: "Wave C - Rules that STAY at MARK_AX".
_BYTE_EMISSION_PREFIXES: Tuple[str, ...] = (
    "OUTPUT_LO",
    "OUTPUT_HI",
    "OUTPUT_HI_THIS_STEP",
    "OUTPUT_HI_PREV_STEP",
)

#: Per-row marker dims that physically only exist at MARK_AX (or other
#: register rows). A rule whose write lands here is doing register
#: bookkeeping at MARK_AX and cannot be moved to STEP_END.
_AX_ROW_MARKER_DIMS: Tuple[str, ...] = (
    "MARK_AX",
    "MARK_PC",
    "MARK_SP",
    "MARK_BP",
    "MARK_MEM",
    "MARK_STACK0",
)


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class MigrationSafetyError(ValueError):
    """Raised when an FFN rule cannot be safely migrated to STEP_END.

    The message lists which write(s) blocked the migration so the
    caller can either (a) reclassify the rule as Wave C "stays at
    MARK_AX" or (b) split it into a STEP_END part + a byte-row part.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rewrite_condition(term: ConditionTerm) -> ConditionTerm:
    """Return a copy of ``term`` with ``MARK_AX`` rewritten to
    ``MARK_SE_ONLY``. Other conditions are unchanged.
    """

    dim = term.dim
    if dim.name == MARK_AX_NAME:
        new_dim = DimRef(name=MARK_SE_NAME, offset=dim.offset)
        return dataclasses.replace(term, dim=new_dim)
    return term


def _rewrite_scope(scope: Optional[str]) -> Optional[str]:
    """Update predicate-DSL ``scope`` strings to reference
    ``MARK_SE_ONLY`` where they previously referenced ``MARK_AX``.

    The scope language is a simple AND/NOT of dim names; word-boundary
    substitution is faithful as long as the dim names are word-isolated
    (never matches inside ``MARK_AX_CARRY`` or similar).
    """

    if scope is None:
        return None
    return re.sub(r"\bMARK_AX\b", MARK_SE_NAME, scope)


def _names_of_writes(writes: Iterable[WriteTerm]) -> Tuple[str, ...]:
    return tuple(w.dim.name for w in writes)


# ---------------------------------------------------------------------------
# FFN rule migration
# ---------------------------------------------------------------------------


def assert_migration_safe(
    rule: FFNRule,
    *,
    allow_step_end_writes_to: Iterable[str] = (),
) -> None:
    """Validate that ``rule`` can be safely migrated to STEP_END.

    The check is structural and conservative:

    * The rule must already gate on ``MARK_AX`` in at least one
      condition (otherwise migration is a no-op or a category error).
    * No write may target a byte-emission slot
      (``OUTPUT_LO`` / ``OUTPUT_HI*``). Those slots are read only at
      per-byte rows; writing them at STEP_END drops the byte.
    * No write may target a per-row register marker dim
      (``MARK_AX``, ``MARK_PC``, ``MARK_SP``, ``MARK_BP``,
      ``MARK_MEM``, ``MARK_STACK0``). Those are row-local.

    Callers can extend ``allow_step_end_writes_to`` to whitelist
    specific output names that look byte-shaped but are actually
    safe (this should be very rare and well-justified).

    Raises :class:`MigrationSafetyError` on the first failure. The
    function is otherwise side-effect free.
    """

    has_mark_ax = any(
        term.dim.name == MARK_AX_NAME for term in rule.conditions
    )
    if not has_mark_ax:
        raise MigrationSafetyError(
            f"rule {rule.name!r}: no MARK_AX condition to migrate"
        )

    allow = frozenset(allow_step_end_writes_to)
    bad_byte_writes = []
    bad_marker_writes = []
    for write in rule.writes:
        name = write.dim.name
        if name in allow:
            continue
        if name in _BYTE_EMISSION_PREFIXES:
            bad_byte_writes.append(name)
        if name in _AX_ROW_MARKER_DIMS:
            bad_marker_writes.append(name)
    if bad_byte_writes:
        raise MigrationSafetyError(
            f"rule {rule.name!r}: writes byte-emission slot(s) "
            f"{bad_byte_writes!r}; cannot fire at STEP_END "
            "(byte rows have already passed). Classify as Wave C "
            "or split into per-byte and STEP_END parts."
        )
    if bad_marker_writes:
        raise MigrationSafetyError(
            f"rule {rule.name!r}: writes row-local marker dim(s) "
            f"{bad_marker_writes!r}; these only exist at the "
            "register's own row."
        )


def migrate_rule_to_step_end(
    rule: FFNRule,
    *,
    relayed_dims: Iterable[str] = (),
    safety_check: bool = True,
    allow_step_end_writes_to: Iterable[str] = (),
) -> FFNRule:
    """Return a copy of ``rule`` gated on ``MARK_SE_ONLY`` instead of
    ``MARK_AX``.

    Parameters
    ----------
    rule:
        Source FFN rule whose conditions include at least one
        ``MARK_AX`` term. Writes / threshold / gate are carried over
        unchanged.
    relayed_dims:
        Names of operand dims that the Wave A relay broadcasts from
        MARK_AX to MARK_SE (e.g. ``OP_LEV``, ``AX_CARRY_LO``,
        ``ALU_HI``, ``CMP``). Listed for documentation and future
        validators; this helper does not currently transform them
        (they already resolve to the right cell at MARK_SE under
        the relay).
    safety_check:
        If True (default), run :func:`assert_migration_safe` first
        and propagate any :class:`MigrationSafetyError`.
    allow_step_end_writes_to:
        Forwarded to :func:`assert_migration_safe`.

    Returns
    -------
    FFNRule
        A new rule with every ``MARK_AX`` condition rewritten and the
        ``scope`` annotation refreshed. ``name`` is augmented with a
        ``_step_end`` suffix so the verifier can tell the rules apart.
    """

    if safety_check:
        assert_migration_safe(
            rule,
            allow_step_end_writes_to=allow_step_end_writes_to,
        )

    # ``relayed_dims`` is accepted (and stored on the closure) for
    # future-validator use. The current rewrite does not need to
    # transform operand dims because the Wave A relay keeps their
    # symbolic names stable across the AX -> SE row boundary.
    _ = frozenset(relayed_dims)

    new_conditions = tuple(_rewrite_condition(c) for c in rule.conditions)
    new_gate_terms = tuple(_rewrite_condition(c) for c in rule.gate_terms)

    new_name: Optional[str]
    if rule.name is not None and not rule.name.endswith("_step_end"):
        new_name = f"{rule.name}_step_end"
    else:
        new_name = rule.name

    new_scope = _rewrite_scope(rule.scope)
    new_dominates_at = (
        None
        if rule.dominates_at is None
        else {k: _rewrite_scope(v) for k, v in rule.dominates_at.items()}
    )

    return dataclasses.replace(
        rule,
        conditions=new_conditions,
        gate_terms=new_gate_terms,
        name=new_name,
        scope=new_scope,
        dominates_at=new_dominates_at,
    )


# ---------------------------------------------------------------------------
# Attention head migration
# ---------------------------------------------------------------------------


def _rewrite_projection_write(
    write: AttentionProjectionWrite,
    *,
    mark_ax_dim_idx: int,
    mark_se_dim_idx: int,
) -> AttentionProjectionWrite:
    if write.dim == mark_ax_dim_idx:
        return dataclasses.replace(write, dim=mark_se_dim_idx)
    return write


def migrate_attention_head_to_step_end(
    spec: DeclarativeAttentionHeadSpec,
    *,
    mark_ax_dim_idx: int,
    mark_se_dim_idx: int,
) -> DeclarativeAttentionHeadSpec:
    """Return a copy of ``spec`` with every Q/K projection write
    that lands on the MARK_AX residual cell rewritten to land on the
    ``MARK_SE_ONLY`` cell.

    Unlike :class:`FFNRule`, attention specs reference dims by integer
    index (already resolved by the allocator) so the caller must
    supply both indices.

    Q and K writes are the natural candidates: Q anchors the row the
    head reads "from" and K anchors the row it attends "to". V and O
    are passed through unchanged -- those project operand cells, not
    marker cells. If a real head needs V/O rewriting, do that step
    explicitly; the helper does NOT touch them.

    The ``head_idx`` / ``alibi_slope`` / ``group_size`` / ``head_dim``
    fields are passed through verbatim.
    """

    new_q = tuple(
        _rewrite_projection_write(
            w,
            mark_ax_dim_idx=mark_ax_dim_idx,
            mark_se_dim_idx=mark_se_dim_idx,
        )
        for w in spec.q
    )
    new_k = tuple(
        _rewrite_projection_write(
            w,
            mark_ax_dim_idx=mark_ax_dim_idx,
            mark_se_dim_idx=mark_se_dim_idx,
        )
        for w in spec.k
    )

    return dataclasses.replace(spec, q=new_q, k=new_k)
