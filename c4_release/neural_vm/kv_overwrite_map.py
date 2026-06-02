"""Phase 8.E.2 — Declarative IR overwrite map.

This module walks the declarative IR (``FFNRule.constant_write`` /
``FFNRule.gated_write`` / ``AttentionHeadIR``) and produces a per-step
``(position, dim_name) -> overwrite_step`` map. The runtime uses the map
at each step boundary to evict KV-cache entries whose value at
``position`` has been semantically overwritten by a later step's write
to the same dim.

The map is **static**: it is precomputed from the IR at compile time,
not from any runtime tensor. Per-step determinism is therefore
automatic — the spec-decode and main-decode paths reach the same
eviction decisions when given the same step index.

Categories (see Phase 8 plan §8.E.1)
-----------------------------------

Each ``(position, dim)`` is tagged with one :class:`OverwriteCategory`
that records *why* the entry is dead at its overwrite step. Consumers
(e.g. ``build_state_from_report``) can filter by category when
constructing the runtime eviction state — e.g. enable only the
byte-identity-safe categories at first.

  * :attr:`OverwriteCategory.REGISTER`
      Register channels (``REG_*``, ``PC_*``, ``AX_*``, ``SP_*``,
      ``BP_*``, ``STACK0_*``, ``MARK_*``, ``OP_*``, ``ADDR_*``,
      ``ALU_*``, ``FETCH_*``, ``BYTE_INDEX_*``, ``EMBED_*``). A later
      step's writer clobbers the same residual cell.
  * :attr:`OverwriteCategory.MEM_CELL`
      Memory cell channels (``MEM_*``). A later step's MEM-write to
      the same dim supersedes the cached entry.
  * :attr:`OverwriteCategory.OUTPUT_SLOT`
      Output decoding slots (``OUTPUT_LO``, ``OUTPUT_HI``,
      ``OUTPUT_*``). A later step's rewrite of the same output nibble
      supersedes the cached entry.
  * :attr:`OverwriteCategory.TRANSIENT_SCRATCH`
      Per-step scratch (``TEMP*``, ``*_SCRATCH``, ``*_THIS_STEP``,
      ``AX_FULL_*``, ``ALU_TEMP*``, ``MUL_TEMP*``, ``DIV_TEMP*``, ...).
      Step-local by construction; the entry is dead at end-of-step
      regardless of whether any later writer touches the dim.
  * :attr:`OverwriteCategory.PREV_STEP`
      Cross-step relay (``*_PREV_STEP``, ``*_PREV``, ``*_LAST_STEP``).
      Written at step S, consumed at step S+1 only. Dead at step S+2.
  * :attr:`OverwriteCategory.NON_PERSISTING`
      Positions whose only writers are transient intermediates with no
      cross-step reader. Dead at end-of-step.

Per the plan, this module is the FOUNDATION for 8.E correctness; the
runtime uses ``build_overwrite_map`` to derive the static eviction
schedule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class OverwriteCategory(str, Enum):
    """Why a ``(position, dim)`` entry is dead at its overwrite step."""

    REGISTER = "register"
    MEM_CELL = "mem_cell"
    OUTPUT_SLOT = "output_slot"
    TRANSIENT_SCRATCH = "transient_scratch"
    PREV_STEP = "prev_step"
    NON_PERSISTING = "non_persisting"
    UNKNOWN = "unknown"


# Dim-name prefix/suffix tables. These are kept in sync with the
# liveness analyzer's tables so the two modules categorise the same
# dims identically.
_REGISTER_PREFIXES: Tuple[str, ...] = (
    "REG_",
    "PC_",
    "AX_CARRY",
    "AX_BYTE",
    "AX_DIGIT",
    "AX_MARK",
    "SP_",
    "BP_",
    "STACK0",
    "MARK_",
    "OP_",
    "ADDR_",
    "ALU_",
    "FETCH_",
    "BYTE_INDEX_",
    "EMBED_",
    "DISPATCH_",
    "DECODE_",
    "OPCODE_",
)
_REGISTER_EXACT: Tuple[str, ...] = (
    "PC",
    "AX",
    "SP",
    "BP",
    "REG_PC",
    "REG_AX",
    "REG_SP",
    "REG_BP",
)

_MEM_CELL_PREFIXES: Tuple[str, ...] = ("MEM_",)

_OUTPUT_SLOT_PREFIXES: Tuple[str, ...] = ("OUTPUT_LO", "OUTPUT_HI", "OUTPUT_")

_TRANSIENT_SCRATCH_PREFIXES: Tuple[str, ...] = (
    "TEMP",
    "ALU_TEMP",
    "MUL_TEMP",
    "DIV_TEMP",
    "MUL_ACCUM",
    "DIV_STAGING",
    "MEM_STAGING",
    "AX_FULL",
)
_TRANSIENT_SCRATCH_SUFFIXES: Tuple[str, ...] = (
    "_THIS_STEP",
    "_SCRATCH",
)

_PREV_STEP_SUFFIXES: Tuple[str, ...] = (
    "_PREV_STEP",
    "_PREV",
    "_LAST_STEP",
)


def categorize_dim(dim_name: str) -> OverwriteCategory:
    """Classify a dim name into one of the §8.E.1 categories.

    Precedence order:

      1. PREV_STEP (suffix match wins — these are explicitly authored
         cross-step relays whose only consumer is step S+1).
      2. TRANSIENT_SCRATCH (suffix match before prefix match because
         e.g. ``OUTPUT_HI_THIS_STEP`` should NOT be classified as
         output slot — it's a per-step transient).
      3. OUTPUT_SLOT.
      4. MEM_CELL.
      5. REGISTER (exact name or prefix).
      6. UNKNOWN otherwise.
    """

    upper = dim_name.upper()
    for suffix in _PREV_STEP_SUFFIXES:
        if upper.endswith(suffix):
            return OverwriteCategory.PREV_STEP
    for suffix in _TRANSIENT_SCRATCH_SUFFIXES:
        if upper.endswith(suffix):
            return OverwriteCategory.TRANSIENT_SCRATCH
    for prefix in _TRANSIENT_SCRATCH_PREFIXES:
        if upper.startswith(prefix):
            return OverwriteCategory.TRANSIENT_SCRATCH
    for prefix in _OUTPUT_SLOT_PREFIXES:
        if upper.startswith(prefix):
            return OverwriteCategory.OUTPUT_SLOT
    for prefix in _MEM_CELL_PREFIXES:
        if upper.startswith(prefix):
            return OverwriteCategory.MEM_CELL
    if upper in _REGISTER_EXACT:
        return OverwriteCategory.REGISTER
    for prefix in _REGISTER_PREFIXES:
        if upper.startswith(prefix):
            return OverwriteCategory.REGISTER
    return OverwriteCategory.UNKNOWN


# ---------------------------------------------------------------------------
# IR walking — local helpers (mirror kv_liveness_analyzer for consistency)
# ---------------------------------------------------------------------------


_EVERY_STEP = "__every__"


def _ffn_rules_from_op(op) -> List:
    """All FFNRules attached to an op's compiler_ir (any layer)."""

    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return []
    if hasattr(ir, "layers"):
        rules: List = []
        for layer in ir.layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "rules"):
                rules.extend(ffn.rules)
        return rules
    if hasattr(ir, "rules"):
        return [r for r in ir.rules if _looks_like_ffn_rule(r)]
    if isinstance(ir, (list, tuple)):
        return [r for r in ir if _looks_like_ffn_rule(r)]
    return []


def _looks_like_ffn_rule(rule) -> bool:
    return (
        hasattr(rule, "conditions")
        and hasattr(rule, "writes")
        and hasattr(rule, "threshold")
    )


def _ffn_rule_writes(rule) -> Set[str]:
    """Dim names written by an FFN rule (only the ``writes`` channel)."""

    names: Set[str] = set()
    for write in getattr(rule, "writes", ()):
        dim = getattr(write, "dim", None)
        if dim is not None:
            names.add(dim.name)
    return names


def _ffn_rule_reads(rule) -> Set[str]:
    names: Set[str] = set()
    for term in getattr(rule, "conditions", ()):
        dim = getattr(term, "dim", None)
        if dim is not None:
            names.add(dim.name)
    gate = getattr(rule, "gate", None)
    if gate is not None:
        names.add(gate.name)
    for term in getattr(rule, "gate_terms", ()):
        dim = getattr(term, "dim", None)
        if dim is not None:
            names.add(dim.name)
    return names


def _attention_heads_from_op(op) -> List:
    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return []
    heads: List = []
    if hasattr(ir, "layers"):
        for layer in ir.layers:
            attn = getattr(layer, "attention", None)
            if attn is None:
                continue
            for head in getattr(attn, "rules", ()):
                heads.append(head)
    return heads


def _op_writes(op) -> Set[str]:
    """Union of declared ``op.writes`` and FFN-rule writes.

    AttentionHeadIR writes are W_o writes — they land on residual dim
    *integers*, not names. Op.writes is the canonical source for those
    cross-name writes.
    """

    names: Set[str] = set(getattr(op, "writes", ()) or ())
    for rule in _ffn_rules_from_op(op):
        names.update(_ffn_rule_writes(rule))
    return names


def _op_reads(op) -> Set[str]:
    names: Set[str] = set(getattr(op, "reads", ()) or ())
    for rule in _ffn_rules_from_op(op):
        names.update(_ffn_rule_reads(rule))
    return names


def _op_step_set(op):
    """Either the ``_EVERY_STEP`` sentinel or a ``frozenset[int]``."""

    step_idx = getattr(op, "step_idx", None)
    if step_idx is None:
        return _EVERY_STEP
    if isinstance(step_idx, int):
        return frozenset({step_idx})
    if isinstance(step_idx, str):
        return _EVERY_STEP
    try:
        ints = frozenset(int(s) for s in step_idx)
        return ints if ints else _EVERY_STEP
    except (TypeError, ValueError):
        return _EVERY_STEP


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverwriteEntry:
    """One entry in the overwrite map.

    Attributes:
        position: The VM step at which the dim was written (cache row).
        dim_name: The residual-stream dim name.
        overwrite_step: The earliest step ``T > position`` at which a
            later writer overwrites this dim. ``None`` when no future
            writer overwrites the dim within ``n_steps``.
        category: Why the entry is dead at ``overwrite_step`` — see
            :class:`OverwriteCategory`. Drives runtime policy filters.
    """

    position: int
    dim_name: str
    overwrite_step: Optional[int]
    category: OverwriteCategory


@dataclass
class OverwriteMap:
    """Per-step ``(position, dim_name) -> OverwriteEntry`` map.

    The map is keyed by ``overwrite_step`` so the runtime can iterate
    "what becomes dead at step T" in O(1):

        for entry in overwrite_map.entries_at_step(t):
            cache.evict(entry.position, entry.dim_name)

    Attributes:
        entries_by_step: ``overwrite_step -> list of OverwriteEntry``.
            Built at compile time; iterated per step at runtime.
        unread_writes: ``(position, dim_name)`` pairs that are written
            at ``position`` and have no later writer in the analysed
            ``n_steps`` window. These are listed in
            ``entries_by_step[None]`` so callers can decide whether to
            evict them at end-of-program.
        categories: ``dim_name -> OverwriteCategory`` — useful for
            stats / report formatting / policy filters.
        n_steps: The step count the map was built for.
    """

    entries_by_step: Dict[Optional[int], List[OverwriteEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )
    categories: Dict[str, OverwriteCategory] = field(default_factory=dict)
    n_steps: int = 0

    def entries_at_step(self, step: int) -> List[OverwriteEntry]:
        """All overwrite entries that fire (become dead) at ``step``."""

        return list(self.entries_by_step.get(step, ()))

    def entries_by_category(
        self, category: OverwriteCategory
    ) -> List[OverwriteEntry]:
        """All entries with the given category, across every overwrite step."""

        out: List[OverwriteEntry] = []
        for entries in self.entries_by_step.values():
            out.extend(e for e in entries if e.category is category)
        return out

    def total_entries(self) -> int:
        return sum(len(v) for v in self.entries_by_step.values())

    def category_counts(self) -> Dict[OverwriteCategory, int]:
        counts: Dict[OverwriteCategory, int] = defaultdict(int)
        for entries in self.entries_by_step.values():
            for entry in entries:
                counts[entry.category] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_overwrite_map(
    ops: Sequence,
    n_steps: int,
) -> OverwriteMap:
    """Build the per-step overwrite map from declarative IR.

    Walks every op in ``ops``, collects per-step writes (FFNRule.writes
    + declared ``op.writes`` — the same union the liveness analyzer
    uses), and emits one :class:`OverwriteEntry` per
    ``(position, dim_name)`` such that:

      * the dim is written at ``position``, AND
      * the dim's :func:`categorize_dim` is in the static-overwrite
        category set (REGISTER, MEM_CELL, OUTPUT_SLOT,
        TRANSIENT_SCRATCH, PREV_STEP), AND
      * the earliest later writer fires at step ``T > position``.

    Per category, the overwrite step is computed as:

      * TRANSIENT_SCRATCH — dead at ``position`` (end-of-step). The
        runtime can evict it immediately; ``overwrite_step = position``.
      * PREV_STEP — dead at ``position + 1`` (consumed at S+1, evict
        after the consuming step). ``overwrite_step = position + 1``
        provided that's ``< n_steps``.
      * REGISTER / MEM_CELL / OUTPUT_SLOT — dead at the earliest
        ``T > position`` where some op anchored at step ``T`` writes
        the same dim. ``overwrite_step = T``.

    NON_PERSISTING is *not* produced as a direct category here — a dim
    that is written but never read becomes evictable at end-of-step via
    the TRANSIENT_SCRATCH path if it matches a scratch prefix, or stays
    UNKNOWN otherwise (conservative).

    Parameters
    ----------
    ops:
        Iterable of declarative ``Operation`` (or look-alike) objects.
        Each op is read-only. Ops with ``step_idx=None`` are treated as
        fires-every-step and contribute writes at every step in
        ``range(n_steps)``.
    n_steps:
        Number of VM steps to reason about. Overwrite steps are
        constrained to ``range(n_steps)``; writers that would fire past
        ``n_steps - 1`` are dropped from the future-writer set.

    Returns
    -------
    OverwriteMap
        Per-step (and per-category) overwrite entries ready to drive
        runtime KV eviction.
    """

    ops_list = list(ops)

    # ---- 1. Per-step writes index --------------------------------------
    # ``writes_at_step[s]`` is the union of dims written by every op
    # anchored at step ``s``. Every-step ops contribute to all
    # ``s in range(n_steps)``.
    writes_at_step: Dict[int, Set[str]] = defaultdict(set)
    every_step_writes: Set[str] = set()
    for op in ops_list:
        op_writes = _op_writes(op)
        if not op_writes:
            continue
        steps = _op_step_set(op)
        if steps == _EVERY_STEP:
            every_step_writes.update(op_writes)
            for s in range(n_steps):
                writes_at_step[s].update(op_writes)
        else:
            for s in steps:
                if 0 <= s < n_steps:
                    writes_at_step[s].update(op_writes)

    # ---- 2. Per-dim future-writers index -------------------------------
    # ``earliest_future_write[s][dim] = smallest T > s with dim in
    # writes_at_step[T]``. Computed by iterating backward.
    earliest_future_write: List[Dict[str, int]] = [
        {} for _ in range(n_steps + 1)
    ]
    # Sentinel cell at ``earliest_future_write[n_steps]`` is intentionally
    # left empty — no step beyond ``n_steps - 1`` writes anything.
    for s in range(n_steps - 1, -1, -1):
        # Start from "what becomes the earliest writer at step s+1 or
        # later"; then overlay step s+1's own writers (those win for
        # queries originating at step s).
        nxt = dict(earliest_future_write[s + 1])
        # Step ``s+1`` writers replace any deeper entry for the same dim.
        if (s + 1) < n_steps:
            for dim in writes_at_step.get(s + 1, set()):
                nxt[dim] = s + 1
        earliest_future_write[s] = nxt

    # ---- 3. Dim universe + categories ----------------------------------
    all_written_dims: Set[str] = set()
    for s in range(n_steps):
        all_written_dims.update(writes_at_step.get(s, set()))
    categories: Dict[str, OverwriteCategory] = {
        d: categorize_dim(d) for d in all_written_dims
    }

    # ---- 4. Emit entries -----------------------------------------------
    entries_by_step: Dict[Optional[int], List[OverwriteEntry]] = defaultdict(
        list
    )

    for position in range(n_steps):
        # Dim is "written at ``position``" iff it appears in
        # ``writes_at_step[position]``. We emit one entry per such dim.
        dims_written = writes_at_step.get(position, set())
        for dim in dims_written:
            category = categories.get(dim, OverwriteCategory.UNKNOWN)

            if category is OverwriteCategory.UNKNOWN:
                # Conservative — analyzer cannot statically prove dead.
                # Do NOT emit a positive overwrite entry. Runtime keeps
                # the row alive until a more refined pass classifies it.
                continue

            if category is OverwriteCategory.TRANSIENT_SCRATCH:
                # Dead at end-of-step regardless of any future writer.
                overwrite_step: Optional[int] = position
            elif category is OverwriteCategory.PREV_STEP:
                # Consumed at S+1, dead at S+1.
                cand = position + 1
                overwrite_step = cand if cand < n_steps else None
            else:
                # REGISTER / MEM_CELL / OUTPUT_SLOT — wait for next
                # writer.
                overwrite_step = earliest_future_write[position].get(dim)

            entries_by_step[overwrite_step].append(
                OverwriteEntry(
                    position=position,
                    dim_name=dim,
                    overwrite_step=overwrite_step,
                    category=category,
                )
            )

    return OverwriteMap(
        entries_by_step=entries_by_step,
        categories=categories,
        n_steps=n_steps,
    )


__all__ = [
    "OverwriteCategory",
    "OverwriteEntry",
    "OverwriteMap",
    "build_overwrite_map",
    "categorize_dim",
]
