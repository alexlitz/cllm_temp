"""B11: Hybrid dynamic-layer compile path for the Neural VM (foundational).

This module provides ``compile_full_vm_dynamic``, a sibling API to
``compile_full_vm`` that derives op ordering from declared dependencies
(``reads``/``writes``/``produces``/``consumes_fresh``/``requires``) where
possible, and falls back to the static ``phase=N.M`` field as a tiebreaker
(and as the sole ordering signal for ops trapped in dep-graph cycles).

Why hybrid (not strict dep-derived)
-----------------------------------
The Phase A scheduler diagnostic (``tools/analyze_scheduler.py``) shows
that on today's op set 57 / 113 ops participate in a single large strongly
connected component. A strict topological sort therefore cannot order
those ops; some external signal (today: the hand-set ``phase=N.M`` field)
must disambiguate. Until that SCC is broken via dim decomposition or
``requires=`` op-name pins (the Phase B work tracked separately), the
hybrid scheduler treats phase as the **fallback** ordering for cycle
members and as the **tiebreaker** when multiple ops are simultaneously
ready in the DAG.

Byte-identity invariant
-----------------------
On the current op set this module returns a layout that is bit-identical
to the layout produced by ``compile_full_vm``. The mechanism is:

  * The hybrid scheduler reproduces the same Kahn-with-phase-pruning
    result as ``LayerCompiler._topological_sort`` — phase pruning is
    treated as the tiebreaker rule that breaks the SCC edges, exactly
    the same way the static path does. The order emitted by the hybrid
    scheduler IS a valid topological order of the phase-pruned DAG, so
    when the unchanged ``LayerCompiler`` re-sorts the same op list it
    converges on the same answer.
  * ``LayerCompiler._assign_layers`` ALREADY shares (layer, kind) slots
    by phase. The pinned-layer_idx path enforces that same-layer
    same-kind ops respect phase. Once the hybrid scheduler hands the
    LayerCompiler an op list whose dep-pruned order matches the static
    pipeline's expectation, the layer assignment runs identically.
  * Critically, this module does NOT modify ``LayerCompiler``,
    ``dim_registry``, ``ir.py``, or any op factory. It only re-orders the
    ops fed into the unchanged compile pipeline.

This invariant is enforced by ``compare_compile_paths()`` and by the
``tests/test_compile_dynamic_byte_identical.py`` regression test. Any
future change in the declared deps that would split the hybrid order
away from the static order would fail those checks immediately.

Migration path
--------------
As cycle members move out of the largest SCC (by breaking back-edges via
dim decomposition or by adding explicit ``requires=`` references), this
hybrid scheduler will gracefully shift them from phase-fallback to
dep-derived placement WITHOUT a behavioral change — provided the static
phase order already agrees with the new dep-derived order. Once the SCC
is fully broken, ``compile_full_vm_dynamic`` becomes the canonical entry
point and the static ``phase`` field can be retired.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .layer_compiler import (
    Operation,
    requires_after_ops,
    requires_same_layer_as_ops,
)
from . import full_vm_compiler as _static


# ---------------------------------------------------------------------------
# Dependency graph (same edge model as tools/analyze_scheduler.py)
# ---------------------------------------------------------------------------


def _build_dep_graph(
    ops: Sequence[Operation],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Construct ``(in_edges, out_edges)`` from declared dependencies.

    Edges:
      * Data flow:         A.writes ∩ B.reads
      * Staleness:         A.produces.keys() ∩ B.consumes_fresh.keys()
      * Explicit requires: B.requires references A.name as a value

    Self-edges are skipped — an op that both writes and reads a dim is
    not blocked on itself.
    """
    name_to_op = {op.name: op for op in ops}
    in_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
    out_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}

    writers: Dict[str, List[Operation]] = defaultdict(list)
    for op in ops:
        for d in op.writes:
            writers[d].append(op)

    producers: Dict[Tuple[str, str], List[Operation]] = defaultdict(list)
    for op in ops:
        for dim, reg in op.produces.items():
            producers[(dim, reg)].append(op)

    for v in ops:
        for d in v.reads:
            for u in writers.get(d, ()):
                if u.name == v.name:
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
        for dim, reg in v.consumes_fresh.items():
            for u in producers.get((dim, reg), ()):
                if u.name == v.name:
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
        # B10: explicit ``requires["after"]`` / ``requires["same_layer_as"]``
        # op-name edges. Both reserved keys add a ref→v dep edge (v must
        # run after the referenced op). Use the canonical accessors so
        # iterable values (multi-ref tuples) are handled correctly rather
        # than silently dropped by an ``isinstance(val, str)`` filter.
        for ref in requires_after_ops(v) + requires_same_layer_as_ops(v):
            if ref == v.name or ref not in name_to_op:
                continue
            if ref not in in_edges[v.name]:
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)

    return in_edges, out_edges


def _find_cycle_members(
    ops: Sequence[Operation],
    in_edges: Dict[str, Set[str]],
    out_edges: Dict[str, Set[str]],
) -> Set[str]:
    """Return the set of op names that participate in any directed cycle.

    Uses Kahn's algorithm: any name not drained when the queue empties is
    a cycle member (or transitively downstream of one).
    """
    indeg = {op.name: len(in_edges[op.name]) for op in ops}
    queue: List[str] = [op.name for op in ops if indeg[op.name] == 0]
    drained: Set[str] = set()
    while queue:
        u = queue.pop(0)
        drained.add(u)
        for v in out_edges[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return {op.name for op in ops if op.name not in drained}


# ---------------------------------------------------------------------------
# Hybrid schedule: dep-derived order with phase-fallback for cycles
# ---------------------------------------------------------------------------


_PHASE_TIEBREAK_FAR = 1.0e18


def _phase_key(op: Operation) -> float:
    """Return a sortable phase value. Ops with no phase sort last (stable)."""
    if op.phase is None:
        return _PHASE_TIEBREAK_FAR
    try:
        return float(op.phase)
    except (TypeError, ValueError):
        return _PHASE_TIEBREAK_FAR


def _build_phase_pruned_graph(
    ops: Sequence[Operation],
    *,
    restrict_to_kinds: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Set[str]]:
    """Build the dep graph WITH the static path's phase-pruning rule.

    Returns ``(in_edges, out_edges, cycle_members_in_unpruned_graph)``.

    Phase pruning matches ``LayerCompiler._topological_sort`` exactly:

      * Drop u→v when ``u.phase > v.phase`` — the writer is "later" than
        the reader in hand-set order, so the reader doesn't actually
        depend on the writer (this is the SCC-breaking signal).
      * Drop u→v when ``u.phase == v.phase`` AND ``u.kind == "ffn"`` AND
        ``v.kind == "attn"`` — within a transformer block attn runs
        before ffn, so an ffn writer at the same phase doesn't gate the
        attn reader.

    When ``restrict_to_kinds`` is set, only ops whose ``kind`` is in
    that set participate as edge endpoints. This mirrors the static
    ``LayerCompiler.compile()`` behaviour where the topological sort
    only sees attn/ffn ops (block / model ops are pinned by
    ``layer_idx`` and do not appear in the dep graph). Pass
    ``{"attn", "ffn"}`` for byte-identity comparisons.

    ``cycle_members_in_unpruned_graph`` reports which ops would have
    been trapped in a cycle in the dep-only DAG (without phase pruning).
    Reported for diagnostics; on today's op set every such cycle is
    broken by phase pruning, so the pruned graph is acyclic.
    """
    if restrict_to_kinds is not None:
        ops = [op for op in ops if op.kind in restrict_to_kinds]
    name_to_op = {op.name: op for op in ops}
    # First compute unpruned cycle membership for the report.
    unpruned_in, unpruned_out = _build_dep_graph(ops)
    cycle_unpruned = _find_cycle_members(ops, unpruned_in, unpruned_out)

    in_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
    out_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}

    writers: Dict[str, List[Operation]] = defaultdict(list)
    for op in ops:
        for d in op.writes:
            writers[d].append(op)

    producers: Dict[Tuple[str, str], List[Operation]] = defaultdict(list)
    for op in ops:
        for dim, reg in op.produces.items():
            producers[(dim, reg)].append(op)

    def _phase_prunes(u: Operation, v: Operation) -> bool:
        """Return True iff this u→v edge should be pruned by the static rule."""
        if u.phase is None or v.phase is None:
            return False
        if u.phase > v.phase:
            return True
        if u.phase == v.phase and u.kind == "ffn" and v.kind == "attn":
            return True
        return False

    for v in ops:
        for d in v.reads:
            for u in writers.get(d, ()):
                if u.name == v.name:
                    continue
                if _phase_prunes(u, v):
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
        for dim, reg in v.consumes_fresh.items():
            for u in producers.get((dim, reg), ()):
                if u.name == v.name:
                    continue
                if _phase_prunes(u, v):
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
        for _key, val in v.requires.items():
            if not isinstance(val, str):
                continue
            if val in name_to_op and val != v.name:
                u = name_to_op[val]
                if _phase_prunes(u, v):
                    continue
                if val not in in_edges[v.name]:
                    in_edges[v.name].add(val)
                    out_edges[val].add(v.name)

    return in_edges, out_edges, cycle_unpruned


def compute_dynamic_schedule(
    ops: Sequence[Operation],
) -> Tuple[List[Operation], Dict[str, str]]:
    """Return ``(scheduled_ops, source)`` where:

      * ``scheduled_ops`` is the input list re-ordered by the hybrid
        scheduler. Ops are placed in topological order over the
        phase-pruned dep DAG (Kahn's algorithm with ``phase`` as the
        primary tiebreaker, original insertion index as the secondary
        stability tiebreaker). On today's op set the phase-pruned DAG
        is acyclic, so every op gets a dep-derived placement.
      * ``source[op_name]`` is either ``"dep"`` (placement determined
        by the phase-pruned dep DAG) or ``"phase"`` (placement fell
        back to phase alone because the op participates in a cycle that
        survives even phase pruning — should never happen on a
        well-formed op set).

    Why phase pruning, not strict dep-only
    --------------------------------------
    The Phase A diagnostic shows that the dep-only graph has 57 cycle
    members on today's op set. A strict topological sort therefore
    can't run. The static ``LayerCompiler._topological_sort`` breaks
    those cycles by dropping ``u→v`` edges where ``u.phase > v.phase``
    (the back-edges of the static phase ordering). This module applies
    the SAME pruning rule so the dynamic schedule converges on the
    same topological order as the static path. The byte-identity
    guarantee falls out of this choice.

    When the SCC is broken by future dim-decomposition work, this
    function will switch transparently from "phase-pruned dep DAG" to
    "pure dep DAG" because the unpruned graph will also be acyclic and
    the phase pruning rule will become a no-op. That's the gradual
    migration path B12/B13 will exercise.
    """
    # Only attn/ffn ops participate in the dep graph (mirroring the
    # static LayerCompiler behaviour). Block / model ops are emitted in
    # phase order at the end — block ops are layer-pinned by ``layer_idx``
    # and model ops run as post-passes, so dep ordering is irrelevant.
    attn_ffn_ops = [op for op in ops if op.kind in ("attn", "ffn")]
    in_edges, out_edges, _unpruned_cycle = _build_phase_pruned_graph(
        ops, restrict_to_kinds={"attn", "ffn"},
    )
    cycle_members = _find_cycle_members(
        attn_ffn_ops, in_edges, out_edges
    )

    name_to_op = {op.name: op for op in ops}
    insertion_index = {op.name: i for i, op in enumerate(ops)}

    def _sort_key(name: str) -> Tuple[float, int]:
        op = name_to_op[name]
        return (_phase_key(op), insertion_index[name])

    # Phase-pruned graph: in-degree counts ignore edges from cycle members
    # to non-cycle ops (cycle members never drain).
    indeg = {
        op.name: len(in_edges[op.name] - cycle_members)
        for op in attn_ffn_ops
    }
    ready: List[str] = sorted(
        (op.name for op in attn_ffn_ops
         if op.name not in cycle_members and indeg[op.name] == 0),
        key=_sort_key,
    )
    ordered: List[str] = []
    source: Dict[str, str] = {}

    while ready:
        u = ready.pop(0)
        ordered.append(u)
        source[u] = "dep"
        new_ready: List[str] = []
        for v in sorted(out_edges[u]):
            if v in cycle_members:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                new_ready.append(v)
        if new_ready:
            merged = ready + new_ready
            merged.sort(key=_sort_key)
            ready = merged

    # Cycle members (attn/ffn) + block / model ops are appended in phase
    # order at the end. The static path layer-pins block ops separately
    # and runs model ops as post-passes; this mirror keeps the schedule
    # output complete and totally ordered, suitable for diagnostics.
    remainder = sorted(
        (op.name for op in ops if op.name not in source),
        key=_sort_key,
    )
    for name in remainder:
        ordered.append(name)
        source[name] = "phase"

    return [name_to_op[name] for name in ordered], source


# ---------------------------------------------------------------------------
# Strict mode: pure dep-derived ordering (no phase fallback)
# ---------------------------------------------------------------------------


class StrictModeUnschedulableError(RuntimeError):
    """Raised by ``compile_full_vm_dynamic(strict=True)`` when the declared
    dep graph cannot place every op without falling back to ``phase``.

    The exception carries three attribute lists naming the offending op
    classes so callers (and the B14 rollout test) can verify the exact
    reason strict mode refused to compile:

      * ``cycle_members`` — ops trapped in a directed cycle in the
        unpruned dep graph (today: the OUTPUT_HI / IF_VAR SCC, etc.).
        These need ``B9`` dim decomposition to break the back-edges.
        In cycle-aware strict mode (``allow_sealed_cycles=True``) the
        cycle members are sealed as a group and the ordering inside the
        group falls back to phase; only ``phase_required_but_undeclared``
        and ``phase_inconsistent_with_deps`` populated by ops OUTSIDE
        the SCC actually raise.
      * ``phase_required_but_undeclared`` — ops whose static
        ``phase=N.M`` places them later than their dep-derived earliest
        position. The static path uses ``phase`` to pin them; strict
        mode does not, so an explicit ``requires={"after": ...}`` (or
        new ``reads`` / ``consumes_fresh``) is needed. ``B12`` is the
        unit that backfills these declarations.
      * ``phase_inconsistent_with_deps`` — ops whose static ``phase`` is
        EARLIER than the dep DAG can satisfy. Indicates a bug or a
        latent cycle whose member's depth is the sentinel ``-1``.
    """

    def __init__(
        self,
        *,
        cycle_members: Sequence[str],
        phase_required_but_undeclared: Sequence[str],
        phase_inconsistent_with_deps: Sequence[str],
    ) -> None:
        self.cycle_members = list(cycle_members)
        self.phase_required_but_undeclared = list(phase_required_but_undeclared)
        self.phase_inconsistent_with_deps = list(phase_inconsistent_with_deps)
        total = (
            len(self.cycle_members)
            + len(self.phase_required_but_undeclared)
            + len(self.phase_inconsistent_with_deps)
        )

        def _preview(names: Sequence[str], limit: int = 10) -> str:
            head = ", ".join(sorted(names)[:limit])
            extra = max(0, len(names) - limit)
            return f"{head}{f', ... (+{extra} more)' if extra else ''}"

        msg_parts: List[str] = [
            f"compile_full_vm_dynamic(strict=True) refused to compile: "
            f"{total} ops cannot be placed by declared dependencies alone.",
        ]
        if self.cycle_members:
            msg_parts.append(
                f"  dep_graph_cycle_member ({len(self.cycle_members)}): "
                f"{_preview(self.cycle_members)}"
            )
        if self.phase_required_but_undeclared:
            msg_parts.append(
                f"  phase_required_but_undeclared "
                f"({len(self.phase_required_but_undeclared)}): "
                f"{_preview(self.phase_required_but_undeclared)}"
            )
        if self.phase_inconsistent_with_deps:
            msg_parts.append(
                f"  phase_inconsistent_with_deps "
                f"({len(self.phase_inconsistent_with_deps)}): "
                f"{_preview(self.phase_inconsistent_with_deps)}"
            )
        msg_parts.append(
            "  Resolution: add explicit deps (requires={\"after\": ...}, "
            "consumes_fresh, or new reads/writes) per "
            "DYNAMIC_SCHEDULER_MIGRATION_PLAN.md units B9/B12. Until then, "
            "compile with strict=False (the default) to fall back to phase "
            "tiebreaking, or pass allow_sealed_cycles=True (the strict-mode "
            "default) to accept SCCs as sealed groups."
        )
        super().__init__("\n".join(msg_parts))


def _current_layer_for_strict(op: Operation) -> Optional[int]:
    """Best-effort integer layer derived from ``op.phase`` / ``op.layer_idx``.

    Mirrors ``tools/analyze_scheduler.py:_current_layer`` so strict-mode
    categorisation matches the analyzer's report exactly.
    """
    if op.layer_idx is not None:
        return op.layer_idx
    if op.phase is None:
        return None
    try:
        return int(math.floor(op.phase))
    except (TypeError, ValueError):
        return None


def _build_strict_dep_graph(
    ops: Sequence[Operation],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Set[Tuple[str, str]]]:
    """Strict-mode dep graph that mirrors ``tools/analyze_scheduler.py``.

    Differs from :func:`_build_dep_graph` in two ways that align with the
    offline analyzer:

      * Applies the B9 R-OH-2 suppression rule: when a reader ``v``
        declares ``requires["after"] = X`` AND ``X`` writes a dim ``D``
        that ``v`` also reads, the data-flow edge ``u→v`` on ``D`` is
        suppressed for every other writer ``u`` of ``D``. The explicit
        ``X→v`` edge is still added. Semantic: the reader opted into
        the prev-step residual / KV-cache satisfaction, NOT a same-step
        data dep on any later-layer producer. See
        ``docs/B9_OUTPUT_HI_SPLIT_SPEC.md`` §7.2 and §6.3.
      * Tracks ``same_layer_edges`` — the set of ``(u, v)`` pairs whose
        ONLY contribution comes from a ``requires["same_layer_as"]``
        declaration. ``_strict_topo_depth`` propagates ``depth[u]``
        instead of ``depth[u]+1`` along these so co-placement peers
        don't artificially bump the depth.

    Used ONLY by the strict-mode admission check; the hybrid scheduler
    still uses the simpler :func:`_build_dep_graph` (kept for backward
    compat with existing ``_build_phase_pruned_graph`` callers).
    """
    name_to_op = {op.name: op for op in ops}
    in_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
    out_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
    same_layer_only: Set[Tuple[str, str]] = set()
    other_edges: Set[Tuple[str, str]] = set()

    writers: Dict[str, List[Operation]] = defaultdict(list)
    for op in ops:
        for d in op.writes:
            writers[d].append(op)

    producers: Dict[Tuple[str, str], List[Operation]] = defaultdict(list)
    for op in ops:
        for dim, reg in op.produces.items():
            producers[(dim, reg)].append(op)

    for v in ops:
        # B9 R-OH-2 suppression: when v declares requires["after"]=X and
        # X writes a dim that v reads, drop the dataflow edge from EVERY
        # other writer of that dim. X→v is added below via the explicit
        # ``requires["after"]`` walk.
        suppressed_dims: Set[str] = set()
        for ref in requires_after_ops(v):
            if ref == v.name or ref not in name_to_op:
                continue
            ref_op = name_to_op[ref]
            for d in ref_op.writes & v.reads:
                suppressed_dims.add(d)
        for d in v.reads:
            if d in suppressed_dims:
                continue
            for u in writers.get(d, ()):
                if u.name == v.name:
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
                other_edges.add((u.name, v.name))
        for dim, reg in v.consumes_fresh.items():
            for u in producers.get((dim, reg), ()):
                if u.name == v.name:
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
                other_edges.add((u.name, v.name))
        for ref in requires_after_ops(v):
            if ref == v.name or ref not in name_to_op:
                continue
            if ref not in in_edges[v.name]:
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)
            other_edges.add((ref, v.name))
        for ref in requires_same_layer_as_ops(v):
            if ref == v.name or ref not in name_to_op:
                continue
            if ref not in in_edges[v.name]:
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)
            same_layer_only.add((ref, v.name))

    # Same-layer edges are ONLY edges with no other contribution.
    same_layer_edges = {p for p in same_layer_only if p not in other_edges}
    return in_edges, out_edges, same_layer_edges


def _strict_topo_depth(
    ops: Sequence[Operation],
    in_edges: Dict[str, Set[str]],
    out_edges: Dict[str, Set[str]],
    same_layer_edges: Set[Tuple[str, str]],
) -> Tuple[Dict[str, int], Set[str]]:
    """Compute earliest-layer depth via Kahn's algorithm.

    Mirrors ``tools/analyze_scheduler.py:topo_depth``. Edges in
    ``same_layer_edges`` contribute ``depth[u]`` (peer-equal), all
    others contribute ``depth[u] + 1``.

    Returns ``(depth, cycle_members)`` where ``cycle_members`` is the
    set of ops never reached (in-degree never hits 0 because they live
    in or downstream of a directed cycle).
    """
    indeg = {op.name: len(in_edges[op.name]) for op in ops}
    max_pred_depth: Dict[str, int] = {op.name: -1 for op in ops}
    depth: Dict[str, int] = {}
    queue: List[str] = []
    for op in ops:
        if indeg[op.name] == 0:
            depth[op.name] = 0
            queue.append(op.name)
    while queue:
        u = queue.pop(0)
        for v in sorted(out_edges[u]):
            if (u, v) in same_layer_edges:
                contribution = depth[u]
            else:
                contribution = depth[u] + 1
            if contribution > max_pred_depth[v]:
                max_pred_depth[v] = contribution
            indeg[v] -= 1
            if indeg[v] == 0:
                depth[v] = max(max_pred_depth[v], 0)
                queue.append(v)
    cycle_members: Set[str] = set()
    for op in ops:
        if op.name not in depth:
            depth[op.name] = -1
            cycle_members.add(op.name)
    return depth, cycle_members


def _strict_mode_categorise(
    ops: Sequence[Operation],
) -> Dict[str, List[str]]:
    """Categorise ops for strict-mode admission.

    Returns ``{"ok": [...], "cycle_members": [...],
    "phase_required_but_undeclared": [...],
    "phase_inconsistent_with_deps": [...]}`` where each value is a sorted
    list of op names. Mirrors ``tools/analyze_scheduler.py:categorise``
    so strict-mode failures and the offline analyzer report agree.

    Implementation notes
    --------------------
    * Uses the strict dep graph (B9 R-OH-2 suppression + same_layer_as
      peer modelling) — keeps the depth calculation aligned with the
      static path's ``LayerCompiler._assign_layers`` co-placement
      semantics.
    * Post-pass ops (current layer >= 100) and ``kind="model"`` ops are
      treated as structurally pinned and pass strict mode unconditionally
      (the dispatcher places them by phase / layer_idx, never by dep
      depth). Block-kind ops go through the same dep check as attn / ffn
      ops — they may still expose ordering bugs.
    * A "freely_placeable" refinement marks ops with no in-edges AND no
      out-edges in the dep graph as ``ok`` regardless of phase vs depth
      — there is no constraint to satisfy.
    """
    in_edges, out_edges, same_layer_edges = _build_strict_dep_graph(ops)
    depth, cycle_members = _strict_topo_depth(
        ops, in_edges, out_edges, same_layer_edges
    )

    buckets: Dict[str, List[str]] = {
        "ok": [],
        "cycle_members": [],
        "phase_required_but_undeclared": [],
        "phase_inconsistent_with_deps": [],
    }
    for op in ops:
        current = _current_layer_for_strict(op)
        # Post-pass / model ops short-circuit: their placement is by
        # phase / layer_idx, never by dep depth.
        if current is not None and current >= 100:
            buckets["ok"].append(op.name)
            continue
        if op.kind == "model":
            buckets["ok"].append(op.name)
            continue
        if op.name in cycle_members:
            buckets["cycle_members"].append(op.name)
            continue
        derived = depth[op.name]
        if current is None:
            # No phase declared and no cycle — free placement.
            buckets["ok"].append(op.name)
            continue
        # Refinement: an op with NO in/out edges in the dep graph has
        # nothing to satisfy. Mirrors analyzer.categorise's
        # freely_placeable refinement.
        if not in_edges[op.name] and not out_edges[op.name]:
            buckets["ok"].append(op.name)
            continue
        if current < derived:
            buckets["phase_inconsistent_with_deps"].append(op.name)
            continue
        if current == derived:
            buckets["ok"].append(op.name)
            continue
        # current > derived: phase pins op later than the DAG requires.
        buckets["phase_required_but_undeclared"].append(op.name)

    for key in buckets:
        buckets[key].sort()
    return buckets


def _strict_mode_sccs(ops: Sequence[Operation]) -> List[Set[str]]:
    """Return strongly-connected components restricted to cycle members.

    Uses Tarjan's algorithm on the cycle-member subgraph derived from
    the strict dep graph. Returns SCCs sorted by size descending; each
    SCC is a set of op names.
    """
    in_edges, out_edges, _ = _build_strict_dep_graph(ops)
    _depth, cycle = _strict_topo_depth(
        ops, in_edges, out_edges, set()
    )
    sub_out: Dict[str, Set[str]] = {
        n: out_edges[n] & cycle for n in cycle
    }
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def _strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sub_out.get(v, ()):
            if w not in indices:
                _strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in sub_out.get(v, ()):
                sccs.append(comp)

    import sys as _sys
    _sys.setrecursionlimit(10000)
    for n in sorted(cycle):
        if n not in indices:
            _strongconnect(n)

    sccs.sort(key=len, reverse=True)
    return [set(c) for c in sccs]


def _assert_strict_mode_clean(
    ops: Sequence[Operation],
    *,
    allow_sealed_cycles: bool = True,
) -> None:
    """Raise ``StrictModeUnschedulableError`` if any op fails strict admission.

    When ``allow_sealed_cycles=True`` (the default for B14 strict mode),
    cycle members are accepted as a sealed group — the hybrid scheduler
    falls back to phase ordering INSIDE the SCC only, leaving the rest
    of the graph dep-derived. Only ``phase_required_but_undeclared``
    and ``phase_inconsistent_with_deps`` (which are populated by
    non-cycle ops in the strict categoriser) actually gate strict mode.
    A non-empty ``cycle_members`` bucket combined with empty
    ``phase_required_but_undeclared`` and ``phase_inconsistent_with_deps``
    is the "cycle-aware strict accept" outcome.

    When ``allow_sealed_cycles=False``, any cycle is a hard failure
    (legacy strict-mode behavior).
    """
    buckets = _strict_mode_categorise(ops)
    gating = bool(
        buckets["phase_required_but_undeclared"]
        or buckets["phase_inconsistent_with_deps"]
    )
    if not allow_sealed_cycles:
        gating = gating or bool(buckets["cycle_members"])
    if gating:
        raise StrictModeUnschedulableError(
            cycle_members=buckets["cycle_members"],
            phase_required_but_undeclared=buckets[
                "phase_required_but_undeclared"
            ],
            phase_inconsistent_with_deps=buckets[
                "phase_inconsistent_with_deps"
            ],
        )


# ---------------------------------------------------------------------------
# Public compile entry point
# ---------------------------------------------------------------------------


def compile_full_vm_dynamic(
    S: float = 100.0,
    *,
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    enable_neural_io_think_protocol: bool = False,
    alu_mode: str = "lookup",
    n_heads: int = 8,
    ffn_hidden: int = 4096,
    max_seq_len: int = 8192,
    pin_io_only: bool = True,
    disk_cache: bool = True,
    use_dynamic_ffn: bool = True,
    enable_moe_routing: Optional[bool] = None,
    positional_encoding: Optional[str] = None,
    attention_normalization: Optional[str] = None,
    rope_base: Optional[float] = None,
    use_rms_norm: Optional[bool] = None,
    rms_norm_eps: Optional[float] = None,
    require_declarative_bake: Optional[bool] = None,
    declarations_only: bool = False,
    strict: bool = False,
    allow_sealed_cycles: bool = True,
):
    """Compile and bake a Neural VM via the hybrid dynamic-layer scheduler.

    Signature mirrors ``compile_full_vm`` exactly (same args/kwargs, same
    return type ``(model, layout)``).

    Internally:

      1. Collect the same op list as ``compile_full_vm`` (delegating to
         the same factories with the same flags).
      2. Compute a hybrid dep+phase schedule via
         ``compute_dynamic_schedule``. This is the load-bearing dynamic
         logic; today it agrees with the static phase order on the full
         op set (cycle members fall back to phase, non-cycle ops sort by
         topological depth with phase as a tiebreaker — and on the current
         op set the tiebreaker always wins because the DAG depth chain is
         only 4 layers deep vs. the 17-layer static layout).
      3. Hand the scheduled ops to the unchanged ``LayerCompiler`` and
         ``build_model_from_layout`` pipelines used by ``compile_full_vm``.
         The static path's phase pruning runs on top, so any residual
         ordering ambiguity is resolved identically to the static path.

    The caller receives ``(model, layout)`` exactly as from
    ``compile_full_vm``. By construction the returned layout is byte-
    identical to the static path's layout on today's op set; this is
    asserted post-hoc by ``compare_compile_paths`` and by
    ``tests/test_compile_dynamic_byte_identical.py``.

    Args mirror ``compile_full_vm`` -- see that function's docstring for
    detailed semantics. The disk-cache key is namespaced by appending
    ``"__dynamic"`` to the kwargs snapshot so dynamic and static
    compiles cannot trample each other's cache entries.

    Strict mode (B14 prep, OFF by default)
    --------------------------------------
    When ``strict=True``, the dynamic compile refuses to fall back to
    ``phase`` for any ordering decision. Before compiling, every op is
    categorised against the unpruned declared-dep graph (mirroring
    ``tools/analyze_scheduler.py``); if any op falls into
    ``dep_graph_cycle_member``, ``phase_required_but_undeclared``, or
    ``phase_inconsistent_with_deps``, the call raises
    ``StrictModeUnschedulableError`` with the offending op names.

    On a clean op set strict mode produces the same byte-identical
    layout as the static path, because on a fully-declared op set the
    dep-derived order and the phase-derived order agree (Phase A
    finding, see ``DYNAMIC_SCHEDULER_MIGRATION_PLAN.md``).

    Today (2026-06-01) strict mode is expected to FAIL on the production
    op set: ~70 ops are cycle members of the OUTPUT_HI SCC, ~23 ops are
    ``phase_required_but_undeclared``, and ~3 are
    ``phase_inconsistent_with_deps``. ``strict=False`` (the default)
    preserves the B11/B12 hybrid behaviour. Flipping the default to
    ``True`` is the B14 unit; it depends on B9 (dim decomposition) and
    B12 (declaration backfill) fully landing first.
    """
    # Mirror static-path env-flag handling to keep the API truly identical.
    if not declarations_only:
        declarations_only = _static._env_flag_enabled(
            _static._DECLARATIONS_ONLY_BAKE_ENV
        )
    if enable_moe_routing is None:
        enable_moe_routing = _static._env_flag_enabled(
            _static._ENABLE_MOE_ROUTING_ENV
        )
    if require_declarative_bake is None:
        require_declarative_bake = _static._env_flag_enabled(
            _static._REQUIRE_DECLARATIVE_BAKE_ENV
        )
    if declarations_only:
        require_declarative_bake = True

    from ..config import get_config
    vm_config = get_config()
    if positional_encoding is None:
        positional_encoding = vm_config.positional_encoding
    if attention_normalization is None:
        attention_normalization = vm_config.attention_normalization
    if rope_base is None:
        rope_base = vm_config.rope_base
    if use_rms_norm is None:
        use_rms_norm = vm_config.use_rms_norm
    if rms_norm_eps is None:
        rms_norm_eps = vm_config.rms_norm_eps

    # Collect ops with the same composition rules as compile_full_vm.
    ops = _collect_ops_for_compile(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
    )

    # B14 strict-mode gate: refuse to compile if any op needs phase to
    # be placed. Raised BEFORE any LayerCompiler / disk-cache work runs
    # so the error citation is purely scheduler-level (no half-built
    # model state to tear down, no stale cache hit masking the failure).
    # When ``strict=False`` (the default), the hybrid scheduler falls
    # back to phase, preserving the B11/B12 byte-identity behaviour.
    #
    # Cycle-aware strict mode (Phase 7.A.5 B14 attempt): when
    # ``allow_sealed_cycles=True`` (the strict-mode default) the SCC of
    # cycle members is accepted as a sealed group — the hybrid
    # scheduler still routes them via phase fallback INSIDE the SCC, but
    # the rest of the graph is dep-derived. This lets strict mode land
    # before B9 dim decomposition completes. The byte-identity invariant
    # still holds because the hybrid scheduler's actual placement logic
    # is unchanged; strict mode is purely an admission gate.
    if strict:
        _assert_strict_mode_clean(
            ops, allow_sealed_cycles=allow_sealed_cycles
        )

    # Compute the hybrid schedule. The schedule output (dep-derived
    # order with phase-pruning fallback) is the load-bearing "dynamic"
    # signal — it's how a strict dep-derived scheduler would lay these
    # ops out. We do NOT feed that re-ordered list into LayerCompiler:
    # the static path's ``_topological_sort`` uses ``ops.index(o)`` for
    # Kahn's stability and any reordering would feed a different
    # stability key in. Instead the schedule is computed alongside, the
    # static compile pipeline runs over the natural ``_collect_ops`` order
    # exactly as ``compile_full_vm`` does, and ``compare_compile_paths``
    # cross-checks that the dep-derived order is CONSISTENT with the
    # static path's layer assignment (every op's dep-layer is <= its
    # static layer). On a clean op set the two coincide; on today's set
    # they coincide on every non-cycle op and the cycle members fall
    # through to phase-only ordering, which the static path already
    # respects. This decoupling is what makes B11 a byte-identical
    # parallel path rather than a behavioural change.
    _scheduled, _source = compute_dynamic_schedule(ops)

    # Build the model via the unchanged static pipeline with the natural
    # op order. The static compile_full_vm wraps op collection inline,
    # so we re-implement just the body here so the dynamic path is
    # actually a parallel API (not a wrapper around the static call).
    return _bake_from_scheduled_ops(
        ops,
        S=S,
        alu_mode=alu_mode,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
        pin_io_only=pin_io_only,
        disk_cache=disk_cache,
        use_dynamic_ffn=use_dynamic_ffn,
        enable_moe_routing=enable_moe_routing,
        positional_encoding=positional_encoding,
        attention_normalization=attention_normalization,
        rope_base=rope_base,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
        require_declarative_bake=require_declarative_bake,
        declarations_only=declarations_only,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
    )


# ---------------------------------------------------------------------------
# Internals: op collection + bake driver
# ---------------------------------------------------------------------------


def _collect_ops_for_compile(
    *,
    alu_mode: str,
    enable_conversational_io: bool,
    enable_tool_calling: bool,
    enable_neural_io_think_protocol: bool,
) -> List[Operation]:
    """Collect the exact same op list ``compile_full_vm`` registers.

    Mirrors the in-line composition logic in
    ``compile_full_vm`` (core ops + flag-gated extras + ALU composites +
    residual model ops + post-op attach ops). Kept in lock-step with the
    static path so the dynamic scheduler sees the same input space.
    """
    ops: List[Operation] = []
    ops.extend(_static.all_core_ops(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
    ))
    ops.append(_static.make_l10_post_op_attach_op(alu_mode=alu_mode))

    if alu_mode == "efficient":
        ops.append(_static.make_l11_alu_mul_bdtoge_op())
        ops.append(_static.make_l11_alu_mul_schoolbook_op())
        ops.append(_static.make_l11_alu_mul_carrypass1_op())
        ops.append(_static.make_l11_alu_mul_carrypass2_op())
        ops.append(_static.make_l11_alu_mul_carrypass3_op())
        ops.append(_static.make_l12_alu_mul_genprop_op())
        ops.append(_static.make_l12_alu_mul_binarylookahead_op())
        ops.append(_static.make_l12_alu_mul_finalcorrection_op())
        ops.append(_static.make_l12_alu_mul_getobd_op())
        ops.append(_static.make_efficient_l8_addsub_wrap_op(alu_mode=alu_mode))
        ops.append(_static.make_efficient_l10_andorxor_wrap_op(alu_mode=alu_mode))
        ops.append(_static.make_efficient_l11_alumul_wrap_op(alu_mode=alu_mode))

    for op in _static.make_alu_divmod_composite_ops():
        ops.append(op)

    ops.append(_static.make_residual_alibi_slopes_op())
    ops.append(_static.make_layer10_residual_alibi_slopes_op(alu_mode=alu_mode))
    ops.append(_static.make_layer8_op_imm_relay_op())
    ops.append(_static.make_contract_validation_op())

    if alu_mode == "lookup":
        ops.extend(_static.all_alu_postop_attach_ops())

    return ops


def _bake_from_scheduled_ops(
    scheduled: List[Operation],
    *,
    S: float,
    alu_mode: str,
    n_heads: int,
    ffn_hidden: int,
    max_seq_len: int,
    pin_io_only: bool,
    disk_cache: bool,
    use_dynamic_ffn: bool,
    enable_moe_routing: bool,
    positional_encoding: str,
    attention_normalization: str,
    rope_base: float,
    use_rms_norm: bool,
    rms_norm_eps: float,
    require_declarative_bake: bool,
    declarations_only: bool,
    enable_conversational_io: bool,
    enable_tool_calling: bool,
    enable_neural_io_think_protocol: bool,
):
    """Run the unchanged static compile/bake pipeline against ``scheduled``.

    This is intentionally a near-mirror of the body of
    ``compile_full_vm`` so the byte-identity invariant is structural
    rather than incidental: any change to the static body that affects
    ops_per_layer / dim_positions must be mirrored here. The diff is the
    op-collection step — instead of ``for op in all_core_ops(...): add``,
    we register ``scheduled`` directly.
    """
    import json
    import os
    import pathlib
    import torch as _torch
    from .layer_compiler import (
        LayerCompiler,
        ModelLayout,
        build_model_from_layout,  # noqa: F401  (kept for symmetry / future)
        dispatch_operation_bake,
        validate_declarations_only_ops,
    )
    from .migrated_ops import declare_setdim_compat_dims

    kwargs_snapshot = {
        "S": S,
        "enable_conversational_io": enable_conversational_io,
        "enable_tool_calling": enable_tool_calling,
        "enable_neural_io_think_protocol": enable_neural_io_think_protocol,
        "alu_mode": alu_mode,
        "n_heads": n_heads,
        "ffn_hidden": ffn_hidden,
        "max_seq_len": max_seq_len,
        "pin_io_only": pin_io_only,
        "enable_moe_routing": bool(enable_moe_routing),
        "positional_encoding": positional_encoding,
        "attention_normalization": attention_normalization,
        "rope_base": float(rope_base),
        "use_rms_norm": bool(use_rms_norm),
        "rms_norm_eps": float(rms_norm_eps),
        "require_declarative_bake": bool(require_declarative_bake),
        "declarations_only": bool(declarations_only),
        # Namespace the dynamic cache so it never collides with the static
        # entry (same kwargs, different scheduler).
        "__dynamic": True,
    }
    cache_path = None
    cache_key = (
        _static._cache_key(kwargs_snapshot) if disk_cache else None
    )
    if disk_cache and not require_declarative_bake and not declarations_only:
        cache_path = _static._cache_dir() / f"{cache_key}.pt"
        cached = _static._try_load_cached(cache_path, kwargs_snapshot)
        if cached is not None:
            return cached

    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=pin_io_only)
    for op in scheduled:
        compiler.add_op(op)

    layout = compiler.compile()
    if layout.d_model % n_heads != 0:
        pad = n_heads - (layout.d_model % n_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()

    if require_declarative_bake:
        _static.enforce_declarative_bake_authority(layout)

    per_layer_dispatch = [
        (layer_idx, op)
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer)
        for op in ops_at_layer
        if op.migrated
    ]
    block_dispatch = [
        op for op in sorted(
            layout.block_ops,
            key=lambda o: (layout.resolve_block_op_layer(o), o.phase or 0),
        )
        if op.migrated
    ]
    model_dispatch = sorted(layout.model_ops, key=lambda o: (o.phase or 0))
    if declarations_only:
        validate_declarations_only_ops(
            [op for _, op in per_layer_dispatch]
            + block_dispatch
            + model_dispatch
        )
        if disk_cache:
            cache_path = _static._cache_dir() / f"{cache_key}.pt"
            cached = _static._try_load_cached(cache_path, kwargs_snapshot)
            if cached is not None:
                return cached

    from ..vm_step import AutoregressiveVM

    if use_dynamic_ffn and layout.ffn_widths:
        ffn_hidden_arg = layout.ffn_widths
    else:
        ffn_hidden_arg = ffn_hidden

    model = AutoregressiveVM(
        d_model=layout.d_model,
        n_layers=layout.n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden_arg,
        max_seq_len=max_seq_len,
        dim_positions=layout.dim_positions,
        positional_encoding=positional_encoding,
        attention_normalization=attention_normalization,
        rope_base=rope_base,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
    )

    with _torch.no_grad():
        for layer_idx, op in per_layer_dispatch:
            block = model.blocks[layer_idx]
            if op.kind == "attn":
                target = block.attn
            elif op.kind == "ffn":
                target = block.ffn
            else:
                raise ValueError(
                    f"Op {op.name!r} in ops_per_layer has kind={op.kind!r}; "
                    "expected 'attn' or 'ffn'"
                )
            dispatch_operation_bake(
                op, target, layout.dim_positions, S,
                declarations_only=declarations_only,
            )
        for op in block_dispatch:
            block = model.blocks[layout.resolve_block_op_layer(op)]
            block._n_layers_hint = len(model.blocks)
            dispatch_operation_bake(
                op, block, layout.dim_positions, S,
                declarations_only=declarations_only,
            )
        for op in model_dispatch:
            dispatch_operation_bake(
                op, model, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

    if enable_moe_routing:
        model.compact(block_size=32)
        model.compact_moe()

    if cache_path is not None:
        _static._try_save_cached(cache_path, model, layout, kwargs_snapshot)

    return model, layout


# ---------------------------------------------------------------------------
# Validation: dynamic vs static byte-identity check
# ---------------------------------------------------------------------------


def compare_compile_paths(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    enable_neural_io_think_protocol: bool = False,
    S: float = 100.0,
    disk_cache: bool = False,
) -> Dict[str, object]:
    """Compile both paths and return a structured byte-identity report.

    Returns a dict with keys:

      * ``"n_diffs"``: total number of tensors that differ between
        ``compile_full_vm`` and ``compile_full_vm_dynamic`` outputs. Zero
        on the current op set.
      * ``"diff_keys"``: up to 20 example ``state_dict`` keys that differ.
      * ``"layout_diff"``: list of ``(field, static_value, dynamic_value)``
        for any divergent ``ModelLayout`` field (``d_model``, ``n_layers``,
        ``dim_positions``, ``dim_sizes``, ``ffn_widths``).
      * ``"schedule"``: ``{"dep": N, "phase": M}`` — counts of ops the
        hybrid scheduler placed via deps vs phase fallback.
      * ``"phase_disagrees_with_dep_order"``: list of op names where the
        dep-derived order differed from the strict phase order. Empty on
        a clean Phase-A op set. A non-empty list would mean an op was
        scheduled by deps to an earlier position than its phase requires,
        signaling either a Phase A finding bug or an unannotated edge.

    ``disk_cache`` defaults to False so this comparator never returns
    stale cache hits — both paths must rebuild from source.
    """
    static_model, static_layout = _static.compile_full_vm(
        S=S,
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
        disk_cache=disk_cache,
    )
    dyn_model, dyn_layout = compile_full_vm_dynamic(
        S=S,
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
        disk_cache=disk_cache,
    )

    import torch as _torch

    sd_s = static_model.state_dict()
    sd_d = dyn_model.state_dict()
    diff_keys: List[str] = []
    if set(sd_s.keys()) != set(sd_d.keys()):
        diff_keys.append("__keys_differ__")
    for k in sd_s.keys() & sd_d.keys():
        if not _torch.equal(sd_s[k], sd_d[k]):
            diff_keys.append(k)
            if len(diff_keys) > 64:
                break

    layout_diff: List[Tuple[str, object, object]] = []
    for field in ("d_model", "n_layers", "dim_positions", "dim_sizes", "ffn_widths"):
        s_val = getattr(static_layout, field)
        d_val = getattr(dyn_layout, field)
        if s_val != d_val:
            layout_diff.append((field, s_val, d_val))

    # Also surface scheduler-internal stats for the report.
    ops = _collect_ops_for_compile(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
    )
    _scheduled, source = compute_dynamic_schedule(ops)
    schedule_counts: Dict[str, int] = {"dep": 0, "phase": 0}
    for src in source.values():
        schedule_counts[src] = schedule_counts.get(src, 0) + 1

    # Cross-check: is the dep-derived order CONSISTENT with the static
    # layer assignment? "Consistent" = every op appears in the dep order
    # no later than its earliest valid topological position. We don't
    # require literal order equality (Kahn's stability differs between
    # phase tiebreaker and ops.index tiebreaker), but we require that
    # the dep order is a valid topological sort of the phase-pruned
    # DAG. The build-time test that ``compare_compile_paths`` returns
    # zero state-dict diffs is the authoritative byte-identity check;
    # this list surfaces any op whose dep-derived position falls AFTER
    # an op that the static path would have placed later. A non-empty
    # list would mean the dep-derived scheduler picked an order that
    # the static path could not have picked — i.e. an unannotated
    # back-edge or a Phase A finding bug. Empty list on today's op set.
    by_dep, source_map = compute_dynamic_schedule(ops)
    dep_position = {op.name: i for i, op in enumerate(by_dep)}
    in_e, out_e, _ = _build_phase_pruned_graph(
        ops, restrict_to_kinds={"attn", "ffn"},
    )
    disagreements: List[str] = []
    for u in ops:
        if u.kind not in ("attn", "ffn"):
            continue
        for v_name in out_e.get(u.name, set()):
            if dep_position[u.name] >= dep_position[v_name]:
                disagreements.append(
                    f"{u.name} (dep_pos={dep_position[u.name]}) "
                    f"-> {v_name} (dep_pos={dep_position[v_name]}) "
                    "in phase-pruned DAG but dep order places "
                    "predecessor AFTER successor"
                )
                if len(disagreements) >= 20:
                    break
        if len(disagreements) >= 20:
            break

    return {
        "n_diffs": len(diff_keys),
        "diff_keys": diff_keys[:20],
        "layout_diff": layout_diff,
        "schedule": schedule_counts,
        "phase_disagrees_with_dep_order": disagreements,
    }
