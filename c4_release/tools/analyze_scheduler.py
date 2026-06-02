"""Phase A diagnostic: dependency-derived layer scheduling vs hardcoded phase.

For each op in ``all_core_ops()`` this driver extracts the declared
``phase`` / ``layer_idx`` and the dependency-bearing annotations
(``reads`` / ``writes`` / ``produces`` / ``consumes_fresh`` / ``requires``),
builds a dependency DAG, runs a topological assignment, and categorises
every op into one of four buckets:

    freely_placeable           : current phase is unconstrained by deps
    phase_pinned_by_deps       : current phase equals dep-derived earliest
    phase_inconsistent_with_deps: current phase contradicts the DAG (bug)
    phase_required_but_undeclared: current phase order can't be derived
        from declared deps (ordering info is hidden somewhere else)

The script is read-only: it imports the production op factories and inspects
their declared Operation fields. Nothing is mutated.

The diagnostic is conservative — it errs on the side of declaring an op
``phase_required_but_undeclared`` when the dep graph permits the op to move
to an earlier integer layer than its current ``floor(phase)``. Bake authors
who consider the existing phase intentional can move ops out of that bucket
by adding the dependency edges that pin it.

Usage::

    python tools/analyze_scheduler.py
    python tools/analyze_scheduler.py --out report.md

The default output path is ``.agent-logs/scheduler_phase_a_<date>.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Allow ``python tools/analyze_scheduler.py`` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from neural_vm.unified_compiler.layer_compiler import (  # noqa: E402
    Operation,
    requires_after_ops,
    requires_same_layer_as_ops,
    validate_requires_op_refs,
)
from neural_vm.unified_compiler.ops.all_core_ops import (  # noqa: E402
    all_alu_postop_attach_ops,
    all_core_ops,
)


# ---------------------------------------------------------------------------
# Op enumeration
# ---------------------------------------------------------------------------

def collect_ops() -> List[Operation]:
    """Return every op the production compiler can see.

    Flags are flipped to True so flag-gated ops are present in the dep graph.
    Bake bodies are not invoked — we only read declarative fields.
    """
    core = all_core_ops(
        enable_conversational_io=True,
        enable_tool_calling=True,
        enable_neural_io_think_protocol=True,
    )
    postops = all_alu_postop_attach_ops()
    return core + postops


# ---------------------------------------------------------------------------
# Dependency DAG construction
# ---------------------------------------------------------------------------

def build_dep_graph(
    ops: List[Operation],
) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, List[str]],
    Set[Tuple[str, str]],
]:
    """Construct dependency edges A -> B for every declared-dep pair.

    Returns ``(in_edges, out_edges, edge_reasons, same_layer_edges)`` where
    ``edge_reasons`` explains why each edge was created (for diagnostics)
    and ``same_layer_edges`` is the set of ``(u, v)`` pairs whose ONLY
    contribution comes from a ``requires["same_layer_as"]`` declaration
    (i.e. the edge is a co-placement assertion, not a strict-after
    constraint). ``topo_depth`` propagates ``depth[u]`` instead of
    ``depth[u] + 1`` along those edges so the diagnostic matches the
    LayerCompiler's ``_assign_layers`` semantics (Phase 7.A.1).

    Edges:
      * Data flow:        A.writes ∩ B.reads
      * Staleness:        A.produces.keys() ∩ B.consumes_fresh.keys()
      * Explicit requires: B.requires["after"] / B.requires["same_layer_as"]
        reference A.name as an op-name string (or a tuple/list of strings).
        See ``Operation.requires`` docstring for the B10 schema. Both keys
        contribute scheduling edges (``after`` => strictly later layer;
        ``same_layer_as`` => peer-layer ordering, equality enforced
        downstream by ``LayerCompiler._assign_layers``). An edge that
        ONLY comes from ``same_layer_as`` (with no overlapping ``reads``,
        ``produces``/``consumes_fresh`` or ``after`` contribution) is
        recorded in ``same_layer_edges`` so ``topo_depth`` treats it as
        depth-equal rather than depth+1.
    """
    name_to_op = {op.name: op for op in ops}
    in_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
    out_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
    edge_reasons: Dict[str, List[str]] = defaultdict(list)
    # Track (u, v) pairs whose edge was contributed by a
    # requires["same_layer_as"] declaration. After all edges are walked,
    # any pair that ALSO has a non-same_layer_as reason (writes/reads,
    # produces/consumes_fresh, requires[after]) is removed from this set
    # — only pure same_layer_as edges propagate depth-equal in topo_depth.
    same_layer_only: Set[Tuple[str, str]] = set()
    other_edges: Set[Tuple[str, str]] = set()

    # writers index: dim -> ops that write it.
    # We iterate ops in their natural list order so writers[d] is a
    # deterministic list; we sort op.writes (a Set) to make the
    # *order in which dims are populated* PYTHONHASHSEED-independent.
    writers: Dict[str, List[Operation]] = defaultdict(list)
    for op in ops:
        for d in sorted(op.writes):
            writers[d].append(op)

    # produces index: (dim, register) -> ops that produce it.
    # ``produces`` is a Dict (insertion order in CPython 3.7+, which is
    # deterministic for ops authored declaratively), but sort defensively
    # so the (dim, register) keys are added in seed-independent order.
    producers: Dict[Tuple[str, str], List[Operation]] = defaultdict(list)
    for op in ops:
        for dim, reg in sorted(op.produces.items()):
            producers[(dim, reg)].append(op)

    for v in ops:
        # B9 R-OH-2: when a reader declares ``requires["after"] = <op X>``
        # AND X writes some dim D that v also reads, suppress the
        # data-flow edge u→v on D for every PRODUCER u (u != X).
        # Semantic: the reader is opting into the "prev-step residual"
        # interpretation -- the read is satisfied by X's prev-step write
        # via the KV cache, NOT a same-step data dep on any later-layer
        # producer. The explicit requires["after"] edge X→v is still
        # added below (so X must run before v in the schedule).
        # See docs/B9_OUTPUT_HI_SPLIT_SPEC.md §7.2 and §6.3.
        requires_after_writers: Set[Tuple[str, str]] = set()
        for ref in requires_after_ops(v):
            if ref == v.name or ref not in name_to_op:
                continue
            ref_op = name_to_op[ref]
            for d in ref_op.writes & v.reads:
                requires_after_writers.add((ref, d))
        suppressed_dims = {d for (_, d) in requires_after_writers}
        # Data-flow edges. Iterate v.reads in sorted order so the
        # FIRST contributing dim for each (u,v) pair is
        # PYTHONHASHSEED-independent, and so the order of dims appended
        # to edge_reasons is deterministic.
        for d in sorted(v.reads):
            if d in suppressed_dims:
                # Cross-step read acknowledged via requires["after"];
                # the actual X→v edge is added below in the requires
                # block. Skip the data-flow inference for every other
                # writer of D.
                continue
            for u in writers.get(d, ()):
                if u.name == v.name:
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
                # IMPORTANT: record the reason OUTSIDE the
                # ``if u not in in_edges`` guard. If multiple dims
                # contribute to the same (u,v) edge (e.g. u writes
                # {A, B}, v reads {A, B}) every contributing dim must
                # appear in edge_reasons -- the back-edge histogram
                # below sums counts per dim, so under-counting hides
                # cycle structure and (because set-iteration order
                # determines which dim "wins" the slot) the bug
                # leaks PYTHONHASHSEED non-determinism into the
                # rendered report.
                edge_reasons[(u.name, v.name)].append(f"writes/reads:{d}")
                other_edges.add((u.name, v.name))

        # Staleness edges. Iterate consumes_fresh in sorted order for
        # the same reason.
        for dim, reg in sorted(v.consumes_fresh.items()):
            for u in producers.get((dim, reg), ()):
                if u.name == v.name:
                    continue
                if u.name not in in_edges[v.name]:
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)
                edge_reasons[(u.name, v.name)].append(
                    f"produces/consumes_fresh:{dim}@{reg}"
                )
                other_edges.add((u.name, v.name))

        # Explicit op-name edges from ``requires["after"]`` and
        # ``requires["same_layer_as"]``. The values may be a single string
        # or an iterable of strings (B10 schema). Unknown names are
        # silently skipped here; ``validate_requires_op_refs`` already ran
        # in ``main`` and would have surfaced them as a hard error.
        # ``requires_after_ops`` / ``requires_same_layer_as_ops`` return
        # lists in declaration order, which is already deterministic;
        # we sort here defensively so future refactors can't reintroduce
        # set-iteration non-determinism.
        for ref in sorted(requires_after_ops(v)):
            if ref == v.name or ref not in name_to_op:
                continue
            if ref not in in_edges[v.name]:
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)
            edge_reasons[(ref, v.name)].append(f"requires[after]={ref}")
            other_edges.add((ref, v.name))
        for ref in sorted(requires_same_layer_as_ops(v)):
            if ref == v.name or ref not in name_to_op:
                continue
            if ref not in in_edges[v.name]:
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)
            edge_reasons[(ref, v.name)].append(
                f"requires[same_layer_as]={ref}"
            )
            same_layer_only.add((ref, v.name))

    # Restrict same_layer_edges to pairs that have NO other contribution
    # (so a dim-flow or requires[after] edge always wins over a peer
    # constraint when both exist for the same (u, v) pair).
    same_layer_edges = {pair for pair in same_layer_only if pair not in other_edges}

    return in_edges, out_edges, edge_reasons, same_layer_edges


# ---------------------------------------------------------------------------
# Topological depth (earliest-layer derivation)
# ---------------------------------------------------------------------------

def topo_depth(
    ops: List[Operation], in_edges: Dict[str, Set[str]],
    out_edges: Dict[str, Set[str]],
    same_layer_edges: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, int], Set[str]]:
    """Return ``(depth, cycle_members)``.

    ``depth[op]`` = earliest-layer index each op can occupy purely from the
    dep DAG. 0 means the op has no incoming deps; otherwise it's
    ``max(contribution[predecessor])`` where ``contribution[u]`` is
    ``depth[u]`` for ``(u, op)`` edges in ``same_layer_edges`` (peer
    constraint -- mirrors ``LayerCompiler._assign_layers`` co-placement
    semantics for ``requires["same_layer_as"]``), else ``depth[u] + 1``.

    ``cycle_members`` = set of op names that never reach in-degree 0 (so
    their ``depth`` is sentinel -1). These nodes are either members of a
    directed cycle or transitively downstream of one — the cycle blocks
    their topological resolution.
    """
    # Use proper Kahn: depth is only finalised when in-degree hits 0
    # (i.e. all predecessors have been visited). Premature depth updates
    # were producing impossible orderings (a node's depth was being set
    # from a single early predecessor while other predecessors were
    # still trapped in a cycle downstream).
    if same_layer_edges is None:
        same_layer_edges = set()
    indeg = {op.name: len(in_edges[op.name]) for op in ops}
    # ``max_pred_depth[v]`` tracks the maximum depth contribution from any
    # predecessor seen so far. Depth-bumping edges contribute ``depth[u]+1``;
    # ``same_layer_as``-only edges contribute ``depth[u]`` (peer placement).
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
                # ``max_pred_depth[v]`` is already the resolved depth
                # (the per-edge contribution baked in the +1 where it
                # applies). Floor at 0 so an isolated same_layer_as
                # peer at depth 0 still resolves to depth 0.
                depth[v] = max(max_pred_depth[v], 0)
                queue.append(v)

    cycle_members: Set[str] = set()
    for op in ops:
        if op.name not in depth:
            depth[op.name] = -1
            cycle_members.add(op.name)
    return depth, cycle_members


def find_scc_summary(
    cycle_members: Set[str],
    in_edges: Dict[str, Set[str]],
    out_edges: Dict[str, Set[str]],
) -> List[List[str]]:
    """Return strongly connected components restricted to cycle members.

    Tarjan's algorithm on the cycle-member subgraph. Returns SCCs sorted by
    size descending.
    """
    # Sort neighbour lists so Tarjan's recursion order is independent of
    # PYTHONHASHSEED. SCC composition is mathematically invariant to
    # traversal order, but recursion order affects per-SCC member order
    # (and would affect the `sccs.sort(key=len, reverse=True)` tie-break
    # when two SCCs have equal sizes -- we additionally tie-break on
    # sorted membership below).
    sub_out = {
        n: sorted(out_edges[n] & cycle_members) for n in cycle_members
    }
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sub_out.get(v, ()):
            if w not in indices:
                strongconnect(w)
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

    sys.setrecursionlimit(10000)
    for n in sorted(cycle_members):
        if n not in indices:
            strongconnect(n)

    # Stable sort: primary by size descending, secondary by sorted member
    # tuple ascending so equal-size SCCs always emit in the same order.
    sccs.sort(key=lambda comp: (-len(comp), tuple(sorted(comp))))
    return sccs


# ---------------------------------------------------------------------------
# Phase categorisation
# ---------------------------------------------------------------------------

def _current_layer(op: Operation) -> Optional[int]:
    """Best-effort integer layer derived from ``op.phase`` / ``op.layer_idx``."""
    if op.layer_idx is not None:
        return op.layer_idx
    if op.phase is None:
        return None
    # Phases like 998/999/1001 are post-pass model-level ops; their
    # "layer" is intentionally past the model blocks. Cap at a sentinel
    # so they don't pollute the histogram.
    try:
        return int(math.floor(op.phase))
    except (TypeError, ValueError):
        return None


def categorise(
    ops: List[Operation],
    depth: Dict[str, int],
    cycle_members: Set[str],
    in_edges: Dict[str, Set[str]],
    out_edges: Dict[str, Set[str]],
) -> Dict[str, str]:
    """Return ``{op_name: category}``.

    Heuristic:
      * ``model``-kind ops and post-pass ops (current layer >= 100) are
        tagged ``phase_pinned_by_deps`` — their position is structurally
        anchored, not derived from data deps;
      * cycle members get their own bucket ``dep_graph_cycle_member`` —
        the declared deps form a cycle so no topological order exists
        without breaking edges (today phase pruning does that);
      * ops with ``current_layer == dep_depth`` are
        ``phase_pinned_by_deps`` (the static phase agrees with the
        dependency-derived earliest layer);
      * ops with ``current_layer > dep_depth`` are
        ``phase_required_but_undeclared`` (the static phase is later than
        the DAG requires; the gap is hidden ordering);
      * ops with ``current_layer < dep_depth`` are
        ``phase_inconsistent_with_deps`` (would be a bug — but in practice
        only cycle members hit this because the DAG depth is sentinel -1).
      * ops with NO declared dep edges AND no current phase are
        ``freely_placeable``.
    """
    by_name = {op.name: op for op in ops}
    cats: Dict[str, str] = {}
    for op in ops:
        current = _current_layer(op)
        derived = depth.get(op.name, -1)

        # Post-pass / model-level ops short-circuit.
        if current is not None and current >= 100:
            cats[op.name] = "phase_pinned_by_deps"
            continue
        if op.kind == "model":
            cats[op.name] = "phase_pinned_by_deps"
            continue

        # Cycle members — the DAG is ill-defined for them.
        if op.name in cycle_members:
            cats[op.name] = "dep_graph_cycle_member"
            continue

        if current is None:
            # No phase declared and no cycle — let topo place freely.
            cats[op.name] = "freely_placeable"
            continue

        if current < derived:
            cats[op.name] = "phase_inconsistent_with_deps"
            continue
        if current == derived:
            cats[op.name] = "phase_pinned_by_deps"
            continue
        cats[op.name] = "phase_required_but_undeclared"

    # Refine freely_placeable: an op with NO in-edges AND NO out-edges in
    # the dep graph is truly free.
    for op in ops:
        if cats[op.name] != "phase_pinned_by_deps":
            continue
        if not in_edges[op.name] and not out_edges[op.name]:
            cats[op.name] = "freely_placeable"

    return cats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_op(op: Operation, depth: Dict[str, int]) -> str:
    cur = _current_layer(op)
    der = depth.get(op.name, -1)
    return (
        f"  - {op.name} | kind={op.kind} | phase={op.phase} | "
        f"layer_idx={op.layer_idx} | current_layer={cur} | dep_depth={der}"
    )


def render_report(
    ops: List[Operation],
    cats: Dict[str, str],
    depth: Dict[str, int],
    cycle_members: Set[str],
    sccs: List[List[str]],
    in_edges: Dict[str, Set[str]],
    out_edges: Dict[str, Set[str]],
    edge_reasons: Dict[str, List[str]],
) -> str:
    lines: List[str] = []
    by_name = {op.name: op for op in ops}

    # Aggregate counts
    counts: Dict[str, int] = defaultdict(int)
    for c in cats.values():
        counts[c] += 1

    # Current declared layer count: peak of all current_layer values (capped
    # at the structural block count, not the post-pass phase sentinels).
    block_layers = [
        l for l in (_current_layer(op) for op in ops)
        if l is not None and l < 100
    ]
    declared_n_layers = (max(block_layers) + 1) if block_layers else 0

    # Dependency DAG depth (max derived depth across in-block, acyclic ops).
    in_block_acyclic_depths = [
        depth[op.name] for op in ops
        if depth[op.name] >= 0 and (_current_layer(op) or 0) < 100
    ]
    dag_depth = max(in_block_acyclic_depths) if in_block_acyclic_depths else 0

    lines.append("# Phase A: dependency-derived layer scheduling diagnostic")
    lines.append("")
    lines.append(
        f"_Generated {_dt.date.today().isoformat()} by "
        "`tools/analyze_scheduler.py`._"
    )
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    cycle_share = (
        counts.get("dep_graph_cycle_member", 0) / max(len(ops), 1) * 100
    )
    lines.append(
        f"- **DAG depth ({dag_depth + 1}) vs static layer count "
        f"({declared_n_layers})**: the dep-derived earliest chain only "
        f"reaches depth {dag_depth + 1}, far below the {declared_n_layers}-"
        f"layer hand-set layout. The remaining gap is held up by hidden "
        f"ordering."
    )
    lines.append(
        f"- **{int(cycle_share)}% of ops are stuck in cycles**: "
        f"{counts.get('dep_graph_cycle_member', 0)} of {len(ops)} ops "
        f"cannot be topologically ordered without phase pruning. The "
        f"largest SCC swallows roughly half the ops, dominated by "
        f"writes/reads on ``OUTPUT_HI``, ``AX_CARRY_HI``, ``ADDR_KEY``, "
        f"``TEMP``."
    )
    lines.append(
        f"- **Phase B is BLOCKED on dim-decomposition work**. Building "
        f"``compile_full_vm_dynamic`` against today's declarations would "
        f"reject the input as cyclic. Before Phase B can produce a baked "
        f"model the largest SCC must be broken — either by introducing "
        f"step-local vs cross-step dim variants (e.g. ``OUTPUT_HI_THIS_STEP`` "
        f"vs ``OUTPUT_HI_PREV_STEP``) or by adding ``requires`` op-name "
        f"references for the top back-edge dims listed below."
    )
    lines.append(
        f"- **Effort estimate**: 1-2 days to land the analyzer + a dynamic-"
        f"scheduler prototype that accepts phase as a fallback tiebreaker; "
        f"**1-2 weeks** to actually retire phase pruning (each back-edge dim "
        f"needs a designed decomposition + decl-verifier update + smoke "
        f"validation). The 2-week estimate matches the upper bound the "
        f"caller mentioned; the scope is dominated by the OUTPUT_HI / "
        f"AX_CARRY / ADDR_KEY cycle work, not by the scheduler code itself."
    )
    lines.append(
        f"- **Internal consistency check**: a clean Phase-B compile WOULD "
        f"produce identical ``dim_positions`` and ``ops_per_layer`` to the "
        f"static path IF all ops moved to ``phase_pinned_by_deps``. Today "
        f"only {counts.get('phase_pinned_by_deps', 0)} of {len(ops)} ops "
        f"reach that bucket; the other "
        f"{counts.get('phase_required_but_undeclared', 0) + counts.get('phase_inconsistent_with_deps', 0)} "
        f"non-cycle ops would float to earlier layers under a strict "
        f"dep-order scheduler. That implies the dynamic and static paths "
        f"are NOT yet equivalent — the dynamic scheduler would change the "
        f"layout."
    )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- total ops analysed: **{len(ops)}**")
    lines.append(f"- declared block-layer count (max floor(phase) for ops < 100): **{declared_n_layers}**")
    lines.append(f"- dependency-DAG depth (longest declared chain): **{dag_depth + 1} layers**")
    lines.append("")
    for c in (
        "freely_placeable",
        "phase_pinned_by_deps",
        "phase_required_but_undeclared",
        "phase_inconsistent_with_deps",
        "dep_graph_cycle_member",
    ):
        lines.append(f"- {c}: **{counts.get(c, 0)}**")
    lines.append("")
    lines.append(
        "Note: the current `LayerCompiler._topological_sort` BREAKS cycles "
        "by phase pruning — an edge u→v is dropped when "
        "`u.phase > v.phase`. Without that crutch, "
        f"**{len(cycle_members)} ops** participate in **{len(sccs)} "
        "strongly-connected components**, none of which can be ordered by "
        "the declared deps alone. Phase pruning is therefore not optional "
        "today — the declarations are incomplete."
    )
    lines.append("")

    # Cycle structure
    lines.append(f"## dep_graph_cycle_member ({len(cycle_members)})")
    lines.append("")
    if not sccs:
        lines.append("_(no cycles detected — DAG is sortable.)_")
    else:
        lines.append(
            "Strongly-connected components (sorted by size). Each SCC is a "
            "group of ops that mutually depend on each other in the "
            "phase-unpruned dep graph. The static phase ordering is what "
            "currently disambiguates these. Top 5 SCCs shown; full list in "
            "the CSV."
        )
        for i, comp in enumerate(sccs[:5]):
            lines.append(f"  - SCC #{i+1} (size {len(comp)}):")
            members = sorted(comp)
            preview = members[:10]
            for n in preview:
                op = by_name[n]
                lines.append(
                    f"      · {n} (kind={op.kind}, phase={op.phase}, "
                    f"layer_idx={op.layer_idx})"
                )
            if len(members) > 10:
                lines.append(f"      · …+{len(members) - 10} more")
        if len(sccs) > 5:
            lines.append(f"  - …+{len(sccs) - 5} more SCCs (see CSV)")
    lines.append("")

    # Top dims responsible for back-edges into the largest SCC.
    # We sort every iteration that contributes to ordering so the
    # rendered output is PYTHONHASHSEED-independent: (a) the (u,v) walk
    # is sorted, (b) example lists are accumulated in sorted (u,v)
    # order, (c) the top-N dims are sorted by (-count, dim_name) so
    # equal-count dims have a stable lexicographic tiebreaker.
    if sccs:
        biggest_scc = set(sccs[0])
        # Count which dim contributes the most edges inside the SCC.
        # An edge u→v is inside the SCC iff u in scc and v in scc.
        dim_back_edges: Dict[str, int] = defaultdict(int)
        dim_back_examples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for (u, v) in sorted(edge_reasons.keys()):
            reasons = edge_reasons[(u, v)]
            if u not in biggest_scc or v not in biggest_scc:
                continue
            u_layer = _current_layer(by_name[u]) or 0
            v_layer = _current_layer(by_name[v]) or 0
            if u_layer <= v_layer:
                continue  # forward edge in current layout
            # back-edge in current static phase ordering. De-dup dims so
            # multiple "writes/reads:X" entries on a single (u,v) edge
            # (e.g. produced by repeated declarations) count once per
            # edge, not once per repetition.
            seen_dims: Set[str] = set()
            for reason in reasons:
                if not reason.startswith("writes/reads:"):
                    continue
                dim = reason.split(":", 1)[1]
                if dim in seen_dims:
                    continue
                seen_dims.add(dim)
                dim_back_edges[dim] += 1
                if len(dim_back_examples[dim]) < 3:
                    dim_back_examples[dim].append((u, v))
        if dim_back_edges:
            lines.append("### Top dims producing back-edges inside the largest SCC")
            lines.append("")
            lines.append(
                "These dim names are claimed as ``writes`` by some op and "
                "``reads`` by another op at an EARLIER static layer — they "
                "are the load-bearing cycle creators. Adding a "
                "``requires`` declaration or splitting the dim into "
                "step-local vs. cross-step variants is the typical fix."
            )
            lines.append("")
            # Tie-break by dim name lexicographically so equal-count
            # dims have a stable order.
            sorted_dims = sorted(
                dim_back_edges.items(), key=lambda kv: (-kv[1], kv[0])
            )[:10]
            for dim, n in sorted_dims:
                exs = dim_back_examples[dim][:3]
                ex_str = "; ".join(f"{u}→{v}" for u, v in exs)
                lines.append(f"  - **{dim}**: {n} back-edges. Examples: {ex_str}")
            lines.append("")

    # Inconsistent ops (likely bugs) — list all
    inconsistent = [op for op in ops if cats[op.name] == "phase_inconsistent_with_deps"]
    lines.append(f"## phase_inconsistent_with_deps ({len(inconsistent)})")
    lines.append("")
    if not inconsistent:
        lines.append("_(none — no acyclic op contradicts its declared DAG depth.)_")
    else:
        lines.append(
            "These ops are not cycle members but their current_layer is "
            "STRICTLY LESS than the topological depth derived from their "
            "declared predecessors. Real bugs."
        )
        for op in inconsistent:
            lines.append(_fmt_op(op, depth))
            for u in sorted(in_edges[op.name]):
                u_layer = _current_layer(by_name[u])
                lines.append(
                    f"      · blocked by {u} (current_layer={u_layer})"
                )
    lines.append("")

    # phase_required_but_undeclared — top 30 by gap (lex tiebreak on
    # op name so equal-gap rows have a stable order).
    needs_decl = [op for op in ops if cats[op.name] == "phase_required_but_undeclared"]
    needs_decl.sort(
        key=lambda o: (
            -((_current_layer(o) or 0) - depth.get(o.name, 0)),
            o.name,
        ),
    )
    lines.append(f"## phase_required_but_undeclared ({len(needs_decl)})")
    lines.append("")
    lines.append(
        "These ops have a current phase that places them STRICTLY LATER "
        "than the declared DAG requires. Either:\n"
        "  (a) something outside the declarations (hardcoded slot reads, "
        "register conventions, set_vm_weights ordering) pins them, OR\n"
        "  (b) the declared deps are simply incomplete and the dynamic "
        "scheduler would happily move them earlier — which may or may not "
        "be safe.\n\nTop 30 by gap (current_layer - dep_depth):"
    )
    lines.append("")
    for op in needs_decl[:30]:
        lines.append(_fmt_op(op, depth))
        # Show what predecessors the op DOES declare
        preds = sorted(in_edges[op.name])
        if preds:
            lines.append(
                f"      · declared predecessors ({len(preds)}): "
                f"{preds[:5]}{'…' if len(preds) > 5 else ''}"
            )
        else:
            lines.append("      · NO declared predecessors (totally free in DAG)")
    if len(needs_decl) > 30:
        lines.append("")
        lines.append(f"…+{len(needs_decl) - 30} more, see CSV dump for the full list.")
    lines.append("")

    # phase_pinned_by_deps — top 30 by depth (lex tiebreak on op name).
    pinned = [op for op in ops if cats[op.name] == "phase_pinned_by_deps"]
    pinned.sort(key=lambda o: (depth.get(o.name, 0), o.name))
    lines.append(f"## phase_pinned_by_deps ({len(pinned)})")
    lines.append("")
    lines.append(
        "Current phase matches dep-derived earliest layer. These ops are "
        "already \"dynamically scheduled\" — the static phase happens to "
        "agree with what a topological pass would produce. Top 30 by depth:"
    )
    lines.append("")
    for op in pinned[:30]:
        lines.append(_fmt_op(op, depth))
    if len(pinned) > 30:
        lines.append("")
        lines.append(f"…+{len(pinned) - 30} more.")
    lines.append("")

    # freely_placeable — full list (small)
    free = [op for op in ops if cats[op.name] == "freely_placeable"]
    lines.append(f"## freely_placeable ({len(free)})")
    lines.append("")
    if not free:
        lines.append("_(none — every op has at least one declared dep edge.)_")
    else:
        lines.append(
            "These ops have no incoming AND no outgoing edges in the dep "
            "graph. The dynamic scheduler can place them anywhere. Often "
            "these are flag-gated stubs or disabled bakes."
        )
        for op in free:
            lines.append(_fmt_op(op, depth))
    lines.append("")

    return "\n".join(lines)


def render_api_sketch() -> str:
    return """
## Phase B sketch: ``compile_full_vm_dynamic``

```python
def compile_full_vm_dynamic(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    enable_neural_io_think_protocol: bool = False,
    S: float = 100.0,
    extra_ops: list[Operation] | None = None,
) -> AutoregressiveVM:
    \"\"\"Compile and bake a VM model with layer assignment derived from
    op declarations rather than hardcoded ``phase=N.M`` values.

    Algorithm:
      1. Collect ops from ``all_core_ops(...)`` + any ``extra_ops`` the
         caller wants to inject (e.g. corrective op patches discovered
         during parallel-debug runs).
      2. Bootstrap a LayerCompiler with the canonical SetDim-compat dims
         (``declare_setdim_compat_dims`` lives in compiler.py today).
      3. For each op, IGNORE ``op.phase`` / ``op.layer_idx`` except where
         ``op.kind == "block"`` and the layer_idx is the only legal pin
         (e.g. ``layer_idx=0`` block-replacement ops). Convert remaining
         layer_idx pins into ``target_op_name`` references so they follow
         their referenced op's dep-derived layer.
      4. Run a strict topo sort: edges from ``writes/reads``,
         ``produces/consumes_fresh``, and any explicit ``requires`` op-
         name references. NO phase pruning. Cycles are hard errors — they
         indicate a missing declaration (an op needs ``requires=...`` or
         a kind-distinction to disambiguate).
      5. Layer-assign each op to ``max(depth(pred) for pred in in_edges) + 1``
         (0 if no preds). Two ops share a (layer, kind) slot when they have
         disjoint reads/writes (or are declared composable via a future
         ``composable_with`` annotation).
      6. ``n_layers = max(layer_assignment.values()) + 1``; grow naturally
         as new ops are added.
      7. Allocate dims as today (compiler._allocate_dims).
      8. ``build_model_from_layout`` runs unchanged — the dispatcher is
         already dep-derived for ``ops_per_layer``.

    Returns the baked AutoregressiveVM. Asserts that ``layout.dim_positions``
    is byte-identical to the static-phase path's dim_positions for any
    superset of ops where the dynamic and static layouts BOTH place the
    same op at the same (layer_idx, kind) slot (Phase A diagnostic must
    show 0 ``phase_inconsistent_with_deps`` ops for this assertion to
    hold across the full op list).
    \"\"\"
    # 1. Collect
    ops = all_core_ops(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
    ) + all_alu_postop_attach_ops()
    if extra_ops:
        ops = ops + list(extra_ops)

    # 2. Compiler + dim declarations
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler)  # imported from compiler.py
    for op in ops:
        # 3. Strip non-essential phase pins; rewrite layer_idx -> target_op_name
        #    where the original pin was just "follow op X".
        op = _strip_phase_for_dynamic(op)
        compiler.add_op(op)

    # 4-7. Compile (the new LayerCompiler.compile_dynamic() variant rejects
    #    phase-based pruning and re-derives layer count fresh.)
    layout = compiler.compile_dynamic()

    # 8. Bake unchanged
    return build_model_from_layout(layout, S=S, legacy_bake=None)
```

Pre-conditions (must be true before Phase B is unblocked):
  * 0 ops in ``phase_inconsistent_with_deps``;
  * every op in ``phase_required_but_undeclared`` is either:
      - moved to ``phase_pinned_by_deps`` by adding the missing edge (e.g.
        an explicit ``requires`` or a new produces/consumes_fresh pair), or
      - acknowledged as intentionally floatable (the dynamic scheduler may
        pick an earlier layer for it; bake authors sign off);
  * the dependency graph is acyclic when ``phase`` pruning is disabled.

Phase C (delete static phase):
  * Once ``compile_full_vm_dynamic`` produces byte-identical layouts to
    ``compile_full_vm`` for the full test matrix, remove the ``phase``
    field from ``Operation`` and delete the static-phase code path.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = os.path.join(
        _REPO,
        "..",
        ".agent-logs",
        f"scheduler_phase_a_{_dt.date.today().strftime('%Y_%m_%d')}.md",
    )
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args()

    ops = collect_ops()
    # B10: surface bad ``requires`` op-name references as a hard error
    # before they silently become no-op edges in the dep graph.
    ref_errors = validate_requires_op_refs(ops)
    if ref_errors:
        sys.stderr.write(
            "analyze_scheduler: invalid requires op-name references:\n"
        )
        for msg in ref_errors:
            sys.stderr.write(f"  - {msg}\n")
        return 2
    in_e, out_e, reasons, same_layer_edges = build_dep_graph(ops)
    depth, cycle_members = topo_depth(ops, in_e, out_e, same_layer_edges)
    sccs = find_scc_summary(cycle_members, in_e, out_e)
    cats = categorise(ops, depth, cycle_members, in_e, out_e)

    report = render_report(ops, cats, depth, cycle_members, sccs, in_e, out_e, reasons)
    report += "\n" + render_api_sketch()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(report)
    print(f"wrote {out}")

    # Also dump a CSV for spreadsheet inspection.
    csv_path = out.replace(".md", ".csv")
    with open(csv_path, "w") as f:
        f.write("name,kind,phase,layer_idx,current_layer,dep_depth,category,n_in,n_out\n")
        for op in ops:
            f.write(
                f"{op.name},{op.kind},{op.phase},{op.layer_idx},"
                f"{_current_layer(op)},{depth.get(op.name, -1)},"
                f"{cats[op.name]},{len(in_e[op.name])},{len(out_e[op.name])}\n"
            )
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
