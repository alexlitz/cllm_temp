"""Phase 7.F.1: Static KV-cache liveness analyzer.

This module reads the declarative IR (``FFNRule`` / ``FFNOp`` /
``AttentionHeadIR`` / ``Operation``) and produces a per-step
``LivenessReport`` describing which KV-cache entries
``(layer, position, head, dim_name)`` are guaranteed dead at step S.

The analyzer is intentionally conservative: it never marks an entry
evictable unless there's *no* later op whose attention K/V projection
could read the dim, and (when ``treat_cycle_members_conservative=True``)
no producer of the entry sits in a dim-level dependency cycle (i.e. dims
that participate in a self-referential write/read loop across ops).

Coverage notes
--------------
This is the first pass. Only a few high-confidence categories produce
``evictable`` entries; everything else falls into ``conservative_keep``:

  * ``TEMP_*`` (and known per-step scratch dims) with no cross-step reader.
  * ``OUTPUT_LO_PREV_STEP`` (or any dim suffixed ``_PREV_STEP``) entries
    whose only consumer is the step immediately after they were written.
    Once that consuming step passes, the entry is dead.
  * Register-marker dims whose later writes overwrite the same residual
    cell (semantic overwrite) without any future K-read.

The analyzer is READ-ONLY; it never mutates the IR.

Algorithm sketch
----------------
1. Walk ``ops`` and build:
   * ``ffn_reads_by_op`` : dim names read by an op's FFNRules (conditions,
     gate, gate_terms).
   * ``ffn_writes_by_op`` : dim names written by an op's FFNRules.
   * ``attn_q_reads_by_op`` / ``attn_kv_reads_by_op`` /
     ``attn_o_writes_by_op`` : dim references used by attention heads.
   * ``op_step`` : the step number for each op (derived from declared
     ``step_idx`` when available, else 0).
2. Compute the dim-level dependency cycle membership:
   * Build a directed graph between dim names: ``A -> B`` if any op reads
     A and writes B. Any dim in a strongly-connected component of size
     >= 2, or with a self-loop, is "cycle member".
3. For each step S and each prospective KV entry ``(L, pos, head, dim)``:
   * If any later step's AttentionHeadIR K/V projection reads ``dim`` -> LIVE.
   * If the entry's dim is cycle-member and
     ``treat_cycle_members_conservative=True`` -> conservative keep.
   * Otherwise classify by category:
       - ``TEMP_*`` / known scratch -> evictable after step.
       - ``*_PREV_STEP`` -> evictable two steps after write.
       - Register-marker overwritten by later position write -> evictable.
       - Default: conservative keep.

The categorisation logic is heuristic and conservative on purpose; it
trades coverage for safety while the declarative coverage catches up.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVEntry:
    """One cached attention K/V entry.

    ``layer`` and ``head`` identify the projection matrix; ``position``
    identifies the token position the entry was written for; ``dim_name``
    is the semantic residual-stream dim the K/V projection sampled (e.g.
    ``"OUTPUT_HI"`` or ``"TEMP_THIS_STEP"``).
    """

    layer: int
    position: int
    head: int
    dim_name: str


@dataclass
class LivenessReport:
    """Per-step evictability report from ``analyze_kv_liveness``.

    Attributes:
        evictable_at_step: ``step -> set of KVEntry that can safely be dropped
            after that step completes``. Membership is conservative: every
            included entry is guaranteed dead under the static-IR view.
        cycle_conservative: KV entries the analyzer left LIVE because the dim
            they encode participates in a producer/consumer cycle (so we
            can't statically prove the next iteration won't read it).
        coverage: fraction of considered ``(step, entry)`` pairs the analyzer
            could classify as definitively evictable. ``1.0`` means the
            analyzer evicts every theoretical entry it inspected; in
            practice the initial implementation lands somewhere lower as it
            keeps unknown dims conservative.
    """

    evictable_at_step: Dict[int, Set[KVEntry]] = field(default_factory=dict)
    cycle_conservative: Set[KVEntry] = field(default_factory=set)
    coverage: float = 0.0


# ---------------------------------------------------------------------------
# Categorisation helpers
# ---------------------------------------------------------------------------


# Dim-name suffixes/prefixes that the analyzer treats as "scratch per
# step" — they hold values that are produced and consumed within one VM
# step and are conceptually dead the moment the step ends, provided no
# later step's attention reads them.
_PER_STEP_SCRATCH_PREFIXES: Tuple[str, ...] = (
    "TEMP",
    "ALU_TEMP",
    "MUL_TEMP",
    "DIV_TEMP",
    "MUL_ACCUM",
    "DIV_STAGING",
    "MEM_STAGING",
)
_PER_STEP_SCRATCH_SUFFIXES: Tuple[str, ...] = (
    "_THIS_STEP",
    "_SCRATCH",
)


# Dim-name suffixes that look like "previous step values" — written at
# step N for consumption at step N+1 only. Once step N+2 begins they're
# dead.
_PREV_STEP_SUFFIXES: Tuple[str, ...] = (
    "_PREV_STEP",
    "_PREV",
    "_LAST_STEP",
)


def _is_per_step_scratch(dim_name: str) -> bool:
    upper = dim_name.upper()
    if any(upper.startswith(prefix) for prefix in _PER_STEP_SCRATCH_PREFIXES):
        return True
    if any(upper.endswith(suffix) for suffix in _PER_STEP_SCRATCH_SUFFIXES):
        return True
    return False


def _is_prev_step_dim(dim_name: str) -> bool:
    upper = dim_name.upper()
    return any(upper.endswith(suffix) for suffix in _PREV_STEP_SUFFIXES)


# ---------------------------------------------------------------------------
# IR walking
# ---------------------------------------------------------------------------


def _ffn_rules_from_op(op) -> List:
    """Walk an op to find FFN rules. Mirrors writer_index._collect_ffn_rules_from_op."""

    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return []

    # CompilerIR with .layers[]
    if hasattr(ir, "layers"):
        rules = []
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


def _attention_heads_from_op(op) -> List:
    """Walk an op to find AttentionHeadIR instances."""

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


def _looks_like_ffn_rule(rule) -> bool:
    return (
        hasattr(rule, "conditions")
        and hasattr(rule, "writes")
        and hasattr(rule, "threshold")
    )


def _ffn_rule_reads(rule) -> Set[str]:
    """All dim names read by an FFN rule (conditions, gate, gate_terms)."""

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


def _ffn_rule_writes(rule) -> Set[str]:
    """All dim names written by an FFN rule."""

    names: Set[str] = set()
    for write in getattr(rule, "writes", ()):
        dim = getattr(write, "dim", None)
        if dim is not None:
            names.add(dim.name)
    return names


def _attn_head_kv_dims(head) -> Set[int]:
    """Residual-stream dim indices the head reads from K and V projections.

    AttentionHeadIR holds resolved dim positions (ints) inside its spec;
    they are useful for cross-checking that a *specific* KVEntry would
    actually be sampled by some head. Returns the union of K and V
    source dims.
    """

    spec = getattr(head, "spec", None)
    if spec is None:
        return set()
    dims: Set[int] = set()
    for write in getattr(spec, "k", ()):
        dims.add(int(write.dim))
    for write in getattr(spec, "v", ()):
        dims.add(int(write.dim))
    return dims


def _attn_head_q_dims(head) -> Set[int]:
    spec = getattr(head, "spec", None)
    if spec is None:
        return set()
    return {int(w.dim) for w in getattr(spec, "q", ())}


def _attn_head_o_dims(head) -> Set[int]:
    spec = getattr(head, "spec", None)
    if spec is None:
        return set()
    return {int(w.out_dim) for w in getattr(spec, "o", ())}


# ---------------------------------------------------------------------------
# Step inference
# ---------------------------------------------------------------------------


def _op_step(op) -> int:
    """Return the step index an op is anchored to, or 0 if unknown."""

    step_idx = getattr(op, "step_idx", None)
    if isinstance(step_idx, int):
        return step_idx
    # Anything non-int (None / "every" / sets / "after_first") is treated
    # as "fires at every step"; we model that as step 0 for the purposes
    # of liveness — the conservative path will keep the entry anyway.
    return 0


# ---------------------------------------------------------------------------
# Dim cycle detection
# ---------------------------------------------------------------------------


def _dim_cycle_members(ops: Sequence) -> Set[str]:
    """Return dims that participate in a producer/consumer cycle.

    A dim D is a cycle member if there exist ops O1, O2, ..., On (n>=1)
    and dims D = D0, D1, ..., Dn = D such that for each i, Oi reads
    Di-1 and writes Di. Self-loops count: a single op that both reads
    and writes D puts D in the cycle set.

    Built using Tarjan-style SCC on the projection of the use/def graph
    onto dim names.
    """

    # Build dim-to-dim edges from each op: read_dim -> write_dim.
    succ: Dict[str, Set[str]] = defaultdict(set)
    all_dims: Set[str] = set()
    for op in ops:
        reads = _all_op_reads(op)
        writes = _all_op_writes(op)
        all_dims.update(reads)
        all_dims.update(writes)
        for r in reads:
            for w in writes:
                succ[r].add(w)

    # Tarjan SCC
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
        for w in succ.get(v, ()):
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
            sccs.append(comp)

    for v in list(all_dims):
        if v not in indices:
            strongconnect(v)

    cycle_dims: Set[str] = set()
    for comp in sccs:
        if len(comp) > 1:
            cycle_dims.update(comp)
            continue
        # singleton — check for self-loop
        only = comp[0]
        if only in succ.get(only, ()):
            cycle_dims.add(only)
    return cycle_dims


def _all_op_reads(op) -> Set[str]:
    """Union of declared op.reads and dim names referenced by the op's IR."""

    names: Set[str] = set(getattr(op, "reads", ()) or ())
    for rule in _ffn_rules_from_op(op):
        names.update(_ffn_rule_reads(rule))
    # Attention heads operate on already-resolved positional ints, so
    # there are no dim names to fold in there. The op.reads declaration
    # is the canonical source for attention-side dim names.
    return names


def _all_op_writes(op) -> Set[str]:
    """Union of declared op.writes and dim names referenced by op's IR."""

    names: Set[str] = set(getattr(op, "writes", ()) or ())
    for rule in _ffn_rules_from_op(op):
        names.update(_ffn_rule_writes(rule))
    return names


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


def analyze_kv_liveness(
    ops: Sequence,
    n_steps: int,
    *,
    treat_cycle_members_conservative: bool = True,
    layers: Optional[Iterable[int]] = None,
    heads: Optional[Iterable[int]] = None,
) -> LivenessReport:
    """Static KV-liveness analysis over a declarative op corpus.

    Parameters
    ----------
    ops:
        Iterable of declarative ``Operation`` (or look-alike) objects.
        Each op is read-only.
    n_steps:
        Number of VM steps to reason about (typically the program's
        unrolled step count). Liveness is computed for steps in
        ``range(n_steps)``.
    treat_cycle_members_conservative:
        When True (default), dims that participate in a producer/consumer
        cycle are *never* marked evictable — even if the per-step
        category would say otherwise. This is the safe default.
    layers, heads:
        Optional restriction of (layer, head) pairs to consider. When
        omitted, the analyzer derives the universe from the
        ``AttentionHeadIR`` instances it finds in ``ops``.

    Returns
    -------
    LivenessReport
        Per-step evictable sets + cycle-conservative kept set + coverage
        ratio.
    """

    ops_list = list(ops)

    # 1) Walk IR.
    ffn_writes_by_op: Dict[str, Set[str]] = {}
    ffn_reads_by_op: Dict[str, Set[str]] = {}
    attn_kv_read_dim_names_by_op: Dict[str, Set[str]] = {}
    attn_heads_seen: Set[Tuple[int, int]] = set()
    for op in ops_list:
        name = getattr(op, "name", repr(op))
        op_ffn_reads: Set[str] = set()
        op_ffn_writes: Set[str] = set()
        for rule in _ffn_rules_from_op(op):
            op_ffn_reads.update(_ffn_rule_reads(rule))
            op_ffn_writes.update(_ffn_rule_writes(rule))
        ffn_reads_by_op[name] = op_ffn_reads
        ffn_writes_by_op[name] = op_ffn_writes

        # For attention K/V reads we record the declared op.reads as the
        # canonical dim-name set (positional integer dims in the spec
        # don't carry semantic names).
        attn_kv_read_dim_names_by_op[name] = set(getattr(op, "reads", ()) or ())

        for head in _attention_heads_from_op(op):
            attn_heads_seen.add(
                (int(getattr(op, "layer_idx", 0) or 0), head.head_idx)
            )

    # 2) Identify cycle dims.
    cycle_dims = _dim_cycle_members(ops_list)

    # 3) Index later-step KV/Q dim-name reads for fast lookup.
    #     later_kv_reads[step] = set of dim names that *some* op at step
    #     > step reads via attention.
    sorted_steps = sorted({_op_step(op) for op in ops_list} | {0})
    max_known_step = max(sorted_steps + [n_steps - 1])

    # For each step S, what attention-KV dim names will be referenced at
    # any step > S?
    attn_reads_at_or_after: Dict[int, Set[str]] = {}
    cumulative_later: Set[str] = set()
    # Iterate descending so we can accumulate "what is read at step >= k"
    # easily.
    by_step: Dict[int, List] = defaultdict(list)
    for op in ops_list:
        by_step[_op_step(op)].append(op)

    # Also remember every write per step for "semantic overwrite" check.
    writes_at_step: Dict[int, Set[str]] = defaultdict(set)
    for step, ops_at in by_step.items():
        for op in ops_at:
            writes_at_step[step].update(ffn_writes_by_op.get(getattr(op, "name", ""), set()))

    for step in range(max_known_step, -1, -1):
        # accumulate reads from this step too — used by callers asking
        # "what's read at step >= S?"; we'll subtract S itself below.
        for op in by_step.get(step, ()):
            cumulative_later.update(
                attn_kv_read_dim_names_by_op.get(getattr(op, "name", ""), set())
            )
            cumulative_later.update(
                ffn_reads_by_op.get(getattr(op, "name", ""), set())
            )
        attn_reads_at_or_after[step] = set(cumulative_later)

    def _read_at_any_step_at_or_after(step: int) -> Set[str]:
        """Dim names referenced by any op at step >= ``step``.

        A KV entry for position ``S`` becomes evictable only once every
        op that could attend back to position ``S`` has run. Since
        attention at step T can attend to all positions ``<=T``, the
        entry for ``S`` stays live for as long as any op at step ``>=S``
        still reads its dim. We therefore check inclusive of ``step``.
        """

        return attn_reads_at_or_after.get(step, set())

    # 4) Determine the (layer, head, dim) universe to consider.
    if layers is None:
        layer_set = {ly for (ly, _) in attn_heads_seen}
        if not layer_set:
            layer_set = {0}
    else:
        layer_set = set(int(x) for x in layers)
    if heads is None:
        head_set = {h for (_, h) in attn_heads_seen}
        if not head_set:
            head_set = {0}
    else:
        head_set = set(int(x) for x in heads)

    # Universe of dim names: any dim ever read or written by any op.
    all_dim_names: Set[str] = set()
    for op in ops_list:
        all_dim_names.update(_all_op_reads(op))
        all_dim_names.update(_all_op_writes(op))

    # 5) Per-step classification.
    evictable_at_step: Dict[int, Set[KVEntry]] = {s: set() for s in range(n_steps)}
    cycle_kept: Set[KVEntry] = set()
    total_considered = 0
    total_evicted = 0

    for step in range(n_steps):
        future_reads = _read_at_any_step_at_or_after(step)
        future_writes: Set[str] = set()
        for later_step in range(step + 1, n_steps):
            future_writes.update(writes_at_step.get(later_step, set()))

        for dim_name in all_dim_names:
            for layer in layer_set:
                for head in head_set:
                    entry = KVEntry(
                        layer=layer,
                        position=step,
                        head=head,
                        dim_name=dim_name,
                    )
                    total_considered += 1
                    if dim_name in future_reads:
                        # Some later op explicitly reads this dim — LIVE.
                        continue
                    if treat_cycle_members_conservative and dim_name in cycle_dims:
                        cycle_kept.add(entry)
                        continue

                    # Categorise: scratch / prev-step / overwrite.
                    if _is_per_step_scratch(dim_name):
                        evictable_at_step[step].add(entry)
                        total_evicted += 1
                        continue
                    if _is_prev_step_dim(dim_name):
                        # PREV_STEP entries written at step S are
                        # readable only at S+1. Once we've passed S+1
                        # they're dead. Mark evictable after step+1.
                        evict_step = step + 1
                        if evict_step < n_steps:
                            evictable_at_step[evict_step].add(entry)
                            total_evicted += 1
                        continue
                    if dim_name in future_writes:
                        # Semantic overwrite: same residual cell will be
                        # written later, and we've already confirmed no
                        # future K-read. Safe to evict.
                        evictable_at_step[step].add(entry)
                        total_evicted += 1
                        continue
                    # Default — conservative keep.

    coverage = (total_evicted / total_considered) if total_considered else 0.0
    return LivenessReport(
        evictable_at_step=evictable_at_step,
        cycle_conservative=cycle_kept,
        coverage=coverage,
    )


__all__ = [
    "KVEntry",
    "LivenessReport",
    "analyze_kv_liveness",
]
