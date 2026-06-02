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
The high-confidence categories that produce ``evictable`` entries:

  * ``TEMP_*`` (and known per-step scratch dims) with no cross-step reader.
  * ``OUTPUT_LO_PREV_STEP`` (or any dim suffixed ``_PREV_STEP``) entries
    whose only consumer is the step immediately after they were written.
    Once that consuming step passes, the entry is dead.
  * Register-marker dims whose later writes overwrite the same residual
    cell (semantic overwrite) without any future K-read.
  * Per-step register channels (``REG_*``, ``OP_*``, ``ADDR_*``,
    ``ALU_*``, ``MEM_*``, ``FETCH_*``, ``BYTE_INDEX_*``, ``EMBED_*``)
    whose every-step writer dominates every reader by phase order. The
    cache entry from step S is provably dead once step S+1's writer
    fires (Phase 7.F.6).

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
     ``step_idx`` when available, else "every step").
   * ``op_phase`` : within-step ordering hint (smaller = earlier).
2. Phase 7.F.6 cycle refinement:
   * A dim D is a "same-step transient" if every reader of D has at
     least one writer that fires same-step with phase <= reader's phase
     (writer dominates reader → value is fresh same-step input, not a
     carry from the previous step). Same-step transients are NOT cycle
     members for KV-cache purposes.
   * A dim D is a true "cross-step cycle" iff:
       - its name ends with one of the cross-step suffixes
         (``_PREV_STEP`` / ``_PREV`` / ``_LAST_STEP``), OR
       - it has a reader with NO dominating same-step writer (writer
         either fires only at specific later steps or always lags the
         reader by phase) and no fires-every-step writer overrides it.
   * SCC self-loops are no longer automatic cycle markers — a self-loop
     where the writer's phase dominates the reader's phase is treated
     as same-step transient.
3. Phase 7.F.6 semantic-overwrite expansion:
   * Treat ops with ``step_idx=None`` (or "every") as writing at every
     step. ``writes_at_step[s]`` for any step ``s`` then includes the
     full every-step writer set, so register channels picked up by
     decode-style writers become evictable after each step's write.
4. For each step S and each prospective KV entry ``(L, pos, head, dim)``:
   * If any later step's AttentionHeadIR K/V projection reads ``dim`` -> LIVE.
   * If the entry's dim is cycle-member and
     ``treat_cycle_members_conservative=True`` -> conservative keep.
   * Otherwise classify by category:
       - ``TEMP_*`` / known scratch -> evictable after step.
       - ``*_PREV_STEP`` -> evictable one step after write (after the
         consuming step passes).
       - Register-marker overwritten by NEXT step's write -> evictable.
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
        attention_read_dim_names: dim names that are read by any
            attention head's K/V projection (across all layers / steps).
            Consumers like ``build_state_from_report`` use this to
            restrict the runtime AND universe — dims that aren't sampled
            by any attention K/V projection cannot keep a cache row
            alive, so they don't need to participate in the
            ``every-dim-must-be-dead`` row-level check.
        attention_read_dim_names_by_layer: per-layer dim universe for
            attention K/V reads. Maps ``layer_idx -> set of dim names``
            read by any attention head pinned to that layer (via
            ``op.layer_idx``). When a layer is missing from this map,
            consumers fall back to the global ``attention_read_dim_names``.
    """

    evictable_at_step: Dict[int, Set[KVEntry]] = field(default_factory=dict)
    cycle_conservative: Set[KVEntry] = field(default_factory=set)
    coverage: float = 0.0
    attention_read_dim_names: Set[str] = field(default_factory=set)
    attention_read_dim_names_by_layer: Dict[int, Set[str]] = field(
        default_factory=dict
    )


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


# Sentinel for "fires at every step" (op.step_idx is None / "every" / etc.).
# We model these as a special marker in the step set; the analyzer treats
# them as if they were members of every step in ``range(n_steps)``.
_EVERY_STEP = "__every__"


def _op_step(op) -> int:
    """Return the step index an op is anchored to, or 0 if unknown.

    Legacy helper kept for backwards-compatible tests that probe a single
    step. New code should call :func:`_op_step_set` which returns a
    structured representation (set of ints OR the ``_EVERY_STEP``
    sentinel).
    """

    step_idx = getattr(op, "step_idx", None)
    if isinstance(step_idx, int):
        return step_idx
    # Anything non-int (None / "every" / sets / "after_first") is treated
    # as "fires at every step"; we model that as step 0 for the purposes
    # of liveness — the conservative path will keep the entry anyway.
    return 0


def _op_step_set(op):
    """Return the structured step set an op fires at.

    Returns either:
      * ``_EVERY_STEP`` — op fires at every step (``step_idx`` is ``None``
        or ``"every"`` or other non-int hint).
      * A ``frozenset[int]`` of concrete step indices when ``step_idx``
        is an int OR an iterable of ints.

    "after_first" is conservatively treated as ``_EVERY_STEP`` (we don't
    know how many steps the program runs for; treating it as every-step
    is the conservative-for-cycles, optimistic-for-eviction choice).
    """

    step_idx = getattr(op, "step_idx", None)
    if step_idx is None:
        return _EVERY_STEP
    if isinstance(step_idx, int):
        return frozenset({step_idx})
    if isinstance(step_idx, str):
        # "every" / "after_first" / other free-form hints → every step.
        return _EVERY_STEP
    try:
        ints = frozenset(int(s) for s in step_idx)
        return ints if ints else _EVERY_STEP
    except (TypeError, ValueError):
        return _EVERY_STEP


def _op_phase(op) -> Optional[float]:
    """Return the op's declared phase ordering hint, or None when unset."""

    phase = getattr(op, "phase", None)
    if phase is None:
        return None
    try:
        return float(phase)
    except (TypeError, ValueError):
        return None


# Register-channel dim-name prefixes whose values are typically refreshed
# every step by a same-step writer (decode, ALU dispatch, address
# resolution, etc.). When the analyzer detects an every-step writer for
# such a dim AND every reader is phase-dominated by the writer, the dim
# becomes an evictable per-step transient (Phase 7.F.6 semantic-overwrite
# expansion). The list is informational — the algorithm relies on the
# phase + every-step-writer signal, not the prefix.
_REGISTER_CHANNEL_PREFIXES: Tuple[str, ...] = (
    "REG_",
    "OP_",
    "ADDR_",
    "ALU_",
    "MEM_",
    "FETCH_",
    "BYTE_INDEX_",
    "EMBED_",
    "MARK_",
)


def _looks_like_register_channel(dim_name: str) -> bool:
    upper = dim_name.upper()
    return any(upper.startswith(p) for p in _REGISTER_CHANNEL_PREFIXES)


# ---------------------------------------------------------------------------
# Dim cycle detection
# ---------------------------------------------------------------------------


def _dim_cycle_members(ops: Sequence) -> Set[str]:
    """Return dims that participate in a TRUE cross-step producer/consumer cycle.

    Phase 7.F.6 refinement: the previous implementation treated every
    self-loop (op reads D, op writes D) as a cycle. Real declarative IR
    has many such "transient" self-loops where the value is overwritten
    same-step (e.g. ``OP_PSH`` written by L5 decode at phase=5 and read
    by L7 dispatch at phase=7 — fresh every step, never a carry).

    Refined rule for dim D being a cross-step cycle member:

      1. **Explicit cross-step alias**: D's name ends with one of
         ``_PREV_STEP`` / ``_PREV`` / ``_LAST_STEP``. These dims are
         specifically authored to relay a previous-step value forward
         (B9 OUTPUT_HI split, B11 OUTPUT_LO split, etc.).
      2. **Phase-undominated reader**: there exists at least one reader
         op R with phase ``Pr`` such that NO writer of D fires same-step
         with phase ``Pw <= Pr``. The reader is therefore consuming a
         value carried over from a previous step (or from no writer at
         all — which means D is a constant cache slot, also cross-step
         from the analyzer's perspective).

    A self-loop where the writer's phase >= the reader's phase (i.e. the
    op overwrites D before any future reader sees it) is **not** a cycle
    — the value at step S is overwritten by step S+1's writer before any
    later attention K projection could sample it.

    Multi-dim SCCs (size >= 2) are still flagged as cycles: those imply
    a producer/consumer chain that the static analyzer cannot decompose
    without more semantic data.
    """

    # Per-dim metadata.
    readers_by_dim: Dict[str, List[Tuple[str, Optional[float], object]]] = (
        defaultdict(list)
    )
    writers_by_dim: Dict[str, List[Tuple[str, Optional[float], object]]] = (
        defaultdict(list)
    )
    all_dims: Set[str] = set()
    # Track (read_dim, write_dim) edges for the multi-dim SCC pass; only
    # cross-op edges contribute (self-loops are handled separately).
    succ: Dict[str, Set[str]] = defaultdict(set)
    for op in ops:
        reads = _all_op_reads(op)
        writes = _all_op_writes(op)
        all_dims.update(reads)
        all_dims.update(writes)
        name = getattr(op, "name", repr(op))
        phase = _op_phase(op)
        steps = _op_step_set(op)
        for r in reads:
            readers_by_dim[r].append((name, phase, steps))
        for w in writes:
            writers_by_dim[w].append((name, phase, steps))
        for r in reads:
            for w in writes:
                if r != w:  # exclude self-loop; covered by the per-dim check
                    succ[r].add(w)

    # 1) Per-dim same-step phase-dominance check.
    def _is_same_step_transient(dim: str) -> bool:
        readers = readers_by_dim.get(dim, [])
        writers = writers_by_dim.get(dim, [])
        if not readers or not writers:
            return False
        # Every reader must have a writer that fires same-step AND has
        # phase <= reader's phase.
        for r_name, r_phase, r_steps in readers:
            if r_phase is None:
                # Reader's phase unknown — we can't prove dominance for
                # this reader. Treat as non-transient.
                return False
            dominated = False
            for w_name, w_phase, w_steps in writers:
                if w_phase is None:
                    continue
                if w_phase > r_phase:
                    continue
                # Check step overlap: writer must fire at every step the
                # reader fires at, OR at least guarantee a same-step
                # write before the reader for the reader's domain.
                if w_steps == _EVERY_STEP:
                    # Writer fires every step → dominates every reader-step.
                    dominated = True
                    break
                if r_steps == _EVERY_STEP:
                    # Reader fires every step; writer fires only at some
                    # steps → there's a step the reader fires without a
                    # same-step writer. Not dominated for this reader.
                    continue
                # Both are concrete step sets. Writer must cover every
                # reader step.
                if r_steps <= w_steps:
                    dominated = True
                    break
            if not dominated:
                return False
        return True

    cycle_dims: Set[str] = set()
    same_step_transient: Set[str] = set()
    for dim in all_dims:
        # Explicit cross-step alias is a cycle member only when it has
        # both a reader and a writer in the corpus — a pure writer (no
        # consumer) cannot form a cycle, and we want the per-step
        # classifier to fall through to the PREV_STEP eviction category
        # in that case.
        if _is_prev_step_dim(dim):
            if readers_by_dim.get(dim) and writers_by_dim.get(dim):
                cycle_dims.add(dim)
            continue
        if _is_same_step_transient(dim):
            same_step_transient.add(dim)
            continue
        # If the dim has a reader, no dominating writer → cross-step cycle.
        if readers_by_dim.get(dim) and writers_by_dim.get(dim):
            cycle_dims.add(dim)

    # 2) Multi-dim SCC pass (size >= 2). A multi-dim SCC implies a
    # producer/consumer chain that is harder to reason about; flag every
    # member as a cycle EXCEPT those we've already shown to be same-step
    # transient.
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

    for comp in sccs:
        if len(comp) > 1:
            for dim in comp:
                if dim not in same_step_transient:
                    cycle_dims.add(dim)
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
    op_has_attention: Dict[str, bool] = {}
    for op in ops_list:
        name = getattr(op, "name", repr(op))
        op_ffn_reads: Set[str] = set()
        op_ffn_writes: Set[str] = set()
        for rule in _ffn_rules_from_op(op):
            op_ffn_reads.update(_ffn_rule_reads(rule))
            op_ffn_writes.update(_ffn_rule_writes(rule))
        ffn_reads_by_op[name] = op_ffn_reads
        ffn_writes_by_op[name] = op_ffn_writes

        op_attn_heads = _attention_heads_from_op(op)
        op_kind = getattr(op, "kind", None)
        # Phase 7.F.6: for KV-cache liveness purposes, only attention K/V
        # reads keep a cached entry alive — FFN reads sample the
        # same-step residual stream, not the K/V cache. An op is treated
        # as an attention reader if it declares ``kind="attn"`` OR if
        # its compiler_ir already resolved to one or more
        # AttentionHeadIR instances (which catches the declarative
        # path); kind="block" and kind="ffn" never extend the cache
        # lifetime via their declared reads.
        is_attn_op = bool(op_attn_heads) or op_kind == "attn"
        op_has_attention[name] = is_attn_op
        if is_attn_op:
            attn_kv_read_dim_names_by_op[name] = set(
                getattr(op, "reads", ()) or ()
            )
        else:
            attn_kv_read_dim_names_by_op[name] = set()

        for head in op_attn_heads:
            attn_heads_seen.add(
                (int(getattr(op, "layer_idx", 0) or 0), head.head_idx)
            )

    # 2) Identify cycle dims.
    cycle_dims = _dim_cycle_members(ops_list)

    # 3) Index per-step reads / writes.
    #
    # Phase 7.F.6 semantic-overwrite expansion: an op with
    # ``step_idx=None`` (or ``"every"``) fires at every step. The
    # previous implementation bucketed those at step 0 only, which left
    # ``writes_at_step[s>0]`` empty for the majority of ops. We now treat
    # every-step ops as members of every step ``s in range(n_steps)`` so
    # the future-writes / future-reads sets reflect the actual VM.
    by_step: Dict[int, List] = defaultdict(list)
    every_step_ops: List = []
    for op in ops_list:
        steps = _op_step_set(op)
        if steps == _EVERY_STEP:
            every_step_ops.append(op)
            for s in range(n_steps):
                by_step[s].append(op)
        else:
            for s in steps:
                by_step[s].append(op)

    # Every-step reads/writes (used both directly and below).
    # Reads here are attention-K/V reads only; FFN reads sample the
    # same-step residual stream, not the cache, so they don't extend a
    # cache entry's lifetime.
    every_step_reads: Set[str] = set()
    every_step_writes: Set[str] = set()
    for op in every_step_ops:
        name = getattr(op, "name", repr(op))
        every_step_reads.update(
            attn_kv_read_dim_names_by_op.get(name, set())
        )
        every_step_writes.update(ffn_writes_by_op.get(name, set()))

    # Per-step writes set used by the semantic-overwrite check.
    writes_at_step: Dict[int, Set[str]] = {}
    for step in range(n_steps):
        wset: Set[str] = set(every_step_writes)
        for op in by_step.get(step, ()):
            if op in every_step_ops:
                continue
            name = getattr(op, "name", repr(op))
            wset.update(ffn_writes_by_op.get(name, set()))
        writes_at_step[step] = wset

    max_known_step = max([n_steps - 1] + list(by_step.keys()) + [0])

    # For each step S, what attention-K/V dim names will be referenced
    # at any step >= S? Only attention reads count here — FFN reads
    # sample the same-step residual stream and do not access the cache.
    attn_reads_at_or_after: Dict[int, Set[str]] = {}
    cumulative_later: Set[str] = set()
    for step in range(max_known_step, -1, -1):
        # Every-step reads contribute at every step.
        cumulative_later.update(every_step_reads)
        # Step-specific reads accumulate as we walk descending.
        for op in by_step.get(step, ()):
            if op in every_step_ops:
                continue
            name = getattr(op, "name", repr(op))
            cumulative_later.update(
                attn_kv_read_dim_names_by_op.get(name, set())
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
        # Phase 7.F.6 semantic-overwrite expansion: loosen the future-write
        # criterion to "written at the very next step" (or any later
        # step). Since every-step writers now populate writes_at_step at
        # every step, the next-step check is sufficient — if step S+1's
        # writer overwrites D, the cache row at position S is provably
        # dead at step S+1's boundary.
        next_step_writes: Set[str] = (
            writes_at_step.get(step + 1, set()) if (step + 1) < n_steps else set()
        )
        future_writes: Set[str] = set(next_step_writes)
        for later_step in range(step + 2, n_steps):
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
                    # Phase 7.F.6: cycle classification fires BEFORE the
                    # future-reads test so that cycle-conservative
                    # bookkeeping is observable even when the dim also
                    # happens to be read at a later step. (A cycle dim is
                    # always live for KV purposes; we just want it
                    # bucketed under ``cycle_conservative`` rather than
                    # vanishing into the "future-reads keep" bucket.)
                    if treat_cycle_members_conservative and dim_name in cycle_dims:
                        cycle_kept.add(entry)
                        continue
                    if dim_name in future_reads:
                        # Some later op explicitly reads this dim — LIVE.
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
                    # Phase 7.F.6: prefer the tight "next-step overwrite"
                    # signal — if step S+1 writes the dim, the position
                    # S value can be evicted at step S. This catches the
                    # register-channel (REG_*, OP_*, ADDR_*, ALU_*, ...)
                    # pattern: an every-step writer refreshes the dim at
                    # the very next step.
                    if dim_name in next_step_writes:
                        evictable_at_step[step].add(entry)
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
    # Phase 7.F.6: expose the attention-K/V-read dim universe so the
    # runtime build_state_from_report can scope the per-row AND to dims
    # that genuinely contribute to a K/V projection.
    attention_read_dim_names: Set[str] = set()
    attention_read_dim_names_by_layer: Dict[int, Set[str]] = defaultdict(set)
    for op in ops_list:
        name = getattr(op, "name", repr(op))
        dims = attn_kv_read_dim_names_by_op.get(name, set())
        if not dims:
            continue
        attention_read_dim_names.update(dims)
        layer_val = getattr(op, "layer_idx", None)
        if isinstance(layer_val, int):
            attention_read_dim_names_by_layer[layer_val].update(dims)
    return LivenessReport(
        evictable_at_step=evictable_at_step,
        cycle_conservative=cycle_kept,
        coverage=coverage,
        attention_read_dim_names=attention_read_dim_names,
        attention_read_dim_names_by_layer=dict(attention_read_dim_names_by_layer),
    )


__all__ = [
    "KVEntry",
    "LivenessReport",
    "analyze_kv_liveness",
]
