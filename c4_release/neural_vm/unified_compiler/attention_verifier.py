"""V2 scaffolding for an attention-side analogue of ``verify_rule_strength``.

Background
----------
``decl_verifier.verify_rule_strength`` /
``decl_verifier.verify_rule_scopes`` give us machine-checkable correctness
for FFN rules: every FFNRule with ``dominates_at`` is checked against the
S-4 writer index. Attention writers have no equivalent today, and several
residual failures across the CAMPAIGN_SUMMARY are pinned to attention-side
sources (L1H5 IN_STEP_FRESH, L7H6 SP_BYTE0_IS_F8, L8 SP gather, L15 nibble
copy attention bleed, etc.).

This module is intentionally a research prototype, not a production API.
For each ``AttentionHeadIR`` it surfaces a list of issue dicts shaped like
``verify_rule_strength``'s output. Heuristics are deliberately approximate
(see "V2 approximations" below); the goal is "does this head dominate at
the dim it claims?" not "exactly reproduce softmax".

V2 approximations
-----------------
* **Effective K-side firing scope** is built from the K projection. Each
  ``AP(slot, dim, weight)`` with positive weight is a positively-scored
  key-side dim — we treat the set of K-side dim *names* as the head's
  preferred firing condition at the *key* position.
* **Effective Q-side firing scope** (new in V2) is built from the Q
  projection by the same rule: positive-weighted Q dims are the active
  *query*-side markers. Heads with disjoint Q scopes cannot fire at the
  same query position simultaneously, so V2 skips competition entirely
  between disjoint-Q heads. This is the **Q-side scope-overlap filter**
  the V1 docstring promised but never wired up. ALiBi slope / softmax
  sink are still ignored — see "V3 wishlist" below.
* **Write magnitude** at a residual dim is ``|o.weight| * sum(|v.weight|)``
  across all V writes sharing the O write's slot. This is the upper
  bound the head can deliver to that dim when its attention probability
  saturates to 1.0 and every contributing source dim is 1.0.
* **Competition** at an output dim covers (a) other AttentionHeadIRs in
  ``ops_for_competition`` that write the same residual dim **and whose
  effective Q scope overlaps the head's effective Q scope**, and (b)
  FFNRules surfaced from the same op list whose ``effective_predicate``
  intersects this head's effective scope and whose ``max_contribution``
  to the same dim is non-zero. When either head has no declared Q
  projection, the filter degrades to V1 behavior (treat as overlapping).
* **Scope ground truth** is read from ``head.metadata['scope']`` (a
  predicate-DSL string) or ``head.metadata['dominates_at']`` (mapping
  output-dim name → predicate string). Both are optional; missing
  declarations are tolerated unless ``require_scope`` /
  ``require_dominates`` is set.

V3 wishlist (limitations)
-------------------------
* Model the softmax probability. Today's bound treats the head as
  delivering its full V value at any firing key, which over-states
  weak heads and under-states sharp heads.
* Use ALiBi slopes to constrain effective scope (distance-bounded
  firing). Several heads (e.g. L1H5 IN_STEP_FRESH) intentionally bound
  themselves through slope, which the V1/V2 effective-scope predicate
  cannot represent.
* Cross-check ``Operation.produces`` declarations against the head's
  declared output dims; mismatch should be an issue kind.
* Honor causal vs non-causal masking (currently the verifier assumes
  any key position is in-scope).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


__all__ = [
    "AttentionWriterEntry",
    "build_attention_writer_index",
    "effective_attention_scope",
    "effective_attention_q_scope",
    "head_write_magnitude",
    "verify_attention_head",
]


# ---------------------------------------------------------------------------
# Writer index for attention heads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionWriterEntry:
    """One residual-dim writer surfaced from an attention head.

    Mirrors the FFN ``WriterEntry`` shape so the strength verifier output
    can quote the competitor by name, magnitude, and effective scope.
    """

    op_name: Optional[str]
    head_name: Optional[str]
    head_idx: int
    output_dim_name: str
    output_dim_offset: int
    magnitude: float
    effective_scope: Tuple[str, ...]   # K-side dim names (positive weights)
    head: Any                          # AttentionHeadIR -- avoid hard import
    # V2: Q-side dim names (positive weights). Empty tuple means "no
    # positively-weighted Q dim was declared" — V2 treats that as a
    # wildcard for the overlap check (preserves V1 behaviour for legacy
    # heads). Defaulted so external V1 positional constructors still work.
    q_effective_scope: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Helpers: dim integer → registry name resolution
# ---------------------------------------------------------------------------


def _dim_int_to_name(
    dim_int: int,
    registry,
) -> Optional[Tuple[str, int]]:
    """Resolve a residual-stream integer column back to (slot_name, offset).

    DimRegistry slots are contiguous; this picks the slot whose range
    covers ``dim_int``. Returns ``None`` when nothing matches (common for
    test fixtures using mock registries or symbolic dims).
    """
    slots = getattr(registry, "slots", None)
    if not slots:
        return None
    for name, slot in slots.items():
        start = getattr(slot, "start", None)
        size = getattr(slot, "size", None)
        if start is None or size is None:
            continue
        if start <= dim_int < start + size:
            return name, dim_int - start
    return None


def _resolve_writes_to_dim_names(
    writes: Sequence[Any],
    registry,
) -> List[Tuple[str, int, float, int]]:
    """Resolve a Q/K/V projection write list to (dim_name, offset, weight, slot)."""
    out: List[Tuple[str, int, float, int]] = []
    for write in writes:
        dim_int = getattr(write, "dim", None)
        if dim_int is None:
            continue
        resolved = _dim_int_to_name(int(dim_int), registry)
        if resolved is None:
            continue
        name, offset = resolved
        out.append((name, offset, float(write.weight), int(write.slot)))
    return out


def _resolve_o_writes(
    writes: Sequence[Any],
    registry,
) -> List[Tuple[str, int, float, int]]:
    """Resolve W_o writes to (dim_name, offset, weight, slot).

    O writes carry ``out_dim`` instead of ``dim`` for the residual column.
    """
    out: List[Tuple[str, int, float, int]] = []
    for write in writes:
        dim_int = getattr(write, "out_dim", None)
        if dim_int is None:
            continue
        resolved = _dim_int_to_name(int(dim_int), registry)
        if resolved is None:
            continue
        name, offset = resolved
        out.append((name, offset, float(write.weight), int(write.slot)))
    return out


# ---------------------------------------------------------------------------
# Effective firing scope (V2: K-side + Q-side)
# ---------------------------------------------------------------------------


def effective_attention_scope(head, registry) -> Tuple[str, ...]:
    """Return the approximate effective K-side firing scope for ``head``.

    The set of K-side dim names with a positive weight. The intuition:
    a positive K write contributes positively to the q·k score when the
    residual stream at the key position has that dim set, so the head
    "prefers keys where any of these K-side dims fire". This is an
    over-approximation — it ignores ALiBi.

    Returned as a sorted tuple so two heads can be compared by identity.
    """
    spec = getattr(head, "spec", None)
    if spec is None:
        return ()
    k_writes = _resolve_writes_to_dim_names(
        getattr(spec, "k", ()), registry
    )
    names: List[str] = []
    for name, _offset, weight, _slot in k_writes:
        if weight > 0.0:
            names.append(name)
    return tuple(sorted(set(names)))


def effective_attention_q_scope(head, registry) -> Tuple[str, ...]:
    """Return the V2 effective Q-side firing scope for ``head``.

    Analogous to ``effective_attention_scope`` but extracted from the Q
    projection: each positively-weighted ``AP(slot, dim, weight)`` in
    ``spec.q`` contributes its dim name. The intuition: a positive Q
    write means the head only produces a non-trivial q·k inner product
    when the residual stream at the *query* position has that dim set,
    so the head "prefers querying from positions where any of these Q-
    side dims fire".

    Used by V2's competition filter: two heads with disjoint non-empty
    Q scopes cannot fire at the same query position, so their O writes
    never collide in practice and they shouldn't be counted as mutual
    competitors. Returned as a sorted tuple so two heads can be compared
    by identity.

    An empty result means "no positively-weighted Q dim was declared"
    (either the head has a uniform/constant Q pattern, or the Q writes
    target dims not present in the registry). V2 treats an empty Q scope
    as a wildcard (fires at every query position) for the overlap check,
    falling back to V1 behavior.
    """
    spec = getattr(head, "spec", None)
    if spec is None:
        return ()
    q_writes = _resolve_writes_to_dim_names(
        getattr(spec, "q", ()), registry
    )
    names: List[str] = []
    for name, _offset, weight, _slot in q_writes:
        if weight > 0.0:
            names.append(name)
    return tuple(sorted(set(names)))


def _q_scopes_overlap(
    a: Sequence[str],
    b: Sequence[str],
) -> bool:
    """Return True when two Q scopes can fire at the same query position.

    Per V2 semantics: an empty scope is a wildcard (treat as overlapping
    with anything). Two non-empty scopes overlap iff they share at least
    one dim name.
    """
    if not a or not b:
        return True
    return bool(set(a) & set(b))


# ---------------------------------------------------------------------------
# Write magnitude (V1; unchanged in V2)
# ---------------------------------------------------------------------------


def head_write_magnitude(
    head,
    output_dim_name: str,
    output_offset: int,
    registry,
) -> float:
    """Upper bound on the residual delta this head can deliver at ``(name, offset)``.

    V1 bound: for each O write targeting ``(name, offset)``, sum across V
    writes sharing the same slot the absolute V weight. Multiply by the O
    weight magnitude. This is the value the head would deliver if its
    softmax probability saturated to 1 and the chosen-key residual were
    all-ones at the V source dims. Returns 0.0 when no O write targets
    the dim.
    """
    spec = getattr(head, "spec", None)
    if spec is None:
        return 0.0

    # Resolve V writes; index by slot.
    v_by_slot: Dict[int, float] = {}
    for _name, _offset, weight, slot in _resolve_writes_to_dim_names(
        getattr(spec, "v", ()), registry
    ):
        v_by_slot[slot] = v_by_slot.get(slot, 0.0) + abs(weight)

    total = 0.0
    for o_name, o_offset, o_weight, slot in _resolve_o_writes(
        getattr(spec, "o", ()), registry
    ):
        if o_name != output_dim_name or o_offset != output_offset:
            continue
        v_total = v_by_slot.get(slot, 0.0)
        total += abs(o_weight) * v_total
    return total


# ---------------------------------------------------------------------------
# Writer index over a collection of attention heads
# ---------------------------------------------------------------------------


def build_attention_writer_index(
    ops: Iterable[Any],
    registry,
) -> Dict[Tuple[str, int], List[AttentionWriterEntry]]:
    """Build a per-(dim_name, offset) writer index for attention heads.

    Each op may carry a ``compiler_ir`` with one or more layers, each of
    which has an ``attention.rules`` list of ``AttentionHeadIR``s. Heads
    that cannot be resolved to dim names (e.g. dim integers outside any
    registry slot range) are silently skipped — callers can address
    those gaps with a registry-completeness audit.

    V2: each ``AttentionWriterEntry`` is also tagged with its effective
    Q-side scope (``q_effective_scope``) so downstream verifiers can
    filter competitors by Q-position overlap without re-walking the head
    spec.
    """
    index: Dict[Tuple[str, int], List[AttentionWriterEntry]] = {}
    for op in ops:
        op_name = getattr(op, "name", None)
        for head in _collect_heads_from_op(op):
            scope = effective_attention_scope(head, registry)
            q_scope = effective_attention_q_scope(head, registry)
            spec = getattr(head, "spec", None)
            if spec is None:
                continue
            for name, offset, o_weight, slot in _resolve_o_writes(
                getattr(spec, "o", ()), registry
            ):
                magnitude = head_write_magnitude(
                    head, name, offset, registry
                )
                entry = AttentionWriterEntry(
                    op_name=op_name,
                    head_name=getattr(head, "name", None),
                    head_idx=int(getattr(spec, "head_idx", -1)),
                    output_dim_name=name,
                    output_dim_offset=offset,
                    magnitude=magnitude,
                    effective_scope=scope,
                    head=head,
                    q_effective_scope=q_scope,
                )
                index.setdefault((name, offset), []).append(entry)
    return index


def _collect_heads_from_op(op) -> List[Any]:
    """Walk an Operation's compiler IR to collect AttentionHeadIRs.

    Handles ``CompilerIR`` (with ``.layers[].attention``), ``AttentionOp``
    directly, lists/tuples, and bare ``AttentionHeadIR``s.
    """
    try:
        from .ir import AttentionHeadIR, AttentionOp, CompilerIR
    except Exception:
        AttentionHeadIR = AttentionOp = CompilerIR = None   # type: ignore

    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return []

    heads: List[Any] = []
    # CompilerIR with .layers[].attention
    if hasattr(ir, "layers"):
        for layer in ir.layers:
            attention = getattr(layer, "attention", None)
            if attention is not None and hasattr(attention, "rules"):
                heads.extend(attention.rules)
        return heads
    # AttentionOp directly
    if AttentionOp is not None and isinstance(ir, AttentionOp):
        return list(ir.rules)
    if isinstance(ir, (list, tuple)):
        for item in ir:
            if AttentionHeadIR is not None and isinstance(item, AttentionHeadIR):
                heads.append(item)
            elif AttentionOp is not None and isinstance(item, AttentionOp):
                heads.extend(item.rules)
        return heads
    if hasattr(ir, "rules"):
        return [r for r in ir.rules]
    if AttentionHeadIR is not None and isinstance(ir, AttentionHeadIR):
        return [ir]
    return []


# ---------------------------------------------------------------------------
# FFN competition (best-effort cross-modality competitor surfacing)
# ---------------------------------------------------------------------------


def _build_ffn_writer_index(
    ops: Iterable[Any],
    registry,
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """Build a FFN writer index once and return a serializable form.

    Returns a dict keyed by ``(name, offset)`` whose values are issue-dict-
    shaped entries. Failures during index construction or dim resolution
    are swallowed silently (matches FFN writer-index policy).
    """
    out: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    try:
        from .writer_index import build_writer_index
    except Exception:
        return out
    try:
        index = build_writer_index(list(ops), registry)
    except Exception:
        return out
    for key, entries in index.items():
        serial: List[Dict[str, Any]] = []
        for entry in entries:
            serial.append({
                "kind": "ffn_rule",
                "op_name": entry.op_name,
                "rule_name": getattr(entry.rule, "name", None),
                "max_contribution": float(entry.max_contribution),
                "effective_scope": str(entry.effective_scope),
            })
        out[key] = serial
    return out


def _ffn_competitors_for_dim(
    ffn_index: Dict[Tuple[str, int], List[Dict[str, Any]]],
    output_dim_name: str,
    output_offset: int,
) -> List[Dict[str, Any]]:
    """Lookup FFN-side competitors from a prebuilt index."""
    return ffn_index.get((output_dim_name, output_offset), [])


# ---------------------------------------------------------------------------
# Scope ground truth from metadata
# ---------------------------------------------------------------------------


def _declared_scope(head) -> Optional[str]:
    metadata = getattr(head, "metadata", None) or {}
    val = metadata.get("scope")
    return str(val) if val is not None else None


def _declared_dominates_at(head, output_dim_name: str) -> Optional[str]:
    metadata = getattr(head, "metadata", None) or {}
    dom = metadata.get("dominates_at")
    if isinstance(dom, Mapping):
        v = dom.get(output_dim_name)
        if v is not None:
            return str(v)
    return _declared_scope(head)


# ---------------------------------------------------------------------------
# Public API: verify_attention_head
# ---------------------------------------------------------------------------


def verify_attention_head(
    head,
    registry,
    *,
    ops_for_competition: Optional[Iterable[Any]] = None,
    margin: float = 1.0,
    require_scope: bool = False,
    require_dominates: bool = False,
    q_scope_filter: bool = True,
) -> List[Dict[str, Any]]:
    """V2 attention-side analogue of ``verify_rule_strength``.

    Walks ``head.spec.o`` and for each declared output dim:
      * resolves the integer column back to (name, offset);
      * computes the head's magnitude;
      * looks up competing attention heads + FFN rules at the same dim
        from ``ops_for_competition``;
      * V2: filters attention competitors by Q-side scope overlap. Two
        heads whose Q projections target disjoint dim names cannot fire
        at the same query position, so they don't actually compete — V2
        skips the strength check for those pairs. Heads with no declared
        Q scope are treated as wildcards (always overlap), which
        preserves V1 behavior for legacy heads. Set
        ``q_scope_filter=False`` to opt back into V1's "magnitude only,
        ignore Q-scope" comparison.
      * emits an issue when the head's magnitude does not dominate the
        strongest *Q-overlapping* competitor + ``margin``.

    Issue kinds returned:
      * ``unresolved_output_dim``   — O write targets a residual column
                                       no registry slot covers.
      * ``empty_value_path``        — O write references slot S, but no V
                                       write feeds slot S; head delivers 0.
      * ``no_scope_declared``       — ``require_scope`` and head metadata
                                       has no ``scope`` entry.
      * ``no_dominates_at_declared`` — ``require_dominates`` and head
                                       metadata has no ``dominates_at``
                                       entry for this dim.
      * ``attention_strength_violation`` — head's magnitude is not
                                       greater than ``competing_max +
                                       margin``. (V2: only Q-overlapping
                                       attention competitors counted.)
      * ``cross_modality_strength_violation`` — head loses to an FFN
                                       rule writing the same dim.

    Each issue dict carries enough context (head name, output dim, my
    magnitude, top competitor, declared scope) for a downstream agent or
    audit script to triage without re-inspecting the head. V2 issues
    additionally carry ``my_q_scope`` / ``competitor_q_scope``.
    """
    issues: List[Dict[str, Any]] = []

    spec = getattr(head, "spec", None)
    if spec is None:
        return issues

    head_name = getattr(head, "name", None)
    head_idx = int(getattr(spec, "head_idx", -1))

    # Slot -> V magnitude map for the empty_value_path check.
    v_by_slot: Dict[int, float] = {}
    for _name, _offset, weight, slot in _resolve_writes_to_dim_names(
        getattr(spec, "v", ()), registry
    ):
        v_by_slot[slot] = v_by_slot.get(slot, 0.0) + abs(weight)

    # Build competition index from other ops (excluding this head's
    # parent op when present via identity check). Built once per call
    # so we don't pay the index cost per output write.
    other_ops = list(ops_for_competition or [])
    attn_index = build_attention_writer_index(other_ops, registry)
    ffn_index = _build_ffn_writer_index(other_ops, registry)
    my_scope = effective_attention_scope(head, registry)
    my_q_scope = effective_attention_q_scope(head, registry)

    # Walk every O write.
    seen_dims = set()
    raw_o = list(getattr(spec, "o", ()))
    if not raw_o:
        # Head declares no output writes at all -- not necessarily a bug
        # (some specs extend an existing head's V slots only), so emit a
        # zero-write note rather than an error.
        return issues

    for o_write in raw_o:
        dim_int = getattr(o_write, "out_dim", None)
        slot = int(getattr(o_write, "slot", -1))
        if dim_int is None:
            continue
        resolved = _dim_int_to_name(int(dim_int), registry)
        if resolved is None:
            issues.append({
                "kind": "unresolved_output_dim",
                "head": head_name,
                "head_idx": head_idx,
                "output_dim_int": int(dim_int),
                "slot": slot,
            })
            continue
        name, offset = resolved
        key = (name, offset)
        if key in seen_dims:
            continue
        seen_dims.add(key)

        # Empty value path: the slot has no V projection writes.
        if v_by_slot.get(slot, 0.0) == 0.0:
            issues.append({
                "kind": "empty_value_path",
                "head": head_name,
                "head_idx": head_idx,
                "output_dim": f"{name}+{offset}",
                "slot": slot,
            })
            # Still proceed -- magnitude will be 0 below, surface
            # strength violations too.

        my_magnitude = head_write_magnitude(head, name, offset, registry)

        # Scope declarations.
        if require_scope and _declared_scope(head) is None:
            issues.append({
                "kind": "no_scope_declared",
                "head": head_name,
                "head_idx": head_idx,
                "output_dim": f"{name}+{offset}",
            })
        if (
            require_dominates
            and _declared_dominates_at(head, name) is None
        ):
            issues.append({
                "kind": "no_dominates_at_declared",
                "head": head_name,
                "head_idx": head_idx,
                "output_dim": f"{name}+{offset}",
            })

        # Attention-side competitors. Identity-skip "this head".
        attn_competitors = [
            entry for entry in attn_index.get(key, [])
            if entry.head is not head
        ]
        # V2: drop competitors whose Q-side scope is disjoint from
        # ours. Two heads with disjoint non-empty Q scopes cannot fire
        # at the same query position, so their O writes can't actually
        # collide at runtime. Heads with an empty Q scope (no positive
        # Q write resolved to the registry) degrade to V1 behaviour
        # (treat as overlapping / wildcard) so we never regress on
        # heads that haven't declared a Q projection.
        if q_scope_filter:
            attn_competitors = [
                entry for entry in attn_competitors
                if _q_scopes_overlap(my_q_scope, entry.q_effective_scope)
            ]
        competing_max_attn = 0.0
        top_attn: Optional[AttentionWriterEntry] = None
        for entry in attn_competitors:
            if entry.magnitude > competing_max_attn:
                competing_max_attn = entry.magnitude
                top_attn = entry

        # FFN-side competitors.
        ffn_competitors = _ffn_competitors_for_dim(
            ffn_index, name, offset,
        )
        competing_max_ffn = 0.0
        top_ffn: Optional[Dict[str, Any]] = None
        for entry in ffn_competitors:
            if entry["max_contribution"] > competing_max_ffn:
                competing_max_ffn = entry["max_contribution"]
                top_ffn = entry

        # Strength check vs attention competitor.
        if my_magnitude < competing_max_attn + margin:
            issues.append({
                "kind": "attention_strength_violation",
                "head": head_name,
                "head_idx": head_idx,
                "output_dim": f"{name}+{offset}",
                "my_magnitude": my_magnitude,
                "competing_max": competing_max_attn,
                "margin": margin,
                "required": competing_max_attn + margin,
                "shortfall": (competing_max_attn + margin) - my_magnitude,
                "top_competitor": (
                    None
                    if top_attn is None
                    else (top_attn.head_name or top_attn.op_name)
                ),
                "top_competitor_head_idx": (
                    None if top_attn is None else top_attn.head_idx
                ),
                "my_effective_scope": list(my_scope),
                "competitor_effective_scope": (
                    [] if top_attn is None else list(top_attn.effective_scope)
                ),
                "my_q_scope": list(my_q_scope),
                "competitor_q_scope": (
                    []
                    if top_attn is None
                    else list(top_attn.q_effective_scope)
                ),
            })

        # Strength check vs FFN competitor.
        if my_magnitude < competing_max_ffn + margin:
            issues.append({
                "kind": "cross_modality_strength_violation",
                "head": head_name,
                "head_idx": head_idx,
                "output_dim": f"{name}+{offset}",
                "my_magnitude": my_magnitude,
                "competing_max": competing_max_ffn,
                "margin": margin,
                "required": competing_max_ffn + margin,
                "shortfall": (competing_max_ffn + margin) - my_magnitude,
                "top_competitor": (
                    None
                    if top_ffn is None
                    else (
                        top_ffn.get("rule_name") or top_ffn.get("op_name")
                    )
                ),
                "top_competitor_kind": "ffn_rule",
            })

    return issues
