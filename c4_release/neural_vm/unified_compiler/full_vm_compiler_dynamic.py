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

This invariant was historically enforced by ``compare_compile_paths()``
and the matching slow byte-identity tests in
``tests/test_compile_dynamic_byte_identical.py``. Both were removed in
Phase 8.G.3 alongside the static phase-pruning body — the remaining
fast scheduler invariants (topo-order, cycle-pruning) live in the same
test module and continue to gate any change to the declared deps that
would break the dep-pruned order.

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
import os
import warnings
from collections import defaultdict
from typing import (
    Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set,
    Tuple,
)

from .layer_compiler import (
    Operation,
    requires_after_ops,
    requires_same_layer_as_ops,
)
from .ssa_dim import base_of, is_ssa_form, parse_ssa_name
from .ir import ModelArchitectureSpec
from . import _legacy_redirect as _static
from .ops.shared import (
    operand_from_memsp_enabled,
    campaign_enabled,
    emit_g5_rbyte_enabled,
)
from ..kv_eviction import KVEvictionPolicy


# ---------------------------------------------------------------------------
# Model-semantics umbrella (compile-flag presets + per-axis overrides).
# See ``docs/MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md`` for the design.
# ---------------------------------------------------------------------------


# Per-axis values. Each axis has a "native" value (today's behavior, the
# byte-identity backward-compat target) and a "qwen" value (the
# Qwen-compatible alternative). The presets snap onto a coherent pair of
# six; per-axis kwargs ablate a single axis off the chosen preset.
_MODEL_SEMANTICS_AXES: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("positional_encoding", ("alibi", "rope")),
    ("softmax_variant",     ("softmax1", "standard")),
    ("normalization",       ("identity", "rmsnorm")),
    ("ffn_variant",         ("native", "swiglu")),
    ("per_head_qk_norm",    ("none", "qwen")),
    ("ffn_routing",         ("single", "composite")),
)


_MODEL_SEMANTICS_PRESETS: Dict[str, Dict[str, str]] = {
    "native": {axis: values[0] for axis, values in _MODEL_SEMANTICS_AXES},
    "qwen":   {axis: values[1] for axis, values in _MODEL_SEMANTICS_AXES},
}


def _validate_semantics_value(axis: str, value: Optional[str]) -> None:
    """Reject an out-of-range per-axis value with a useful error."""
    if value is None:
        return
    valid = dict(_MODEL_SEMANTICS_AXES).get(axis)
    if valid is None:  # pragma: no cover - defensive
        return
    if value not in valid:
        raise ValueError(
            f"compile_full_vm_dynamic({axis}=...) must be one of "
            f"{valid!r}; got {value!r}"
        )


def _resolve_semantics_flags(
    *,
    preset: Optional[str],
    positional_encoding: Optional[str],
    softmax_variant: Optional[str],
    attention_normalization: Optional[str],
    normalization: Optional[str],
    use_rms_norm: Optional[bool],
    ffn_variant: Optional[str],
    per_head_qk_norm: Optional[str],
    ffn_routing: Optional[str],
    enable_moe_routing: Optional[bool],
    arch: Optional[ModelArchitectureSpec],
) -> Tuple[Optional[str], Optional[str], Optional[bool], Optional[bool]]:
    """Collapse the umbrella flags onto the legacy bake-pipeline kwargs.

    Returns the legacy-shaped tuple
    ``(positional_encoding, attention_normalization, use_rms_norm,
    enable_moe_routing)`` that the downstream bake already consumes.

    Backward-compatibility contract: when ``preset is None`` and every
    per-axis umbrella flag is ``None``, this function returns the input
    legacy kwargs unchanged. This is the byte-identity gate that
    ``tests/test_compile_flag_parity.py::test_preset_native_byte_identical_to_default``
    exercises.

    Errors:
    - ``preset=`` with ``arch=`` is a hard error (mirrors the existing
      ``arch=`` vs individual-kwarg gate).
    - Any per-axis value outside the documented set is a ``ValueError``.
    - ``per_head_qk_norm="qwen"`` raises ``NotImplementedError`` — the
      flag is reserved on the signature but the lowering lands with the
      per-axis design doc (see §6 of the design doc).
    """
    # Per-axis value validation runs first so the most specific error
    # surfaces above the preset / arch-gate machinery.
    _validate_semantics_value("positional_encoding", positional_encoding)
    _validate_semantics_value("softmax_variant", softmax_variant)
    _validate_semantics_value("normalization", normalization)
    _validate_semantics_value("ffn_variant", ffn_variant)
    _validate_semantics_value("per_head_qk_norm", per_head_qk_norm)
    _validate_semantics_value("ffn_routing", ffn_routing)

    umbrella_supplied = {
        name: value
        for name, value in (
            ("preset", preset),
            ("softmax_variant", softmax_variant),
            ("normalization", normalization),
            ("ffn_variant", ffn_variant),
            ("per_head_qk_norm", per_head_qk_norm),
            ("ffn_routing", ffn_routing),
        )
        if value is not None
    }

    # Pure backward-compat fast path: no umbrella flag supplied AND no
    # positional_encoding (the one umbrella flag that aliases an existing
    # legacy kwarg) -- pass the legacy kwargs through unchanged.
    if not umbrella_supplied and positional_encoding is None:
        return (
            positional_encoding,
            attention_normalization,
            use_rms_norm,
            enable_moe_routing,
        )

    if umbrella_supplied and arch is not None:
        raise TypeError(
            "compile_full_vm_dynamic(arch=...) is mutually exclusive with "
            "the model-semantics umbrella flags "
            f"{sorted(umbrella_supplied)}. Pass either ``arch=`` or the "
            "``preset=``/per-axis flags, not both. See "
            "``docs/MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md`` for the "
            "migration story."
        )

    if preset is not None and preset not in _MODEL_SEMANTICS_PRESETS:
        raise ValueError(
            f"compile_full_vm_dynamic(preset=...) must be one of "
            f"{sorted(_MODEL_SEMANTICS_PRESETS)}; got {preset!r}"
        )

    # Build a sparse resolved dict that holds only the axes the caller
    # explicitly opted into (via ``preset=`` or a per-axis kwarg). Axes
    # the caller did NOT name are left untouched: the legacy kwarg /
    # env default flows through unchanged. This preserves the "only
    # ablate what I named" composition rule documented in §3 of the
    # design doc.
    resolved: Dict[str, str] = {}
    if preset is not None:
        resolved.update(_MODEL_SEMANTICS_PRESETS[preset])
    per_axis_overrides = {
        "positional_encoding": positional_encoding,
        "softmax_variant":     softmax_variant,
        "normalization":       normalization,
        "ffn_variant":         ffn_variant,
        "per_head_qk_norm":    per_head_qk_norm,
        "ffn_routing":         ffn_routing,
    }
    for axis, value in per_axis_overrides.items():
        if value is not None:
            resolved[axis] = value

    # Per-axis variant implementations land separately. For this round
    # only the legacy-aliased axes are wired; the others are reserved.
    #
    # An axis set via a DIRECT per-axis kwarg (e.g. ``ffn_variant='swiglu'``)
    # is a hard ``NotImplementedError``: the caller asked for a specific
    # semantic the bake cannot honour, so raise rather than silently
    # degrade. But when the axis is reached via ``preset="qwen"`` (a
    # coarse-grained shortcut), let the compile proceed with the legacy
    # bake for the unwired axes -- this matches the partial-implementation
    # story documented at ``MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md``
    # §1.2: presets are the umbrella surface, per-axis kwargs are the
    # contractual surface. The preset path's bake completes as the
    # individual axes land.
    if ffn_variant == "swiglu":
        raise NotImplementedError(
            "ffn_variant='swiglu' is reserved on the compile signature; "
            "implementation lands with QWEN_SWIGLU_REPACK_PROTOTYPE_2026_06_07.md."
        )
    if per_head_qk_norm == "qwen":
        raise NotImplementedError(
            "per_head_qk_norm='qwen' is reserved on the compile signature; "
            "implementation lands with the per-axis design doc (see §6 of "
            "MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md)."
        )

    # Map onto the legacy bake-pipeline kwargs. An axis the caller did
    # not opt into stays at its incoming legacy-kwarg value.
    new_positional_encoding = resolved.get(
        "positional_encoding", positional_encoding
    )

    if "softmax_variant" in resolved:
        softmax_to_legacy = {"softmax1": "softmax1", "standard": "softmax"}
        new_attention_normalization = softmax_to_legacy[
            resolved["softmax_variant"]
        ]
    else:
        new_attention_normalization = attention_normalization

    if "normalization" in resolved:
        new_use_rms_norm: Optional[bool] = (
            resolved["normalization"] == "rmsnorm"
        )
    else:
        new_use_rms_norm = use_rms_norm

    if "ffn_routing" in resolved:
        new_enable_moe_routing: Optional[bool] = (
            resolved["ffn_routing"] == "composite"
        )
    else:
        new_enable_moe_routing = enable_moe_routing

    return (
        new_positional_encoding,
        new_attention_normalization,
        new_use_rms_norm,
        new_enable_moe_routing,
    )


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
    * Phase-as-slot-share refinement (mirrors ``_topological_sort``): for
      ops with an explicit ``phase=N`` but ``layer_idx=None``, the phase
      is a slot-share key, not a hard layer pin.
      ``LayerCompiler._assign_layers`` places such ops at
      ``max(dep-derived earliest, slot-share constraint)`` and uses
      ``phase`` only to break ties within a ``(layer, kind)`` slot. So
      ``current < derived`` is NOT a strict-mode violation when the op
      has no hard ``layer_idx`` pin — the dispatcher simply slides the
      op to ``derived``. This mirrors the static path's
      ``_topological_sort`` phase-pruning semantics (phase orders ops
      INSIDE an SCC / same layer; it never overrides dep-derived layer
      assignment). The categoriser flags
      ``phase_inconsistent_with_deps`` only for ``layer_idx``-pinned ops,
      where the static path cannot reconcile a phase < derived gap by
      sliding the op later.
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
            # Phase-as-slot-share refinement (see docstring): when the
            # op has no ``layer_idx`` pin, ``phase`` is a slot-share key
            # not a hard layer pin. ``_assign_layers`` will slide the op
            # to ``derived`` (or later) and use ``phase`` only for
            # intra-slot ordering. This mirrors ``_topological_sort``'s
            # phase-pruning rules: phase orders ops within the same SCC
            # / layer but never overrides dep-derived placement.
            if op.layer_idx is None:
                buckets["ok"].append(op.name)
                continue
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
# Step-1 safety: cross-step reads with same-step writers
# ---------------------------------------------------------------------------


# Sentinel used by the step-1 safety check to indicate "no warning category
# filter was supplied". Exposed via module-level constant so callers can pass
# it through.
class CrossStepReadWarning(UserWarning):
    """Emitted at compile time when an op declares a cross-step (``X.*.-1``)
    read of a residual dim that ALSO has a same-step writer scheduled in the
    same compile.

    On VM step 1 (the first step) there is no previous step, so a cross-step
    read returns the residual's zero-init value rather than a meaningful
    producer-written value. If a same-step writer exists, the bake author
    very often *meant* to read the freshly-written same-step value (or to OR
    same-step ∪ prev-step), but the IR's ``.*.-1`` form silently falls
    through to 0 on step 1.

    This warning surfaces the class-of-bug at compile time WITHOUT changing
    runtime behaviour. Each warning names the consuming op, the cross-step
    SSA dim name, and the set of same-step writer ops so the bake author can
    audit whether the step-1 zero-propagation is intentional. The historical
    discovery that motivated this check is the OPCODE_BYTE_LO cross-step
    read in ``l5_ops.py:opcode_decode_ffn`` whose same-step writer is
    ``layer5_fetch``: on step 1 the decoder reads 0 because L5 fetch's
    write hadn't entered the prev-step residual.
    """


class CrossStepReadError(Exception):
    """Raised by ``compile_full_vm_dynamic(strict=True)`` (Step 4) when the
    cross-step safety check finds at least one ``(consumer, ssa_read)``
    pair that is NOT in the
    ``cross_step_baseline_allowlist`` (or the bundled
    :data:`CROSS_STEP_BASELINE_ALLOWLIST` when none is passed).

    This is the strict-mode promotion of :class:`CrossStepReadWarning`
    (Step 4 of ``IR_INCREMENTAL_IMPROVEMENTS.md``). Distinguishing
    ``DIM`` (step=0) from ``DIM.*.-1`` (step=-1) as separate logical
    values means the compiler must refuse to compile when an op reads a
    cross-step alias whose base dim ALSO has a same-step writer — unless
    that pair has been explicitly whitelisted for incremental migration.

    The error message lists every offending ``(consumer, ssa_read)`` pair
    along with the same-step writers so the bake author can either:

    * Migrate the read (replace ``X.*.-1`` with ``X`` or
      ``OR(X, X.*.-1)``), removing the line from the allowlist as the
      cause is fixed; or
    * Add the pair to the caller-supplied allowlist with a TODO comment
      pointing at the migration tracking doc.

    Attributes
    ----------
    unallowed_findings:
        ``[(consumer_op_name, ssa_dim_name, same_step_writers), ...]`` —
        the findings that escaped the allowlist. The full set of all
        findings (allowlisted + unallowed) is available on the
        ``all_findings`` attribute.
    all_findings:
        Every finding the safety check produced, regardless of allowlist
        status. Lets a downstream tool re-derive the allowlist by
        filtering the runtime-rejected pairs.
    """

    def __init__(
        self,
        unallowed_findings: List[Tuple[str, str, Tuple[str, ...]]],
        all_findings: List[Tuple[str, str, Tuple[str, ...]]],
    ):
        self.unallowed_findings = unallowed_findings
        self.all_findings = all_findings
        # Cap message preview to avoid 80-line tracebacks for the same
        # underlying bug class. The full list is on the attribute.
        preview = "\n".join(
            f"  - {consumer!r} reads {ssa_read!r}; same-step writers: "
            f"{list(writers)!r}"
            for (consumer, ssa_read, writers) in unallowed_findings[:10]
        )
        extra = max(0, len(unallowed_findings) - 10)
        if extra:
            preview = f"{preview}\n  ... (+{extra} more)"
        super().__init__(
            f"compile_full_vm_dynamic(strict=True) refused to compile: "
            f"{len(unallowed_findings)} cross-step read(s) with same-step "
            f"writers are not in the baseline allowlist (Step 4 of "
            f"IR_INCREMENTAL_IMPROVEMENTS.md). On VM step 1 the cross-step "
            f"alias resolves to 0; if a same-step writer exists the bake "
            f"author likely meant the same-step value. Either migrate the "
            f"read OR add the pair to "
            f"``cross_step_baseline_allowlist`` with a TODO. Findings:\n"
            f"{preview}"
        )


# ---------------------------------------------------------------------------
# Step 4 cross-step baseline allowlist
# ---------------------------------------------------------------------------
#
# Each entry is a ``(consumer_op_name, ssa_dim_name)`` tuple that the safety
# check has agreed to ignore — i.e. it WILL still produce a warning, but it
# WILL NOT cause ``compile_full_vm_dynamic(strict=True)`` to raise
# :class:`CrossStepReadError`. The list pinned the 82 findings present at
# Step-4 landing (alu_mode='lookup', io/tool/think flags all off); every
# entry is a TODO to migrate the read away from the cross-step alias or
# document why the step-1 zero-propagation is intentional.
#
# Adding entries: only as a last resort when the cause cannot be fixed in
# the same change. Removing entries: do this aggressively — every removal
# is a step toward Step 4's "no cross-step zero-prop bugs at compile" goal.
# A regression test
# (``tests/test_compile_cross_step_safety.py::test_baseline_allowlist_does_not_grow``)
# pins ``len(CROSS_STEP_BASELINE_ALLOWLIST) <= 82`` as the ratchet.
#
# Step 4 follow-up: entries that have been audited and confirmed
# INTENTIONAL cross-step reads (see ``CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md``)
# are migrated OUT of this baseline and INTO
# :data:`CROSS_STEP_DOCUMENTED_SAFE` (a dict with per-entry
# justifications). Both sets together form the effective allowlist used
# by :func:`_emit_cross_step_safety_warnings`; the baseline ratchet only
# counts this set, so each documented-safe migration shrinks the
# remaining migration backlog visible to the ratchet.
#
# TODO(step4-migration): each entry below corresponds to a real cross-step
# read in the production op set that should eventually be migrated to one
# of:
#   * a same-step read of the base dim (when the same-step writer is the
#     intended producer),
#   * an explicit ``OR(X.*.-1, X)`` aggregation (when both same-step and
#     prev-step writes are legitimate inputs), or
#   * a documented self-writer back-edge (which the safety check already
#     excludes — these entries are the residual non-self-writer findings).
# Track migrations in ``docs/IR_INCREMENTAL_IMPROVEMENTS.md`` Step 4
# section.
CROSS_STEP_BASELINE_ALLOWLIST: FrozenSet[Tuple[str, str]] = frozenset({
    ('l10_post_ops_combined', 'OUTPUT_HI.*.-1'),
    ('l10_post_ops_combined', 'OUTPUT_LO.*.-1'),
    ('l10_post_ops_combined', 'TEMP.*.-1'),
    ('layer10_alu', 'ALU_HI.*.-1'),
    ('layer10_byte_passthrough_bake', 'TEMP.*.-1'),
    ('layer10_carry_relay_bake', 'CARRY.*.-1'),
    ('layer10_psh_stack0_passthrough_bake', 'OUTPUT_HI.*.-1'),
    ('layer10_psh_stack0_passthrough_bake', 'OUTPUT_LO.*.-1'),
    ('layer10_stack0_byte_relay_bake', 'TEMP.*.-1'),
    ('layer12_mul_combine', 'TEMP.*.-1'),
    ('layer3_carry_forward_attn', 'EMBED_HI.*.-1'),
    ('layer3_carry_forward_attn', 'EMBED_LO.*.-1'),
    ('layer3_ffn', 'EMBED_HI.*.-1'),
    ('layer3_ffn', 'EMBED_LO.*.-1'),
    ('layer3_ffn', 'TEMP.*.-1'),
    ('layer6_routing_ffn', 'AX_CARRY_HI.*.-1'),
    ('layer6_routing_ffn', 'AX_CARRY_LO.*.-1'),
    ('layer6_routing_ffn', 'OUTPUT_HI.*.-1'),
    ('layer6_routing_ffn', 'OUTPUT_LO.*.-1'),
    ('layer6_routing_ffn', 'TEMP.*.-1'),
    ('layer7_memory_heads', 'AX_CARRY_HI.*.-1'),
    ('layer7_memory_heads', 'AX_CARRY_LO.*.-1'),
    ('layer7_memory_heads', 'TEMP.*.-1'),
    ('layer8_alu', 'ALU_LO.*.-1'),
    ('layer8_head6_ax_carry_refresh', 'OUTPUT_HI.*.-1'),
    ('layer8_head6_ax_carry_refresh', 'OUTPUT_LO.*.-1'),
    ('layer8_mem_to_alu', 'ADDR_B0_HI.*.-1'),
    ('layer8_mem_to_alu', 'ADDR_B1_HI.*.-1'),
    ('layer8_mem_to_alu', 'ADDR_B2_HI.*.-1'),
    ('layer9_alu', 'ALU_HI.*.-1'),
    ('layer9_alu', 'ALU_LO.*.-1'),
    ('layer9_alu', 'CARRY.*.-1'),
    ('opcode_decode_ffn', 'OPCODE_BYTE_LO.*.-1'),
    # TODO(step4-migration): post_l9_bz_bnz_pc_override OUTPUT_LO.*.-1 added
    # in commit b82462c5 (2026-06-03 cluster D BZ/BNZ move) after the round-2
    # audit, so the round-2 ratchet missed it. Same Tier A pattern as the
    # other OUTPUT_LO.*.-1 readers in this set: every L3+ OUTPUT_LO writer
    # fires before this op's post-L9 placement, so the prev-step alias here
    # is the same potential step-1 zero-prop bug class. Migration needs a
    # bake-side fix (rename the cancel gate to a non-aliasing form).
    ('post_l9_bz_bnz_pc_override', 'OUTPUT_LO.*.-1'),
    ('putchar_think_protocol', 'AX_CARRY_HI.*.-1'),
    ('putchar_think_protocol', 'AX_CARRY_LO.*.-1'),
})


# ---------------------------------------------------------------------------
# Cross-step DOCUMENTED-SAFE annotations
# ---------------------------------------------------------------------------
#
# Each entry is a ``(consumer_op_name, ssa_dim_name)`` pair that the
# safety check produces a finding for, but which has been *audited* (see
# ``docs/CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md``) and confirmed as an
# INTENTIONAL cross-step read: the same-step "writer" reported by the
# global writer scan runs *later in the same VM step* than the reader, so
# the ``.*.-1`` alias correctly resolves to the prior step's value (not
# the current step's not-yet-produced write). Step 1 then sees the
# initialised residual rather than a stale 0 (or sees 0 by design because
# the prev-step residual is the boot value).
#
# The dict value is a short justification — typically pointing at the
# comment in the op factory that explains the cross-step semantics. These
# pairs are MERGED into the effective allowlist by
# :func:`_emit_cross_step_safety_warnings`, so they continue to suppress
# the strict-mode error, but they no longer count toward the
# :data:`CROSS_STEP_BASELINE_ALLOWLIST` migration backlog (the ratchet
# test only inspects the baseline allowlist).
#
# Add an entry HERE rather than to the baseline when the cross-step read
# is the correct semantics (writer is genuinely later-in-step, or the
# residual is a step-boundary durable). Add to the baseline only when
# the read is a TODO for migration.
CROSS_STEP_DOCUMENTED_SAFE: Dict[Tuple[str, str], str] = {
    ('layer4_pc_relay', 'ADDR_KEY.*.-1'):
        "Reader is L4; sole producer is L14 (layer14_addr_key_neural_decode "
        "/ layer14_clear_addr_key_pollution), which runs AFTER L4 in the "
        "same step. The PC-marker residual carries prev-step's L14 write. "
        "See ops/l4_ops.py:211-219.",
    ('format_pointer_extraction', 'IO_IN_OUTPUT_MODE.*.-1'):
        "null_terminator_detection (phase=10.6) writes IO_IN_OUTPUT_MODE "
        "for the NEXT step's gating. Same numeric slot via SSA alias; "
        "byte-identical bake. See ops/l7_ops.py:608-611.",
    ('format_position_counter', 'IO_IN_OUTPUT_MODE.*.-1'):
        "null_terminator_detection (phase=10.6) is the sole writer and "
        "runs AFTER this op (phase=8.5) in the same step; reader sees "
        "the prev-step residual by design. See ops/l8_ops.py:1240-1244.",
    ('format_string_fetch_head', 'IO_IN_OUTPUT_MODE.*.-1'):
        "null_terminator_detection (phase=10.6) stages "
        "IO_IN_OUTPUT_MODE for the NEXT step's L9 format-fetch gating. "
        "Breaks the null_terminator_detection -> format_string_fetch_head "
        "SCC. See ops/l9_ops.py:1522-1527.",
    ('lev_detector_head', 'TEMP.*.-1'):
        "L8 form-2 control-flow head whose job is to detect the PRIOR "
        "instruction step's LEV opcode. The cross-step TEMP read is the "
        "core design: it materialises the prev-step saved-PC staging "
        "through the KV cache. See ops/control_flow_heads.py:252-261.",
    ('lev_detector_head', 'ADDR_B0_LO.*.-1'):
        "Same design as TEMP.*.-1 above — the LEV detector head reads "
        "prev-step ADDR_B0_LO from the saved-BP staging. "
        "See ops/control_flow_heads.py:252-261.",
    ('lev_detector_head', 'ADDR_B0_HI.*.-1'):
        "Same design as TEMP.*.-1 above — the LEV detector head reads "
        "prev-step ADDR_B0_HI from the saved-BP staging. "
        "See ops/control_flow_heads.py:252-261.",
    ('layer7_operand_gather', 'OUTPUT_LO.*.-1'):
        "L7 fires before any same-step OUTPUT_LO writer (L8+/L14+). The "
        "attended row is a prior step's marker; its residual OUTPUT_LO "
        "carries the previous step's value. Retires 31 OUTPUT_LO "
        "back-edges from later layers. See ops/l7_ops.py:167-175.",
    ('layer7_operand_gather', 'OUTPUT_HI.*.-1'):
        "L7 fires before any same-step OUTPUT_HI writer (L8+). Same "
        "prev-step semantics as OUTPUT_LO above. "
        "See ops/l7_ops.py:176-189.",
    ('layer14_addr_key_neural_decode', 'ADDR_B0_HI.*.-1'):
        "Sole same-step writer is L15 store_stack0_sp_byte0_addr "
        "(phase=15.2), which runs AFTER L14. Same numeric slot 206; "
        "byte-identical bake. See ops/l14_ops.py:3080-3084.",
    ('layer14_addr_key_neural_decode', 'ADDR_B0_LO.*.-1'):
        "Same design as ADDR_B0_HI above — L15 store_stack0_sp_byte0_addr "
        "is the sole same-step writer. See ops/l14_ops.py:3083-3084.",
    ('layer14_mem_generation', 'ADDR_B0_HI.*.-1'):
        "L15 store_stack0_sp_byte0_addr (phase=15.2) writes ADDR_B0_HI "
        "AFTER L14 in the same step. All earlier ADDR_B0_HI writers "
        "(L4/L8/L9/L13) resolve to the same numeric slot 206 as forward "
        "edges. See ops/l14_ops.py:739-746.",
    ('layer14_mem_generation', 'ADDR_B0_LO.*.-1'):
        "Same design as ADDR_B0_HI above — L15 store_stack0_sp_byte0_addr "
        "is the same-step LATER writer. See ops/l14_ops.py:747-750.",
    ('layer15_memory_lookup', 'TEMP.*.-1'):
        "L11 mul_partial / L14 temp_clear are the same-step writers but "
        "land AFTER L15 under dynamic scheduling; the TEMP residual the "
        "lookup heads consume is the PRIOR step's value carried through "
        "the KV cache. See ops/l15_ops.py:559-569.",
    ('layer15_store_stack0_sp_byte0_addr', 'OUTPUT_HI.*.-1'):
        "L16 lev_routing and tail_bit32_result_correction both fire AFTER "
        "L15 in the same step. Head 12 attends back to the prior-step "
        "STACK0/SP-marker token whose cached OUTPUT_HI is the "
        "prev-step value. See ops/l15_ops.py:666-673.",
    ('layer16_lev_routing', 'ADDR_B0_LO.*.-1'):
        "L16 is the last layer; the LEV routing materialises PC/BP/SP "
        "from the previous step's marker residuals via the KV cache. "
        "Same-step L15 store_stack0_sp_byte0_addr is the writer the "
        "global scan flags but dynamic scheduling lands it BEFORE L16, "
        "so the resolved read is genuine prev-step. "
        "See ops/l16_ops.py:1664-1678.",
    ('layer16_lev_routing', 'ADDR_B0_HI.*.-1'):
        "Same design as ADDR_B0_LO above — genuine prev-step LEV "
        "semantics. See ops/l16_ops.py:1671.",
    ('layer16_lev_routing', 'TEMP.*.-1'):
        "Same design — L11 mul_partial / L14 temp_clear writers run "
        "earlier than L16 (forward edges, resolved at the same numeric "
        "slot); the LEV routing reads the previous step's TEMP residual "
        "delivered via the KV cache. See ops/l16_ops.py:1669-1670.",
    # ----- Round 2 migrations (TODO backlog -> documented-safe) -----
    # Subset 1: consumer-is-topology-anchor (writes-only-declared, bake_fn
    # returns immediately — see make_*_dep_anchor_op factories). For these
    # ops the cross-step read has ZERO runtime impact: the declared reads
    # exist purely to fix the dep-graph slot the LayerCompiler reserves;
    # the bake itself produces no weight rows. Same-step earlier writers
    # therefore cannot "poison" any downstream computation through this
    # consumer, regardless of who else writes the base dim.
    ('_layer3_ffn_dep_anchor', 'EMBED_HI.*.-1'):
        "Topology anchor: writes=set(), bake_fn returns immediately "
        "(see ops/l3_ops.py:1032-1034). The EMBED_HI.*.-1 read is "
        "declarative-only — it sizes the dep-graph slot so layer3_ffn "
        "(kind='block', target_op_name='_layer3_ffn_dep_anchor') resolves "
        "to L3; the anchor's own bake produces no weight contribution.",
    ('_layer3_ffn_dep_anchor', 'EMBED_LO.*.-1'):
        "Same design as EMBED_HI.*.-1 above — topology anchor, no bake "
        "side-effect. See ops/l3_ops.py:1049-1052 (Phase 9.B mirror of "
        "layer3_ffn's EMBED_LO rename).",
    ('_layer3_ffn_dep_anchor', 'OP_LEV.*.-1'):
        "Topology anchor (no bake side-effect) AND all writers run later: "
        "_opcode_decode_ffn_dep_anchor / opcode_decode_ffn are L5 ops, "
        "anchor sits at L3. See ops/l3_ops.py:1045 (Phase 8.A OP_LEV "
        "PREV_STEP rationale: L5 owns OP_LEV; L3 anchor reads prev-step).",
    ('_layer3_ffn_dep_anchor', 'TEMP.*.-1'):
        "Topology anchor (no bake side-effect). The same-step earlier "
        "writer layer3_carry_forward_attn writes TEMP but the anchor's "
        "read cannot propagate because writes=set(). See "
        "ops/l3_ops.py:1044 (Phase 8.A.6 v2 TEMP_PREV_STEP rationale).",
    ('_layer6_attn_dep_anchor', 'AX_CARRY_HI.*.-1'):
        "Topology anchor: writes={CMP, ALU_LO, ALU_HI} but bake_fn returns "
        "immediately (see ops/l6_ops.py; the AX_CARRY reads are kept as "
        ".*.-1 SSA aliases to keep the anchor's earliest landable layer "
        "at L6 — purely structural). The anchor's bake produces no weight "
        "contribution; same-step writers cannot poison anything downstream.",
    ('_layer6_attn_dep_anchor', 'AX_CARRY_LO.*.-1'):
        "Same design as AX_CARRY_HI.*.-1 above — topology anchor, no bake "
        "side-effect. The AX_CARRY_LO/HI .*.-1 form is documented in the "
        "factory as 'kept so a same-step writer at L6 cannot push this "
        "anchor's earliest landable layer past L6'.",
    ('_layer6_ffn_dep_anchor', 'AX_CARRY_HI.*.-1'):
        "Topology anchor for layer6_routing_ffn (block, target_op_name="
        "'_layer6_ffn_dep_anchor'): writes are dep-graph mirrors, bake_fn "
        "returns immediately. The cross-step AX_CARRY_HI read is purely "
        "declarative; it cannot affect runtime weights.",
    ('_layer6_ffn_dep_anchor', 'AX_CARRY_LO.*.-1'):
        "Same design as AX_CARRY_HI.*.-1 above — topology anchor, no bake "
        "side-effect.",
    ('_layer6_ffn_dep_anchor', 'CMP.*.-1'):
        "Same design — topology anchor (no bake side-effect). See "
        "ops/l6_ops.py make_layer6_ffn_dep_anchor_op; the CMP.*.-1 read "
        "mirrors layer6_routing_ffn's own cross-step CMP read so the dep "
        "graph treats them identically.",
    ('_layer11_ffn_dep_anchor', 'ALU_LO.*.-1'):
        "Topology anchor for layer11_mul_partial (block, target_op_name="
        "'_layer11_ffn_dep_anchor'). bake_fn returns immediately; "
        "writes={TEMP} are dep-graph mirrors, not a runtime computation. "
        "The cross-step ALU_LO read is declarative-only.",
    ('_layer12_ffn_dep_anchor', 'TEMP.*.-1'):
        "Topology anchor for layer12_mul_combine (block, target_op_name="
        "'_layer12_ffn_dep_anchor'). bake_fn returns immediately. The "
        "cross-step TEMP read is declarative-only and cannot poison any "
        "downstream computation. See ops/l12_ops.py make_layer12_ffn_dep_"
        "anchor_op.",
    ('_opcode_decode_ffn_dep_anchor', 'OPCODE_BYTE_LO.*.-1'):
        "Topology anchor for opcode_decode_ffn (block, target_op_name="
        "'_opcode_decode_ffn_dep_anchor'). bake_fn returns immediately "
        "(ops/l5_ops.py:869-871) so the cross-step read is declarative-"
        "only. Additionally the sole writer layer5_fetch runs LATER in "
        "step than this anchor: anchor@L5(attn,0,1), fetch@L5(block,1,9). "
        "The actual opcode_decode_ffn op (block, same layer LATER intra) "
        "still appears in the BASELINE as the active bug record.",
    # Subset 2: all-writers-later-than-reader, confirmed by inspecting
    # the compiled layout (LayerCompiler.ops_per_layer + block-op layer
    # resolution). The cross-step alias correctly resolves to the prior
    # step's value through the KV cache because no same-step writer of
    # the base dim runs before the reader.
    ('layer3_carry_forward_attn', 'OP_LEV.*.-1'):
        "Sole writers are L5 ops (_opcode_decode_ffn_dep_anchor / "
        "opcode_decode_ffn), both LATER than L3 reader. See "
        "ops/l3_ops.py:733-738: 'OP_LEV_PREV_STEP marks the OP_LEV read "
        "as cross-step relative to L5 opcode_decode_ffn... PC-increment "
        "/ carry-correction rules use the previous step's OP_LEV decode "
        "as the skip-increment-on-LEV suppressor.'",
    ('layer3_carry_forward_attn', 'OUTPUT_HI.*.-1'):
        "L3 carry-forward attn is at layer 3 (attn, intra=0); every "
        "OUTPUT_HI writer runs LATER (L10/L14/L17 block ops, plus "
        "layer3_ffn at (3,1,4)). The prev-step OUTPUT_HI residual through "
        "the KV cache is the genuine input. See "
        "ops/l3_ops.py:1225-1236 (OUTPUT_*.*.-1 cross-step rationale).",
    ('layer3_carry_forward_attn', 'OUTPUT_LO.*.-1'):
        "Same design as OUTPUT_HI.*.-1 above — every OUTPUT_LO writer runs "
        "LATER than the L3 reader in the same step. The cross-step alias "
        "delivers the prior step's OUTPUT_LO residual via the KV cache.",
    ('layer3_ffn', 'OP_LEV.*.-1'):
        "Sole writers are L5 ops (_opcode_decode_ffn_dep_anchor / "
        "opcode_decode_ffn), both at later layers than the L3 FFN reader. "
        "See ops/l3_ops.py:733-738 — the L3 FFN PC-increment uses the "
        "PREVIOUS step's OP_LEV decode as the skip-increment-on-LEV "
        "suppressor; that's exactly what the prev-step alias delivers.",
    ('layer6_routing_ffn', 'DIV_STAGING.*.-1'):
        "Sole writer is layer10_alu at L13 (block, intra=38); reader is "
        "layer6_routing_ffn at L6 (block, intra=12). Writer runs MUCH "
        "later — the cross-step alias delivers the prior step's "
        "DIV_STAGING residual via the KV cache.",
    # ----- Round 3 migrations (TODO backlog -> documented-safe) -----
    # Subset 3A: consumer is a ``declarative_authority="topology_anchor"``
    # op whose ``compiler_ir`` has zero rules and whose ``bake_fn`` body is
    # ``return``. These ops declare ``reads`` / ``writes`` solely to fix
    # the LayerCompiler dep-graph slot for downstream consumers; no weight
    # rows are emitted, so the cross-step read has ZERO runtime impact
    # regardless of any same-step writer. Equivalent in spirit to Round 2
    # Subset 1 (``*_dep_anchor`` consumers); these are the *non-anchor-
    # named* sibling anchors. Verified by inspecting each op factory in
    # ``ops/l{6,10}_ops.py``.
    ('layer6_attn', 'AX_CARRY_HI.*.-1'):
        "Topology anchor: ``declarative_authority='topology_anchor'`` + "
        "empty ``CompilerIR()``; ``bake_fn`` body is ``return`` (see "
        "ops/l6_ops.py:2562-2604 ``make_layer6_attn_op``). Actual L6 attn "
        "weight bake happens in ``layer6_attn_bake`` (kind='model', "
        "phase=998.5). Declared ``writes={CMP, AX_CARRY_LO, AX_CARRY_HI}`` "
        "are dep-graph mirrors; cross-step read has zero runtime impact.",
    ('layer6_attn', 'AX_CARRY_LO.*.-1'):
        "Same design as AX_CARRY_HI.*.-1 above — topology anchor, "
        "``bake_fn`` returns immediately. See "
        "ops/l6_ops.py:2580-2587 for the AX_CARRY_*_PREV_STEP rationale "
        "comment (Phase 8.A targeted).",
    ('layer6_relay_heads', 'AX_CARRY_HI.*.-1'):
        "Topology anchor for L6 head 6/7 STACK0<-AX relay: "
        "``declarative_authority='topology_anchor'`` + empty "
        "``CompilerIR()``; ``bake_fn`` body is ``return`` (see "
        "ops/l6_ops.py:3022-3063 ``make_layer6_relay_heads_op``). Actual "
        "bake happens in ``layer6_relay_heads_bake`` (kind='model', "
        "phase=998.6). Cross-step read has zero runtime impact.",
    ('layer6_relay_heads', 'AX_CARRY_LO.*.-1'):
        "Same design as AX_CARRY_HI.*.-1 above — topology anchor, no "
        "bake side-effect. See ops/l6_ops.py:3041-3042 for the "
        "AX_CARRY_*_PREV_STEP rationale (Phase 8.A targeted).",
    ('layer10_byte_passthrough', 'TEMP.*.-1'):
        "Topology anchor for L10 head 1 AX-byte passthrough: "
        "``declarative_authority='topology_anchor'`` + empty "
        "``CompilerIR()``; ``bake_fn`` body is ``return`` (see "
        "ops/l10_ops.py:1944-1974 ``make_layer10_byte_passthrough_op``). "
        "Actual bake lives in ``layer10_byte_passthrough_bake`` "
        "(kind='block', target_op_name='layer10_byte_passthrough'); the "
        "anchor's read has zero runtime impact. The ``_bake`` sibling "
        "still appears in BASELINE as the active record.",
    ('layer10_carry_relay', 'CARRY.*.-1'):
        "Topology anchor for L10 head 0 carry relay: "
        "``declarative_authority='topology_anchor'`` + empty "
        "``CompilerIR()``; ``bake_fn`` body is ``return None`` (see "
        "ops/l10_ops.py:1902-1941 ``make_layer10_carry_relay_op``). "
        "Actual bake lives in ``layer10_carry_relay_bake``; this "
        "anchor's CARRY.*.-1 read sizes the dep-graph slot but emits "
        "no weight rows. The ``_bake`` sibling stays in BASELINE.",
    ('_layer10_attn_anchor', 'CARRY.*.-1'):
        "Phase 3 (mem cluster fix, 2026-06-05) sibling topology anchor "
        "for the L10 attn family: "
        "``declarative_authority='topology_anchor'`` + empty "
        "``CompilerIR()``; ``bake_fn`` body is ``return None`` (see "
        "ops/l10_ops.py ``make_layer10_attn_anchor_op``). Mirrors "
        "``layer10_carry_relay``'s CARRY.*.-1 read (decoupled from the "
        "L10 FFN family — see docs/MEMORY_PHASE2_BLOCKER_2026_06_05.md). "
        "The anchor sizes the dep-graph slot but emits no weight rows.",
    ('layer10_stack0_byte_relay', 'TEMP.*.-1'):
        "Topology anchor for L10 stack byte relays: "
        "``declarative_authority='topology_anchor'`` + empty "
        "``CompilerIR()``; ``bake_fn`` body is ``return None`` (see "
        "ops/l10_ops.py:2611-2640 ``make_layer10_stack0_byte_relay_op``). "
        "Actual bake lives in ``layer10_stack0_byte_relay_bake``; this "
        "anchor's read sizes the dep-graph slot but emits no weight "
        "rows. The ``_bake`` sibling stays in BASELINE.",
    # Subset 3B: all-writers-later-than-reader confirmed by the
    # LayerCompiler-resolved fire order. Reader at fire-tuple
    # (layer, kind_order={attn:0,ffn:1,block:2}, intra) compared to
    # writers' fire-tuples; every writer fires strictly later, so the
    # ``.*.-1`` alias delivers the previous step's residual via the KV
    # cache by design (no same-step write has landed when the reader
    # fires).
    ('layer8_mem_to_alu', 'ADDR_B0_LO.*.-1'):
        "All same-step writers fire LATER than the L8.45 (block, intra=3) "
        "reader: ``_layer13_attn_dep_anchor`` (L16 attn), "
        "``layer13_mem_addr_gather`` (L16 block), "
        "``layer15_store_stack0_sp_byte0_addr`` (L18 block), "
        "``layer8_sp_gather_bake`` (L9 block intra=7), "
        "``layer9_lev_addr_relay`` (L10 block), "
        "``layer9_lev_bp_to_pc_relay`` (L10 block). The prev-step "
        "residual via KV cache is the genuine input (slot shared with "
        "ADDR_B0_LO; bake byte-identical). See ops/l8_ops.py:2437-2443.",
    ('layer8_mem_to_alu', 'ADDR_B1_LO.*.-1'):
        "All same-step writers fire LATER than the L8.45 reader: "
        "``_layer13_attn_dep_anchor`` (L16 attn), "
        "``layer13_mem_addr_gather`` (L16 block), "
        "``layer8_sp_gather_bake`` (L9 block intra=7). See "
        "ops/l8_ops.py:2444-2449 for the ADDR_B{1,2}_*_PREV_STEP "
        "rationale (Phase 8.A continuation).",
    ('layer8_mem_to_alu', 'ADDR_B2_LO.*.-1'):
        "Same design as ADDR_B1_LO.*.-1 above — all writers "
        "(``_layer13_attn_dep_anchor`` L16, ``layer13_mem_addr_gather`` "
        "L16, ``layer8_sp_gather_bake`` L9.7) fire LATER than the L8.45 "
        "reader. The cross-step alias delivers the prior step's "
        "ADDR_B2_LO residual via the KV cache.",
    # Subset 3C: same-layer earlier writers are all topology anchors
    # (no bake side-effect); every real-bake writer fires LATER than the
    # reader. The reader's cross-step read is unambiguous because no
    # weight-emitting writer has landed when the reader fires.
    ('layer6_routing_ffn', 'CMP.*.-1'):
        "Reader at L6 (block, intra=3). The two same-layer-earlier "
        "writers — ``_layer6_attn_dep_anchor`` (L6 attn intra=0) and "
        "``layer6_attn`` (L6 attn intra=1) — are BOTH topology anchors "
        "(``declarative_authority='topology_anchor'`` + empty IR + "
        "``bake_fn=return``), so neither emits weight rows. The sole "
        "real-bake CMP writer (``layer9_alu`` at L10) fires LATER than "
        "this reader. The prev-step CMP residual via KV cache (BZ/BNZ "
        "branch decision committed at end of prev step) is the genuine "
        "input. See ops/l6_ops.py:2696-2703 (Phase 8.A CMP_PREV_STEP "
        "rationale).",
    ('layer8_sp_gather_bake', 'CMP.*.-1'):
        "Reader at L9 (block, intra=7). Same-layer-earlier writers are "
        "``_layer6_attn_dep_anchor`` and ``layer6_attn`` (both topology "
        "anchors at L6, no bake side-effect). The sole real-bake CMP "
        "writer (``layer9_alu`` at L10.2) fires LATER than this reader. "
        "The cross-step alias delivers the prev-step CMP (the BZ/BNZ "
        "branch decision just committed). See ops/l8_ops.py:1702-1712 "
        "(Phase 8.A CMP_PREV_STEP rationale).",
    ('layer13_ax_byte1_dump_carry', 'H1.*.-1'):
        "AX byte-1 register-dump carry. The carry head's ENTIRE purpose is "
        "to read the PREVIOUS VM step's ``H1`` one-hot (born at L9/block 10 "
        "on the producing fresh-AX step, held through block 39 via the KV "
        "cache) and copy it forward into ``H1_PREV_STEP`` so the carried-step "
        "byte-1 dump can re-emit it. The only same-step ``H1`` writer is "
        "``layer0_threshold_attn`` (L0), which produces the CURRENT step's "
        "(empty, on a carried step) one-hot — exactly the value we must NOT "
        "read. The ``.*.-1`` SSA alias correctly resolves to the prior "
        "step's value via the KV cache; reading the same-step dim would "
        "defeat the carry. This is the intentional cross-step read that "
        "breaks the H1-write 2-cycle (the head WRITES the distinct band "
        "``H1_PREV_STEP``, read by nobody upstream). See "
        "ops/l11_ops.py make_layer11_ax_byte1_dump_carry_op and "
        "docs/AX_BYTE1_DUMP_CARRY_H1_WRITE_CYCLE_2026_06_13.md.",
    ('layer13_ax_byte1_dump_carry', 'H2.*.-1'):
        "AX byte-1 register-dump carry VALUE-GENERALISATION (byte-1 5..11): "
        "the H2-band partner of the ``H1.*.-1`` read above. The fresh-step "
        "byte-1 emission one-hot is SPREAD across the LM head's H1/H2/H3 "
        "marker-distance bands (v 0..4 -> H1, 5..11 -> H2, 12..15 -> H3); the "
        "H1-only carry dropped byte-1 >= 5. This reads the PREVIOUS step's "
        "``H2`` one-hot via the ``.*.-1`` SSA alias (the only same-step ``H2`` "
        "writer is the L0 ``layer0_threshold_attn`` topology producer, which "
        "holds the CURRENT carried-step empty one-hot — the value we must NOT "
        "read) and copies it into the distinct ``H2_PREV_STEP`` band (read by "
        "nobody upstream -> no back-edge / no 2-cycle). See ops/l11_ops.py "
        "make_layer11_ax_byte1_dump_carry_op and "
        "docs/AX_BYTE1_DUMP_CARRY_LANDED_2026_06_13.md.",
    ('layer13_ax_byte1_dump_carry', 'H3.*.-1'):
        "AX byte-1 register-dump carry VALUE-GENERALISATION (byte-1 12..15): "
        "the H3-band partner of the ``H1.*.-1`` / ``H2.*.-1`` reads above. "
        "Same rationale: reads the PREVIOUS step's ``H3`` one-hot via the "
        "``.*.-1`` SSA alias and copies it into the distinct ``H3_PREV_STEP`` "
        "band (read by nobody upstream). See ops/l11_ops.py "
        "make_layer11_ax_byte1_dump_carry_op.",
    ('stack0_byte0_dump_carry', 'H1.*.-1'):
        "STACK0 byte-0 cross-step carry (the SOLE survivor of the deleted Root 2 "
        "register-dump machinery). The carry head reads the PREVIOUS VM step's "
        "STACK0-marker ``H1`` (high nibble) one-hot and copies it forward into "
        "``STACK0_B0_H1_PREV``. The ``.*.-1`` SSA alias resolves to the prior "
        "step's value via the KV cache; the head WRITES the distinct band "
        "``STACK0_B0_H1_PREV`` (read by nobody upstream) -> no back-edge. Kept "
        "because ``STACK0_B0_H1_PREV`` + ``STACK0_B0_H3_PREV`` are read LIVE by "
        "the campaign MUL multi-byte L19 boost as the literal-vs-var_mul "
        "discriminator. See ops/l11_ops.py make_stack0_byte0_dump_carry_op.",
    ('stack0_byte0_dump_carry', 'H3.*.-1'):
        "STACK0 byte-0 cross-step carry: the LOW-nibble partner of the "
        "``H1.*.-1`` read above. Copies the prev step's STACK0-marker ``H3`` "
        "(low nibble) one-hot into ``STACK0_B0_H3_PREV``. Same rationale. See "
        "ops/l11_ops.py make_stack0_byte0_dump_carry_op.",
    ('bp_save_prev_carry', 'CLEAN_EMBED_LO.*.-1'):
        "ENT saved-BP store cross-step carry (BP_SAVE_PREV — the func/nested/"
        "rec/var LI-from-frame 37-token desync). The carry head reads the "
        "PREVIOUS VM step's BP byte-k ``CLEAN_EMBED_LO`` (the low nibble of the "
        "clean old_BP, emitted correctly as the prev step's BP byte token but "
        "whose same-step OUTPUT residual is later nuked to 0xFF by the tail "
        "corruptor) and copies it forward into the distinct ``BP_SAVE_PREV`` "
        "band (read ONLY by the gated dump FFN -> no back-edge). The same-step "
        "``CLEAN_EMBED_LO`` is the FRESH ENT-step value (0xFF garbage / wrong "
        "row), so the ``.*.-1`` SSA alias (prev step's value via the KV cache) "
        "is exactly what we must read. The first ENT-bearing step is never VM "
        "step 0 (it follows a JSR), so the step-1 zero-propagation does not "
        "apply. See ops/l11_ops.py make_bp_save_prev_carry_op.",
    ('bp_save_prev_carry', 'CLEAN_EMBED_HI.*.-1'):
        "ENT saved-BP store cross-step carry (BP_SAVE_PREV): the HIGH-nibble "
        "partner of the ``CLEAN_EMBED_LO.*.-1`` read above. Copies the prev "
        "step's BP byte-k ``CLEAN_EMBED_HI`` (high nibble of old_BP) into "
        "``BP_SAVE_PREV+16..31``. Same rationale. See ops/l11_ops.py "
        "make_bp_save_prev_carry_op.",
}


def _find_cross_step_reads_with_same_step_writers(
    ops: Sequence[Operation],
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Return ``[(consumer_op_name, ssa_dim_name, same_step_writers), ...]``.

    For each op in ``ops``, scan its declared ``reads`` for SSA names whose
    ``step_offset`` is non-zero (the cross-step alias, e.g.
    ``OPCODE_BYTE_LO.*.-1``). Then look across ALL ops for any op whose
    ``writes`` contain the base dim (unversioned form). When at least one
    such writer exists, emit a tuple naming the reader, the SSA read, and
    the writer ops.

    The check is purely declarative: it only inspects ``reads`` / ``writes``
    sets on the op set. It does NOT consult the schedule, dim positions, or
    runtime state.

    Notes
    -----
    * Self-writers are excluded — an op that both writes ``X`` and reads
      ``X.*.-1`` is not a step-1 safety problem (the cross-step read is
      asking for the PRIOR step's own write, which is the canonical
      back-edge pattern).
    * Writes are matched against the *base dim only*. Today the corpus
      writes with unversioned names exclusively, so this is a strict
      match. If future ops adopt versioned writes, ``base_of`` strips
      the suffix before comparison.
    """
    # writers[base_dim] -> [op.name, ...]
    writers: Dict[str, List[str]] = defaultdict(list)
    for op in ops:
        for w in op.writes:
            base = base_of(w) if is_ssa_form(w) else w
            writers[base].append(op.name)

    findings: List[Tuple[str, str, Tuple[str, ...]]] = []
    for op in ops:
        for r in op.reads:
            if not is_ssa_form(r):
                continue
            try:
                parsed = parse_ssa_name(r)
            except ValueError:
                # Malformed SSA name — let the rest of the compile pipeline
                # surface the structured error; the safety check stays quiet.
                continue
            if parsed.step_offset == 0:
                continue
            base = parsed.base_dim
            ws = writers.get(base, ())
            # Exclude self-writers: an op that reads its own prior-step
            # write is the canonical back-edge case (not a step-1 bug).
            same_step = tuple(sorted(w for w in ws if w != op.name))
            if not same_step:
                continue
            findings.append((op.name, r, same_step))
    return findings


def _emit_cross_step_safety_warnings(
    ops: Sequence[Operation],
    *,
    enabled: bool = True,
    limit: Optional[int] = None,
    strict_error: bool = False,
    allowlist: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Run the cross-step safety check and emit one warning per finding.

    Returns the findings list (always — even when ``enabled=False``) so the
    caller can count / log without re-running the analysis. When
    ``enabled=True`` (the default) each finding is also emitted via
    ``warnings.warn`` with the :class:`CrossStepReadWarning` category. A
    fixed-format message names the consumer op, the SSA read, and the
    same-step writer set so a downstream agent / log scraper can pattern-
    match the structured fields.

    ``limit``: if set, only the first ``limit`` findings emit warnings (all
    findings are still returned). Use ``None`` (the default) to emit them
    all — the production op set has ~5-15 findings so the volume is
    bounded.

    Step 4 strict-error path (``strict_error=True``)
    ------------------------------------------------
    When the caller sets ``strict_error=True``, every finding whose
    ``(consumer, ssa_read)`` is NOT in ``allowlist`` is treated as a
    HARD ERROR: the function raises :class:`CrossStepReadError` after
    emitting the per-finding warnings. The error carries both the
    unallowed-finding subset and the full findings list as attributes
    so a caller can post-process (e.g. for incremental migration).

    ``allowlist``: an iterable of ``(consumer, ssa_read)`` tuples to
    suppress from the hard-error gate. When ``None`` (the default) the
    bundled :data:`CROSS_STEP_BASELINE_ALLOWLIST` UNION
    :data:`CROSS_STEP_DOCUMENTED_SAFE` is used — together they cover the
    82 findings present at Step-4 landing (the baseline is the
    migration backlog; documented-safe is the audited-intentional
    subset). Pass an empty iterable to disable both (i.e. promote every
    finding to an error). Pass a custom iterable to extend or replace
    them for a specific compile.

    The warning emission is independent of ``strict_error`` — every
    finding still emits a :class:`CrossStepReadWarning` so allowlisted
    entries remain visible in the warnings stream and analyzers can
    still pattern-match them.
    """
    findings = _find_cross_step_reads_with_same_step_writers(ops)
    if enabled:
        for i, (consumer, ssa_read, same_step_writers) in enumerate(findings):
            if limit is not None and i >= limit:
                break
            writers_preview = ", ".join(same_step_writers[:5])
            extra = max(0, len(same_step_writers) - 5)
            if extra:
                writers_preview = f"{writers_preview}, ... (+{extra} more)"
            msg = (
                f"Cross-step read {ssa_read!r} in op {consumer!r} may return "
                f"0 on VM step 1 because same-step writer(s) for base dim "
                f"{base_of(ssa_read)!r} exist: [{writers_preview}]. Did you "
                f"mean to OR the cross-step alias with the same-step dim, or "
                f"read the same-step dim directly? See "
                f"CrossStepReadWarning for the step-1 zero-propagation "
                f"class-of-bug."
            )
            warnings.warn(msg, CrossStepReadWarning, stacklevel=2)
    if strict_error:
        # Resolve the allowlist. ``None`` -> baked-in baseline UNION
        # the documented-safe set (entries audited as intentional
        # cross-step reads; see :data:`CROSS_STEP_DOCUMENTED_SAFE`).
        # Any explicit iterable (including an empty one) -> use as-is,
        # which lets a caller pass ``[]`` to promote every finding to
        # an error (audited or not) for the strictest no-cross-step
        # bake-author workflow.
        if allowlist is None:
            effective_allowlist: FrozenSet[Tuple[str, str]] = (
                CROSS_STEP_BASELINE_ALLOWLIST
                | frozenset(CROSS_STEP_DOCUMENTED_SAFE.keys())
            )
        else:
            effective_allowlist = frozenset(allowlist)
        unallowed = [
            (consumer, ssa_read, writers)
            for (consumer, ssa_read, writers) in findings
            if (consumer, ssa_read) not in effective_allowlist
        ]
        if unallowed:
            raise CrossStepReadError(
                unallowed_findings=unallowed,
                all_findings=findings,
            )
    return findings


# ---------------------------------------------------------------------------
# Public compile entry point
# ---------------------------------------------------------------------------


# In-process memo for compile_full_vm_dynamic results, keyed by a structural
# hash of the public kwargs. The disk cache lives at ``~/.cache/c4_release/
# compiled_vm/<sha256>.pt`` (see ``_legacy_redirect._cache_dir`` /
# ``_cache_key``) and short-circuits the ~60-120 s bake to a ~3 s torch.load
# of an ~830 MB pickle, but the dynamic entry still pays ~1.3 s for
# ``_collect_ops_for_compile`` BEFORE the disk-cache check, plus the
# deserialisation cost on every call. When a single process compiles with the
# same kwargs more than once -- the conftest ``_pure_neural_runner_model``
# session fixture, the class-scoped runner fixtures in
# ``tests/test_suite_1096_pytest.py``, and any in-process retry loop -- this
# memo returns the already-built ``(model, layout)`` in O(hash) time.
#
# Cross-process callers (fresh pytest invocations) still hit the disk cache;
# this memo only short-circuits same-process repeats. The key is the structural
# kwargs dict (the same one passed into ``_cache_key``), so cache invalidation
# tracks the disk-cache key 1:1 without needing to rehash source bytes.
_INPROC_COMPILE_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _build_cache_key_snapshot(
    *,
    S,
    enable_conversational_io,
    enable_tool_calling,
    enable_neural_io_think_protocol,
    alu_mode,
    n_heads,
    ffn_hidden,
    max_seq_len,
    pin_io_only,
    enable_moe_routing,
    positional_encoding,
    attention_normalization,
    rope_base,
    use_rms_norm,
    rms_norm_eps,
    require_declarative_bake,
    declarations_only,
    kv_eviction_policy,
    kv_eviction_n_steps,
    extra_residual_dims,
):
    """Single source of truth for the compile cache-invalidation snapshot.

    Both the in-process memo (``compile_full_vm_dynamic``) and the on-disk cache
    (``_bake_from_scheduled_ops`` -> ``_static._cache_key``) key on this dict, so
    any C4_* flag / build kwarg that changes the baked weights (or the model
    geometry) MUST appear here exactly once, and the two call sites MUST use the
    identical key set -- otherwise a flag registered on only one layer silently
    lets that layer hand back a stale model when the flag is toggled.

    Reconciled 2026-07-13: the former dual inline snapshots had drifted
    (``C4_L15_LOOKUP_CMP_VETO`` / ``C4_STACK0_NEXT_ARITH`` were memo-only;
    ``C4_ADDSUB_DECLARATIVE`` / ``C4_L10_ENT_AXCARRY`` were disk-only), a
    cache-collision hazard. This builder is their union (byte-identical values
    for every shared key). The ``sort_keys=True`` in both key hashers makes the
    dict order here irrelevant.
    """
    return {
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
        # Phase 7.F.2: include eviction policy in the cache key so OFF and
        # STATIC_LIVENESS builds don't share a serialised model (mirrors the
        # static-path snapshot in ``compile_full_vm``).
        "kv_eviction_policy": KVEvictionPolicy(kv_eviction_policy).value,
        "kv_eviction_n_steps": int(kv_eviction_n_steps),
        # Phase 10.B: include wrapper-expansion env flag so the merged-path
        # (~17-block, post_ops folded into Sequential FFNs) and the
        # expanded-path (~31-block, dedicated wrapper TransformerBlocks)
        # never share a serialised cache entry. See
        # ``make_expand_wrapper_blocks_op`` for the dispatch.
        "C4_DISABLE_WRAPPER_EXPANSION": (
            os.environ.get("C4_DISABLE_WRAPPER_EXPANSION") == "1"
        ),
        # Qwen R1 (docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md §R1).
        # When ``C4_QWEN_EXPORT_COMPAT=1`` the dim registry exposes a
        # NORM_COMPENSATOR slot and ``norm_compensator_seed`` (phase
        # 1400) seeds it with K=1000.0 on every token + zeros the
        # corresponding W_o / W_down rows. This changes d_model and
        # weight contents, so cache keys MUST diverge between flag
        # states.
        "C4_QWEN_EXPORT_COMPAT": (
            os.environ.get("C4_QWEN_EXPORT_COMPAT") == "1"
        ),
        # AX byte-1 register-dump emission flag (DEFAULT-ON; opt out with =0,
        # output-affecting, no source change): the ON / OFF builds must never
        # share a serialised entry.
        "C4_AX_BYTE1_DUMP": (
            os.environ.get("C4_AX_BYTE1_DUMP", "1") != "0"
        ),
        # AX byte-1 sign-extension delivery on a negative LEA-local frame address
        # (#343; DEFAULT-ON in the campaign config, opt out =0, output-affecting,
        # no source change): the L10 ADD high-byte adder sign-ext NOT-blocker pair.
        # The ON / OFF builds must never share a serialised entry.
        "C4_AX_BYTE1_SIGNEXT_LEA": (
            os.environ.get("C4_AX_BYTE1_SIGNEXT_LEA", "1") != "0"
        ),
        # AX high-byte (byte-2/3) all-step zero-default (DEFAULT-OFF; opt in with
        # =1, output-affecting, no source change): appends the l11
        # ``make_ax_hibyte_clear_allstep_op`` FFN + omits the redundant
        # ``H*_DUMP_OUT`` LM-head columns. The ON / OFF builds must never share a
        # serialised entry.
        "C4_AX_HIBYTE_CLEAR": (
            os.environ.get("C4_AX_HIBYTE_CLEAR", "0") != "0"
        ),
        # (C4_STACK0_B0_DUMP removed 2026-07: the STACK0-b0 dump machinery it
        # gated was deleted as provably-dead in the 30-token frame, so the flag
        # no longer affects the build and is no longer part of the cache key.)
        # Declarative L8 ADD/SUB wrap flag (DEFAULT-OFF; opt in with =1,
        # swaps the imperative AddSub5StageBlock for the DeclarativeAddSubBlock
        # composite, output-affecting, no source change): the ON / OFF builds
        # must never share a serialised entry.
        "C4_ADDSUB_DECLARATIVE": (
            os.environ.get("C4_ADDSUB_DECLARATIVE", "0") == "1"
        ),
        # Imperative AddSub byte-0 OUTPUT dominant-amplitude write (DEFAULT-ON,
        # opt out =0, output-affecting on the add/sub MARK_AX OUTPUT_LO/HI byte-0
        # one-hot): writes the block-10 result at amplitude 30.0 instead of 2.0
        # to out-vote the downstream L9 ALU_LO->OUTPUT_LO leak. The ON / OFF
        # builds must never share a serialised entry.
        "C4_ADDSUB_DUMP_BOOST": (
            os.environ.get("C4_ADDSUB_DUMP_BOOST", "1") != "0"
        ),
        # AX byte-1 FULL-WIDTH emission (DEFAULT-OFF, opt in =1,
        # output-affecting + geometry-affecting): adds the AX_BYTE1_FULL_WIDE
        # band, the un-aliased LM-head columns 16..255, and the L25-tail
        # band-FILL FFN. The ON / OFF builds must never share a serialised entry.
        "C4_AX_BYTE1_FULL_WIDTH": (
            os.environ.get("C4_AX_BYTE1_FULL_WIDTH", "0") != "0"
        ),
        # post-ENT SP-byte1=0xff H1+2 hardening (framing-recovery; DEFAULT-ON,
        # opt out with =0, output-affecting): the ON / OFF builds must never
        # share a serialised entry.
        "C4_ENT_SP_BYTE1_FF_H1_HARDEN": (
            os.environ.get("C4_ENT_SP_BYTE1_FF_H1_HARDEN", "1") != "0"
        ),
        # L8 ADJ-lo AX-marker blocker (DEFAULT-OFF, opt in =1, output-affecting on
        # post-LEV ADJ AX rows): the ON / OFF builds must never share a
        # serialised entry. Ships with the C4_L15_LEV func chain.
        "C4_L8_ADJ_LO_AX_MARKER_BLOCKER": (
            os.environ.get("C4_L8_ADJ_LO_AX_MARKER_BLOCKER", "1") == "1"
        ),
        # PSH-of-argument value-source AX lock (DEFAULT-ON, opt out =0,
        # output-affecting on the call-arg PSH store value): the ON / OFF
        # builds must never share a serialised entry.
        "C4_PSH_ARG_VAL_AX": (
            os.environ.get("C4_PSH_ARG_VAL_AX", "1") != "0"
        ),
        # L15 head-0 LI/LC-load suppressor inert (DEFAULT-ON, opt out =0,
        # output-affecting on func/nested/rec/var LI loads): the ON / OFF
        # builds must never share a serialised entry.
        "C4_L15_LI_SUPPR_INERT": (
            os.environ.get("C4_L15_LI_SUPPR_INERT", "1") != "0"
        ),
        # L15 LEV PC-restore head 14 (DEFAULT-OFF, opt in =1): grows L15
        # memory-lookup attention 14 -> 15 heads (changes num_heads + the L15
        # W_q/W_k/W_v/W_o shapes), so the ON / OFF builds must never share a
        # serialised entry.
        "C4_L15_LEV_PC_RESTORE": (
            os.environ.get("C4_L15_LEV_PC_RESTORE", "1") != "0"
        ),
        # L15 head-16 SI/SC store address-provenance CAM (DEFAULT-OFF, opt in
        # =1): grows L15 memory-lookup attention 16 -> 17 heads (changes
        # num_heads + the L15 W_q/W_k/W_v/W_o shapes), so the ON / OFF builds
        # must never share a serialised entry.
        "C4_SI_STORE_ADDR": (
            os.environ.get("C4_SI_STORE_ADDR") == "1"
        ),
        # L15 head-0 OP_SI/OP_SC store-row veto (DEFAULT-OFF, opt in =1): two
        # extra slot-0 Q writes on head 0 (var_three ``SI b`` stray OP_LI_RELAY).
        # Same head shape, different W_q, so ON / OFF must never share an entry.
        "C4_VAR_THREE_LI": (
            os.environ.get("C4_VAR_THREE_LI") == "1"
        ),
        # L15 head-16 SI-store-addr CAM null-high-address candidate veto
        # (DEFAULT-OFF, opt in =1): one slot-32 K veto on ADDR_B0_HI+0 so the
        # store-addr CAM fails-closed on the callee ENT-frame phantom (ADDR_B0=0)
        # at a call-site-arg LI (func_max/min operand-b). Different W_k, so ON /
        # OFF must never share an entry. Only installs when C4_SI_STORE_ADDR is
        # on (head 16 exists); OFF is byte-identical to golden.
        "C4_LI_VALUE_LOAD": (
            os.environ.get("C4_LI_VALUE_LOAD") == "1"
        ),
        # Derived-CAM MEMORY umbrella (DEFAULT-OFF, opt in =1): floors ON the
        # clean binary-address-CAM memory-fix heads (C4_SI_STORE_ADDR +
        # C4_VAR_THREE_LI). Changes the built head set, so ON / OFF must never
        # share a memo / disk entry.
        "C4_DERIVE_MEMORY": (
            os.environ.get("C4_DERIVE_MEMORY", "0") != "0"
        ),
        # absdiff arg-b LI value byte-0 LO-nibble de-contaminate (DEFAULT-OFF,
        # opt in =1). Flag-OFF registers NO rules -> byte-identical to golden.
        # Own key so ON/OFF never share a memo / disk entry.
        "C4_ABSDIFF_FIX": (
            os.environ.get("C4_ABSDIFF_FIX") == "1"
        ),
        # L15 LEV address-widening on head 14 (DEFAULT-OFF, opt in =1):
        # output-affecting on the LEV PC marker and on LI/LC load rows. Sub-knobs
        # fold into the same key so any retune invalidates the entry.
        "C4_L15_LEV_ADDR_WIDEN": (
            os.environ.get("C4_L15_LEV_ADDR_WIDEN", "1") != "0",
            os.environ.get("C4_L15_LEV_B0_BOOST", "8"),
            os.environ.get("C4_L15_LEV_JSR_DISC", "100"),
            os.environ.get("C4_L15_LEV_BYTE0_SELECT", "400"),
            os.environ.get("C4_L15_LEV_PC_ONLY", "1") != "0",
        ),
        # LEV (function-return) AX byte-1 stale-carry dump kill (DEFAULT-ON,
        # opt out =0, output-affecting on the func/nested/rec EXIT value): adds a
        # 3rd AX_CARRY_OVERFLOW unit firing on Σ AX_CARRY >= 3.0. The ON / OFF
        # builds must never share a serialised entry.
        "C4_LEV_AX_BYTE1_KILL": (
            os.environ.get("C4_LEV_AX_BYTE1_KILL", "1") != "0"
        ),
        # LEA-local multi-local E8 guard (DEFAULT-OFF, opt in =1, output-
        # affecting on the var multi-local LEA-from-frame address): the ON / OFF
        # builds must never share a serialised entry. See l10_ops.py
        # ``_lea_local_e8_multilocal_guard_enabled``.
        "C4_LEA_LOCAL_E8_MULTILOCAL_GUARD": (
            os.environ.get("C4_LEA_LOCAL_E8_MULTILOCAL_GUARD", "0") != "0"
        ),
        # ENT-step AX-dump 0xE8/0x02 (744) sentinel-slam guard (#311; DEFAULT-ON
        # in the campaign config, opt out =0; output-affecting on the main-ENT
        # AX byte-0/byte-1 dump rows): the ON / OFF builds must never share a
        # serialised entry. See l10_ops.py ``_tail_lea_e8_ent_guard_enabled``.
        "C4_TAIL_LEA_E8_ENT_GUARD": (
            os.environ.get("C4_TAIL_LEA_E8_ENT_GUARD", "1") != "0"
            and os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
        ),
        # L10 tail byte-0x39 STACK0-restore store-context guard (DEFAULT-OFF,
        # opt in =1, output-affecting on the binary-op STACK0 byte-0 emit): the
        # ON / OFF builds bake the byte_39_from_e8_addr rule with different
        # conditions/threshold, so they must never share a serialised entry.
        # See l10_ops.py ``mul_stack0_byte39_guard_enabled``.
        "C4_MUL_STACK0_BYTE39_GUARD": (
            os.environ.get("C4_MUL_STACK0_BYTE39_GUARD", "0") != "0"
        ),
        # (``C4_MUL_L11_SE_RECOVER`` cache-key entry removed 2026-07-14 — the
        # ``MulOperandSeRecoverFFN`` wrap it isolated was DELETED 2026-07-13
        # (root-fixed by ``C4_ALU_OPERAND_SURVIVE``), so ON / OFF now produce
        # byte-identical builds and no longer need separate serialised entries.
        # See docs/L10_ATTN_GLUE_2026_07_13.md dead-flag sweep.)
        # si/li LOAD byte-1 ADDRESS-leak discriminator (Inc-2, campaign-ON, opt
        # out =0): adds L10 head-1 slot 83 so ON / OFF builds must never share a
        # serialised entry. See shared.sili_cam_b1_enabled.
        "C4_SILI_CAM_B1": (
            os.environ.get("C4_SILI_CAM_B1", "1") != "0"
        ),
        # IMM full-derivation (task #392, DEFAULT-OFF): whole IMM opcode from
        # spec — decode + marker_broadcast relay + value_route (zero
        # hand-authored IMM rules). Implies C4_DERIVE_DECODE. Byte-identical to
        # the hand path, different source. ON / OFF must never share a
        # serialised entry. See shared.derive_imm_enabled.
        "C4_DERIVE_IMM": (
            os.environ.get("C4_DERIVE_IMM", "0") != "0"
        ),
        # BITWISE one-formula derivation (task #449, DEFAULT-ON): OR/XOR/AND
        # result nibbles from the single BLOG_SPEC §568 per-bit formula
        # ``c_a*a + c_b*b + c_ab*a*b``. The enumerated ``operator`` dispatch is
        # deleted; ``C4_DERIVE_BITWISE=0`` is a legacy no-op kill-switch.
        # Byte-identical golden e50521f3. See shared.derive_bitwise_enabled.
        "C4_DERIVE_BITWISE": (
            os.environ.get("C4_DERIVE_BITWISE", "1") != "0"
        ),
        # SHIFT full-derivation (task #448, DEFAULT-ON): the L13 SHL/SHR lookup
        # result is derived from BLOG_SPEC §599 (multiply/divide by powers of two
        # + mod-by-floor). The hand-authored ``(v << s) & 0xFF`` / ``v >> s``
        # lambdas have been deleted; ``C4_DERIVE_SHIFT=0`` is a legacy no-op
        # kill-switch. Byte-identical golden e50521f3. See
        # shared.derive_shift_enabled.
        "C4_DERIVE_SHIFT": (
            os.environ.get("C4_DERIVE_SHIFT", "1") != "0"
        ),
        # COMPARISON family derived from ONE zero-detector (task #446,
        # DEFAULT-ON). The per-op default+override hand enumeration has been
        # deleted; ``C4_DERIVE_CMP=0`` is a legacy no-op kill-switch.
        # Byte-identical golden e50521f3. See shared.derive_cmp_enabled.
        "C4_DERIVE_CMP": (
            os.environ.get("C4_DERIVE_CMP", "1") != "0"
        ),
        # CONTROL-family PC-override gate derivations (task #391 + #395,
        # DEFAULT-OFF). RE-DERIVATION (behaviour-correct, NOT byte-identical), so
        # ON / OFF must never share a memo / disk entry. See
        # shared.derive_{jmp,control}_enabled / derive_gate_scales_enabled.
        "C4_DERIVE_JMP": (
            os.environ.get("C4_DERIVE_JMP", "0") != "0"
        ),
        "C4_DERIVE_CONTROL": (
            os.environ.get("C4_DERIVE_CONTROL", "0") != "0"
        ),
        "C4_DERIVE_GATE_SCALES": (
            os.environ.get("C4_DERIVE_GATE_SCALES", "0") != "0"
        ),
        "C4_PC_OVERRIDE_K": os.environ.get("C4_PC_OVERRIDE_K", "1"),
        "C4_ACTSCALE_JSON": os.environ.get("C4_ACTSCALE_JSON", ""),
        # si/li 16-bit LOAD byte-1 value RESTORE (Inc-2 part-c, campaign-ON, opt
        # out =0, BAKE-affecting): registers the LI_RELOAD_B1 band + the capture /
        # restore PureFFN ops so ON / OFF builds STRUCTURALLY differ (d_model +
        # extra blocks) and must never share a serialised entry. See
        # shared.sili_b1_restore_enabled.
        "C4_SILI_B1_RESTORE": (
            os.environ.get("C4_SILI_B1_RESTORE", "1") != "0"
        ),
        # SI/SC store-AX byte-0 OUTPUT zero-default OVERRIDE (var_mul step-9,
        # campaign-ON, opt out =0, BAKE-affecting): rewrites the
        # l16_store_ax_carry_lo write tuples so ON / OFF builds bake different
        # L16 FFN weights and must never share a serialised entry. See
        # shared.store_ax_b0_override_enabled.
        "C4_STORE_AX_B0_OVERRIDE": (
            os.environ.get("C4_STORE_AX_B0_OVERRIDE", "1") != "0"
        ),
        # SI/SC store-AX byte-0 OVERRIDE V2 = the clean store-only
        # discriminator (default OFF, opt in =1, BAKE-affecting): appends
        # ALU/cmp opcode anti-conditions to store_ax_conditions AND switches
        # the l16_store_ax_carry_lo write to the override form, so the ON/OFF
        # builds bake different L16 FFN weights and must never share a
        # serialised entry. See shared.store_ax_b0_override_v2_enabled.
        "C4_STORE_AX_B0_OVERRIDE_V2": (
            os.environ.get("C4_STORE_AX_B0_OVERRIDE_V2", "1") != "0"
        ),
        # absdiff / func-return AX byte-1 OUTPUT_LO stale-marker de-contamination
        # (DEFAULT OFF, opt in =1, STRUCTURALLY-affecting): registers TWO extra
        # L10-tail PureFFN post_ops (a bounded ABSDIFF_RET_LEAK l6-crush flag +
        # the corrector) so ON / OFF builds STRUCTURALLY differ (extra FFN ops +
        # blocks) and must never share a serialised entry. Flag-OFF registers NO
        # rules -> byte-identical to golden. Held DEFAULT-OFF in the 2026-07
        # round-1 land: ON regressed test_lea_basic (crush-band over-fires on the
        # ENT-frame LEA row). See shared.absdiff_ret_byte1_enabled.
        "C4_ABSDIFF_RET_BYTE1": (
            os.environ.get("C4_ABSDIFF_RET_BYTE1", "0") == "1"
        ),
        # func/var/loop/gcd/nested step-0 JSR-step BP byte-3 = 0x00 CLEAR
        # (campaign-ON, opt out =0, STRUCTURALLY-affecting): registers an extra
        # L10-tail PureFFN post_op so ON / OFF builds STRUCTURALLY differ (an
        # extra FFN op + block) and must never share a serialised entry. Gated on
        # the campaign prerequisites (``C4_NO_STACK0_EMIT`` +
        # ``C4_OPERAND_FROM_MEMSP``) so the non-campaign / golden build is
        # byte-identical. See shared.jsr_bp_byte3_clear_enabled.
        "C4_JSR_BP_BYTE3_CLEAR": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"
            and os.environ.get("C4_JSR_BP_BYTE3_CLEAR", "1") != "0"
        ),
        # loop_sum in-loop 2nd-local ``LEA &sum`` byte-0 0xE0 RESTORE (#330,
        # campaign-ON, opt out =0, BAKE-affecting): registers the
        # l10_loop_lea_b0_e0 PureFFN post_op so ON / OFF builds STRUCTURALLY
        # differ (an extra FFN op + block) and must never share a serialised
        # entry. Gated on the campaign prerequisites so the non-campaign /
        # golden build is byte-identical. See shared.loop_lea_b0_e0_restore_enabled.
        "C4_LOOP_LEA_B0_E0": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"
            and os.environ.get("C4_LOOP_LEA_B0_E0", "1") != "0"
        ),
        # PROJECT_0XE8_SLAM Phase-2: MULTIPLICATIVE OP_LEA gate on the
        # loop_lea_b0_e8 / _e0 restore ops (DEFAULT OFF, opt in =1,
        # BAKE-affecting): adds a per-unit gate to EVERY unit in both rule
        # families so the discriminator's W_gate / b_gate weights differ ON vs
        # OFF and the two serialised models must NEVER share an entry. Gated on
        # the campaign prerequisites so the non-campaign / golden build is
        # byte-identical. See shared.loop_lea_oplea_gate_enabled.
        "C4_LOOP_LEA_OPLEA_GATE": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"
            and os.environ.get("C4_LOOP_LEA_OPLEA_GATE", "1") != "0"
        ),
        # func re-read-LEA ``&b`` byte-0 0xE8 over-fire FIX (campaign-ON, opt out
        # =0, BAKE-affecting): adds an ``OP_ENT`` condition + a +60 threshold bump
        # to the L10 ``e8_alubp_memsp`` writer so ON / OFF builds bake different
        # L25-tail FFN weights and must never share a serialised entry. Gated on
        # the campaign prerequisites so the non-campaign / golden build is
        # byte-identical. See l10_ops._lea_e8_first_ent_gate_enabled.
        "C4_LEA_E8_FIRST_ENT_GATE": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"
            and os.environ.get("C4_LEA_E8_FIRST_ENT_GATE", "1") != "0"
        ),
        # nested callee-ENT AX-dump over-fire FIX (#342, campaign-config, DEFAULT
        # OFF, opt in =1, BAKE-affecting): appends a FETCHED-ENT
        # ``OPCODE_BYTE_LO+6`` NOT-blocker to the L10 ``e8_alubp_memsp`` writer so
        # ON / OFF builds bake different L25-tail FFN weights and must never share
        # a serialised entry. Gated on the campaign prerequisites so the
        # non-campaign / golden build is byte-identical. DEFAULT OFF (does NOT
        # track the campaign floor). See l10_ops._lea_e8_nested_ent_axdump_enabled.
        "C4_NESTED_ENT_AXDUMP": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"
            and os.environ.get("C4_NESTED_ENT_AXDUMP", "0") != "0"
        ),
        # Operand-CAM frame-depth ALU-amplifier umbrella (campaign OFF by
        # default, opt in =1, BAKE-affecting): ``C4_OPCAM_FRAME=1`` turns on the
        # L10 multi-param LEA byte-0 ALU-AMPLIFIER (+4 L25-tail FFN rules), so
        # ON / OFF builds bake a different-width tail bank and must never share a
        # serialised entry. Mirrors ``l10_ops._lea_byte0_alu_amplify_enabled``:
        # the dedicated ``C4_LEA_BYTE0_ALU_AMPLIFY`` kill-switch (=0/=1)
        # overrides the umbrella; both require the campaign STACK0-drop, so the
        # non-campaign / golden build is byte-identical.
        "C4_OPCAM_FRAME": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and (
                os.environ.get("C4_LEA_BYTE0_ALU_AMPLIFY") != "0"
                if os.environ.get("C4_LEA_BYTE0_ALU_AMPLIFY") is not None
                else os.environ.get("C4_OPCAM_FRAME", "0") != "0"
            )
        ),
        # func_max/func_min CMP loaded-operand-A clean (task #428, DEFAULT-OFF,
        # opt in =1, MODULE-affecting): installs CmpLoadedOperandCleanFFN on the
        # L9 block.ffn so the ON / OFF models are structurally different modules
        # and must never share a serialised entry. Gated on the campaign
        # prerequisite ``C4_NO_STACK0_EMIT`` so the non-campaign / golden build is
        # byte-identical. See shared.func_cmp_operand_clean_enabled.
        "C4_FUNC_CMP_OPERAND_CLEAN": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_FUNC_CMP_OPERAND_CLEAN", "0") == "1"
        ),
        # DERIVED clean-one-hot operand delivery (CBC, DEFAULT-OFF, opt in =1,
        # MODULE-affecting): installs CleanOperandOneHotFFN on the L8 main
        # block.ffn so the ON / OFF models are structurally different modules and
        # must never share a serialised entry. Gated on the campaign prerequisite
        # C4_NO_STACK0_EMIT so the non-campaign / golden build is byte-identical.
        # See shared.clean_operand_enabled.
        "C4_CLEAN_OPERAND": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_CLEAN_OPERAND", "0") != "0"
        ),
        # Arithmetic-only clean operand delivery (CBC pass-gain, DEFAULT-ON, opt
        # out =0, MODULE-affecting): installs the SAME CleanOperandOneHotFFN wrap
        # with an ARITHMETIC-ONLY op_dims tuple (no cmp dims) so the ON model's
        # wrap module differs from both flag-OFF and the full C4_CLEAN_OPERAND
        # variant and must never share a serialised entry. Gated on the campaign
        # prerequisite C4_NO_STACK0_EMIT so the non-campaign / golden build is
        # byte-identical. Default is ON only in the campaign config; escape hatch
        # C4_CLEAN_OPERAND_ADD=0 reproduces golden e50521f3. See
        # shared.clean_operand_add_enabled.
        "C4_CLEAN_OPERAND_ADD": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_CLEAN_OPERAND_ADD", "1") != "0"
        ),
        # Bitwise clean operand delivery (campaign-ON, opt out =0,
        # MODULE-affecting): ADDS OP_AND/OP_OR/OP_XOR to the CleanOperandOneHotFFN
        # op_dims so the ON model's wrap module differs and must never share a
        # serialised entry. Gated on C4_NO_STACK0_EMIT so the golden build is
        # byte-identical. See shared.clean_operand_bitwise_enabled.
        "C4_CLEAN_OPERAND_BITWISE": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_CLEAN_OPERAND_BITWISE", "1") != "0"
        ),
        # SC/LC byte-0 reload (campaign-ON, opt out =0, BAKE-affecting): adds
        # OP_LC to the L15 head-0 #318 keystone slot-103 Q gate so ON / OFF
        # builds must never share a serialised entry. See
        # l15_ops._l15_sclc_byte0_on.
        "C4_SCLC_LC_B0": (
            os.environ.get(
                "C4_SCLC_LC_B0",
                "1" if os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
                else "0",
            ) != "0"
        ),
        # Multi-arg first-param LI JSR-phantom value-row penalty (campaign-ON, opt
        # out =0): adds L15 head-0 slot 104 (a MARK_AX-gated -OP_JSR K penalty) so
        # the genuine PSH'd-arg store row out-scores the callee JSR/ENT-step
        # phantom MEM rows. ON / OFF builds bake a different W_q/W_k row and must
        # never share a serialised entry. See l15_ops._l15_li_jsr_phantom_penalty_on.
        "C4_L15_LI_JSR_PHANTOM": (
            os.environ.get(
                "C4_L15_LI_JSR_PHANTOM",
                "1" if os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
                else "0",
            ) != "0"
        ),
        # C4_SHIFT_OUTPUT_B0_CLEAR: now INERT (the ShiftOutputClearFFN wrap it
        # gated was DELETED 2026-07-13; the SHR OUTPUT byte-0 leak is cancelled at
        # its L11 source by C4_OUTPUT_B0_NOLEAK). Kept as an always-same cache-key
        # term for serialised-key stability; a follow-up dead-flag sweep can drop it.
        "C4_SHIFT_OUTPUT_B0_CLEAR": (
            os.environ.get("C4_SHIFT_OUTPUT_B0_CLEAR", "1") != "0"
        ),
        # L11 OUTPUT byte-0 no-leak root (DEFAULT-ON, BAKE-affecting): when active
        # a 1-unit L11 post-op FFN is baked (cancelling OUTPUT byte-0 on the OP_SHR
        # compute row) so the ON / OFF builds differ and must never share a
        # serialised entry. See shared.output_b0_noleak_enabled.
        "C4_OUTPUT_B0_NOLEAK": (
            os.environ.get("C4_OUTPUT_B0_NOLEAK", "1") != "0"
        ),
        # if_var GT-FALSE 0xF-leak guard (#339, campaign-ON, opt out =0,
        # BAKE-affecting): when active the L10 ordering-engine ``hi_lt`` (CMP+0)
        # blocker DROPS its ``ALU_HI+15`` veto term, so the ON / OFF builds bake
        # different FFN weights and must never share a serialised entry. Gated on
        # the campaign prerequisite ``C4_NO_STACK0_EMIT`` so the non-campaign /
        # flag-OFF build is byte-identical to golden. See
        # shared.cmp_hi_lt_alu15_leak_guard_enabled.
        "C4_CMP_HI_LT_ALU15_GUARD": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_CMP_HI_LT_ALU15_GUARD", "1") != "0"
        ),
        # if_var GT-TRUE lo_lt-leak guard (DEFAULT-ON, opt out =0,
        # BAKE-affecting): when active the ComparisonCombine GT/GE
        # ``(hi_eq AND lo_lt)`` override threshold is RAISED 2.5 -> 2.75, so the
        # ON / OFF builds bake different ``b_up`` and must never share a
        # serialised entry. Gated on the campaign prerequisite
        # ``C4_NO_STACK0_EMIT`` so the non-campaign / flag-OFF build is
        # byte-identical to golden. See shared.cmp_gt_lo_lt_hieq_guard_enabled.
        "C4_CMP_GT_LO_LT_HIEQ_GUARD": (
            os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
            and os.environ.get("C4_CMP_GT_LO_LT_HIEQ_GUARD", "1") != "0"
        ),
        # if_var BZ/BNZ branch-target byte-0 HIGH-NIBBLE correction (#430;
        # DEFAULT-ON, opt out =0, BAKE-affecting): adds 32 odd-FETCH_HI
        # correction units to post_l9_bz_bnz_pc_override so a BZ/BNZ target index
        # >= 16 keeps byte-0's high nibble (BZ 16 -> PC 130, not 2). The ON / OFF
        # builds bake different FFN weights and must never share a serialised
        # entry. See l6_ops._ifvar_bz_hi_nibble_enabled.
        "C4_IFVAR_BZ_HI_NIBBLE": (
            os.environ.get("C4_IFVAR_BZ_HI_NIBBLE", "1") != "0"
        ),
        # Multilocal-ENT AX byte-0 source fix (DEFAULT-ON in campaign, opt out
        # =0, BAKE-affecting): when active ``make_l10_ent_axcarry_op`` appends a
        # PureFFN post_op to the L25 tail block that re-asserts the carried AX
        # over the leaked-LEA materializer on the multilocal main-ENT step, so
        # the ON / OFF builds STRUCTURALLY differ (extra post_op / FFN units)
        # and must never share a serialised entry. See
        # l10_ops._l10_ent_axcarry_enabled.
        "C4_L10_ENT_AXCARRY": (
            os.environ.get("C4_L10_ENT_AXCARRY", "1") != "0"
            and os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"
        ),
        # L7 head-1 re-read-LEA BP-frame RE-SHARPEN (func_add/mul/square/max/
        # min; DEFAULT-ON in campaign, opt out =0, BAKE-affecting): adds a Q/K
        # scoring slot to the SHARED L7 operand-gather head 1. The ON / OFF
        # builds differ in W_q/W_k so they must NEVER share a serialised entry.
        # Gated on ``operand_from_memsp_enabled()`` (DEFAULT-ON post-flip;
        # reading the raw env ``== "1"`` would default OFF and let the campaign
        # ON bake collide with a non-campaign cache entry). See
        # shared.func_lea_reread_bp_resharpen_enabled.
        "C4_FUNC_LEA_REREAD_BP_RESHARPEN": (
            operand_from_memsp_enabled()
            and os.environ.get("C4_FUNC_LEA_REREAD_BP_RESHARPEN", "1") != "0"
        ),
        # Single CAMPAIGN-CONFIG entry point (DEFAULT-OFF; opt in with =1):
        # ``C4_CAMPAIGN=1`` OR-ins the ON floor for the coherent 30-token
        # campaign set (no_stack0_emit + operand_from_memsp + si_store_addr +
        # operand_cam_fix — see ops.shared.campaign_enabled). Turning it on
        # STRUCTURALLY changes the build, so the ON / OFF builds MUST NEVER
        # share a serialised entry. DEFAULT-OFF -> golden (35-tok) build is
        # byte-identical (key unchanged when unset).
        "C4_CAMPAIGN": campaign_enabled(),
        # EMIT-G5 R-BYTE ``ax_bytes_zero`` fold verification toggle (DEFAULT OFF,
        # opt in =1). BYTE-NEUTRAL: both states bake the identical weights (golden
        # ``e50521f3``), but the flag routes the spec-driven generator through a
        # SECOND independent code path (see l14_ops._make_ax_bytes_zero_op /
        # ops.shared.emit_g5_rbyte_enabled). Keyed here so the ON / OFF builds
        # never share a serialised entry — a cross-state hit would silently skip
        # the flag-ON path during the byte-identity golden check. See
        # docs/EMIT_G5_ROLLOUT_2026_07_13.md.
        "C4_EMIT_G5_RBYTE": emit_g5_rbyte_enabled(),
        # CLEAN_EMITTER generic all-marker-row OUTPUT sink (DEFAULT-ON, opt out
        # =0, BAKE-affecting): appends a 32-unit PureFFN post_op to the L25 tail
        # block, so the ON / OFF builds have different state_dicts and MUST NEVER
        # share a serialised entry. See l0_ops._clean_emitter_enabled.
        "C4_CLEAN_EMITTER": (os.environ.get("C4_CLEAN_EMITTER", "1") != "0"),
        # M8 pilot: STACK0 store-loaded byte-writeback ENUMERATED (255 per-value
        # AND) -> COMPUTED (32 per-nibble route) collapse (DEFAULT-ON, kill-switch
        # =0). Changes the L10-tail FFN hidden_dim AND the emitted OUTPUT delta,
        # so the ON / OFF builds have different state_dicts and MUST NEVER share
        # a serialised entry. See l10_ops._stack0_store_loaded_computed_enabled.
        "C4_STACK0_STORE_LOADED_COMPUTED": (
            os.environ.get("C4_STACK0_STORE_LOADED_COMPUTED", "1") != "0"
        ),
        # STACK0 pop-loaded / store-top-e8 byte-writeback ENUMERATED (255
        # per-value AND) -> COMPUTED (32 per-nibble route) collapses (siblings of
        # the M8 pilot, DEFAULT-ON). Each changes the L10-tail FFN hidden_dim AND
        # the emitted OUTPUT delta, so the ON / OFF builds have different
        # state_dicts and MUST NEVER share a serialised entry. See
        # l10_ops._stack0_pop_loaded_computed_enabled /
        # _stack0_store_e8_computed_enabled.
        "C4_STACK0_POP_LOADED_COMPUTED": (
            os.environ.get("C4_STACK0_POP_LOADED_COMPUTED", "1") != "0"
        ),
        "C4_STACK0_STORE_E8_COMPUTED": (
            os.environ.get("C4_STACK0_STORE_E8_COMPUTED", "1") != "0"
        ),
        # GAP-PRIMITIVE #3 pilot: STACK0 store-top-e0 CROSS-LANE ALU->OUTPUT
        # ENUMERATED (254 per-value AND) -> COMPUTED (32 per-nibble route)
        # collapse (DEFAULT-ON, kill-switch =0). Changes the L10-tail FFN hidden_dim
        # AND the emitted OUTPUT delta, so the ON / OFF builds have different
        # state_dicts and MUST NEVER share a serialised entry. See
        # l10_ops._stack0_store_top_e0_computed_enabled.
        "C4_STACK0_STORE_TOP_E0_COMPUTED": (
            os.environ.get("C4_STACK0_STORE_TOP_E0_COMPUTED", "1") != "0"
        ),
        # wide_mul_byte1_preserve byte-writeback ENUMERATED (256 per-value AND)
        # -> COMPUTED (32 per-nibble route) collapse (DEFAULT-OFF, opt in =1).
        # Changes the L10-tail FFN hidden_dim (256 -> 32) AND the emitted OUTPUT
        # delta, so the ON / OFF builds have different state_dicts and MUST
        # NEVER share a serialised entry. See
        # l10_ops._wide_mul_byte1_computed_enabled.
        "C4_WIDE_MUL_BYTE1_COMPUTED": (
            os.environ.get("C4_WIDE_MUL_BYTE1_COMPUTED", "0") == "1"
        ),
        # M8 collapse for the L14 ADDR_KEY nibble decode: the load-query lo+hi
        # ENUMERATED (2 x 16x16 = 512 per-(lo,hi) AND) bank -> COMPUTED (2 x 32
        # per-nibble route) collapse (DEFAULT-OFF, opt in =1). Changes the L14
        # addr_key FFN hidden_dim (512 -> 64) AND the ADDR_KEY firing/write, so
        # the ON / OFF builds have different state_dicts and MUST NEVER share a
        # serialised entry. See l14_ops._l14_byte_computed_enabled.
        "C4_L14_BYTE_COMPUTED": (
            os.environ.get("C4_L14_BYTE_COMPUTED", "0") != "0"
        ),
        # GAP-PRIMITIVE #2: multi_pass MUL cascade replacing the L11 mul-partial
        # / L12 mul-combine lookup (DEFAULT-OFF, opt in =1). Adds the
        # MUL_MULTIPASS_WS 240-dim workspace band AND swaps block.ffn for the
        # 7-pass cascade, so the ON / OFF builds have different state_dicts and
        # MUST NEVER share a serialised entry. See ops/shared.mul_multipass_enabled.
        "C4_MUL_MULTIPASS": (
            os.environ.get("C4_MUL_MULTIPASS", "0") == "1"
        ),
        # GAP-PRIMITIVE #2 (DIV): multi_pass long-division cascade replacing the
        # L10 FlattenedDivMod composite (DEFAULT-OFF, opt in =1). Adds the
        # DIV_MULTIPASS_WS 1728-dim workspace + result-lane bands AND swaps the
        # divmod post_op for the 43-pass cascade, so the ON / OFF builds have
        # different state_dicts and MUST NEVER share a serialised entry. See
        # ops/shared.div_multipass_enabled.
        "C4_DIV_MULTIPASS": (
            os.environ.get("C4_DIV_MULTIPASS", "0") == "1"
        ),
        # MUL byte-1 delivery (DEFAULT-OFF, opt in =1): OP_MUL/OP_SHL W_up
        # blocker on the ``layer14_jsr_ax_bytes_zero`` clear units (idx104/idx106
        # spurious-OP_JSR-leak). Output-affecting → ON / OFF builds must never
        # share a serialised cache entry. See ops/l14_ops._mul_b1_delivery_enabled.
        "C4_MUL_B1_DELIVERY": (
            os.environ.get("C4_MUL_B1_DELIVERY", "0") != "0"
        ),
        # Combined ALU operand-survival fix (DEFAULT-ON, opt out =0,
        # WEIGHT+BAKE-affecting): the block-15 L9-clear operand spare
        # (l9_ops._alu_operand_survive_enabled -- extra W_up NOT-blocker
        # columns) AND the block-17 head-4 CMP Q-veto
        # (model_ops.make_cmp_h4_qveto_op -- overwrites head 4's W_q at the six
        # CMP columns with -1e5). Both survive operand-A through the two
        # ALU-clear crushes so CmpOperandSeRecoverFFN / MulOperandSeRecoverFFN
        # are inert and deleted. ON / OFF bake different block-15 W_up +
        # block-17 W_q, so they must NEVER share a memo / disk entry. OFF (=0)
        # reproduces the fork-point golden. See docs/SERECOVER_DELETE_2026_07_13.md.
        "C4_ALU_OPERAND_SURVIVE": (
            os.environ.get("C4_ALU_OPERAND_SURVIVE", "1") != "0"
        ),
        # Whether OP_LT is in the L9 raw-band clear spare (DEFAULT-OFF; LT
        # excluded fixes func_min id675 while EQ/NE/GT/LE/GE stay spared so
        # if_eq/bool_and hold). Weight-affecting (adds/removes the OP_LT -1e6
        # blocker column on the block-15 clear), so ON/OFF must not share a memo
        # entry. Only meaningful when C4_ALU_OPERAND_SURVIVE is ON. See
        # docs/FUNCMIN_FIX_2026_07_13.md.
        "C4_ALU_OPERAND_SURVIVE_CMP_RAW": (
            os.environ.get("C4_ALU_OPERAND_SURVIVE_CMP_RAW", "0") != "0"
        ),
        # Auto-widen: extra residual bands change d_model / n_heads, so a
        # widened model must never share a serialised cache entry with the
        # baseline (or with a different requested band set).
        "extra_residual_dims": (
            tuple(sorted(extra_residual_dims.items()))
            if extra_residual_dims else None
        ),
        # L15 li_lc_stack0_h0 lookup-head comparison-step veto (bool_and
        # id=1087, DEFAULT-ON, opt out =0, BAKE-affecting): when active the
        # head-0 slot-0 Q discriminator gains six OP_<cmp> * -1e6 veto cells,
        # so the ON / OFF builds bake a different W_q row and must never share
        # a memo entry. See shared.l15_lookup_cmp_veto_enabled.
        "C4_L15_LOOKUP_CMP_VETO": (
            os.environ.get("C4_L15_LOOKUP_CMP_VETO", "1") != "0"
        ),
        # Consumer-opcode LOOKAHEAD (#221; DEFAULT-ON, opt out =0): adds the
        # PC+8 chain + lookahead fetch head + arith-decode flag + AX->STACK0
        # relay + prior-arith latch + dump-block flag bands AND the dump's
        # STACK0_B0_DUMP_BLOCK blocker (output-affecting on multi-op arith
        # intermediate-operand frames). On / off builds must NEVER share a
        # memo / disk entry -- the band presence changes d_model and the dump
        # gate changes emission, so this flag toggles geometry AND output.
        "C4_STACK0_NEXT_ARITH": (
            os.environ.get("C4_STACK0_NEXT_ARITH", "1") != "0"
        ),
        # Namespace the dynamic cache so it never collides with the static
        # entry (same kwargs, different scheduler).
        "__dynamic": True,
    }


def _inproc_cache_key(snapshot: dict) -> str:
    """Stable SHA1 of a kwargs snapshot for in-process memoisation."""
    import hashlib as _hashlib
    import json as _json
    payload = _json.dumps(snapshot, sort_keys=True, default=repr).encode("utf-8")
    return _hashlib.sha1(payload).hexdigest()


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
    arch: Optional[ModelArchitectureSpec] = None,
    positional_encoding: Optional[str] = None,
    attention_normalization: Optional[str] = None,
    rope_base: Optional[float] = None,
    use_rms_norm: Optional[bool] = None,
    rms_norm_eps: Optional[float] = None,
    require_declarative_bake: Optional[bool] = None,
    declarations_only: bool = False,
    kv_eviction_policy=None,
    kv_eviction_n_steps: int = 64,
    strict: bool = True,
    allow_sealed_cycles: bool = True,
    model_shape_constraint=None,
    target_shape_overrides=None,
    d_model_packing: bool = False,
    d_model_packing_target: Optional[int] = None,
    cross_step_baseline_allowlist: Optional[Iterable[Tuple[str, str]]] = None,
    # Model-semantics umbrella flags (see
    # ``docs/MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md``). All default to
    # ``None`` so existing callers see no behavior change. ``preset=`` is
    # the convenience shortcut; the six per-axis flags are the per-axis
    # override surface. Mixing ``preset=`` with ``arch=`` is rejected like
    # the existing ``arch=`` vs individual-kwarg gate.
    preset: Optional[str] = None,
    softmax_variant: Optional[str] = None,
    normalization: Optional[str] = None,
    ffn_variant: Optional[str] = None,
    per_head_qk_norm: Optional[str] = None,
    ffn_routing: Optional[str] = None,
    extra_residual_dims: Optional[Mapping[str, int]] = None,
):
    """Compile and bake a Neural VM via the hybrid dynamic-layer scheduler.

    Signature mirrors ``compile_full_vm`` exactly (same args/kwargs, same
    return type ``(model, layout)``). As of Phase 8.G.3 this is the only
    implementation of ``compile_full_vm`` — that entry point is a thin
    unconditional redirect here after the static phase-pruning body was
    deleted.

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
    ``compile_full_vm`` (which is now a thin redirect to this function
    after the Phase 8.G.3 static-body deletion). The historical
    byte-identity gate (``compare_compile_paths``) was removed alongside
    the static body; the surviving fast scheduler invariants in
    ``tests/test_compile_dynamic_byte_identical.py`` continue to gate
    dep-graph regressions.

    Args mirror ``compile_full_vm`` -- see that function's docstring for
    detailed semantics. The disk-cache key is namespaced by appending
    ``"__dynamic"`` to the kwargs snapshot so dynamic and static
    compiles cannot trample each other's cache entries.

    Strict mode (B14, ON by default as of Phase 7.A.5 default-flip)
    ----------------------------------------------------------------
    When ``strict=True`` (the default since Phase 7.A.5), the dynamic
    compile refuses to fall back to ``phase`` for any ordering decision.
    Before compiling, every op is categorised against the unpruned
    declared-dep graph (mirroring ``tools/analyze_scheduler.py``); if any
    op falls into ``dep_graph_cycle_member`` (and ``allow_sealed_cycles``
    is False), ``phase_required_but_undeclared``, or
    ``phase_inconsistent_with_deps``, the call raises
    ``StrictModeUnschedulableError`` with the offending op names.

    On a clean op set strict mode produces the same byte-identical
    layout as the static path, because on a fully-declared op set the
    dep-derived order and the phase-derived order agree (Phase A
    finding, see ``DYNAMIC_SCHEDULER_MIGRATION_PLAN.md``).

    Cycle-aware admission (``allow_sealed_cycles=True``, also the
    default) accepts the OUTPUT_HI / IF_VAR SCC as a sealed group: the
    hybrid scheduler still falls back to phase ordering INSIDE the SCC,
    but every op OUTSIDE the SCC must be cleanly placeable from declared
    deps alone. Today's production op set has ~92 cycle members and 0
    non-cycle ``phase_required_but_undeclared`` /
    ``phase_inconsistent_with_deps`` ops, so the default
    ``strict=True, allow_sealed_cycles=True`` admits the compile and
    produces a byte-identical layout to the prior strict-off path.

    Passing ``strict=False`` restores the pre-Phase-7.A.5 behaviour
    (no admission gate). Passing ``allow_sealed_cycles=False`` restores
    the legacy "any cycle is a failure" behaviour, which today fails on
    the production op set until B9 (dim decomposition) completes.

    Target-shape overrides (shape-only rebuild)
    --------------------------------------------
    ``target_shape_overrides`` (when not ``None``) is a
    ``ModelShapeConstraint`` whose pinned fields drive a POST-COMPILE
    rebuild of the returned VM into a fresh ``AutoregressiveVM`` with
    the target ``d_model`` / ``n_layers`` / ``n_heads`` / ``head_dim``
    / ``intermediate_size`` / ``vocab_size``. The rebuilt model has
    zero-initialized weights — it is NOT semantically equivalent to the
    compiled VM. This is the minimum viable path that lets the HF
    state-dict export adapters (Mixtral, Llama) be wired end-to-end
    without rewriting the allocator stack to natively emit
    Mixtral-shaped weights. See ``_rebuild_to_target_shape`` for the
    documented gap (the allocator-native path is tracked separately).

    ``model_shape_constraint`` (validation-only) and
    ``target_shape_overrides`` (rebuild) compose: when both are set,
    the rebuild runs first, then the constraint validates the rebuilt
    model. Use ``target_shape_overrides`` alone for "shape-match HF
    export"; use ``model_shape_constraint`` alone to assert the
    naturally-allocated shape matches an external envelope.
    """
    # Mirror static-path env-flag handling to keep the API truly identical.
    if not declarations_only:
        declarations_only = _static._env_flag_enabled(
            _static._DECLARATIONS_ONLY_BAKE_ENV
        )
    # Phase 7.F.2: default to OFF (preserves byte-identity with the
    # historical baseline). Accept either a ``KVEvictionPolicy`` member
    # or its string value; the static path is symmetric.
    if kv_eviction_policy is None:
        kv_eviction_policy = KVEvictionPolicy.OFF
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

    # Model-semantics umbrella (see
    # ``docs/MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md``). Resolves the
    # six per-axis flags + the ``preset=`` shortcut into the existing
    # ``positional_encoding`` / ``attention_normalization`` / ``use_rms_norm``
    # / ``enable_moe_routing`` kwargs that the bake pipeline already
    # consumes. The expansion is purely additive: when every umbrella flag
    # is ``None`` (the default for existing callers) this block is a no-op
    # and the downstream bake is byte-identical to the historical path.
    (
        positional_encoding,
        attention_normalization,
        use_rms_norm,
        enable_moe_routing,
    ) = _resolve_semantics_flags(
        preset=preset,
        positional_encoding=positional_encoding,
        softmax_variant=softmax_variant,
        attention_normalization=attention_normalization,
        normalization=normalization,
        use_rms_norm=use_rms_norm,
        ffn_variant=ffn_variant,
        per_head_qk_norm=per_head_qk_norm,
        ffn_routing=ffn_routing,
        enable_moe_routing=enable_moe_routing,
        arch=arch,
    )

    # ------------------------------------------------------------------
    # Auto-widen: extra residual bands requested by an op / caller.
    # ------------------------------------------------------------------
    # ``extra_residual_dims`` maps ``name -> size`` for fresh residual
    # bands that should be appended past the natural d_model. Declaring
    # them grows d_model automatically (the layout's d_model is the
    # highest dim end), and the head-dim-preserving alignment in
    # ``_bake_from_scheduled_ops`` then rounds d_model up to a multiple of
    # the base head_dim and ADDS heads — so an op never has to hardcode
    # d_model to claim a fresh band. The dims are bump-pointer allocated
    # at the tail; the op's rules reference them by name through
    # ``layout.dim_positions``. This is the API for AX ``H1_PREV_STEP``,
    # MUL ``MUL_RESULT_HI``, and any future over-width family.
    if extra_residual_dims:
        for _name, _size in extra_residual_dims.items():
            if not isinstance(_name, str) or not _name:
                raise ValueError(
                    f"extra_residual_dims: name must be a non-empty str "
                    f"(got {_name!r})"
                )
            if not isinstance(_size, int) or _size <= 0:
                raise ValueError(
                    f"extra_residual_dims[{_name!r}]: size must be a "
                    f"positive int (got {_size!r})"
                )

    # ------------------------------------------------------------------
    # Auto-collect op-local residual bands (replaces the central dict).
    # ------------------------------------------------------------------
    # Band-adding ops no longer hand-edit a shared dict here. Each op declares
    # the over-width residual band(s) it needs LOCALLY, next to the op that
    # reads/writes them, via ``register_residual_band`` in
    # ``ops/residual_band_registry.py`` (populated at op-module import time —
    # ``all_core_ops`` wildcard-imports every ``lN_ops`` module). The registry
    # is auto-collected here: ``collect_registered_residual_bands`` evaluates
    # each band's flag predicate FRESH (so flag-off builds omit the band and
    # stay byte-identical) and returns the union in registration order.
    #
    # The current production registry holds: the AX byte-1 carry bands
    # (``H1_PREV_STEP`` / ``H1_DUMP_OUT`` / ``AX_CARRY_OVERFLOW``, l11_ops,
    # always present), the Root 2 STACK0 byte-0 carry bands (``STACK0_B0_*``,
    # l11_ops, always present), and the flag-gated width=2 MUL result band
    # (``MUL_RESULT_HI_LO/HI``, alu_ops, ``C4_MUL_WIDTH2``-gated). Every band
    # routes through the SINGLE head-dim-preserving auto-widen in
    # ``_bake_from_scheduled_ops`` (``base_head_dim`` captured from the BASE
    # layout BEFORE any extra band is appended, so the widen rounds d_model up
    # to a multiple of the BASE head_dim and ADDS heads instead of
    # repartitioning every existing head — declaring them via
    # ``declare_setdim_compat_dims`` would re-derive head_dim from the widened
    # width and scramble attention content -> regresses test_bnz_branch). The
    # collected set flows into the disk/in-proc cache key below (which hashes
    # ``extra_residual_dims`` but NOT ``C4_MUL_WIDTH2`` / ``C4_AX_BYTE1_DUMP``
    # directly), so flag-on and flag-off builds never share a serialised entry.
    #
    # The COMBINED widen (14 AX/Root2-flag dims + 28 Root2 PREV/DUMP dims always
    # + 32 MUL dims when enabled) rounds head-dim-preservingly (base head_dim
    # 109): 872 -> 981 (n_heads 8 -> 9). All existing dims are
    # byte-behaviour-identical (every new band is zero on every row the prior
    # model touched). Threaded by NAME via ``layout.dim_positions`` -- they must
    # NOT be op-declared ``declare_dim``s. To add a new band-adding op, call
    # ``register_residual_band`` at that op's module top — never touch this file.
    # See docs/RESIDUAL_BAND_REGISTRY_2026_06_13.md.
    #
    # AX byte-1 VALUE-GENERALISATION (byte-1 0..15): the carry now ALSO copies
    # the H2/H3 marker-distance one-hots (``H2/H3_PREV_STEP`` ->
    # ``H2/H3_DUMP_OUT``, +28 dims) because the fresh-step byte-1 emission
    # one-hot is SPREAD across H1 (v 0..4) / H2 (v 5..11) / H3 (v 12..15). Those
    # four bands are now ALSO declared op-locally via ``register_residual_band``
    # in ``ops/l11_ops.py`` (next to the H1 pair), so they auto-collect here.
    from .ops.residual_band_registry import (
        collect_registered_residual_bands,
        collect_never_share_band_names,
        collect_alibi_base_residual_bands,
    )
    _merged_extra = collect_registered_residual_bands()
    # The ALiBi-slope BASE band set: bands active with every C4_* flag at its
    # DEFAULT. Used to derive the slope base head count so a non-default
    # over-width flag (e.g. C4_AX_BYTE1_FULL_WIDTH) does not shift existing
    # heads' ALiBi slopes (see collect_alibi_base_residual_bands +
    # _bake_from_scheduled_ops base_n_heads). Caller/env extra_residual_dims
    # are folded in below so an explicitly-requested band is part of the base.
    _alibi_base_extra = collect_alibi_base_residual_bands()
    # Escape hatch: ``C4_DISABLE_AX_CARRY_BANDS=1`` drops the auto-collected
    # production bands (A/B diagnostics only — the carry ops then reference
    # undeclared dims, so this is for layout/geometry comparison, not a runnable
    # build). The caller-passed / env ``extra_residual_dims`` below is preserved.
    if os.environ.get("C4_DISABLE_AX_CARRY_BANDS") == "1":
        _merged_extra = {}
        _alibi_base_extra = {}
    if extra_residual_dims:
        _merged_extra.update(extra_residual_dims)
        # An explicitly-requested band is part of the geometry the existing
        # heads are authored against, so it belongs in the slope base too.
        _alibi_base_extra.update(extra_residual_dims)
    extra_residual_dims = _merged_extra or None
    # Carry/dump bands that must keep a private dim-liveness slot — threaded
    # into the compiler's never-share set inside ``_bake_from_scheduled_ops``
    # (no class-level ``_LIVENESS_NEVER_SHARE_NAMES`` hand-edit needed).
    _never_share_band_names = collect_never_share_band_names()

    from ..config import get_config
    vm_config = get_config()

    # V2 vision (Phase 8.X): a single ``arch=ModelArchitectureSpec(...)`` may
    # be passed in lieu of the individual architectural kwargs
    # (``positional_encoding=``, ``attention_normalization=``,
    # ``rope_base=``, ``use_rms_norm=``, ``rms_norm_eps=``). Mixing the two
    # surfaces is rejected with an explicit error rather than silently
    # privileging one — the migration story is "pick one path per call site".
    # The individual kwargs remain the back-compat surface; new callers
    # should prefer ``arch=`` (which composes per-layer overrides cleanly
    # via :class:`LayerSpec`).
    if arch is not None:
        _explicit_arch_kwargs = {
            name: value
            for name, value in (
                ("positional_encoding", positional_encoding),
                ("attention_normalization", attention_normalization),
                ("rope_base", rope_base),
                ("use_rms_norm", use_rms_norm),
                ("rms_norm_eps", rms_norm_eps),
            )
            if value is not None
        }
        if _explicit_arch_kwargs:
            raise TypeError(
                "compile_full_vm_dynamic(arch=...) is mutually exclusive "
                "with the individual architectural kwargs "
                f"{sorted(_explicit_arch_kwargs)}. Pass either an "
                "``arch=ModelArchitectureSpec(...)`` instance OR the "
                "individual ``positional_encoding=``/"
                "``attention_normalization=``/``rope_base=``/"
                "``use_rms_norm=``/``rms_norm_eps=`` kwargs, not both."
            )
        # Project the spec back onto the legacy 5-kwarg surface that the
        # downstream ``_bake_from_scheduled_ops`` / ``_rebuild_to_target_shape``
        # APIs still consume. This keeps the runtime path byte-identical
        # to the historical kwarg path — a spec is just a typed name for
        # the same five values.
        positional_encoding = arch.positional_encoding.kind
        attention_normalization = arch.attention_activation.softmax_kind
        rope_base = float(arch.positional_encoding.rope_base)
        use_rms_norm = arch.norm_pre_attention.kind == "rmsnorm"
        rms_norm_eps = float(arch.norm_pre_attention.eps)
    else:
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

    # In-process memo short-circuit. Mirrors the kwargs snapshot built inside
    # ``_bake_from_scheduled_ops`` at the disk-cache lookup so the two layers
    # invalidate together. Skips the ~1.3 s ``_collect_ops_for_compile`` pass
    # and the ~3 s ``torch.load`` deserialisation on every subsequent call in
    # the same process. Cross-process callers (fresh pytest invocations) still
    # fall through to the disk cache. Disabled when ``disk_cache=False``
    # (test paths that explicitly want a fresh compile) and when
    # ``require_declarative_bake`` / ``declarations_only`` / a
    # ``target_shape_overrides`` rebuild is requested (those paths post-process
    # the model and would alias if we handed back a memoised reference).
    _inproc_snapshot = None
    if (
        disk_cache
        and not require_declarative_bake
        and not declarations_only
        and target_shape_overrides is None
        and model_shape_constraint is None
        and not d_model_packing
    ):
        _inproc_snapshot = _build_cache_key_snapshot(
            S=S,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            enable_neural_io_think_protocol=enable_neural_io_think_protocol,
            alu_mode=alu_mode,
            n_heads=n_heads,
            ffn_hidden=ffn_hidden,
            max_seq_len=max_seq_len,
            pin_io_only=pin_io_only,
            enable_moe_routing=enable_moe_routing,
            positional_encoding=positional_encoding,
            attention_normalization=attention_normalization,
            rope_base=rope_base,
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            require_declarative_bake=require_declarative_bake,
            declarations_only=declarations_only,
            kv_eviction_policy=kv_eviction_policy,
            kv_eviction_n_steps=kv_eviction_n_steps,
            extra_residual_dims=extra_residual_dims,
        )
        _memo_key = _inproc_cache_key(_inproc_snapshot)
        _memo_hit = _INPROC_COMPILE_CACHE.get(_memo_key)
        if _memo_hit is not None:
            _cached_model, _cached_layout = _memo_hit
            # Re-attach KV eviction state mirrors the disk-cache hit path in
            # ``_bake_from_scheduled_ops``; the cached model may have been
            # built in this process with a different policy/step count.
            _static._attach_kv_eviction_state(
                _cached_model,
                _cached_layout,
                kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
                n_steps=int(kv_eviction_n_steps),
            )
            return _cached_model, _cached_layout

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
    # ``_topological_sort`` uses ``ops.index(o)`` for Kahn's stability
    # and any reordering would feed a different stability key in.
    # Instead the schedule is computed alongside, and the compile pipeline
    # runs over the natural ``_collect_ops`` order. On today's op set the
    # two coincide on every non-cycle op and the cycle members fall
    # through to phase-only ordering. This decoupling is what made B11 a
    # byte-identical parallel path rather than a behavioural change
    # (Phase 8.G.3 cut the static fallback that gated this claim, leaving
    # the dynamic path as the single bake entry point).
    _scheduled, _source = compute_dynamic_schedule(ops)

    # Step-1 safety: scan for cross-step reads (``X.*.-1``) whose base dim
    # ALSO has a same-step writer in the scheduled op set. On VM step 1
    # there is no previous step, so the cross-step alias resolves to 0;
    # when a same-step writer exists, the bake author very often meant to
    # consume that fresh write instead. Emitted via ``warnings.warn`` with
    # the ``CrossStepReadWarning`` category so callers can filter or
    # promote to errors via the stdlib ``warnings`` filter mechanism. The
    # canonical motivating case is the OPCODE_BYTE_LO read in
    # ``opcode_decode_ffn`` (see ``ops/l5_ops.py``) whose same-step writer
    # is ``layer5_fetch`` — on step 1 the decoder reads 0.
    #
    # Step 4 (``IR_INCREMENTAL_IMPROVEMENTS.md``): when ``strict=True``,
    # the warning becomes a HARD ERROR (``CrossStepReadError``). The
    # caller can pass ``cross_step_baseline_allowlist`` to whitelist the
    # historical findings while migrating them off cross-step reads; the
    # bundled :data:`CROSS_STEP_BASELINE_ALLOWLIST` (82 entries at Step-4
    # landing) is used when no explicit allowlist is supplied. Passing
    # ``cross_step_baseline_allowlist=[]`` promotes every finding to an
    # error (useful for new bake authors who want zero cross-step
    # zero-prop risk).
    _emit_cross_step_safety_warnings(
        _scheduled,
        strict_error=strict,
        allowlist=cross_step_baseline_allowlist,
    )

    # Build the model via the unchanged static pipeline with the natural
    # op order. The static compile_full_vm wraps op collection inline,
    # so we re-implement just the body here so the dynamic path is
    # actually a parallel API (not a wrapper around the static call).
    model, layout = _bake_from_scheduled_ops(
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
        kv_eviction_policy=kv_eviction_policy,
        kv_eviction_n_steps=kv_eviction_n_steps,
        d_model_packing=d_model_packing,
        d_model_packing_target=d_model_packing_target,
        extra_residual_dims=extra_residual_dims,
        never_share_band_names=_never_share_band_names,
        alibi_base_extra_dims=(_alibi_base_extra or None),
    )

    # Post-compile shape rebuild. When the caller hands in
    # ``target_shape_overrides`` (a ``ModelShapeConstraint``), REPLACE the
    # compiled VM with a freshly-initialized ``AutoregressiveVM`` whose
    # geometry matches the overrides. This is a SHAPE-ONLY rebuild — the
    # baked VM weights are NOT copied into the new model, so semantic
    # equivalence is sacrificed for shape compatibility. Use case:
    # producing a target-architecture-shaped VM for HF state-dict export
    # (Mixtral, Llama) without changing the VM's allocator-derived
    # natural shape.
    if target_shape_overrides is not None:
        from ..verification.model_shape_constraint import ModelShapeConstraint as _MSC
        if not isinstance(target_shape_overrides, _MSC):
            raise TypeError(
                "target_shape_overrides must be a ModelShapeConstraint instance, "
                f"got {type(target_shape_overrides).__name__}"
            )
        model = _rebuild_to_target_shape(
            model,
            target_shape_overrides,
            max_seq_len=max_seq_len,
            positional_encoding=positional_encoding,
            attention_normalization=attention_normalization,
            rope_base=rope_base,
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
        )

    # Post-compile shape-constraint check. When the caller hands in a
    # ``ModelShapeConstraint``, diff it against the compiled model and
    # raise ``ModelShapeMismatchError`` on any mismatch. The check runs
    # AFTER the bake so it sees the real ``num_heads`` / ``head_dim``
    # the allocator emitted (Phase 8.O.1/8.O.2 dynamic heads + GQA), not
    # just the caller's intent.
    if model_shape_constraint is not None:
        from ..verification.model_shape_constraint import (
            ModelShapeConstraint as _MSC,
            ModelShapeMismatchError,
            validate_against_shape,
        )
        if not isinstance(model_shape_constraint, _MSC):
            raise TypeError(
                "model_shape_constraint must be a ModelShapeConstraint instance, "
                f"got {type(model_shape_constraint).__name__}"
            )
        mismatches = validate_against_shape(model, model_shape_constraint)
        if mismatches:
            raise ModelShapeMismatchError(
                mismatches, target=model_shape_constraint.target
            )

    # Populate the in-process memo so a subsequent call with the same kwargs
    # in this process skips ``_collect_ops_for_compile`` + ``torch.load``.
    # Only populated when the early-cache snapshot path was taken (the post-
    # compile rebuild / packing branches are explicitly excluded above).
    if _inproc_snapshot is not None:
        _INPROC_COMPILE_CACHE[_inproc_cache_key(_inproc_snapshot)] = (model, layout)

    return model, layout


# ---------------------------------------------------------------------------
# Internals: target-shape rebuild
# ---------------------------------------------------------------------------


def _rebuild_to_target_shape(
    compiled_model,
    overrides,
    *,
    max_seq_len: int,
    positional_encoding: str,
    attention_normalization: str,
    rope_base: float,
    use_rms_norm: bool,
    rms_norm_eps: float,
):
    """Build a fresh ``AutoregressiveVM`` with the override geometry.

    SHAPE-ONLY rebuild — returns a new model with target
    ``d_model`` / ``n_layers`` / ``n_heads`` / ``head_dim`` /
    ``ffn_hidden`` / ``vocab_size`` taken from ``overrides``, falling
    back to the compiled model's values when a field is ``None``. The
    new model's weights are nn.Parameter zeros (PureFFN's default init),
    so it is NOT semantically equivalent to the compiled VM. The intent
    is to produce a model whose ``state_dict`` shape matches a target
    HF architecture (Mixtral, Llama) for export adapters.

    Gap documented loudly: a real "compile-time shape override" would
    have the allocator emit a baked, semantics-preserving model with
    the target geometry (padding dims up, rebuilding attention with
    GQA, padding FFN widths uniformly, padding vocab). That requires
    reworking the dim allocator, head allocator, and FFN allocator to
    accept a target envelope and route excess capacity to padding
    rather than to functional ops. This rebuild is the minimum viable
    path that lets the HF export adapters be wired end-to-end without
    that allocator rework.

    Override fields:
      * ``d_model`` — total residual width (must equal num_heads * head_dim)
      * ``num_hidden_layers`` — n_layers
      * ``num_attention_heads`` — n_heads
      * ``head_dim`` — per-head width (overrides d_model // n_heads)
      * ``intermediate_size`` — FFN hidden_dim (applied uniformly)
      * ``vocab_size`` — embedding + lm_head vocab
      * ``num_key_value_heads`` — not honored yet (MHA only; raises if
        set != num_attention_heads). GQA wiring through the runtime
        attention module is a separate item.
    """
    from ..vm_step import AutoregressiveVM

    d_model = overrides.d_model if overrides.d_model is not None else int(compiled_model.d_model)
    n_layers = (
        overrides.num_hidden_layers
        if overrides.num_hidden_layers is not None
        else len(compiled_model.blocks)
    )
    if overrides.num_attention_heads is not None:
        n_heads = int(overrides.num_attention_heads)
    else:
        n_heads = int(getattr(compiled_model.blocks[0].attn, "num_heads", 8))
    head_dim = overrides.head_dim
    if head_dim is not None:
        if int(head_dim) * n_heads != d_model:
            raise ValueError(
                f"target_shape_overrides: head_dim={head_dim} * "
                f"num_attention_heads={n_heads} = {int(head_dim) * n_heads}, "
                f"but d_model={d_model}. Mixtral requires num_attention_heads * "
                "head_dim == hidden_size."
            )
    if overrides.num_key_value_heads is not None:
        if int(overrides.num_key_value_heads) != n_heads:
            raise NotImplementedError(
                f"target_shape_overrides: num_key_value_heads="
                f"{overrides.num_key_value_heads} != num_attention_heads="
                f"{n_heads}. GQA rebuild is not yet implemented; pass MHA "
                "(num_key_value_heads == num_attention_heads) for now."
            )
    if overrides.intermediate_size is not None:
        ffn_hidden = int(overrides.intermediate_size)
    else:
        widths = [
            int(getattr(b.ffn, "hidden_dim", 0)) for b in compiled_model.blocks
        ]
        # Step 3 (literal-fallback lint, audit 2026-06-03): historically
        # the empty-widths branch substituted a literal ``4096`` —
        # which silently seeded the rebuild with a stale shape any time
        # ``compiled_model.blocks`` was empty (synthetic / partial
        # rebuild fixtures). Demand an explicit override instead of
        # papering over the missing topology.
        if not widths:
            raise ValueError(
                "rebuild target: cannot derive ffn_hidden because "
                "compiled_model.blocks is empty. Pass "
                "target_shape_overrides.intermediate_size explicitly. "
                "Bare-literal fallback (4096) removed by Step 3 (see "
                "docs/LITERAL_FALLBACK_AUDIT.md)."
            )
        ffn_hidden = max(widths)
    vocab_size = (
        int(overrides.vocab_size)
        if overrides.vocab_size is not None
        else int(getattr(compiled_model, "vocab_size", 256))
    )

    ffn_widths: Dict[int, int] = {}
    for layer_idx, ov in overrides.per_layer_overrides.items():
        if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(
                f"target_shape_overrides.per_layer_overrides has layer_idx="
                f"{layer_idx}, but rebuild has {n_layers} layers."
            )
        if (
            ov.get("num_attention_heads") is not None
            or ov.get("num_key_value_heads") is not None
            or ov.get("head_dim") is not None
        ):
            raise NotImplementedError(
                "target_shape_overrides: per-layer head/head_dim overrides "
                "are not supported by the rebuild path (AutoregressiveVM "
                "uses one n_heads / head_dim across all blocks)."
            )
        if ov.get("intermediate_size") is not None:
            ffn_widths[layer_idx] = int(ov["intermediate_size"])

    rebuilt = AutoregressiveVM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_widths if ffn_widths else ffn_hidden,
        max_seq_len=max_seq_len,
        positional_encoding=positional_encoding,
        attention_normalization=attention_normalization,
        rope_base=rope_base,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
    )
    return rebuilt


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
        # Campaign loaded-operand ADD ALU cell-15 contaminant clear
        # CBC Phase 1: DERIVED clean-one-hot operand delivery. Snaps
        # ALU_LO/HI + AX_CARRY_LO/HI to a clean per-nibble one-hot. Two flags,
        # both gated on this campaign so flag-OFF is golden byte-identical:
        #   * C4_CLEAN_OPERAND      — feasibility, cleans ALL binary-op + cmp rows
        #   * C4_CLEAN_OPERAND_ADD  — CBC pass-gain (DEFAULT-ON), cleans ONLY the
        #     arithmetic (ADD/SUB/MUL/DIV/MOD) rows, leaving the CMP calibration
        #     intact.
        # This SUBSUMES the former ADD-only address-leak corrector
        # (``LoadedOperandAddHi15ClearFFN``): the clean-snap zeros every non-argmax
        # ALU_HI cell on the ADD/SUB/MUL/DIV/MOD MARK_AX rows BEFORE any hi-nibble
        # clear could run, so the old cell-13/15 (and all-16) clears were provably
        # inert once ``C4_CLEAN_OPERAND_ADD`` went DEFAULT-ON (proven: forward
        # max_abs_diff=0.0 on ADD rows; fast-gate byte-identical corpus verdicts).
        # Must come AFTER efficient_l8_addsub_wrap (whose d_model read sees the raw
        # L8 PureFFN before this wrap). See shared.clean_operand_enabled /
        # shared.clean_operand_add_enabled.
        ops.append(_static.make_clean_operand_op())
        # Campaign func_max/func_min CMP loaded-operand-A two-hot clean (task
        # #428). Wraps the L9 block.ffn (CMP nibble-comparator factory) to clean
        # the SE_ALU operand-A band on the MARK_SE_ONLY cmp row before the L9 CMP
        # rules read it. Must come AFTER layer9_alu (whose FFN it wraps).
        # Explicit-opt-in campaign flag; flag-OFF leaves block.ffn untouched
        # (golden byte-identical). See shared.func_cmp_operand_clean_enabled.
        ops.append(_static.make_cmp_loaded_operand_clean_op())
        ops.append(_static.make_efficient_l10_andorxor_wrap_op(alu_mode=alu_mode))
        ops.append(_static.make_efficient_l11_alumul_wrap_op(alu_mode=alu_mode))

    for op in _static.make_alu_divmod_composite_ops(alu_mode=alu_mode):
        ops.append(op)
    # Bug #36 declarative wrapper: no-op bake, consolidates the
    # FlattenedDivMod composite's reads/writes for dim_contracts_audit.
    ops.append(_static.make_layer10_divmod_op())

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
    kv_eviction_policy: KVEvictionPolicy = KVEvictionPolicy.OFF,
    kv_eviction_n_steps: int = 64,
    d_model_packing: bool = False,
    d_model_packing_target: Optional[int] = None,
    extra_residual_dims: Optional[Mapping[str, int]] = None,
    never_share_band_names: Optional[Set[str]] = None,
    alibi_base_extra_dims: Optional[Mapping[str, int]] = None,
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

    kwargs_snapshot = _build_cache_key_snapshot(
        S=S,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
        alu_mode=alu_mode,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
        pin_io_only=pin_io_only,
        enable_moe_routing=enable_moe_routing,
        positional_encoding=positional_encoding,
        attention_normalization=attention_normalization,
        rope_base=rope_base,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
        require_declarative_bake=require_declarative_bake,
        declarations_only=declarations_only,
        kv_eviction_policy=kv_eviction_policy,
        kv_eviction_n_steps=kv_eviction_n_steps,
        extra_residual_dims=extra_residual_dims,
    )
    cache_path = None
    cache_key = (
        _static._cache_key(kwargs_snapshot) if disk_cache else None
    )
    if disk_cache and not require_declarative_bake and not declarations_only:
        cache_path = _static._cache_dir() / f"{cache_key}.pt"
        cached = _static._try_load_cached(cache_path, kwargs_snapshot)
        if cached is not None:
            # Phase 7.F.2: defensive re-attach of KV eviction state on cache
            # hit, mirroring the static-path behaviour.
            cached_model, cached_layout = cached
            _static._attach_kv_eviction_state(
                cached_model,
                cached_layout,
                kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
                n_steps=int(kv_eviction_n_steps),
            )
            return cached_model, cached_layout

    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=pin_io_only)
    # Forward-declare the extra residual band NAMES so the ops added below
    # validate even though the bands are positioned AFTER the loop (at the
    # tail, by the auto-widen path). Their tail placement must be computed
    # from the FULL op-dim layout (NOT the compat-dims-only one), so the
    # actual ``declare_dim`` happens post-loop. ``extra_residual_dims`` here
    # carries BOTH band families: the AX byte-1 carry bands
    # (``H1_PREV_STEP`` / ``H1_DUMP_OUT``, always present) and — when
    # ``C4_MUL_WIDTH2`` is on — the MUL byte-1 result bands
    # (``MUL_RESULT_HI_LO`` / ``MUL_RESULT_HI_HI``), read by the L13
    # ``layer13_mul_result_hi_relay``. Both are resolved post-loop.
    if extra_residual_dims:
        compiler.pending_extra_dims.update(extra_residual_dims.keys())
    # Thread the op-local registry's ``never_share`` band names into this
    # compile's dim-liveness allocator so carry/dump bands keep PRIVATE slots
    # (replaces the hand-listed class-level ``_LIVENESS_NEVER_SHARE_NAMES``
    # entries). See ``ops/residual_band_registry.py``.
    if never_share_band_names:
        compiler.add_never_share_names(never_share_band_names)
    for op in scheduled:
        compiler.add_op(op)

    # ------------------------------------------------------------------
    # Establish the canonical (base) ``head_dim``.
    # ------------------------------------------------------------------
    # The natural layout d_model (e.g. 869) is first padded up to a
    # multiple of ``n_heads`` — exactly the legacy alignment — to obtain
    # the canonical production geometry (869 -> 872, head_dim = 872 / 8 =
    # 109). ``base_head_dim`` is captured from THAT padded width because
    # it is the load-bearing invariant the ops author against (every head
    # spec writes rows at ``head_idx * head_dim``). The auto-widen below
    # preserves THIS head_dim; it must never be re-derived from a width
    # that already includes the extra bands. The extra bands (AX + MUL) are
    # NOT yet declared at this point — they're only forward-declared as names
    # in ``compiler.pending_extra_dims`` — so this layout is the clean BASE
    # layout (no extra-band width folded in). The head-dim-preserving widen
    # below therefore rounds up to a multiple of the BASE head_dim and ADDS
    # heads instead of repartitioning existing heads.
    layout = compiler.compile()
    if layout.d_model % n_heads != 0:
        pad = n_heads - (layout.d_model % n_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()
    base_head_dim = layout.d_model // n_heads
    if base_head_dim <= 0:
        base_head_dim = layout.d_model  # degenerate single-head fallback

    # ------------------------------------------------------------------
    # ALiBi-slope base head count (over-width-band-invariant).
    # ------------------------------------------------------------------
    # The auto-widen below rounds the widened d_model up to a multiple of
    # ``base_head_dim`` and DERIVES a larger ``n_heads`` (it ADDS trailing
    # heads). The ALiBi-slope formula ``2**(-8/N * (i+1))`` must keep ``N``
    # PINNED to the head count of the build WITHOUT the flag-gated over-width
    # bands, otherwise flipping such a band on (e.g.
    # ``C4_AX_BYTE1_FULL_WIDTH`` adds the 256-cell ``AX_BYTE1_FULL_WIDE`` band
    # -> n_heads 10 -> 13) would shift EVERY existing head's slope (head 0
    # 0.574 -> 0.653) and silently perturb every globally-sized attention
    # block (the mul_overflow / shl_8bit tail regression). The slope base is
    # the n_heads obtained by widening with only the ALWAYS-ON bands
    # (``alibi_base_extra_dims``; the flag-gated emission bands excluded), so
    # it equals the default build's n_heads on BOTH flag-off and flag-on
    # compiles. ``AutoregressiveVM`` receives it as ``alibi_base_heads``.
    # (L15's num_heads-keyed head layout is insulated separately by its own
    # ``l15_attention_resize`` op, which recomputes its slopes from a fixed
    # target_num_heads.) When ``alibi_base_extra_dims is None`` the base
    # equals the FULL widen -> byte-identical to the historical path.
    _base_widen_dims = (
        alibi_base_extra_dims
        if alibi_base_extra_dims is not None
        else extra_residual_dims
    )
    _base_extra_total = (
        sum(int(s) for s in _base_widen_dims.values())
        if _base_widen_dims else 0
    )
    # Round (base layout d_model + always-on extra width) up to a multiple of
    # base_head_dim (mirrors the widen alignment below) and derive n_heads.
    _base_widened = layout.d_model + _base_extra_total
    if _base_widened % base_head_dim != 0:
        _base_widened += base_head_dim - (_base_widened % base_head_dim)
    base_n_heads = _base_widened // base_head_dim

    # ------------------------------------------------------------------
    # Auto-widen: append caller/op-requested residual bands at the tail.
    # ------------------------------------------------------------------
    # Bump-pointer allocated (no pin) so they land past the highest existing
    # dim, growing d_model automatically — an op never has to hardcode d_model
    # to claim a fresh band. (Pinning them at an explicit tail position is
    # WRONG: it defeats the default-ON dim-liveness slot-sharing allocator,
    # which then refuses to compact and blows up d_model — e.g. efficient-mode
    # 920 -> 1635, relocating ALU_LO past the efficient-ALU's 512-wide W_proj.
    # The forward-declared names — both AX (``H1_PREV_STEP`` / ``H1_DUMP_OUT``)
    # and MUL (``MUL_RESULT_HI_*``) — are now resolved by this declare.)
    if extra_residual_dims:
        for _name, _size in extra_residual_dims.items():
            compiler.declare_dim(_name, int(_size))
        layout = compiler.compile()

        # ------------------------------------------------------------------
        # Head-dim-preserving alignment of the widened d_model.
        # ------------------------------------------------------------------
        # The attention reshape ``x.view(B, S, num_heads, head_dim)``
        # splits the residual stream into ``num_heads`` contiguous bands
        # of width ``head_dim = d_model // num_heads``. ``head_dim`` fixes
        # which residual dims belong to which head, so widening with a
        # FIXED ``num_heads`` (the legacy multiple-of-n_heads pad) changes
        # ``head_dim``, repartitions every head and scrambles all
        # attention content even though no new dim is read (confirmed: a
        # naive widen to 920 regresses test_lea_basic; residual diverges
        # by ~0.063 starting at block 1).
        #
        # Instead we round the widened d_model up to a multiple of the
        # BASE ``head_dim`` and derive ``num_heads = d_model // head_dim``.
        # A widen therefore ADDS heads (the new trailing band is all-zero
        # and contributes nothing) while every existing head keeps its
        # exact dim span and weights — byte-behaviour-identical on
        # existing dims (verified: logits diff == 0, smoke 49/2 including
        # bnz + lea at d_model 872 -> 981, n_heads 8 -> 9).
        if layout.d_model % base_head_dim != 0:
            widen_pad = base_head_dim - (layout.d_model % base_head_dim)
            compiler.declare_dim("_widen_pad", widen_pad)
            layout = compiler.compile()
        n_heads = layout.d_model // base_head_dim

    # Phase 10.A: optionally repack unpinned dims via best-fit decreasing
    # so the residual stream is tighter. Pinned IO dims (declared via
    # ``declare_setdim_compat_dims(pin_io_only=True)``) keep their
    # positions byte-identically; unpinned scratch dims relocate to
    # recover slack. The packed pool width is rounded up to ``n_heads``
    # so the attention head splits remain integer. An explicit
    # ``d_model_packing_target`` is honored exactly and must already be
    # a multiple of n_heads.
    if d_model_packing:
        from ..dim_allocator import pack_layout_dims

        target = d_model_packing_target
        if target is not None and target % n_heads != 0:
            raise ValueError(
                f"compile_full_vm_dynamic(d_model_packing_target="
                f"{target}) must be a multiple of n_heads={n_heads}"
            )
        new_positions, packed_d_model = pack_layout_dims(
            dim_positions=layout.dim_positions,
            dim_sizes=layout.dim_sizes,
            pinned_dims=getattr(compiler, "_pinned", {}) or {},
            alias_map=getattr(compiler, "_aliases", {}) or {},
            current_d_model=layout.d_model,
            target_d_model=target,
        )
        if target is None and packed_d_model % n_heads != 0:
            packed_d_model += n_heads - (packed_d_model % n_heads)
        layout = ModelLayout(
            d_model=packed_d_model,
            n_layers=layout.n_layers,
            ops_per_layer=layout.ops_per_layer,
            dim_positions=new_positions,
            dim_sizes=layout.dim_sizes,
            block_ops=layout.block_ops,
            model_ops=layout.model_ops,
            ffn_widths=layout.ffn_widths,
        )

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
                # Phase 7.F.2: defensive re-attach of KV eviction state on
                # cache hit (mirrors the static-path declarations_only branch).
                cached_model, cached_layout = cached
                _static._attach_kv_eviction_state(
                    cached_model,
                    cached_layout,
                    kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
                    n_steps=int(kv_eviction_n_steps),
                )
                return cached_model, cached_layout

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
        # Pin the ALiBi-slope base to the PRE-widen head count so the
        # auto-widen's trailing padding heads (added when an over-width band
        # crosses a head-dim multiple) do not shift existing heads' slopes.
        alibi_base_heads=base_n_heads,
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

    # Phase 7.F.2: attach per-attention KVEvictionState artifacts. OFF is
    # a no-op (preserves byte-identity with the historical baseline), so
    # the static and dynamic paths emit identical models when the policy
    # is unset. STATIC_LIVENESS runs the analyzer and projects the report
    # onto every block.attn — same code path as the static implementation.
    _static._attach_kv_eviction_state(
        model,
        layout,
        kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
        n_steps=int(kv_eviction_n_steps),
    )

    if cache_path is not None:
        _static._try_save_cached(cache_path, model, layout, kwargs_snapshot)

    return model, layout
