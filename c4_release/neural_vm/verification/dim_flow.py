"""Dim-flow analyzer: static enumeration of every op that reads / writes a
residual dim slot across the compiled layout.

This is the static-analysis tool the A3.X memory cluster chain has been
missing. For a given dim slot (e.g. ``STACK0_BYTE_VAL_1_LO``) it answers:

  * Who writes this dim? At what layer? With what gate conditions?
  * Who reads this dim? At what layer? In what role (Q / K / V / FFN gate
    input / FFN condition)?
  * Are the writer's effective gates compatible with the reader's
    effective gates?
  * Is anyone writing 0 to this dim (zeroing-writer hypothesis)?

The analyzer walks ``ModelLayout.ops_per_layer`` + ``block_ops`` +
``model_ops`` and inspects every op's ``compiler_ir`` (built via
``compiler_ir`` directly, or by invoking ``compiler_ir_factory`` with the
layout's ``dim_positions``). Both FFN rules (``FFNRule.writes`` /
``FFNRule.conditions`` / ``FFNRule.gate*``) and attention head specs
(``DeclarativeAttentionHeadSpec.q`` / ``.k`` / ``.v`` / ``.o``) are
covered.

Public API:

* :func:`enumerate_dim_writers` — every op that writes ``dim_name``,
  ordered by layer, with gate-condition summary.
* :func:`enumerate_dim_readers` — every op that reads ``dim_name``,
  ordered by layer, with read-role (FFN condition / FFN gate / attn-Q /
  attn-K / attn-V).
* :func:`trace_dim_flow` — producer-consumer chain (writers grouped,
  then readers grouped, in layer order).
* :func:`find_zeroing_writers` — writers whose net contribution is zero
  (the A3.5 "zeroing-writer" hypothesis surface).
* :func:`format_report` — human-readable report for CLI consumption.

Design notes:

* "Layer" here is the integer placement chosen by the dynamic compiler
  (``ModelLayout.ops_per_layer`` index), not the historical "layer N" in
  the op name. This is the correct number for runtime block ordering.
* Op-level declared ``reads`` / ``writes`` sets are NOT used: they
  routinely omit dims that the compiled IR actually touches (e.g.
  ``layer14_mem_generation`` does not list ``STACK0_BYTE_VAL_1_LO`` in
  ``reads`` even though the head spec wires it into V/O). The analyzer
  walks the IR directly.
* Compiler-ir-factory ops have their IR built lazily once via
  ``compiler_ir_factory(dim_positions, head_dim)``. The default
  ``head_dim=64`` matches the standard transformer block.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..unified_compiler.ir import CompilerIR, FFNOp, FFNRule


__all__ = [
    "DimWriter",
    "DimReader",
    "DimFlow",
    "enumerate_dim_writers",
    "enumerate_dim_readers",
    "trace_dim_flow",
    "find_zeroing_writers",
    "format_report",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimWriter:
    """One op that writes the dim. ``write_value_expr`` is a short string
    summarizing the write (e.g. ``"+3.0"``, ``"V slot 1 * O slot 1"``).

    ``gate_conditions`` is a list of human-readable condition strings —
    for FFN rules that's the rule's gate inputs; for attn heads it's the
    Q-side dims with positive weight.
    """

    layer_idx: int
    op_name: str
    op_kind: str           # "ffn" | "attn"
    rule_name: Optional[str]
    write_value_expr: str
    gate_conditions: Tuple[str, ...]
    offset: int = 0


@dataclass(frozen=True)
class DimReader:
    """One op that reads the dim. ``read_role`` is the semantic role:

    * ``ffn_condition`` — FFNRule.conditions input
    * ``ffn_gate`` — FFNRule.gate / gate_terms input
    * ``attn_Q`` — DeclarativeAttentionHeadSpec.q
    * ``attn_K`` — DeclarativeAttentionHeadSpec.k
    * ``attn_V`` — DeclarativeAttentionHeadSpec.v
    """

    layer_idx: int
    op_name: str
    op_kind: str           # "ffn" | "attn"
    rule_name: Optional[str]
    read_role: str
    gate_conditions: Tuple[str, ...]
    weight: float
    offset: int = 0


@dataclass
class DimFlow:
    """Producer-consumer chain for a single dim slot."""

    dim_name: str
    dim_position: Optional[int]
    writers: List[DimWriter] = field(default_factory=list)
    readers: List[DimReader] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------


def _coerce_layout(compiler_or_layout):
    """Accept either a LayerCompiler-like object with ``.compile()`` /
    ``layout`` attribute, or a ModelLayout directly. Returns the
    ``ModelLayout``.
    """

    if compiler_or_layout is None:
        raise ValueError("dim_flow: compiler_or_layout is None")
    # Already a layout (ops_per_layer attribute is the marker).
    if hasattr(compiler_or_layout, "ops_per_layer") and hasattr(
        compiler_or_layout, "dim_positions"
    ):
        return compiler_or_layout
    if hasattr(compiler_or_layout, "layout") and hasattr(
        compiler_or_layout.layout, "ops_per_layer"
    ):
        return compiler_or_layout.layout
    if hasattr(compiler_or_layout, "compile"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return compiler_or_layout.compile()
    raise TypeError(
        f"dim_flow: cannot coerce {type(compiler_or_layout).__name__!r} "
        "to a ModelLayout (need .ops_per_layer + .dim_positions, or "
        "a LayerCompiler with .compile())"
    )


def _walk_ops_with_layers(layout) -> List[Tuple[int, Any]]:
    """Return ``[(layer_idx, op), ...]`` for every op in the layout.

    Block ops are placed at the layer their bake targets (resolved via
    ``layout.resolve_block_op_layer`` when available). Model ops carry
    ``layer_idx=None`` — they're returned as ``layer_idx=-1`` so they
    sort at the top of the report (model-wide bakes).
    """

    out: List[Tuple[int, Any]] = []
    for layer_idx, ops_at in enumerate(layout.ops_per_layer):
        for op in ops_at:
            out.append((layer_idx, op))
    for op in getattr(layout, "block_ops", ()):
        try:
            ly = layout.resolve_block_op_layer(op)
        except Exception:
            ly = getattr(op, "layer_idx", None)
            if ly is None:
                ly = -1
        out.append((int(ly), op))
    for op in getattr(layout, "model_ops", ()):
        ly = getattr(op, "layer_idx", None)
        if ly is None:
            ly = -1
        out.append((int(ly), op))
    return out


def _materialize_op_ir(op, dim_positions: Mapping[str, int], head_dim: int):
    """Return the op's ``compiler_ir`` if present, otherwise invoke the
    factory. Returns ``None`` when the op has neither.
    """

    ir = getattr(op, "compiler_ir", None)
    if ir is not None:
        return ir
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is None:
        return None
    try:
        return factory(dim_positions, head_dim)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# IR walkers
# ---------------------------------------------------------------------------


def _ffn_rules_from_ir(ir) -> List[FFNRule]:
    if ir is None:
        return []
    rules: List[FFNRule] = []
    if hasattr(ir, "layers"):
        for layer in ir.layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "rules"):
                rules.extend(ffn.rules)
        return rules
    if isinstance(ir, FFNOp):
        return list(ir.rules)
    if isinstance(ir, (list, tuple)):
        for item in ir:
            if isinstance(item, FFNRule):
                rules.append(item)
            elif isinstance(item, FFNOp):
                rules.extend(item.rules)
        return rules
    if hasattr(ir, "rules"):
        return [r for r in ir.rules if isinstance(r, FFNRule)]
    return []


def _attention_heads_from_ir(ir) -> List[Any]:
    if ir is None:
        return []
    heads: List[Any] = []
    if hasattr(ir, "layers"):
        for layer in ir.layers:
            attention = getattr(layer, "attention", None)
            if attention is not None and hasattr(attention, "rules"):
                heads.extend(attention.rules)
        return heads
    if hasattr(ir, "rules"):
        # AttentionOp directly.
        return [r for r in ir.rules]
    return []


def _ffn_gate_conditions(rule: FFNRule) -> Tuple[str, ...]:
    """Human-readable list of the rule's gate/condition inputs."""

    cs: List[str] = []
    for ct in rule.conditions:
        cs.append(f"{ct.dim.key()}*{ct.weight:g}")
    if rule.gate is not None:
        cs.append(f"GATE:{rule.gate.key()}*{rule.gate_weight:g}")
    for ct in rule.gate_terms:
        cs.append(f"GATE_T:{ct.dim.key()}*{ct.weight:g}")
    if rule.threshold:
        cs.append(f"thr={rule.threshold:g}")
    return tuple(cs)


def _dim_int_to_name(
    dim_int: int,
    dim_positions: Mapping[str, int],
    dim_sizes: Mapping[str, int],
) -> Optional[Tuple[str, int]]:
    """Reverse-lookup a residual-column int back to (dim_name, offset).

    Uses the layout's ``dim_positions`` + ``dim_sizes`` so aliases land
    on their canonical name. Returns ``None`` when no slot contains the
    integer (e.g. padding).
    """

    for name, pos in dim_positions.items():
        size = dim_sizes.get(name, 1)
        if pos <= dim_int < pos + size:
            return name, dim_int - pos
    return None


def _attn_q_scope(spec, dim_positions, dim_sizes) -> Tuple[str, ...]:
    """Dim names with positive Q weight — the head's effective Q-side
    firing scope (mirrors ``attention_verifier.effective_attention_q_scope``
    but uses the layout's dim tables directly so no DimRegistry is needed).
    """

    names: List[str] = []
    for w in getattr(spec, "q", ()):
        if w.weight <= 0.0:
            continue
        resolved = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
        if resolved is not None:
            names.append(resolved[0])
    return tuple(sorted(set(names)))


def _attn_k_scope(spec, dim_positions, dim_sizes) -> Tuple[str, ...]:
    names: List[str] = []
    for w in getattr(spec, "k", ()):
        if w.weight <= 0.0:
            continue
        resolved = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
        if resolved is not None:
            names.append(resolved[0])
    return tuple(sorted(set(names)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enumerate_dim_writers(
    compiler,
    dim_name: str,
    *,
    head_dim: int = 64,
) -> List[DimWriter]:
    """Every op that writes ``dim_name`` across all layers of the compiled
    layout. Ordered by layer index.

    Sources covered:

    * FFNRule.writes targeting ``dim_name`` (any offset).
    * AttentionHead.spec.o (O projection) targeting any residual column
      that lives inside the ``dim_name`` slot range.

    ``head_dim`` is passed to any ``compiler_ir_factory`` that needs it
    (default 64 = transformer block head dimension).
    """

    layout = _coerce_layout(compiler)
    dim_positions = layout.dim_positions
    dim_sizes = getattr(layout, "dim_sizes", {})

    if dim_name not in dim_positions:
        return []
    start = dim_positions[dim_name]
    size = dim_sizes.get(dim_name, 1)
    end = start + size

    out: List[DimWriter] = []

    for layer_idx, op in _walk_ops_with_layers(layout):
        ir = _materialize_op_ir(op, dim_positions, head_dim)
        if ir is None:
            continue

        # FFN side.
        for rule in _ffn_rules_from_ir(ir):
            for wt in rule.writes:
                if wt.dim.name != dim_name:
                    continue
                if wt.weight == 0.0:
                    continue
                out.append(DimWriter(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="ffn",
                    rule_name=getattr(rule, "name", None),
                    write_value_expr=f"{wt.weight:+g}",
                    gate_conditions=_ffn_gate_conditions(rule),
                    offset=wt.dim.offset,
                ))

        # Attention side: walk O writes.
        for head in _attention_heads_from_ir(ir):
            spec = getattr(head, "spec", None)
            if spec is None:
                continue
            q_scope = _attn_q_scope(spec, dim_positions, dim_sizes)
            k_scope = _attn_k_scope(spec, dim_positions, dim_sizes)
            # Build a "V slot -> aggregate weight" map so the report can
            # surface the head's max delivered value.
            v_mag_by_slot: Dict[int, float] = {}
            for vw in getattr(spec, "v", ()):
                v_mag_by_slot[vw.slot] = (
                    v_mag_by_slot.get(vw.slot, 0.0) + abs(vw.weight)
                )
            for ow in getattr(spec, "o", ()):
                if not (start <= int(ow.out_dim) < end):
                    continue
                offset = int(ow.out_dim) - start
                vmag = v_mag_by_slot.get(ow.slot, 0.0)
                expr = (
                    f"O slot {ow.slot} * {ow.weight:+g}"
                    f" (V mag {vmag:g})"
                )
                gate_conds = tuple(
                    [f"Q:{n}" for n in q_scope]
                    + [f"K:{n}" for n in k_scope]
                )
                out.append(DimWriter(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="attn",
                    rule_name=getattr(head, "name", None)
                    or f"head_{getattr(spec, 'head_idx', '?')}",
                    write_value_expr=expr,
                    gate_conditions=gate_conds,
                    offset=offset,
                ))

    out.sort(key=lambda w: (w.layer_idx, w.op_kind, w.op_name, w.offset))
    return out


def enumerate_dim_readers(
    compiler,
    dim_name: str,
    *,
    head_dim: int = 64,
) -> List[DimReader]:
    """Every op that reads ``dim_name`` across all layers. Ordered by layer.

    Sources covered:

    * FFNRule.conditions / FFNRule.gate / FFNRule.gate_terms.
    * AttentionHead.spec.q / .k / .v (Q/K/V projection inputs).
    """

    layout = _coerce_layout(compiler)
    dim_positions = layout.dim_positions
    dim_sizes = getattr(layout, "dim_sizes", {})

    if dim_name not in dim_positions:
        return []
    start = dim_positions[dim_name]
    size = dim_sizes.get(dim_name, 1)
    end = start + size

    out: List[DimReader] = []

    for layer_idx, op in _walk_ops_with_layers(layout):
        ir = _materialize_op_ir(op, dim_positions, head_dim)
        if ir is None:
            continue

        # FFN side.
        for rule in _ffn_rules_from_ir(ir):
            rule_name = getattr(rule, "name", None)
            gate_conds = _ffn_gate_conditions(rule)
            for ct in rule.conditions:
                if ct.dim.name != dim_name:
                    continue
                out.append(DimReader(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="ffn",
                    rule_name=rule_name,
                    read_role="ffn_condition",
                    gate_conditions=gate_conds,
                    weight=ct.weight,
                    offset=ct.dim.offset,
                ))
            if rule.gate is not None and rule.gate.name == dim_name:
                out.append(DimReader(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="ffn",
                    rule_name=rule_name,
                    read_role="ffn_gate",
                    gate_conditions=gate_conds,
                    weight=rule.gate_weight,
                    offset=rule.gate.offset,
                ))
            for ct in rule.gate_terms:
                if ct.dim.name != dim_name:
                    continue
                out.append(DimReader(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="ffn",
                    rule_name=rule_name,
                    read_role="ffn_gate",
                    gate_conditions=gate_conds,
                    weight=ct.weight,
                    offset=ct.dim.offset,
                ))

        # Attention side: walk Q/K/V projection inputs.
        for head in _attention_heads_from_ir(ir):
            spec = getattr(head, "spec", None)
            if spec is None:
                continue
            head_name = (
                getattr(head, "name", None)
                or f"head_{getattr(spec, 'head_idx', '?')}"
            )
            q_scope = _attn_q_scope(spec, dim_positions, dim_sizes)
            k_scope = _attn_k_scope(spec, dim_positions, dim_sizes)
            gate_conds = tuple(
                [f"Q:{n}" for n in q_scope]
                + [f"K:{n}" for n in k_scope]
            )
            for role, projection in (
                ("attn_Q", getattr(spec, "q", ())),
                ("attn_K", getattr(spec, "k", ())),
                ("attn_V", getattr(spec, "v", ())),
            ):
                for w in projection:
                    if not (start <= int(w.dim) < end):
                        continue
                    offset = int(w.dim) - start
                    out.append(DimReader(
                        layer_idx=layer_idx,
                        op_name=getattr(op, "name", "<anonymous>"),
                        op_kind="attn",
                        rule_name=head_name,
                        read_role=role,
                        gate_conditions=gate_conds,
                        weight=float(w.weight),
                        offset=offset,
                    ))

    out.sort(key=lambda r: (r.layer_idx, r.op_kind, r.op_name, r.read_role,
                            r.offset))
    return out


def trace_dim_flow(
    compiler,
    dim_name: str,
    *,
    head_dim: int = 64,
) -> DimFlow:
    """Producer-consumer chain for ``dim_name``: writers (sorted by layer)
    followed by readers (sorted by layer). The ``DimFlow.writers`` /
    ``.readers`` lists are ready to use; ``DimFlow.dim_position`` exposes
    the layout slot index (or ``None`` if the dim is not declared).
    """

    layout = _coerce_layout(compiler)
    return DimFlow(
        dim_name=dim_name,
        dim_position=layout.dim_positions.get(dim_name),
        writers=enumerate_dim_writers(layout, dim_name, head_dim=head_dim),
        readers=enumerate_dim_readers(layout, dim_name, head_dim=head_dim),
    )


def find_zeroing_writers(
    compiler,
    dim_name: str,
    *,
    head_dim: int = 64,
) -> List[DimWriter]:
    """Writers whose net contribution to ``dim_name`` is zero.

    Two definitions of "zero":

    1. **Explicit zero**: an FFN write whose ``weight == 0`` (defensive
       — already filtered out of ``enumerate_dim_writers``).
    2. **Effective zero**: an attention head whose V-side magnitude at
       the O write's slot is zero. The head can't deliver any value
       through that slot — the O write is dead.
    """

    layout = _coerce_layout(compiler)
    dim_positions = layout.dim_positions
    dim_sizes = getattr(layout, "dim_sizes", {})

    if dim_name not in dim_positions:
        return []
    start = dim_positions[dim_name]
    size = dim_sizes.get(dim_name, 1)
    end = start + size

    out: List[DimWriter] = []

    for layer_idx, op in _walk_ops_with_layers(layout):
        ir = _materialize_op_ir(op, dim_positions, head_dim)
        if ir is None:
            continue

        # FFN side: explicit zero writes.
        for rule in _ffn_rules_from_ir(ir):
            for wt in rule.writes:
                if wt.dim.name != dim_name:
                    continue
                if wt.weight != 0.0:
                    continue
                out.append(DimWriter(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="ffn",
                    rule_name=getattr(rule, "name", None),
                    write_value_expr="0 (explicit)",
                    gate_conditions=_ffn_gate_conditions(rule),
                    offset=wt.dim.offset,
                ))

        # Attention side: O write whose V-slot magnitude is zero.
        for head in _attention_heads_from_ir(ir):
            spec = getattr(head, "spec", None)
            if spec is None:
                continue
            v_mag_by_slot: Dict[int, float] = {}
            for vw in getattr(spec, "v", ()):
                v_mag_by_slot[vw.slot] = (
                    v_mag_by_slot.get(vw.slot, 0.0) + abs(vw.weight)
                )
            for ow in getattr(spec, "o", ()):
                if not (start <= int(ow.out_dim) < end):
                    continue
                vmag = v_mag_by_slot.get(ow.slot, 0.0)
                if vmag > 0.0 and ow.weight != 0.0:
                    continue
                offset = int(ow.out_dim) - start
                head_name = (
                    getattr(head, "name", None)
                    or f"head_{getattr(spec, 'head_idx', '?')}"
                )
                out.append(DimWriter(
                    layer_idx=layer_idx,
                    op_name=getattr(op, "name", "<anonymous>"),
                    op_kind="attn",
                    rule_name=head_name,
                    write_value_expr=(
                        f"O slot {ow.slot} * {ow.weight:+g} (V mag 0)"
                    ),
                    gate_conditions=(),
                    offset=offset,
                ))

    out.sort(key=lambda w: (w.layer_idx, w.op_kind, w.op_name, w.offset))
    return out


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt_writer(w: DimWriter) -> str:
    gates = ", ".join(w.gate_conditions) if w.gate_conditions else "<none>"
    return (
        f"  L{w.layer_idx:02d} [{w.op_kind:4s}] {w.op_name}"
        f"::{w.rule_name or '-'}"
        f" +offset={w.offset} value={w.write_value_expr}"
        f"\n      gates: {gates}"
    )


def _fmt_reader(r: DimReader) -> str:
    gates = ", ".join(r.gate_conditions) if r.gate_conditions else "<none>"
    return (
        f"  L{r.layer_idx:02d} [{r.op_kind:4s}] {r.op_name}"
        f"::{r.rule_name or '-'}"
        f" role={r.read_role} offset={r.offset} w={r.weight:+g}"
        f"\n      gates: {gates}"
    )


def format_report(
    flow: DimFlow,
    *,
    zeroing_writers: Optional[Sequence[DimWriter]] = None,
) -> str:
    """Human-readable producer-consumer report for ``flow.dim_name``."""

    lines: List[str] = []
    lines.append(f"=== dim flow: {flow.dim_name} ===")
    pos = flow.dim_position
    lines.append(
        f"slot position: {pos if pos is not None else '<not declared>'}"
    )
    lines.append("")

    lines.append(f"WRITERS ({len(flow.writers)}):")
    if not flow.writers:
        lines.append("  <no writers — dim is never set>")
    else:
        for w in flow.writers:
            lines.append(_fmt_writer(w))
    lines.append("")

    lines.append(f"READERS ({len(flow.readers)}):")
    if not flow.readers:
        lines.append("  <no readers — dim is never consumed>")
    else:
        for r in flow.readers:
            lines.append(_fmt_reader(r))
    lines.append("")

    if zeroing_writers is not None:
        lines.append(f"ZEROING WRITERS ({len(zeroing_writers)}):")
        if not zeroing_writers:
            lines.append("  <none>")
        else:
            for w in zeroing_writers:
                lines.append(_fmt_writer(w))
        lines.append("")

    # Producer/consumer ordering summary.
    if flow.writers and flow.readers:
        first_write = min(w.layer_idx for w in flow.writers)
        first_read = min(r.layer_idx for r in flow.readers)
        if first_read < first_write:
            lines.append(
                f"WARN: first read at L{first_read:02d} precedes first "
                f"write at L{first_write:02d} — reader sees stale / "
                f"prev-step value."
            )
        else:
            lines.append(
                f"OK: writes start at L{first_write:02d}, reads start "
                f"at L{first_read:02d}."
            )

    return "\n".join(lines)
