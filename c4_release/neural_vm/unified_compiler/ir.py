"""Declarative compiler IR shared by weight lowering and symbolic execution.

This module is intentionally small: it defines the first common data model
that can be lowered into neural weights and interpreted symbolically. The
goal is to make operation authors describe *what* should happen once, then
use separate backends for:

* neural weight generation, and
* symbolic/debug execution.

The first supported primitive is an FFN-style conditional write because many
legacy bakes are SwiGLU AND-gates of the form:

    if sum(condition dims) >= threshold:
        output dims += constant or gated source value
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    from .primitives import DeclarativeAttentionHeadSpec


_SILU_ONE_INPUT = 1.278464542761074


@dataclass(frozen=True)
class DimRef:
    """Reference to one residual dimension cell."""

    name: str
    offset: int = 0

    @classmethod
    def parse(cls, value: str) -> "DimRef":
        if "+" not in value:
            return cls(value, 0)
        name, offset = value.rsplit("+", 1)
        return cls(name, int(offset))

    def key(self) -> str:
        return f"{self.name}+{self.offset}"

    def resolve(self, dim_positions: Mapping[str, int]) -> int:
        return dim_positions[self.name] + self.offset


@dataclass(frozen=True)
class ConditionTerm:
    """Weighted condition input for a symbolic/neural rule."""

    dim: DimRef
    weight: float = 1.0


@dataclass(frozen=True)
class WriteTerm:
    """Weighted output write for a symbolic/neural rule."""

    dim: DimRef
    weight: float


@dataclass(frozen=True)
class FFNRule:
    """One SwiGLU-compatible conditional write."""

    conditions: Tuple[ConditionTerm, ...]
    threshold: float
    writes: Tuple[WriteTerm, ...]
    gate: Optional[DimRef] = None
    gate_weight: float = 1.0
    gate_terms: Tuple[ConditionTerm, ...] = ()
    gate_bias: float = 1.0
    name: Optional[str] = None

    @classmethod
    def constant_write(
        cls,
        *,
        conditions: Sequence[Tuple[str, float]],
        threshold: float,
        writes: Sequence[Tuple[str, float]],
        name: Optional[str] = None,
    ) -> "FFNRule":
        return cls(
            conditions=tuple(
                ConditionTerm(DimRef.parse(dim), weight)
                for dim, weight in conditions
            ),
            threshold=threshold,
            writes=tuple(
                WriteTerm(DimRef.parse(dim), weight)
                for dim, weight in writes
            ),
            name=name,
        )

    @classmethod
    def gated_write(
        cls,
        *,
        conditions: Sequence[Tuple[str, float]],
        threshold: float,
        gate: Optional[str] = None,
        writes: Sequence[Tuple[str, float]],
        gate_terms: Sequence[Tuple[str, float]] = (),
        gate_weight: float = 1.0,
        gate_bias: float = 0.0,
        name: Optional[str] = None,
    ) -> "FFNRule":
        return cls(
            conditions=tuple(
                ConditionTerm(DimRef.parse(dim), weight)
                for dim, weight in conditions
            ),
            threshold=threshold,
            gate=DimRef.parse(gate) if gate is not None else None,
            gate_weight=gate_weight,
            gate_terms=tuple(
                ConditionTerm(DimRef.parse(dim), weight)
                for dim, weight in gate_terms
            ),
            gate_bias=gate_bias,
            writes=tuple(
                WriteTerm(DimRef.parse(dim), weight)
                for dim, weight in writes
            ),
            name=name,
        )


@dataclass
class FFNOp:
    """Declarative FFN operation made of conditional-write rules."""

    rules: List[FFNRule] = field(default_factory=list)

    def append(self, rule: FFNRule) -> "FFNOp":
        self.rules.append(rule)
        return self

    @property
    def hidden_units(self) -> int:
        return len(self.rules)


@dataclass(frozen=True)
class FFNComparisonIssue:
    """One classified symbolic-vs-lowered FFN comparison failure."""

    kind: str
    message: str


@dataclass
class FFNComparisonReport:
    """Diagnostic result for ``compare_symbolic_to_lowered_ffn``."""

    ok: bool
    issues: List[FFNComparisonIssue] = field(default_factory=list)
    symbolic_state: Dict[str, float] = field(default_factory=dict)
    lowered_state: Dict[str, float] = field(default_factory=dict)
    layer_idx: int = 0

    @property
    def failure_kinds(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(issue.kind for issue in self.issues))

    @property
    def primary_failure_kind(self) -> Optional[str]:
        return self.issues[0].kind if self.issues else None

    def format(self) -> str:
        status = "OK" if self.ok else "DRIFT"
        lines = [
            f"=== CompilerIR FFN comparison ({status}, layer={self.layer_idx}) ==="
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.kind}] {issue.message}")
        return "\n".join(lines)


@dataclass
class AttentionOp:
    """Declarative attention operation made of head-level specs.

    The underlying write representation remains
    ``primitives.DeclarativeAttentionHeadSpec`` so this prototype can carry
    existing attention-head bakes through ``CompilerIR`` without migrating
    production bake code.
    """

    rules: List["AttentionHeadIR"] = field(default_factory=list)

    def add_head(
        self,
        spec_or_head,
        *,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "AttentionHeadIR":
        head = _coerce_attention_head_ir(
            spec_or_head,
            name=name,
            metadata=metadata,
        )
        self.rules.append(head)
        return head

    def append(
        self,
        spec_or_head,
        *,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "AttentionOp":
        self.add_head(spec_or_head, name=name, metadata=metadata)
        return self

    def extend(self, specs_or_heads: Iterable[object]) -> "AttentionOp":
        for spec_or_head in specs_or_heads:
            self.add_head(spec_or_head)
        return self

    @property
    def heads(self) -> Tuple["AttentionHeadIR", ...]:
        return tuple(self.rules)


@dataclass(frozen=True)
class AttentionHeadIR:
    """CompilerIR adapter for one declarative attention-head spec."""

    spec: "DeclarativeAttentionHeadSpec"
    name: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def head_idx(self) -> int:
        return int(self.spec.head_idx)

    def debug_report(self, HD: int) -> "AttentionHeadReport":
        base = self.head_idx * HD
        q = tuple(
            AttentionMatrixWrite(
                matrix="W_q",
                row=base + write.slot,
                col=write.dim,
                head_idx=self.head_idx,
                slot=write.slot,
                source_dim=write.dim,
                output_dim=None,
                weight=write.weight,
            )
            for write in self.spec.q
        )
        k = tuple(
            AttentionMatrixWrite(
                matrix="W_k",
                row=base + write.slot,
                col=write.dim,
                head_idx=self.head_idx,
                slot=write.slot,
                source_dim=write.dim,
                output_dim=None,
                weight=write.weight,
            )
            for write in self.spec.k
        )
        v = tuple(
            AttentionMatrixWrite(
                matrix="W_v",
                row=base + write.slot,
                col=write.dim,
                head_idx=self.head_idx,
                slot=write.slot,
                source_dim=write.dim,
                output_dim=None,
                weight=write.weight,
            )
            for write in self.spec.v
        )
        o = tuple(
            AttentionMatrixWrite(
                matrix="W_o",
                row=write.out_dim,
                col=base + write.slot,
                head_idx=self.head_idx,
                slot=write.slot,
                source_dim=None,
                output_dim=write.out_dim,
                weight=write.weight,
            )
            for write in self.spec.o
        )
        return AttentionHeadReport(
            name=self.name,
            head_idx=self.head_idx,
            metadata=dict(self.metadata),
            q=q,
            k=k,
            v=v,
            o=o,
        )


@dataclass(frozen=True)
class AttentionMatrixWrite:
    """One resolved attention matrix write for debug/reporting."""

    matrix: str
    row: int
    col: int
    head_idx: int
    slot: int
    source_dim: Optional[int]
    output_dim: Optional[int]
    weight: float


@dataclass(frozen=True)
class AttentionHeadReport:
    """Structured Q/K/V/O write report for one attention head."""

    name: Optional[str]
    head_idx: int
    metadata: Mapping[str, object]
    q: Tuple[AttentionMatrixWrite, ...] = field(default_factory=tuple)
    k: Tuple[AttentionMatrixWrite, ...] = field(default_factory=tuple)
    v: Tuple[AttentionMatrixWrite, ...] = field(default_factory=tuple)
    o: Tuple[AttentionMatrixWrite, ...] = field(default_factory=tuple)

    @property
    def all_writes(self) -> Tuple[AttentionMatrixWrite, ...]:
        return self.q + self.k + self.v + self.o


@dataclass(frozen=True)
class AttentionDebugReport:
    """Diagnostic result for attention IR metadata inspection."""

    layer_idx: int
    heads: Tuple[AttentionHeadReport, ...] = field(default_factory=tuple)

    @property
    def matrix_write_counts(self) -> Dict[str, int]:
        counts = {"W_q": 0, "W_k": 0, "W_v": 0, "W_o": 0}
        for head in self.heads:
            for write in head.all_writes:
                counts[write.matrix] = counts.get(write.matrix, 0) + 1
        return counts

    def format(self) -> str:
        lines = [
            f"=== CompilerIR attention report "
            f"(layer={self.layer_idx}, heads={len(self.heads)}) ==="
        ]
        for head in self.heads:
            label = head.name or f"head_{head.head_idx}"
            counts = {
                "Q": len(head.q),
                "K": len(head.k),
                "V": len(head.v),
                "O": len(head.o),
            }
            lines.append(
                f"  {label}: head={head.head_idx} "
                f"Q={counts['Q']} K={counts['K']} "
                f"V={counts['V']} O={counts['O']}"
            )
        return "\n".join(lines)


@dataclass
class SymbolicResidualState:
    """Position-indexed residual stream used by symbolic declaration runs.

    ``positions[pos][dim]`` stores the residual value for one concrete token
    position and one resolved residual dimension. This is deliberately closer
    to the lowered Q/K/V/FFN matrices than ``symbolic_ffn``'s named dict API,
    so attention and FFN rules can share one execution substrate.
    """

    positions: List[Dict[int, float]] = field(default_factory=list)
    pos_ids: Optional[List[int]] = None

    @classmethod
    def empty(cls, n_positions: int) -> "SymbolicResidualState":
        return cls([{} for _ in range(n_positions)])

    @classmethod
    def from_named_positions(
        cls,
        named_positions: Sequence[Mapping[str, float]],
        dim_positions: Mapping[str, int],
    ) -> "SymbolicResidualState":
        """Build a symbolic state from ``{"DIM+offset": value}`` maps."""

        positions: List[Dict[int, float]] = []
        for named in named_positions:
            resolved: Dict[int, float] = {}
            for key, value in named.items():
                resolved[DimRef.parse(key).resolve(dim_positions)] = float(value)
            positions.append(resolved)
        return cls(positions)

    def copy(self) -> "SymbolicResidualState":
        return SymbolicResidualState(
            [dict(pos) for pos in self.positions],
            None if self.pos_ids is None else list(self.pos_ids),
        )

    def __len__(self) -> int:
        return len(self.positions)

    def get(self, pos_idx: int, dim: int) -> float:
        return float(self.positions[pos_idx].get(dim, 0.0))

    def add(self, pos_idx: int, dim: int, value: float) -> None:
        if value == 0.0:
            return
        slot = self.positions[pos_idx]
        slot[dim] = float(slot.get(dim, 0.0) + value)

    def set(self, pos_idx: int, dim: int, value: float) -> None:
        self.positions[pos_idx][dim] = float(value)

    def named_position(
        self,
        pos_idx: int,
        dim_positions: Mapping[str, int],
        *,
        dim_sizes: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, float]:
        """Return one position using ``DIM+offset`` keys for diagnostics."""

        return {
            _dim_key_from_position(dim, dim_positions, dim_sizes): value
            for dim, value in sorted(self.positions[pos_idx].items())
        }


@dataclass(frozen=True)
class SymbolicAttentionChoice:
    """One selected source row for symbolic hardmax attention."""

    layer_idx: int
    head_idx: int
    query_pos: int
    key_pos: Optional[int]
    score: float
    sink_selected: bool = False


@dataclass
class SymbolicDeclarativeRunReport:
    """Structured result for a declaration-only symbolic run."""

    initial_state: SymbolicResidualState
    final_state: SymbolicResidualState
    executed_ops: List[str] = field(default_factory=list)
    unsupported_ops: List[str] = field(default_factory=list)
    attention_choices: List[SymbolicAttentionChoice] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported_ops

    def format(self) -> str:
        status = "OK" if self.ok else "UNSUPPORTED"
        lines = [
            f"=== Symbolic declarative run ({status}) ===",
            f"Executed ops: {len(self.executed_ops)}",
            f"Unsupported ops: {len(self.unsupported_ops)}",
        ]
        for name in self.executed_ops:
            lines.append(f"  EXEC: {name}")
        for name in self.unsupported_ops:
            lines.append(f"  UNSUPPORTED: {name}")
        return "\n".join(lines)


class SymbolicDeclarativeRunner:
    """Execute declaration IR without baking or running neural weights.

    The runner is intentionally strict about production layouts: an op without
    ``compiler_ir`` is reported as unsupported unless it is a topology anchor.
    That gives us the isolation split we need while the migration is ongoing:
    unsupported declaration coverage vs. declaration semantics vs. neural
    lowering/runtime drift.
    """

    def __init__(
        self,
        *,
        head_dim: int = 64,
        causal: bool = True,
        attention_mode: str = "hardmax",
        alibi_slopes: Optional[Mapping[Tuple[int, int], float]] = None,
    ):
        if attention_mode not in ("hardmax", "softmax1"):
            raise ValueError("attention_mode must be 'hardmax' or 'softmax1'")
        self.head_dim = head_dim
        self.causal = causal
        self.attention_mode = attention_mode
        self.alibi_slopes = dict(alibi_slopes or {})

    def run_ir(
        self,
        ir: "CompilerIR",
        state: SymbolicResidualState,
        *,
        layer_indices: Optional[Sequence[int]] = None,
        dim_positions: Optional[Mapping[str, int]] = None,
    ) -> SymbolicDeclarativeRunReport:
        out = state.copy()
        choices: List[SymbolicAttentionChoice] = []
        layers = (
            list(layer_indices)
            if layer_indices is not None
            else list(range(len(ir.layers)))
        )
        for layer_idx in layers:
            out, layer_choices = ir.symbolic_attention_positions(
                out,
                self.head_dim,
                layer_idx=layer_idx,
                alibi_slopes={
                    head_idx: slope
                    for (decl_layer, head_idx), slope in self.alibi_slopes.items()
                    if decl_layer == layer_idx
                },
                causal=self.causal,
                mode=self.attention_mode,
            )
            choices.extend(layer_choices)
            if dim_positions is not None:
                out = ir.symbolic_ffn_positions(
                    out,
                    dim_positions,
                    layer_idx=layer_idx,
                )
        return SymbolicDeclarativeRunReport(
            initial_state=state,
            final_state=out,
            executed_ops=["<compiler_ir>"],
            attention_choices=choices,
        )

    def run_layout(
        self,
        layout,
        state: SymbolicResidualState,
        *,
        dim_positions: Optional[Mapping[str, int]] = None,
    ) -> SymbolicDeclarativeRunReport:
        if dim_positions is None:
            dim_positions = getattr(layout, "dim_positions", None)
        if dim_positions is None:
            raise ValueError("dim_positions is required for layout symbolic runs")

        out = state.copy()
        executed: List[str] = []
        unsupported: List[str] = []
        choices: List[SymbolicAttentionChoice] = []

        for layer_idx, ops in enumerate(getattr(layout, "ops_per_layer", [])):
            for op in ops:
                op_ir = _operation_compiler_ir(
                    op,
                    dim_positions=dim_positions,
                    head_dim=self.head_dim,
                )
                if op_ir is None:
                    if getattr(op, "declarative_authority", None) in {
                        "structural_model",
                        "topology_anchor",
                    }:
                        continue
                    unsupported.append(getattr(op, "name", repr(op)))
                    continue
                ir_layer = layer_idx if layer_idx < len(op_ir.layers) else 0
                out, layer_choices = op_ir.symbolic_attention_positions(
                    out,
                    self.head_dim,
                    layer_idx=ir_layer,
                    alibi_slopes={
                        head_idx: slope
                        for (decl_layer, head_idx), slope
                        in self.alibi_slopes.items()
                        if decl_layer == layer_idx
                    },
                    causal=self.causal,
                    mode=self.attention_mode,
                )
                choices.extend(layer_choices)
                out = op_ir.symbolic_ffn_positions(
                    out,
                    dim_positions,
                    layer_idx=ir_layer,
                )
                executed.append(getattr(op, "name", repr(op)))

        for op in getattr(layout, "block_ops", []):
            op_ir = _operation_compiler_ir(
                op,
                dim_positions=dim_positions,
                head_dim=self.head_dim,
            )
            if op_ir is None:
                if getattr(op, "declarative_authority", None) in {
                    "structural_model",
                    "topology_anchor",
                }:
                    continue
                unsupported.append(getattr(op, "name", repr(op)))
                continue
            out = op_ir.symbolic_ffn_positions(out, dim_positions, layer_idx=0)
            executed.append(getattr(op, "name", repr(op)))

        for op in getattr(layout, "model_ops", []):
            op_ir = _operation_compiler_ir(
                op,
                dim_positions=dim_positions,
                head_dim=self.head_dim,
            )
            if op_ir is None:
                if getattr(op, "declarative_authority", None) in {
                    "structural_model",
                    "topology_anchor",
                }:
                    continue
                unsupported.append(getattr(op, "name", repr(op)))
                continue
            out = op_ir.symbolic_ffn_positions(out, dim_positions, layer_idx=0)
            executed.append(getattr(op, "name", repr(op)))

        return SymbolicDeclarativeRunReport(
            initial_state=state,
            final_state=out,
            executed_ops=executed,
            unsupported_ops=unsupported,
            attention_choices=choices,
        )


@dataclass
class LayerSpec:
    """Declarative work assigned to one logical compiler layer."""

    ffn: FFNOp = field(default_factory=FFNOp)
    attention: AttentionOp = field(default_factory=AttentionOp)


@dataclass
class CompilerIR:
    """A symbolic-and-lowerable program fragment."""

    layers: List[LayerSpec] = field(default_factory=list)

    def layer(self, index: int) -> LayerSpec:
        while len(self.layers) <= index:
            self.layers.append(LayerSpec())
        return self.layers[index]

    def lower_ffn(
        self,
        ffn,
        dim_positions: Mapping[str, int],
        *,
        layer_idx: int = 0,
        start_unit: int = 0,
        S: float = 100.0,
        write_scale: float = 1.0,
    ) -> int:
        """Lower one layer's FFN rules into a ``PureFFN``-style module."""

        unit = start_unit
        for rule in self.layer(layer_idx).ffn.rules:
            for term in rule.conditions:
                ffn.W_up.data[unit, term.dim.resolve(dim_positions)] = (
                    S * term.weight
                )
            ffn.b_up.data[unit] = -S * rule.threshold

            if rule.gate is None:
                ffn.b_gate.data[unit] = rule.gate_bias
            else:
                ffn.W_gate.data[unit, rule.gate.resolve(dim_positions)] = (
                    rule.gate_weight
                )
                ffn.b_gate.data[unit] = rule.gate_bias
            for term in rule.gate_terms:
                ffn.W_gate.data[unit, term.dim.resolve(dim_positions)] = (
                    term.weight
                )

            for write in rule.writes:
                ffn.W_down.data[write.dim.resolve(dim_positions), unit] = (
                    write.weight * write_scale
                )
            unit += 1
        return unit

    def symbolic_ffn(
        self,
        state: Mapping[str, float],
        *,
        layer_idx: int = 0,
    ) -> Dict[str, float]:
        """Symbolically apply one layer's FFN rules to a residual-state map."""

        out: Dict[str, float] = dict(state)
        for rule in self.layer(layer_idx).ffn.rules:
            score = sum(
                _state_value(state, term.dim) * term.weight
                for term in rule.conditions
            )
            if score < rule.threshold:
                continue
            gate_value = rule.gate_bias
            if rule.gate is not None:
                gate_value += _state_value(state, rule.gate) * rule.gate_weight
            gate_value += sum(
                _state_value(state, term.dim) * term.weight
                for term in rule.gate_terms
            )
            for write in rule.writes:
                key = write.dim.key()
                out[key] = out.get(key, 0.0) + gate_value * write.weight
        return out

    def required_ffn_units(self, *, layer_idx: int = 0) -> int:
        return self.layer(layer_idx).ffn.hidden_units

    def lower_attention(
        self,
        attn,
        HD: int,
        *,
        layer_idx: int = 0,
    ) -> int:
        """Lower one layer's attention specs into attention projection weights."""

        from .primitives import Primitives

        heads = self.layer(layer_idx).attention.rules
        for head in heads:
            Primitives.generate_attention_head(attn, head.spec, HD)
        return len(heads)

    def attention_debug_report(
        self,
        HD: int,
        *,
        layer_idx: int = 0,
    ) -> AttentionDebugReport:
        """Return structured Q/K/V/O write metadata for one attention layer."""

        return AttentionDebugReport(
            layer_idx=layer_idx,
            heads=tuple(
                head.debug_report(HD)
                for head in self.layer(layer_idx).attention.rules
            ),
        )

    def symbolic_attention_positions(
        self,
        state: SymbolicResidualState,
        HD: int,
        *,
        layer_idx: int = 0,
        alibi_slopes: Optional[Mapping[int, float]] = None,
        causal: bool = True,
        mode: str = "hardmax",
        sink_score: float = 0.0,
    ) -> Tuple[SymbolicResidualState, List[SymbolicAttentionChoice]]:
        """Symbolically apply one layer's attention specs to all positions.

        ``hardmax`` selects the strongest key only when it beats the softmax1
        sink score. ``softmax1`` keeps numeric weights, including the sink
        denominator, and is useful when a head intentionally blends sources.
        """

        if mode not in ("hardmax", "softmax1"):
            raise ValueError("mode must be 'hardmax' or 'softmax1'")
        out = state.copy()
        choices: List[SymbolicAttentionChoice] = []
        slopes = dict(alibi_slopes or {})
        scale = 1.0 / math.sqrt(float(HD))

        for head in self.layer(layer_idx).attention.rules:
            spec = head.spec
            slope = float(slopes.get(spec.head_idx, 0.0))
            for q_pos in range(len(state)):
                q_slots = _attention_projection_slots(
                    state, q_pos, spec.q
                )
                key_positions = range(q_pos + 1) if causal else range(len(state))
                scored_keys = []
                for k_pos in key_positions:
                    k_slots = _attention_projection_slots(
                        state, k_pos, spec.k
                    )
                    score = _slot_dot(q_slots, k_slots) * scale
                    if slope:
                        q_abs = _absolute_pos(state, q_pos)
                        k_abs = _absolute_pos(state, k_pos)
                        score -= slope * abs(q_abs - k_abs)
                    scored_keys.append((score, k_pos))

                if mode == "hardmax":
                    if not scored_keys:
                        choices.append(SymbolicAttentionChoice(
                            layer_idx, spec.head_idx, q_pos, None,
                            sink_score, sink_selected=True,
                        ))
                        continue
                    best_score, best_pos = max(
                        scored_keys,
                        key=lambda item: (item[0], item[1]),
                    )
                    if best_score <= sink_score:
                        choices.append(SymbolicAttentionChoice(
                            layer_idx, spec.head_idx, q_pos, None,
                            best_score, sink_selected=True,
                        ))
                        continue
                    values = _attention_projection_slots(
                        state, best_pos, spec.v
                    )
                    _apply_attention_output(out, q_pos, values, spec.o)
                    choices.append(SymbolicAttentionChoice(
                        layer_idx, spec.head_idx, q_pos, best_pos, best_score
                    ))
                    continue

                values_accum: Dict[int, float] = {}
                if scored_keys:
                    max_score = max(
                        max(score for score, _ in scored_keys),
                        sink_score,
                    )
                    denom = math.exp(sink_score - max_score)
                    exp_scores = []
                    for score, key_pos in scored_keys:
                        weight = math.exp(score - max_score)
                        exp_scores.append((weight, key_pos, score))
                        denom += weight
                    for weight, key_pos, _ in exp_scores:
                        values = _attention_projection_slots(
                            state, key_pos, spec.v
                        )
                        for slot, value in values.items():
                            values_accum[slot] = (
                                values_accum.get(slot, 0.0)
                                + (weight / denom) * value
                            )
                    best_score, best_pos = max(
                        scored_keys,
                        key=lambda item: (item[0], item[1]),
                    )
                    choices.append(SymbolicAttentionChoice(
                        layer_idx,
                        spec.head_idx,
                        q_pos,
                        best_pos if best_score > sink_score else None,
                        best_score,
                        sink_selected=best_score <= sink_score,
                    ))
                _apply_attention_output(out, q_pos, values_accum, spec.o)

        return out, choices

    def symbolic_ffn_positions(
        self,
        state: SymbolicResidualState,
        dim_positions: Mapping[str, int],
        *,
        layer_idx: int = 0,
    ) -> SymbolicResidualState:
        """Symbolically apply one layer's FFN rules to every token position."""

        out = state.copy()
        for pos_idx in range(len(state)):
            for rule in self.layer(layer_idx).ffn.rules:
                score = sum(
                    state.get(pos_idx, term.dim.resolve(dim_positions))
                    * term.weight
                    for term in rule.conditions
                )
                if score < rule.threshold:
                    continue
                gate_value = rule.gate_bias
                if rule.gate is not None:
                    gate_value += (
                        state.get(pos_idx, rule.gate.resolve(dim_positions))
                        * rule.gate_weight
                    )
                gate_value += sum(
                    state.get(pos_idx, term.dim.resolve(dim_positions))
                    * term.weight
                    for term in rule.gate_terms
                )
                for write in rule.writes:
                    out.add(
                        pos_idx,
                        write.dim.resolve(dim_positions),
                        gate_value * write.weight,
                    )
        return out


def compare_symbolic_to_lowered_ffn(
    ir_or_rule,
    dim_positions: Mapping[str, int],
    state: Optional[Mapping[str, float]] = None,
    *,
    layer_idx: int = 0,
    ffn=None,
    lower: bool = True,
    start_unit: int = 0,
    dim: Optional[int] = None,
    S: float = 100.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> FFNComparisonReport:
    """Compare symbolic FFN semantics with a lowered ``PureFFN`` forward.

    ``ir_or_rule`` may be a ``CompilerIR`` or a single ``FFNRule``. When
    ``ffn`` is omitted, the helper creates a fresh ``PureFFN`` and lowers the
    selected rules into it. When ``ffn`` is supplied, set ``lower=False`` to
    validate and run an already-lowered module.

    Failure kinds are intentionally coarse:
    ``declaration_semantics`` means the IR references unresolved dimensions;
    ``lowering`` means the PureFFN weights do not match the IR lowering
    contract; ``weight_output_mismatch`` means the weights match the contract
    but the PureFFN output disagrees with symbolic execution.
    """
    ir = _coerce_compiler_ir(ir_or_rule, layer_idx=layer_idx)
    report = FFNComparisonReport(ok=False, layer_idx=layer_idx)
    rules = ir.layer(layer_idx).ffn.rules
    normalized_state = _normalize_state(
        _synthetic_ffn_state(rules, S=S) if state is None else state
    )

    declaration_issues = _validate_ffn_declarations(
        rules, dim_positions, normalized_state
    )
    if declaration_issues:
        report.issues.extend(declaration_issues)
        return report

    resolved_dim = _resolve_model_dim(
        rules, dim_positions, normalized_state, dim=dim, ffn=ffn
    )

    if ffn is None:
        from c4_release.neural_vm.base_layers import PureFFN

        ffn = PureFFN(
            dim=resolved_dim,
            hidden_dim=start_unit + ir.required_ffn_units(layer_idx=layer_idx),
        )

    if lower:
        ir.lower_ffn(
            ffn,
            dim_positions,
            layer_idx=layer_idx,
            start_unit=start_unit,
            S=S,
        )

    lowering_issues = _validate_lowered_ffn(
        rules,
        ffn,
        dim_positions,
        start_unit=start_unit,
        S=S,
        atol=atol,
        rtol=rtol,
    )
    if lowering_issues:
        report.issues.extend(lowering_issues)
        return report

    try:
        symbolic_state = ir.symbolic_ffn(normalized_state, layer_idx=layer_idx)
    except Exception as exc:
        report.issues.append(FFNComparisonIssue(
            "declaration_semantics",
            f"symbolic execution failed: {exc!r}",
        ))
        return report

    lowered_state, output_issues = _run_lowered_ffn_comparison(
        ffn,
        dim_positions,
        normalized_state,
        symbolic_state,
        rules,
        dim=resolved_dim,
        atol=atol,
        rtol=rtol,
    )
    report.symbolic_state = symbolic_state
    report.lowered_state = lowered_state
    report.issues.extend(output_issues)
    report.ok = not report.issues
    return report


def _coerce_compiler_ir(ir_or_rule, *, layer_idx: int) -> CompilerIR:
    if isinstance(ir_or_rule, CompilerIR):
        return ir_or_rule
    if isinstance(ir_or_rule, FFNRule):
        ir = CompilerIR()
        ir.layer(layer_idx).ffn.append(ir_or_rule)
        return ir
    raise TypeError(
        "compare_symbolic_to_lowered_ffn expects CompilerIR or FFNRule"
    )


def _coerce_attention_head_ir(
    spec_or_head,
    *,
    name: Optional[str],
    metadata: Optional[Mapping[str, object]],
) -> AttentionHeadIR:
    if isinstance(spec_or_head, AttentionHeadIR):
        if name is None and metadata is None:
            return spec_or_head
        return AttentionHeadIR(
            spec=spec_or_head.spec,
            name=name if name is not None else spec_or_head.name,
            metadata=(
                dict(spec_or_head.metadata)
                if metadata is None
                else dict(metadata)
            ),
        )

    from .primitives import DeclarativeAttentionHeadSpec

    if isinstance(spec_or_head, DeclarativeAttentionHeadSpec):
        return AttentionHeadIR(
            spec=spec_or_head,
            name=name,
            metadata={} if metadata is None else dict(metadata),
        )
    raise TypeError(
        "AttentionOp expects AttentionHeadIR or DeclarativeAttentionHeadSpec"
    )


def _state_value(state: Mapping[str, float], dim: DimRef) -> float:
    key = dim.key()
    if key in state:
        return float(state[key])
    if dim.offset == 0 and dim.name in state:
        return float(state[dim.name])
    return 0.0


def _absolute_pos(state: SymbolicResidualState, pos_idx: int) -> int:
    if state.pos_ids is None:
        return pos_idx
    return int(state.pos_ids[pos_idx])


def _attention_projection_slots(
    state: SymbolicResidualState,
    pos_idx: int,
    writes: Sequence[object],
) -> Dict[int, float]:
    slots: Dict[int, float] = {}
    for write in writes:
        value = state.get(pos_idx, write.dim) * float(write.weight)
        if value == 0.0:
            continue
        slots[write.slot] = slots.get(write.slot, 0.0) + value
    return slots


def _slot_dot(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(slot, 0.0) for slot, value in left.items())


def _apply_attention_output(
    state: SymbolicResidualState,
    pos_idx: int,
    values: Mapping[int, float],
    writes: Sequence[object],
) -> None:
    for write in writes:
        state.add(
            pos_idx,
            write.out_dim,
            values.get(write.slot, 0.0) * float(write.weight),
        )


def _operation_compiler_ir(op, *, dim_positions, head_dim: int):
    op_ir = getattr(op, "compiler_ir", None)
    if op_ir is not None:
        return op_ir
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is None:
        return None
    return factory(dim_positions, head_dim)


def _dim_key_from_position(
    dim: int,
    dim_positions: Mapping[str, int],
    dim_sizes: Optional[Mapping[str, int]],
) -> str:
    if dim_sizes is not None:
        matches = [
            (start, name, dim_sizes.get(name, 1))
            for name, start in dim_positions.items()
            if start <= dim < start + dim_sizes.get(name, 1)
        ]
    else:
        matches = [
            (start, name, max(1, dim - start + 1))
            for name, start in dim_positions.items()
            if start <= dim
        ]
    if not matches:
        return f"<dim:{dim}>"
    start, name, _ = max(matches, key=lambda item: item[0])
    return f"{name}+{dim - start}"


def _normalize_state(state: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in state.items():
        dim = DimRef.parse(key)
        out[dim.key()] = float(value)
    return out


def _synthetic_ffn_state(rules: Sequence[FFNRule], *, S: float) -> Dict[str, float]:
    state: Dict[str, float] = {}
    margin = _SILU_ONE_INPUT / S
    for rule in rules:
        for term in rule.conditions:
            key = term.dim.key()
            if key not in state:
                state[key] = 0.0
        if rule.gate is not None:
            state.setdefault(rule.gate.key(), 1.0)
        for term in rule.gate_terms:
            state.setdefault(term.dim.key(), 1.0)

        score = sum(state.get(term.dim.key(), 0.0) * term.weight
                    for term in rule.conditions)
        needed = rule.threshold + margin - score
        for term in rule.conditions:
            if term.weight == 0.0:
                continue
            key = term.dim.key()
            state[key] = state.get(key, 0.0) + needed / term.weight
            break
    return state


def _validate_ffn_declarations(
    rules: Sequence[FFNRule],
    dim_positions: Mapping[str, int],
    state: Mapping[str, float],
) -> List[FFNComparisonIssue]:
    issues: List[FFNComparisonIssue] = []
    for rule_idx, rule in enumerate(rules):
        label = rule.name or f"rule_{rule_idx}"
        for role, dims in (
            ("condition", [term.dim for term in rule.conditions]),
            ("write", [term.dim for term in rule.writes]),
            (
                "gate",
                ([rule.gate] if rule.gate is not None else [])
                + [term.dim for term in rule.gate_terms],
            ),
        ):
            for dim in dims:
                if dim.name not in dim_positions:
                    issues.append(FFNComparisonIssue(
                        "declaration_semantics",
                        f"{label} {role} references undeclared dim "
                        f"{dim.name!r}",
                    ))
    for key in state:
        dim = DimRef.parse(key)
        if dim.name not in dim_positions:
            issues.append(FFNComparisonIssue(
                "declaration_semantics",
                f"state references undeclared dim {dim.name!r}",
            ))
    return issues


def _resolve_model_dim(
    rules: Sequence[FFNRule],
    dim_positions: Mapping[str, int],
    state: Mapping[str, float],
    *,
    dim: Optional[int],
    ffn,
) -> int:
    if dim is not None:
        return dim
    if ffn is not None:
        return int(ffn.W_up.shape[1])

    max_pos = -1
    for pos in dim_positions.values():
        max_pos = max(max_pos, int(pos))
    for rule in rules:
        for ref in _rule_dim_refs(rule):
            max_pos = max(max_pos, ref.resolve(dim_positions))
    for key in state:
        max_pos = max(max_pos, DimRef.parse(key).resolve(dim_positions))
    return max_pos + 1


def _rule_dim_refs(rule: FFNRule) -> Iterable[DimRef]:
    for term in rule.conditions:
        yield term.dim
    if rule.gate is not None:
        yield rule.gate
    for term in rule.gate_terms:
        yield term.dim
    for write in rule.writes:
        yield write.dim


def _validate_lowered_ffn(
    rules: Sequence[FFNRule],
    ffn,
    dim_positions: Mapping[str, int],
    *,
    start_unit: int,
    S: float,
    atol: float,
    rtol: float,
) -> List[FFNComparisonIssue]:
    issues: List[FFNComparisonIssue] = []
    hidden_dim = int(ffn.W_up.shape[0])
    model_dim = int(ffn.W_up.shape[1])
    for rule_idx, rule in enumerate(rules):
        unit = start_unit + rule_idx
        label = rule.name or f"rule_{rule_idx}"
        if unit >= hidden_dim:
            issues.append(FFNComparisonIssue(
                "lowering",
                f"{label} needs hidden unit {unit}, but ffn has "
                f"{hidden_dim} units",
            ))
            continue

        expected_up = [0.0] * model_dim
        expected_gate = [0.0] * model_dim
        expected_down = [0.0] * model_dim
        for term in rule.conditions:
            expected_up[term.dim.resolve(dim_positions)] += S * term.weight
        if rule.gate is not None:
            expected_gate[rule.gate.resolve(dim_positions)] += rule.gate_weight
        for term in rule.gate_terms:
            expected_gate[term.dim.resolve(dim_positions)] += term.weight
        for write in rule.writes:
            expected_down[write.dim.resolve(dim_positions)] += write.weight

        _check_vector(
            issues, ffn.W_up.detach()[unit], expected_up,
            f"{label} W_up[{unit}]", "lowering", atol, rtol,
        )
        _check_scalar(
            issues, float(ffn.b_up.detach()[unit]),
            -S * rule.threshold,
            f"{label} b_up[{unit}]", "lowering", atol, rtol,
        )
        _check_vector(
            issues, ffn.W_gate.detach()[unit], expected_gate,
            f"{label} W_gate[{unit}]", "lowering", atol, rtol,
        )
        _check_scalar(
            issues, float(ffn.b_gate.detach()[unit]),
            rule.gate_bias,
            f"{label} b_gate[{unit}]", "lowering", atol, rtol,
        )
        _check_vector(
            issues, ffn.W_down.detach()[:, unit], expected_down,
            f"{label} W_down[:, {unit}]", "lowering", atol, rtol,
        )
    return issues


def _check_scalar(
    issues: List[FFNComparisonIssue],
    observed: float,
    expected: float,
    label: str,
    kind: str,
    atol: float,
    rtol: float,
) -> None:
    if abs(observed - expected) <= atol + rtol * abs(expected):
        return
    issues.append(FFNComparisonIssue(
        kind,
        f"{label}: expected {expected:.8g}, observed {observed:.8g}",
    ))


def _check_vector(
    issues: List[FFNComparisonIssue],
    observed,
    expected: Sequence[float],
    label: str,
    kind: str,
    atol: float,
    rtol: float,
) -> None:
    for idx, expected_value in enumerate(expected):
        observed_value = float(observed[idx])
        if abs(observed_value - expected_value) <= (
            atol + rtol * abs(expected_value)
        ):
            continue
        issues.append(FFNComparisonIssue(
            kind,
            f"{label}[{idx}]: expected {expected_value:.8g}, "
            f"observed {observed_value:.8g}",
        ))


def _run_lowered_ffn_comparison(
    ffn,
    dim_positions: Mapping[str, int],
    state: Mapping[str, float],
    symbolic_state: Mapping[str, float],
    rules: Sequence[FFNRule],
    *,
    dim: int,
    atol: float,
    rtol: float,
) -> Tuple[Dict[str, float], List[FFNComparisonIssue]]:
    import torch

    x = torch.zeros(1, 1, dim, dtype=ffn.W_up.dtype, device=ffn.W_up.device)
    for key, value in state.items():
        pos = DimRef.parse(key).resolve(dim_positions)
        x[..., pos] = value

    with torch.no_grad():
        y = ffn(x)

    keys = set(symbolic_state) | set(state)
    for rule in rules:
        for ref in _rule_dim_refs(rule):
            keys.add(ref.key())

    lowered_state: Dict[str, float] = {}
    issues: List[FFNComparisonIssue] = []
    expected_positions = set()
    for key in sorted(keys):
        pos = DimRef.parse(key).resolve(dim_positions)
        expected_positions.add(pos)
        expected = float(symbolic_state.get(key, state.get(key, 0.0)))
        observed = float(y[..., pos].item())
        lowered_state[key] = observed
        if abs(observed - expected) <= atol + rtol * abs(expected):
            continue
        issues.append(FFNComparisonIssue(
            "weight_output_mismatch",
            f"{key}: symbolic expected {expected:.8g}, "
            f"lowered observed {observed:.8g}",
        ))

    delta = (y - x).detach().abs().reshape(-1)
    for pos, value in enumerate(delta.tolist()):
        if pos in expected_positions or value <= atol:
            continue
        issues.append(FFNComparisonIssue(
            "weight_output_mismatch",
            f"residual position {pos}: unexpected lowered delta "
            f"{value:.8g}",
        ))
    return lowered_state, issues


__all__ = [
    "AttentionDebugReport",
    "AttentionHeadIR",
    "AttentionHeadReport",
    "AttentionMatrixWrite",
    "AttentionOp",
    "CompilerIR",
    "ConditionTerm",
    "DimRef",
    "FFNComparisonIssue",
    "FFNComparisonReport",
    "FFNOp",
    "FFNRule",
    "LayerSpec",
    "SymbolicAttentionChoice",
    "SymbolicDeclarativeRunReport",
    "SymbolicDeclarativeRunner",
    "SymbolicResidualState",
    "WriteTerm",
    "compare_symbolic_to_lowered_ffn",
]
