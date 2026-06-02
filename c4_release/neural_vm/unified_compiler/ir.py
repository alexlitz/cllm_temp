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
    scope: Optional[str] = None  # NEW: predicate-DSL string declaring
    # positions where this rule is intended to fire. Checked by F-7's
    # verify_rule_scopes. None means "no declared scope — verifier won't
    # check".
    dominates_at: Optional[Mapping[str, str]] = None  # NEW: per-output-dim
    # dominance scope. Maps output_dim_name → predicate string. If None,
    # falls back to scope for all of rule.writes.

    def dominates_at_for(self, output_dim_name: str) -> Optional[str]:
        """Return the dominance scope predicate for ``output_dim_name``,
        or the rule's general ``scope`` if ``dominates_at`` doesn't specify
        one, or ``None`` if no scope at all."""
        if (
            self.dominates_at is not None
            and output_dim_name in self.dominates_at
        ):
            return self.dominates_at[output_dim_name]
        return self.scope

    @classmethod
    def constant_write(
        cls,
        *,
        conditions: Sequence[Tuple[str, float]],
        threshold: float,
        writes: Sequence[Tuple[str, float]],
        name: Optional[str] = None,
        scope: Optional[str] = None,
        dominates_at: Optional[Mapping[str, str]] = None,
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
            scope=scope,
            dominates_at=dominates_at,
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
        scope: Optional[str] = None,
        dominates_at: Optional[Mapping[str, str]] = None,
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
            scope=scope,
            dominates_at=dominates_at,
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


@dataclass(frozen=True)
class AttentionComparisonIssue:
    """One classified symbolic-vs-lowered attention comparison failure."""

    kind: str
    message: str


@dataclass
class AttentionComparisonReport:
    """Diagnostic result for ``compare_symbolic_to_lowered_attn``.

    Mirror of :class:`FFNComparisonReport` for attention heads. ``issues`` is
    a coarse classification:

    * ``declaration_semantics`` — the head spec references slots/dims that
      cannot be lowered into a ``PureAttention`` of the resolved shape.
    * ``lowering`` — the lowered Q/K/V/O matrices do not match the values the
      ``AttentionHeadIR`` would produce via ``Primitives.generate_attention_head``.
    * ``weight_output_mismatch`` — the matrices match the lowering contract
      but ``PureAttention.forward`` disagrees with the symbolic execution on
      at least one ``(query_pos, output_dim)``.
    """

    ok: bool
    issues: List[AttentionComparisonIssue] = field(default_factory=list)
    symbolic_state: Optional["SymbolicResidualState"] = None
    lowered_state: Optional["SymbolicResidualState"] = None
    layer_idx: int = 0
    head_dim: int = 0
    num_heads: int = 0

    @property
    def failure_kinds(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(issue.kind for issue in self.issues))

    @property
    def primary_failure_kind(self) -> Optional[str]:
        return self.issues[0].kind if self.issues else None

    def format(self) -> str:
        status = "OK" if self.ok else "DRIFT"
        lines = [
            f"=== CompilerIR attention comparison "
            f"({status}, layer={self.layer_idx}, "
            f"heads={self.num_heads}, HD={self.head_dim}) ==="
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


@dataclass(frozen=True)
class TokenEmbeddingRule:
    """One declarative write into a model-level token table.

    ``TokenEmbeddingRule`` is the embedding/head-bake analogue of
    :class:`FFNRule`: instead of conditionally adding values to one residual
    position based on other residual cells, it unconditionally writes a fixed
    set of ``(dim, value)`` pairs into a per-token row of a model-level table
    (``model.embed.embed.weight`` or ``model.head.weight``), or into a per-token
    scalar (``model.head.bias``).

    Three ``target`` values are supported:

    * ``embed`` — writes into ``model.embed.embed.weight[token_ids, dim]``.
      ``writes`` is a tuple of ``(dim_name+offset, value)`` pairs resolved via
      ``dim_positions`` (just like :class:`FFNRule.writes`).
    * ``head_weight`` — writes into ``model.head.weight[token_ids, dim]``.
      ``writes`` uses the same ``(dim_name+offset, value)`` form.
    * ``head_bias`` — writes into ``model.head.bias[token_ids]``. ``writes`` is
      either empty or a single ``("", value)`` entry; the convenience
      constructor :meth:`bias_write` builds the canonical form.

    A rule fans out across every token in ``token_ids``. The lowering is
    accumulative (``+=``) so multiple rules may target the same cell, mirroring
    how :class:`FFNRule` lowering accumulates ``W_up``/``W_down`` contributions.
    """

    target: str  # Literal["embed", "head_weight", "head_bias"]
    token_ids: Tuple[int, ...]
    writes: Tuple[WriteTerm, ...]
    name: Optional[str] = None

    @classmethod
    def embed_write(
        cls,
        *,
        token_ids,
        writes: Sequence[Tuple[str, float]],
        name: Optional[str] = None,
    ) -> "TokenEmbeddingRule":
        return cls(
            target="embed",
            token_ids=_coerce_token_ids(token_ids),
            writes=tuple(
                WriteTerm(DimRef.parse(dim), float(weight))
                for dim, weight in writes
            ),
            name=name,
        )

    @classmethod
    def head_weight_write(
        cls,
        *,
        token_ids,
        writes: Sequence[Tuple[str, float]],
        name: Optional[str] = None,
    ) -> "TokenEmbeddingRule":
        return cls(
            target="head_weight",
            token_ids=_coerce_token_ids(token_ids),
            writes=tuple(
                WriteTerm(DimRef.parse(dim), float(weight))
                for dim, weight in writes
            ),
            name=name,
        )

    @classmethod
    def head_bias_write(
        cls,
        *,
        token_ids,
        value: float,
        name: Optional[str] = None,
    ) -> "TokenEmbeddingRule":
        return cls(
            target="head_bias",
            token_ids=_coerce_token_ids(token_ids),
            writes=(WriteTerm(DimRef("", 0), float(value)),),
            name=name,
        )

    def __post_init__(self):
        if self.target not in ("embed", "head_weight", "head_bias"):
            raise ValueError(
                f"TokenEmbeddingRule.target must be one of "
                f"'embed' | 'head_weight' | 'head_bias', got {self.target!r}"
            )
        if self.target == "head_bias":
            if len(self.writes) != 1 or self.writes[0].dim.name != "":
                raise ValueError(
                    "TokenEmbeddingRule(target='head_bias') expects exactly "
                    "one write with an empty DimRef; use "
                    "TokenEmbeddingRule.head_bias_write"
                )


def _coerce_token_ids(token_ids) -> Tuple[int, ...]:
    if isinstance(token_ids, int):
        return (int(token_ids),)
    return tuple(int(t) for t in token_ids)


@dataclass(frozen=True)
class TokenEmbeddingComparisonIssue:
    """One classified symbolic-vs-lowered token-embedding failure."""

    kind: str
    message: str


@dataclass
class TokenEmbeddingComparisonReport:
    """Diagnostic result for ``compare_symbolic_to_lowered_embedding``."""

    ok: bool
    issues: List[TokenEmbeddingComparisonIssue] = field(default_factory=list)

    @property
    def failure_kinds(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(issue.kind for issue in self.issues))

    @property
    def primary_failure_kind(self) -> Optional[str]:
        return self.issues[0].kind if self.issues else None

    def format(self) -> str:
        status = "OK" if self.ok else "DRIFT"
        lines = [f"=== CompilerIR token-embedding comparison ({status}) ==="]
        for issue in self.issues:
            lines.append(f"  [{issue.kind}] {issue.message}")
        return "\n".join(lines)


@dataclass
class LayerSpec:
    """Declarative work assigned to one logical compiler layer."""

    ffn: FFNOp = field(default_factory=FFNOp)
    attention: AttentionOp = field(default_factory=AttentionOp)


@dataclass
class CompilerIR:
    """A symbolic-and-lowerable program fragment."""

    layers: List[LayerSpec] = field(default_factory=list)
    embeddings: List[TokenEmbeddingRule] = field(default_factory=list)

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
                ffn.W_up.data[unit, term.dim.resolve(dim_positions)] += (
                    S * term.weight
                )
            ffn.b_up.data[unit] = -S * rule.threshold

            if rule.gate is None:
                ffn.b_gate.data[unit] = rule.gate_bias
            else:
                ffn.W_gate.data[unit, rule.gate.resolve(dim_positions)] += (
                    rule.gate_weight
                )
                ffn.b_gate.data[unit] = rule.gate_bias
            for term in rule.gate_terms:
                ffn.W_gate.data[unit, term.dim.resolve(dim_positions)] += (
                    term.weight
                )

            for write in rule.writes:
                ffn.W_down.data[write.dim.resolve(dim_positions), unit] += (
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

    def lower_token_embeddings(
        self,
        model,
        dim_positions: Mapping[str, int],
        *,
        rules: Optional[Sequence[TokenEmbeddingRule]] = None,
    ) -> int:
        """Lower model-level :class:`TokenEmbeddingRule` entries into ``model``.

        Mirrors :meth:`lower_ffn` / :meth:`lower_attention` for the embedding
        / head bakes. ``model`` must expose ``model.embed.embed.weight``,
        ``model.head.weight``, and ``model.head.bias`` (the standard
        ``NeuralVMEmbedding`` + ``nn.Linear`` head shape used throughout this
        codebase).

        When ``rules`` is None the IR's own ``embeddings`` list is used. Returns
        the number of (rule, token_id) write batches applied — useful as a
        sanity check.
        """

        chosen = list(self.embeddings if rules is None else rules)
        applied = 0
        embed_weight = model.embed.embed.weight if chosen else None
        head_weight = getattr(model, "head", None)
        for rule in chosen:
            for token_id in rule.token_ids:
                if rule.target == "embed":
                    for write in rule.writes:
                        col = write.dim.resolve(dim_positions)
                        embed_weight.data[token_id, col] += float(write.weight)
                elif rule.target == "head_weight":
                    for write in rule.writes:
                        col = write.dim.resolve(dim_positions)
                        head_weight.weight.data[token_id, col] += float(write.weight)
                elif rule.target == "head_bias":
                    # Validated by __post_init__ to be one bias write.
                    head_weight.bias.data[token_id] += float(rule.writes[0].weight)
                else:  # pragma: no cover — guarded in __post_init__
                    raise ValueError(f"unknown target {rule.target!r}")
                applied += 1
        return applied

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


def compare_symbolic_to_lowered_attn(
    ir_or_head,
    head_dim: int,
    *,
    layer_idx: int = 0,
    attn=None,
    lower: bool = True,
    num_heads: Optional[int] = None,
    dim: Optional[int] = None,
    state: Optional["SymbolicResidualState"] = None,
    n_positions: int = 2,
    causal: bool = True,
    score_amplitude: float = 50.0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> AttentionComparisonReport:
    """Compare symbolic attention semantics with a lowered ``PureAttention`` forward.

    ``ir_or_head`` may be a ``CompilerIR``, an :class:`AttentionHeadIR`, or a
    raw ``DeclarativeAttentionHeadSpec`` (the latter two are wrapped into a
    single-head ``CompilerIR`` at ``layer_idx``). When ``attn`` is omitted the
    helper allocates a fresh ``PureAttention(dim, num_heads)`` and lowers the
    selected heads into it. Pass an already-baked ``PureAttention`` together
    with ``lower=False`` to validate an external module.

    The comparison is the attention analogue of
    :func:`compare_symbolic_to_lowered_ffn`:

    1. Validate that the heads only reference slots in ``range(head_dim)`` and
       Q/K/V/W_o positions that fit in ``num_heads * head_dim``. Mismatches
       are reported as ``declaration_semantics``.
    2. After lowering, check that each ``W_q[base+slot, dim]``,
       ``W_k[..., dim]``, ``W_v[..., dim]`` and ``W_o[out_dim, base+slot]``
       equals the IR-declared value. Mismatches are ``lowering`` failures.
    3. Build a synthetic ``SymbolicResidualState`` (or accept one) that fires
       every head with high amplitude (so softmax converges to hardmax), then
       diff per ``(query_pos, output_dim)`` between
       ``symbolic_attention_positions`` (hardmax) and ``PureAttention(x)``.
       Mismatches are ``weight_output_mismatch`` failures.

    The amplitude knob ``score_amplitude`` controls how strongly the synthetic
    Q row dominates the softmax distribution. The defaults give roughly
    ``softmax_winner / softmax_runner_up ~ exp(score_amplitude / sqrt(HD))``.
    """

    ir = _coerce_compiler_ir_attn(ir_or_head, layer_idx=layer_idx)
    heads = ir.layer(layer_idx).attention.rules
    report = AttentionComparisonReport(
        ok=False, layer_idx=layer_idx, head_dim=head_dim
    )

    if not heads:
        report.ok = True
        return report

    resolved_num_heads, resolved_dim = _resolve_attn_shape(
        heads, head_dim, num_heads=num_heads, dim=dim, attn=attn,
    )
    report.num_heads = resolved_num_heads

    declaration_issues = _validate_attn_declarations(
        heads, head_dim=head_dim, num_heads=resolved_num_heads,
        model_dim=resolved_dim,
    )
    if declaration_issues:
        report.issues.extend(declaration_issues)
        return report

    if attn is None:
        from c4_release.neural_vm.base_layers import PureAttention

        attn = PureAttention(
            dim=resolved_dim,
            num_heads=resolved_num_heads,
            causal=causal,
        )

    if lower:
        ir.lower_attention(attn, head_dim, layer_idx=layer_idx)

    lowering_issues = _validate_lowered_attn(
        heads,
        attn,
        head_dim=head_dim,
        atol=atol,
        rtol=rtol,
    )
    if lowering_issues:
        report.issues.extend(lowering_issues)
        return report

    if state is None:
        state = _synthetic_attention_state(
            heads,
            head_dim=head_dim,
            n_positions=n_positions,
            score_amplitude=score_amplitude,
        )

    try:
        # PureAttention.forward has no softmax1 sink; symbolic hardmax must
        # therefore disable its sink (``sink_score=-inf``) so every query
        # position picks a real key, matching what plain softmax does on a
        # masked single-key row.
        symbolic_state, _choices = ir.symbolic_attention_positions(
            state,
            head_dim,
            layer_idx=layer_idx,
            causal=causal,
            mode="hardmax",
            sink_score=-math.inf,
        )
    except Exception as exc:
        report.issues.append(AttentionComparisonIssue(
            "declaration_semantics",
            f"symbolic execution failed: {exc!r}",
        ))
        return report

    lowered_state, output_issues = _run_lowered_attn_comparison(
        attn,
        state,
        symbolic_state,
        heads,
        head_dim=head_dim,
        model_dim=resolved_dim,
        causal=causal,
        atol=atol,
        rtol=rtol,
    )
    report.symbolic_state = symbolic_state
    report.lowered_state = lowered_state
    report.issues.extend(output_issues)
    report.ok = not report.issues
    return report


def compare_symbolic_to_lowered_embedding(
    ir_or_rule,
    dim_positions: Mapping[str, int],
    *,
    model=None,
    vocab_size: Optional[int] = None,
    d_model: Optional[int] = None,
    lower: bool = True,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> TokenEmbeddingComparisonReport:
    """Compare declared :class:`TokenEmbeddingRule` writes with lowered weights.

    The token-embedding analogue of :func:`compare_symbolic_to_lowered_ffn`.
    For each rule we expect the lowered model to contain ``sum(rule.weight)``
    at every ``(token, resolved_dim)`` it targets — accumulated across rules
    that hit the same cell, exactly as ``lower_token_embeddings`` accumulates.

    Failure kinds:

    * ``declaration_semantics`` — a rule references a dim not in
      ``dim_positions``, or a ``token_id`` outside the model's vocab range.
    * ``lowering`` — the lowered weights (``model.embed.embed.weight`` /
      ``model.head.weight`` / ``model.head.bias``) disagree with the IR sum.
    """

    ir = _coerce_compiler_ir_embedding(ir_or_rule)
    report = TokenEmbeddingComparisonReport(ok=False)
    rules = list(ir.embeddings)
    if not rules:
        report.ok = True
        return report

    decl_issues = _validate_embedding_declarations(
        rules,
        dim_positions,
        vocab_size=vocab_size,
    )
    if decl_issues:
        report.issues.extend(decl_issues)
        return report

    resolved_vocab, resolved_d_model = _resolve_embedding_shape(
        rules,
        dim_positions,
        vocab_size=vocab_size,
        d_model=d_model,
        model=model,
    )

    if model is None:
        model = _build_synthetic_embed_model(resolved_vocab, resolved_d_model)

    if lower:
        ir.lower_token_embeddings(model, dim_positions)

    lowering_issues = _validate_lowered_embedding(
        rules,
        model,
        dim_positions,
        atol=atol,
        rtol=rtol,
    )
    if lowering_issues:
        report.issues.extend(lowering_issues)
        return report

    report.ok = True
    return report


def _coerce_compiler_ir_embedding(ir_or_rule) -> CompilerIR:
    if isinstance(ir_or_rule, CompilerIR):
        return ir_or_rule
    if isinstance(ir_or_rule, TokenEmbeddingRule):
        ir = CompilerIR()
        ir.embeddings.append(ir_or_rule)
        return ir
    raise TypeError(
        "compare_symbolic_to_lowered_embedding expects CompilerIR or "
        "TokenEmbeddingRule"
    )


def _validate_embedding_declarations(
    rules: Sequence[TokenEmbeddingRule],
    dim_positions: Mapping[str, int],
    *,
    vocab_size: Optional[int],
) -> List[TokenEmbeddingComparisonIssue]:
    issues: List[TokenEmbeddingComparisonIssue] = []
    for rule_idx, rule in enumerate(rules):
        label = rule.name or f"embed_rule_{rule_idx}"
        if rule.target in ("embed", "head_weight"):
            for write in rule.writes:
                if write.dim.name not in dim_positions:
                    issues.append(TokenEmbeddingComparisonIssue(
                        "declaration_semantics",
                        f"{label} write references undeclared dim "
                        f"{write.dim.name!r}",
                    ))
        if vocab_size is not None:
            for token_id in rule.token_ids:
                if not (0 <= int(token_id) < int(vocab_size)):
                    issues.append(TokenEmbeddingComparisonIssue(
                        "declaration_semantics",
                        f"{label} token_id {token_id} out of vocab "
                        f"range [0, {vocab_size})",
                    ))
    return issues


def _resolve_embedding_shape(
    rules: Sequence[TokenEmbeddingRule],
    dim_positions: Mapping[str, int],
    *,
    vocab_size: Optional[int],
    d_model: Optional[int],
    model,
) -> Tuple[int, int]:
    if model is not None:
        embed_w = model.embed.embed.weight
        return int(embed_w.shape[0]), int(embed_w.shape[1])

    max_token = -1
    for rule in rules:
        for token_id in rule.token_ids:
            max_token = max(max_token, int(token_id))
    inferred_vocab = max_token + 1 if max_token >= 0 else 1
    resolved_vocab = vocab_size if vocab_size is not None else inferred_vocab

    max_dim = -1
    for pos in dim_positions.values():
        max_dim = max(max_dim, int(pos))
    for rule in rules:
        if rule.target == "head_bias":
            continue
        for write in rule.writes:
            if write.dim.name in dim_positions:
                max_dim = max(max_dim, write.dim.resolve(dim_positions))
    inferred_d_model = max_dim + 1 if max_dim >= 0 else 1
    resolved_d_model = d_model if d_model is not None else inferred_d_model
    return resolved_vocab, resolved_d_model


def _build_synthetic_embed_model(vocab_size: int, d_model: int):
    """Return a minimal duck-typed ``model`` with the fields lowering touches.

    Avoids importing the full ``NeuralVMEmbedding`` + ``AutoregressiveVM`` stack
    in unit tests / quick comparisons. The synthetic model exposes
    ``model.embed.embed.weight`` and ``model.head.{weight,bias}`` as plain
    ``torch.nn`` modules, which is what ``lower_token_embeddings`` writes into.
    """

    import torch
    import torch.nn as nn

    class _SyntheticEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            with torch.no_grad():
                self.embed.weight.zero_()

    class _SyntheticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = _SyntheticEmbed()
            self.head = nn.Linear(d_model, vocab_size)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.zero_()

    return _SyntheticModel()


def _validate_lowered_embedding(
    rules: Sequence[TokenEmbeddingRule],
    model,
    dim_positions: Mapping[str, int],
    *,
    atol: float,
    rtol: float,
) -> List[TokenEmbeddingComparisonIssue]:
    """Diff expected per-cell sums against the lowered weights."""

    embed_w = model.embed.embed.weight.detach()
    head_w = model.head.weight.detach()
    head_b = model.head.bias.detach()

    # Build expected accumulators keyed by (target, token, col_or_None).
    expected: Dict[Tuple[str, int, Optional[int]], float] = {}
    for rule in rules:
        for token_id in rule.token_ids:
            if rule.target == "head_bias":
                key = ("head_bias", int(token_id), None)
                expected[key] = expected.get(key, 0.0) + float(
                    rule.writes[0].weight
                )
                continue
            for write in rule.writes:
                col = write.dim.resolve(dim_positions)
                key = (rule.target, int(token_id), col)
                expected[key] = expected.get(key, 0.0) + float(write.weight)

    issues: List[TokenEmbeddingComparisonIssue] = []
    for (target, token_id, col), expected_value in expected.items():
        if target == "embed":
            observed = float(embed_w[token_id, col])
            label = f"embed[{token_id}, {col}]"
        elif target == "head_weight":
            observed = float(head_w[token_id, col])
            label = f"head.weight[{token_id}, {col}]"
        else:
            observed = float(head_b[token_id])
            label = f"head.bias[{token_id}]"
        if abs(observed - expected_value) <= atol + rtol * abs(expected_value):
            continue
        issues.append(TokenEmbeddingComparisonIssue(
            "lowering",
            f"{label}: expected {expected_value:.8g}, observed {observed:.8g}",
        ))
    return issues


def _coerce_compiler_ir_attn(ir_or_head, *, layer_idx: int) -> CompilerIR:
    if isinstance(ir_or_head, CompilerIR):
        return ir_or_head
    if isinstance(ir_or_head, AttentionHeadIR):
        ir = CompilerIR()
        ir.layer(layer_idx).attention.append(ir_or_head)
        return ir

    from .primitives import DeclarativeAttentionHeadSpec

    if isinstance(ir_or_head, DeclarativeAttentionHeadSpec):
        ir = CompilerIR()
        ir.layer(layer_idx).attention.append(ir_or_head)
        return ir
    raise TypeError(
        "compare_symbolic_to_lowered_attn expects CompilerIR, "
        "AttentionHeadIR, or DeclarativeAttentionHeadSpec"
    )


def _resolve_attn_shape(
    heads: Sequence["AttentionHeadIR"],
    head_dim: int,
    *,
    num_heads: Optional[int],
    dim: Optional[int],
    attn,
) -> Tuple[int, int]:
    if attn is not None:
        attn_dim = int(attn.W_q.shape[0])
        attn_heads = int(getattr(attn, "num_heads", attn_dim // head_dim))
        return attn_heads, attn_dim

    max_head_idx = -1
    max_residual_dim = -1
    for head in heads:
        spec = head.spec
        max_head_idx = max(max_head_idx, int(spec.head_idx))
        for write in spec.q + spec.k + spec.v:
            max_residual_dim = max(max_residual_dim, int(write.dim))
        for write in spec.o:
            max_residual_dim = max(max_residual_dim, int(write.out_dim))

    inferred_heads = max(max_head_idx + 1, 1)
    resolved_heads = num_heads if num_heads is not None else inferred_heads
    inferred_dim = max(
        resolved_heads * head_dim,
        max_residual_dim + 1 if max_residual_dim >= 0 else 0,
    )
    resolved_dim = dim if dim is not None else inferred_dim
    # PureAttention requires dim % num_heads == 0 with dim/num_heads == head_dim.
    if resolved_dim < resolved_heads * head_dim:
        resolved_dim = resolved_heads * head_dim
    return resolved_heads, resolved_dim


def _validate_attn_declarations(
    heads: Sequence["AttentionHeadIR"],
    *,
    head_dim: int,
    num_heads: int,
    model_dim: int,
) -> List[AttentionComparisonIssue]:
    issues: List[AttentionComparisonIssue] = []
    for h_idx, head in enumerate(heads):
        spec = head.spec
        label = head.name or f"head_{h_idx}"
        if not (0 <= int(spec.head_idx) < num_heads):
            issues.append(AttentionComparisonIssue(
                "declaration_semantics",
                f"{label}: head_idx {spec.head_idx} out of range "
                f"[0, {num_heads})",
            ))
            continue
        for role, writes in (
            ("q", spec.q), ("k", spec.k), ("v", spec.v),
        ):
            for write in writes:
                if not (0 <= int(write.slot) < head_dim):
                    issues.append(AttentionComparisonIssue(
                        "declaration_semantics",
                        f"{label} {role} write slot {write.slot} "
                        f"out of range [0, {head_dim})",
                    ))
                if not (0 <= int(write.dim) < model_dim):
                    issues.append(AttentionComparisonIssue(
                        "declaration_semantics",
                        f"{label} {role} write dim {write.dim} "
                        f"out of range [0, {model_dim})",
                    ))
        for write in spec.o:
            if not (0 <= int(write.slot) < head_dim):
                issues.append(AttentionComparisonIssue(
                    "declaration_semantics",
                    f"{label} o write slot {write.slot} "
                    f"out of range [0, {head_dim})",
                ))
            if not (0 <= int(write.out_dim) < model_dim):
                issues.append(AttentionComparisonIssue(
                    "declaration_semantics",
                    f"{label} o write out_dim {write.out_dim} "
                    f"out of range [0, {model_dim})",
                ))
    return issues


def _synthetic_attention_state(
    heads: Sequence["AttentionHeadIR"],
    *,
    head_dim: int,
    n_positions: int,
    score_amplitude: float,
) -> "SymbolicResidualState":
    """Build a synthetic ``SymbolicResidualState`` that fires every head.

    The state splits into two roles so symbolic hardmax and lowered softmax
    pick the same key for every query:

    * **Source** positions (everything except the last) hold ``1.0`` at every
      K-dim and ``1 + 0.1 * pos_idx`` at every V-dim. The per-position V
      scaling means each source position contributes a distinct value, so the
      comparison cannot mask a key-routing bug.
    * **Query** (final) position holds ``score_amplitude`` at every Q-dim and
      ``0.0`` everywhere else. With K-dims silent at the query row, no later
      row competes for the softmax mass and the lowered output collapses to
      the chosen source.

    When a dim plays multiple roles (e.g. ``q`` and ``k`` share a residual
    cell), the source value wins on source rows and the amplitude wins on the
    query row, matching the projection sums symbolic execution computes.
    """

    n_positions = max(2, int(n_positions))
    q_dims = set()
    k_dims = set()
    v_dims = set()
    for head in heads:
        spec = head.spec
        for write in spec.q:
            q_dims.add(int(write.dim))
        for write in spec.k:
            k_dims.add(int(write.dim))
        for write in spec.v:
            v_dims.add(int(write.dim))

    last = n_positions - 1
    positions: List[Dict[int, float]] = [{} for _ in range(n_positions)]
    for pos_idx in range(n_positions):
        if pos_idx == last:
            # Query row: only q-dims active.
            for dim in q_dims:
                positions[pos_idx][dim] = float(score_amplitude)
        else:
            v_scale = 1.0 + 0.1 * float(pos_idx)
            for dim in k_dims:
                positions[pos_idx][dim] = 1.0
            for dim in v_dims:
                # V wins ties with K when a dim plays both roles, so each
                # source row still produces a unique V signal.
                positions[pos_idx][dim] = float(v_scale)
    return SymbolicResidualState(positions)


def _validate_lowered_attn(
    heads: Sequence["AttentionHeadIR"],
    attn,
    *,
    head_dim: int,
    atol: float,
    rtol: float,
) -> List[AttentionComparisonIssue]:
    """Mirror ``_validate_lowered_ffn``: check Q/K/V/O weights match heads."""

    issues: List[AttentionComparisonIssue] = []
    for h_idx, head in enumerate(heads):
        spec = head.spec
        label = head.name or f"head_{h_idx}"
        base = int(spec.head_idx) * head_dim
        for role, writes, matrix_name in (
            ("q", spec.q, "W_q"),
            ("k", spec.k, "W_k"),
            ("v", spec.v, "W_v"),
        ):
            matrix = getattr(attn, matrix_name).detach()
            for write in writes:
                observed = float(matrix[base + int(write.slot), int(write.dim)])
                expected = float(write.weight)
                if abs(observed - expected) > (atol + rtol * abs(expected)):
                    issues.append(AttentionComparisonIssue(
                        "lowering",
                        f"{label} {matrix_name}[{base + int(write.slot)}, "
                        f"{int(write.dim)}]: expected {expected:.8g}, "
                        f"observed {observed:.8g}",
                    ))
        w_o = attn.W_o.detach()
        for write in spec.o:
            observed = float(w_o[int(write.out_dim), base + int(write.slot)])
            expected = float(write.weight)
            if abs(observed - expected) > (atol + rtol * abs(expected)):
                issues.append(AttentionComparisonIssue(
                    "lowering",
                    f"{label} W_o[{int(write.out_dim)}, "
                    f"{base + int(write.slot)}]: expected {expected:.8g}, "
                    f"observed {observed:.8g}",
                ))
    return issues


def _run_lowered_attn_comparison(
    attn,
    state: "SymbolicResidualState",
    symbolic_state: "SymbolicResidualState",
    heads: Sequence["AttentionHeadIR"],
    *,
    head_dim: int,
    model_dim: int,
    causal: bool,
    atol: float,
    rtol: float,
) -> Tuple["SymbolicResidualState", List[AttentionComparisonIssue]]:
    """Run ``PureAttention(x)`` and diff against symbolic per (pos, out_dim).

    Mirror of :func:`_run_lowered_ffn_comparison`. The lowered state is
    captured per-position via ``SymbolicResidualState`` so downstream tooling
    can pretty-print it.
    """

    import torch

    n_positions = len(state)
    x = torch.zeros(
        1, n_positions, model_dim,
        dtype=attn.W_q.dtype, device=attn.W_q.device,
    )
    for pos_idx in range(n_positions):
        for dim, value in state.positions[pos_idx].items():
            if 0 <= dim < model_dim:
                x[0, pos_idx, dim] = float(value)

    # Many PureAttention subclasses install a causal mask via the ``mask``
    # buffer (e.g. ``-inf`` above the diagonal). Our symbolic backend
    # implements causality with a key range cap, so we install a parallel
    # mask if the existing buffer is all zeros (the base PureAttention
    # default).
    mask = attn.mask
    seq_mask = mask[:n_positions, :n_positions]
    if causal and bool(torch.all(seq_mask == 0)):
        causal_mask = torch.full(
            (n_positions, n_positions), float("-inf"),
            dtype=mask.dtype, device=mask.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # Patch the slice in-place so PureAttention.forward sees the mask.
        with torch.no_grad():
            attn.mask[:n_positions, :n_positions] = causal_mask
        _restore_mask = True
    else:
        _restore_mask = False

    try:
        with torch.no_grad():
            y = attn(x)
    finally:
        if _restore_mask:
            with torch.no_grad():
                attn.mask[:n_positions, :n_positions] = 0.0

    # Collect every output_dim that any head writes to.
    output_dims = set()
    for head in heads:
        for write in head.spec.o:
            output_dims.add(int(write.out_dim))

    lowered_positions: List[Dict[int, float]] = []
    for pos_idx in range(n_positions):
        lowered_positions.append({
            int(d): float(y[0, pos_idx, d].item())
            for d in sorted(output_dims | set(state.positions[pos_idx].keys()))
        })
    lowered_state = SymbolicResidualState(lowered_positions)

    issues: List[AttentionComparisonIssue] = []
    for pos_idx in range(n_positions):
        for out_dim in sorted(output_dims):
            observed = float(y[0, pos_idx, out_dim].item())
            expected = float(symbolic_state.get(pos_idx, out_dim))
            tol = atol + rtol * abs(expected)
            if abs(observed - expected) <= tol:
                continue
            issues.append(AttentionComparisonIssue(
                "weight_output_mismatch",
                f"pos={pos_idx} out_dim={out_dim}: "
                f"symbolic expected {expected:.8g}, "
                f"lowered observed {observed:.8g}",
            ))
    return lowered_state, issues


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
    # ``_synthetic_ffn_state`` builds a state that satisfies every rule's
    # threshold by piling per-condition activations on a small set of dims.
    # For huge-rule layers (fan-in > a few hundred), those accumulated values
    # can exceed fp32 range and the cast into the lowered ``x`` tensor raises
    # ``RuntimeError: value cannot be converted to type float without
    # overflow``. That is a harness limitation, not a semantic bug in the IR,
    # so classify it as ``_synthetic_state_overflow`` and bail out before
    # running the lowered forward.
    try:
        for key, value in state.items():
            pos = DimRef.parse(key).resolve(dim_positions)
            x[..., pos] = value
    except (RuntimeError, OverflowError) as exc:
        return {}, [FFNComparisonIssue(
            "_synthetic_state_overflow",
            f"synthetic FFN state overflowed lowered fp32 input tensor "
            f"({type(exc).__name__}: {str(exc).splitlines()[0]}); "
            f"likely too many rules sharing a condition dim",
        )]

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
    "AttentionComparisonIssue",
    "AttentionComparisonReport",
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
    "TokenEmbeddingComparisonIssue",
    "TokenEmbeddingComparisonReport",
    "TokenEmbeddingRule",
    "WriteTerm",
    "compare_symbolic_to_lowered_attn",
    "compare_symbolic_to_lowered_embedding",
    "compare_symbolic_to_lowered_ffn",
]
