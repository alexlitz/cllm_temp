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

import enum
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    from .primitives import DeclarativeAttentionHeadSpec


_SILU_ONE_INPUT = 1.278464542761074


class StepWindowConstraint(enum.Enum):
    """Declares the VM-step window an attention head is allowed to read.

    The autoregressive VM emits ``Token.STEP_TOKENS = 35`` tokens per
    VM step (PC + AX + SP + BP + STACK0 + MEM + STEP_END). Compute is
    expected to fire at STEP_END within a single step's window:
    same-step compute reads same-step inputs, and memory persistence
    across steps is the exception (LI/SI memory lookups, MEM relays).

    Attention heads that mix tokens across multiple step windows
    without an ALiBi recency slope or an explicit step-boundary
    K-suppressor are vulnerable to cross-step dilution — the IMM
    relay bug at L8 head 4 is the canonical example: prior-step
    MARK_AX positions diluted the current step's relay below the
    multibyte_routing threshold until commit b23f818c added
    ``alibi_slope=0.5``.

    Values:

    * ``CURRENT_STEP_ONLY`` — head must only attend within its own
      35-token window. The verifier expects an ALiBi slope >=
      :data:`STEP_WINDOW_MIN_ALIBI_SLOPE` (so prior-step positions
      decay below the softmax mass) or an explicit step-boundary
      suppressor (a negative K weight on ``MARK_SE_ONLY`` /
      ``MARK_CS``). This is the default for compute-intent heads.
    * ``PREV_STEP_OK`` — head may attend to the immediately prior
      step's tokens (e.g. cross-step register relays). Verifier
      records this as informational; no slope requirement.
    * ``ANY_STEP`` — head may attend across all prior steps with no
      decay constraint. Used by memory-lookup heads (memory persists
      across steps by design). The verifier still warns when an
      ``ANY_STEP`` head looks compute-intent (heuristic: large relay
      V/O weights with no MEM marker reads).
    """

    CURRENT_STEP_ONLY = "current_step_only"
    PREV_STEP_OK = "prev_step_ok"
    ANY_STEP = "any_step"


# Minimum ALiBi slope considered "strong enough" to confine a head's
# attention mass to the current 35-token step window. Derived from
# Token.STEP_TOKENS=35 and the L8 op_imm_relay slope (0.5): at
# distance 35 the decay term ``0.5 * 35 = 17.5`` exceeds the relay K
# score gap (~30) enough that softmax mass on prior-step MARK_AX rows
# falls below 1%. Heads with smaller slopes will be flagged unless
# they carry a K-side step-boundary suppressor (read of MARK_SE_ONLY
# or MARK_CS with a sufficiently negative weight).
STEP_WINDOW_MIN_ALIBI_SLOPE = 0.5

# Names of K-marker dims whose negative-weighted reads count as
# explicit "step-boundary suppressor" structure for the verifier.
# Reading any of these with a sufficiently negative weight pushes
# prior-step (or program-start) tokens out of the softmax mass even
# without an ALiBi slope.
STEP_BOUNDARY_K_SUPPRESSOR_DIMS = ("MARK_SE_ONLY", "MARK_CS", "MARK_SE")


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


@dataclass
class FFNPass:
    """One pass of a :class:`MultiPassOp` — a named ``FFNOp``.

    A pass is a single SwiGLU forward: its rules read the residual state as
    left by the PRIOR pass (its conditions may reference a *workspace* band
    written by earlier passes) and write into the workspace or the result
    bands. The pass boundary is the mechanism that expresses a cross-pass
    carry chain: a rule in pass ``k`` can read a carry dim that a rule in
    pass ``k-1`` wrote, which a single-forward FFN lookup fundamentally
    cannot (a lane cannot read a carry its own forward emits).
    """

    name: str
    ffn: FFNOp = field(default_factory=FFNOp)

    @property
    def hidden_units(self) -> int:
        return self.ffn.hidden_units


@dataclass
class MultiPassOp:
    """Declarative multi-pass FFN cascade — the ``multi_pass_rules`` IR.

    A sequence of :class:`FFNPass` stages applied to the SAME residual
    state in order. Each pass is a full SwiGLU forward that reads the
    residual left by the previous pass (including a *workspace* band the
    earlier passes populate) and accumulates its writes back into the
    residual. This is the construct the single-pass ``FFNRule`` lowering
    cannot express: the cross-pass CARRY CHAIN of wide MUL (schoolbook
    partial-product + column-carry passes) and wide DIV (long-division
    shift-subtract iterations) needs a lane in pass ``k`` to read a value a
    lane in pass ``k-1`` computed — a data dependency that only a staged
    sequence of forwards realizes.

    The workspace band (an op-local residual band, e.g. ``MUL_WS``) is a
    scratch region that lives only across the passes of one op; passes
    write partial sums / carries there and the final pass reads it to
    assemble the result. See ``docs/DSL_W5_MULDIV_LIMIT.md`` Path 1 and
    ``docs/semantic_spec_ALU.md`` G3/G4.
    """

    passes: List[FFNPass] = field(default_factory=list)
    workspace_band: Optional[str] = None
    name: Optional[str] = None

    def add_pass(self, name: str, rules: Iterable[FFNRule] = ()) -> "FFNPass":
        p = FFNPass(name=name, ffn=FFNOp(list(rules)))
        self.passes.append(p)
        return p

    @property
    def hidden_units(self) -> int:
        """Total FFN units across every pass (one per rule)."""
        return sum(p.hidden_units for p in self.passes)

    @property
    def num_passes(self) -> int:
        return len(self.passes)

    def as_flat_ir(self) -> "CompilerIR":
        """Lower each pass onto its own :class:`CompilerIR` layer.

        Pass ``k`` becomes layer ``k`` so the existing
        ``CompilerIR.lower_ffn`` / ``symbolic_ffn`` per-layer machinery
        drives each pass; the caller applies the layers in order (each a
        stacked SwiGLU forward over the running residual). This keeps the
        multi-pass construct a THIN layer over the proven single-pass
        lowering — no new lowering backend, just N stacked FFN passes.
        """
        ir = CompilerIR()
        for idx, p in enumerate(self.passes):
            ir.layer(idx).ffn.rules.extend(p.ffn.rules)
        return ir

    def run_symbolic(
        self, state: Mapping[str, float]
    ) -> Dict[str, float]:
        """Apply every pass in order to ``state``, threading the residual.

        Returns the residual-state map after the final pass. Each pass sees
        the writes of all prior passes (this is where the cross-pass carry
        chain resolves). Uses the same ``symbolic_ffn`` per-layer executor
        the single-pass path uses, so the symbolic result is the exact
        contract the lowered stacked-FFN forward must reproduce.
        """
        ir = self.as_flat_ir()
        cur: Dict[str, float] = dict(state)
        for idx in range(len(self.passes)):
            cur = ir.symbolic_ffn(cur, layer_idx=idx)
        return cur


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

    In addition to the per-head specs in ``rules``, an op may attach
    :class:`RuntimeAttentionFragment` entries to express attention bakes
    whose shape depends on the live attention block's geometry (e.g.
    ``attn.num_heads``). Fragments carry a name + bake_fn; the lowerer
    emits each attached fragment in order.

    Phase 7.C.2 introduced fragments with ``runtime_predicate`` lambdas
    that gated emission at lowering time. DSL Wave W7 moves that
    branching one level up: the IR-builder takes the shape variable as
    an argument (e.g. ``_layer15_memory_lookup_ir(.., num_heads=...)``)
    and uses plain Python ``if`` to choose which fragments to attach.
    ``runtime_predicate`` is deprecated and no in-tree builder sets it.
    """

    rules: List["AttentionHeadIR"] = field(default_factory=list)
    fragments: List["RuntimeAttentionFragment"] = field(default_factory=list)

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

    def add_fragment(
        self,
        fragment: "RuntimeAttentionFragment",
    ) -> "RuntimeAttentionFragment":
        """Attach a runtime-shape conditional bake fragment to this op."""

        self.fragments.append(fragment)
        return fragment

    @property
    def heads(self) -> Tuple["AttentionHeadIR", ...]:
        return tuple(self.rules)

    @property
    def runtime_fragments(self) -> Tuple["RuntimeAttentionFragment", ...]:
        return tuple(self.fragments)

    def shape(self) -> Tuple[int, int]:
        """Return ``(num_q_heads, num_kv_heads)`` (Phase 8.O.2 GQA).

        Computed from the registered :class:`AttentionHeadIR` specs:

        * ``num_q_heads = max(spec.head_idx)+1`` across all rules — the
          Q-row footprint the lowered ``attn.W_q`` needs.
        * ``num_kv_heads = max(spec.kv_head_idx)+1`` across all rules —
          the K/V-row footprint the lowered ``attn.W_k`` / ``W_v`` needs.

        At ``group_size=1`` (every spec's default) the two counts are
        equal and the lowering site stays byte-identical with vanilla
        MHA. With ``group_size > 1`` (GQA) the K/V count shrinks.

        Empty op (``rules=()``) returns ``(0, 0)``.
        """
        if not self.rules:
            return (0, 0)
        max_q = max(int(h.spec.head_idx) for h in self.rules)
        max_kv = max(int(h.spec.kv_head_idx) for h in self.rules)
        return (max_q + 1, max_kv + 1)


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
class RuntimeAttentionFragment:
    """An imperative attention-head fragment carried by an :class:`AttentionOp`.

    Phase 7.C.2 (Option B) — some legacy attention bakes (notably L15
    ``memory_lookup``) write Q/K/V/O across heads whose presence depends
    on the live attention shape (``attn.num_heads``). The per-head
    :class:`DeclarativeAttentionHeadSpec` shape can't express that
    branching, so the imperative writer is kept as a callable and the
    IR carries it as a *fragment*. All variants are visible at the IR
    site rather than hidden inside a bake function that branches at the
    weight-write site.

    DSL Wave W7 — the per-fragment ``runtime_predicate`` escape hatch
    is deprecated. Shape branching now happens at IR-build time inside
    the (parameterized) factory: the builder takes the shape variable
    (e.g. ``num_heads``) as an argument, uses plain Python ``if`` to
    select which fragments to add, and lowering emits every attached
    fragment unconditionally. ``runtime_predicate`` defaults to ``None``
    ("always emit"), and is retained for backward compatibility only --
    new fragments should leave it ``None`` and rely on the builder's
    Python branching to do the selection. :meth:`should_emit` still
    honours a non-``None`` predicate, so any pre-W7 caller continues to
    work, but no in-tree call site sets it as of this wave.

    Attributes
    ----------
    name:
        Human-readable identifier (e.g. ``"l15_memory_lookup.heads_0_3"``).
        Surfaced in debug reports and used as a deduplication key by
        downstream tooling.
    bake_fn:
        Callable that writes weights. Called as
        ``bake_fn(attn, dim_positions, HD, S)`` so it has the same
        information the legacy imperative helper consumed. Must be
        idempotent within a single bake pass — fragments may be invoked
        multiple times if the surrounding IR is composed.
    runtime_predicate:
        Deprecated since DSL Wave W7. Callable ``(attn) -> bool``. The
        fragment is emitted iff this returns ``True``. Defaults to
        ``None`` meaning "always emit". New fragments should leave it
        ``None``.
    metadata:
        Optional opaque mapping for downstream tooling (audit reports,
        symbolic execution stubs, etc.). The lowerer never consults it.
        Convention: ``shape="num_heads >= 12"`` (a human-readable
        marker) is used by W7-era builders to record the build-time
        shape gate that selected the fragment.
    """

    name: str
    bake_fn: Callable[..., None]
    runtime_predicate: Optional[Callable[[object], bool]] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def should_emit(self, attn) -> bool:
        if self.runtime_predicate is None:
            return True
        return bool(self.runtime_predicate(attn))


@dataclass(frozen=True)
class StructuralOp:
    """Declarative description of a whole-block structural transform.

    Phase 8.I closing-audit primitive: some legacy bakes do work that
    isn't a per-head Q/K/V/O write or a per-unit FFN rule -- they reshape
    the block (e.g. ``nn.Parameter`` resizes that grow ``attn.num_heads``)
    and then run a follow-up imperative pass that depends on the new
    shape. ``StructuralOp`` carries the structural intent as data so
    those bakes lower through :meth:`CompilerIR.lower_structural_ops`
    instead of writing weights directly from a bake_fn.

    Attributes
    ----------
    kind:
        The structural transform identifier. Today only
        ``"attention_resize"`` is recognised; the lowerer raises on
        unknown kinds so a typo can't silently no-op.
    target_num_heads:
        Desired ``attn.num_heads`` after the resize. The lowerer
        re-allocates ``W_q`` / ``W_k`` / ``W_v`` / ``W_o`` and the
        ``alibi_slopes`` buffer to match.
    small_num_heads:
        Alternative head count used when ``block._n_layers_hint`` is at
        or below :attr:`layers_threshold`. ``None`` means "always use
        ``target_num_heads``".
    layers_threshold:
        ``n_layers_hint`` boundary that selects between
        :attr:`target_num_heads` (larger build) and
        :attr:`small_num_heads` (16-layer smoke build).
    alibi_pin_value:
        Value the first :attr:`alibi_pin_count` ALiBi slopes are pinned
        to after the resize. ``None`` disables the pin.
    alibi_pin_count:
        How many of the leading ALiBi slope slots receive
        :attr:`alibi_pin_value`. Defaults to 4 (the L15 load-head count).
    follow_up:
        Optional callable ``follow_up(block, dim_positions, S)`` run
        after the structural resize completes. Used by L15 to stash the
        attention-head allocator and call
        ``_suppress_l15_lookup_during_current_store_generation``. The
        follow-up callable is opaque to the lowerer: it can be a thin
        imperative helper as long as the structural shape comes from
        the IR.
    metadata:
        Opaque mapping for downstream tooling (audit reports, symbolic
        execution stubs). The lowerer never consults it.
    """

    kind: str
    target_num_heads: int = 0
    small_num_heads: Optional[int] = None
    layers_threshold: Optional[int] = None
    alibi_pin_value: Optional[float] = None
    alibi_pin_count: int = 4
    follow_up: Optional[Callable[..., None]] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def resolve_num_heads(self, n_layers_hint: Optional[int]) -> int:
        """Return the head count this op should resize to for ``n_layers_hint``."""

        if (
            self.small_num_heads is not None
            and self.layers_threshold is not None
            and n_layers_hint is not None
            and n_layers_hint <= self.layers_threshold
        ):
            return int(self.small_num_heads)
        return int(self.target_num_heads)


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


@dataclass(frozen=True)
class PositionalEncodingSpec:
    """Declarative positional-encoding choice for an attention block.

    V2 / Phase 8 architectural toggle. The legacy
    ``compile_full_vm_dynamic(..., positional_encoding="alibi", rope_base=...)``
    kwarg path is shape-equivalent to a ``PositionalEncodingSpec`` that
    lives in the IR. Lifting it into the IR makes the choice auditable
    (no kwarg flowing through ten internal helpers) and round-trippable
    (a tool can serialize the IR and recover the architectural intent).

    The default value (``kind="alibi"``) is byte-identical to the
    historical default — the existing ``VMConfig.alibi_mode`` produces
    the same effective attention slopes.
    """

    kind: Literal["rope", "alibi", "sinusoidal", "learned", "none"] = "alibi"
    rope_base: float = 10000.0
    alibi_slopes: Optional[Tuple[float, ...]] = None

    _VALID_KINDS = ("rope", "alibi", "sinusoidal", "learned", "none")

    def __post_init__(self) -> None:
        if self.kind not in self._VALID_KINDS:
            raise ValueError(
                "PositionalEncodingSpec.kind must be one of "
                f"{self._VALID_KINDS}; got {self.kind!r}"
            )
        if self.alibi_slopes is not None and not isinstance(
            self.alibi_slopes, tuple
        ):
            raise TypeError(
                "PositionalEncodingSpec.alibi_slopes must be a tuple of "
                "floats (use tuple(...) when constructing from a list); got "
                f"{type(self.alibi_slopes).__name__}"
            )


@dataclass(frozen=True)
class AttentionActivationSpec:
    """Declarative attention denominator choice.

    ``softmax`` is the canonical attention normalization. ``softmax1`` is
    the +1-in-denominator variant used by every C4-VM attention head as
    of Phase 8 (see ``BLOG_SPEC.md`` §"Long Division via Attention" for
    the construction that depends on the sink). ``none`` is the no-op
    "raw scores" variant, exposed so tests can verify the toggle plumbs
    through the IR.
    """

    softmax_kind: Literal["softmax", "softmax1", "none"] = "softmax1"
    div_mode: Literal["long_div", "log_softmax1"] = "long_div"

    _VALID_KINDS = ("softmax", "softmax1", "none")
    _VALID_DIV_MODES = ("long_div", "log_softmax1")

    def __post_init__(self) -> None:
        if self.softmax_kind not in self._VALID_KINDS:
            raise ValueError(
                "AttentionActivationSpec.softmax_kind must be one of "
                f"{self._VALID_KINDS}; got {self.softmax_kind!r}"
            )
        if self.div_mode not in self._VALID_DIV_MODES:
            raise ValueError(
                "AttentionActivationSpec.div_mode must be one of "
                f"{self._VALID_DIV_MODES}; got {self.div_mode!r}"
            )
        # Mirror ``VMConfig.__post_init__``: log_softmax1 div requires the
        # softmax1 sink, otherwise the 1/n attention construction has no
        # geometric backing.
        if (
            self.div_mode == "log_softmax1"
            and self.softmax_kind != "softmax1"
        ):
            raise ValueError(
                "AttentionActivationSpec.div_mode='log_softmax1' requires "
                "softmax_kind='softmax1'; got softmax_kind="
                f"{self.softmax_kind!r}."
            )


@dataclass(frozen=True)
class NormSpec:
    """Declarative pre-attention / pre-FFN normalization.

    The C4 VM historically baked LayerNorm into every block; Phase 8.O
    added an ``use_rms_norm`` toggle for the open-model-like target.
    ``kind="none"`` is exposed so a caller can construct a transformer
    block without any pre-norm (used by some teacher-forced eval paths
    that pre-apply norm at the outer level).
    """

    kind: Literal["rmsnorm", "layernorm", "none"] = "layernorm"
    eps: float = 1e-6

    _VALID_KINDS = ("rmsnorm", "layernorm", "none")

    def __post_init__(self) -> None:
        if self.kind not in self._VALID_KINDS:
            raise ValueError(
                "NormSpec.kind must be one of "
                f"{self._VALID_KINDS}; got {self.kind!r}"
            )
        if not (self.eps > 0.0):
            raise ValueError(
                f"NormSpec.eps must be > 0; got {self.eps!r}"
            )


@dataclass(frozen=True)
class FFNActivationSpec:
    """Declarative FFN activation + expansion ratio.

    ``relu`` is the default (matches PureFFN's historical ReLU gate),
    ``gelu`` for open-model parity, ``swiglu`` for Llama/Mixtral-style
    gated SiLU FFNs. ``ffn_expansion_ratio`` is the multiplier from
    ``d_model`` to ``ffn_hidden`` (4.0 is the standard transformer
    ratio; SwiGLU variants typically use 8/3 ≈ 2.667).
    """

    kind: Literal["relu", "gelu", "swiglu"] = "relu"
    ffn_expansion_ratio: float = 4.0
    # Absolute ``ffn_hidden`` overrides the expansion ratio when set. This
    # mirrors the legacy ``compile_full_vm_dynamic(ffn_hidden=4096, ...)``
    # kwarg path: callers that know the absolute width (HF state-dict
    # exports, fixed-layout bakes) can pin it explicitly; otherwise the
    # ratio derives the width from ``d_model``.
    ffn_hidden: Optional[int] = None

    _VALID_KINDS = ("relu", "gelu", "swiglu")

    def __post_init__(self) -> None:
        if self.kind not in self._VALID_KINDS:
            raise ValueError(
                "FFNActivationSpec.kind must be one of "
                f"{self._VALID_KINDS}; got {self.kind!r}"
            )
        if not (self.ffn_expansion_ratio > 0.0):
            raise ValueError(
                "FFNActivationSpec.ffn_expansion_ratio must be > 0; "
                f"got {self.ffn_expansion_ratio!r}"
            )
        if self.ffn_hidden is not None and not (self.ffn_hidden > 0):
            raise ValueError(
                "FFNActivationSpec.ffn_hidden must be > 0 or None; "
                f"got {self.ffn_hidden!r}"
            )

    def resolve_hidden(self, d_model: int) -> int:
        """Return the effective FFN hidden width for ``d_model``.

        ``ffn_hidden`` (if set) wins; otherwise derive from the expansion
        ratio. This is the single point of truth for the lowering pipeline
        and the HF export adapters.
        """
        if self.ffn_hidden is not None:
            return int(self.ffn_hidden)
        return int(round(self.ffn_expansion_ratio * d_model))


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """Model-wide architectural toggles, lifted out of compile kwargs.

    V2 compliance: the historical
    ``compile_full_vm_dynamic(positional_encoding=..., use_rms_norm=...,
    rope_base=..., ffn_hidden=..., div_mode=..., attention_normalization=...)``
    kwargs encode architectural choices that aren't weight-shaped, so
    they shouldn't live as compile kwargs. This dataclass groups them
    into the IR so a caller can describe a target architecture
    declaratively.

    Per-layer overrides live on :class:`LayerSpec`'s
    ``attention_pos_encoding`` / ``attention_activation`` /
    ``norm_pre_attention`` / ``norm_pre_ffn`` / ``ffn_activation`` fields
    — if a layer pins one of those, it wins; otherwise the
    model-wide spec applies.

    The default value is byte-identical to the historical
    ``VMConfig.alibi_mode`` factory: ALiBi attention, softmax1 denom,
    LayerNorm pre-norm, ReLU FFN with 4x expansion.
    """

    positional_encoding: PositionalEncodingSpec = field(
        default_factory=PositionalEncodingSpec
    )
    attention_activation: AttentionActivationSpec = field(
        default_factory=AttentionActivationSpec
    )
    norm_pre_attention: NormSpec = field(default_factory=NormSpec)
    norm_pre_ffn: NormSpec = field(default_factory=NormSpec)
    ffn_activation: FFNActivationSpec = field(
        default_factory=FFNActivationSpec
    )

    @classmethod
    def from_compile_kwargs(
        cls,
        *,
        positional_encoding: Optional[str] = None,
        attention_normalization: Optional[str] = None,
        rope_base: Optional[float] = None,
        use_rms_norm: Optional[bool] = None,
        rms_norm_eps: Optional[float] = None,
        ffn_activation_kind: Optional[str] = None,
        ffn_expansion_ratio: Optional[float] = None,
    ) -> "ModelArchitectureSpec":
        """Construct from ``compile_full_vm_dynamic``'s kwarg surface.

        Mirrors the kwarg names so a caller migrating from the legacy
        kwarg path can do so by replacing the kwargs with a single
        ``ModelArchitectureSpec.from_compile_kwargs(...)`` call. Any
        kwarg left as ``None`` falls back to the dataclass default —
        which is byte-identical to ``VMConfig.alibi_mode``.

        ``positional_encoding="hybrid"`` is the legacy 3-valued string;
        callers wiring a non-hybrid IR can either resolve it ahead of
        time (per-layer overrides on :class:`LayerSpec`) or pass
        ``kind="alibi"`` and rely on the layer-level override path.
        Hybrid as a kind isn't supported in :class:`PositionalEncodingSpec`
        because the IR forces the per-layer decision to be visible.
        """
        pos_kind: Literal[
            "rope", "alibi", "sinusoidal", "learned", "none"
        ] = "alibi"
        if positional_encoding == "rope":
            pos_kind = "rope"
        elif positional_encoding == "alibi" or positional_encoding is None:
            pos_kind = "alibi"
        elif positional_encoding == "hybrid":
            # Hybrid resolves at the layer level; the model-wide default
            # describes the majority path (alibi for L0-L2, rope for
            # the rest — call it rope here since the rest dominates).
            pos_kind = "rope"
        else:
            # Allow direct passthrough of the new kinds.
            pos_kind = positional_encoding  # type: ignore[assignment]
        pos = PositionalEncodingSpec(
            kind=pos_kind,
            rope_base=rope_base if rope_base is not None else 10000.0,
        )

        if attention_normalization == "softmax":
            act_kind: Literal["softmax", "softmax1", "none"] = "softmax"
        elif attention_normalization == "softmax1" or attention_normalization is None:
            act_kind = "softmax1"
        else:
            act_kind = attention_normalization  # type: ignore[assignment]
        attn_act = AttentionActivationSpec(softmax_kind=act_kind)

        if use_rms_norm is True:
            norm_kind: Literal["rmsnorm", "layernorm", "none"] = "rmsnorm"
        elif use_rms_norm is False or use_rms_norm is None:
            norm_kind = "layernorm"
        else:  # pragma: no cover - defensive
            norm_kind = "layernorm"
        eps = rms_norm_eps if rms_norm_eps is not None else 1e-6
        norm_pre_attn = NormSpec(kind=norm_kind, eps=eps)
        norm_pre_ffn = NormSpec(kind=norm_kind, eps=eps)

        ffn_kind: Literal["relu", "gelu", "swiglu"] = "relu"
        if ffn_activation_kind in ("relu", "gelu", "swiglu"):
            ffn_kind = ffn_activation_kind  # type: ignore[assignment]
        ffn_act = FFNActivationSpec(
            kind=ffn_kind,
            ffn_expansion_ratio=(
                ffn_expansion_ratio if ffn_expansion_ratio is not None else 4.0
            ),
        )

        return cls(
            positional_encoding=pos,
            attention_activation=attn_act,
            norm_pre_attention=norm_pre_attn,
            norm_pre_ffn=norm_pre_ffn,
            ffn_activation=ffn_act,
        )


@dataclass
class LayerSpec:
    """Declarative work assigned to one logical compiler layer.

    The architectural-toggle fields (``attention_pos_encoding``,
    ``attention_activation``, ``norm_pre_attention``, ``norm_pre_ffn``,
    ``ffn_activation``) are per-layer overrides. When set, they take
    precedence over the :class:`ModelArchitectureSpec` on the enclosing
    :class:`CompilerIR`. The default ``None`` means "inherit from the
    model-wide spec", which is byte-identical to the pre-spec behaviour.
    """

    ffn: FFNOp = field(default_factory=FFNOp)
    attention: AttentionOp = field(default_factory=AttentionOp)
    structural_ops: List[StructuralOp] = field(default_factory=list)
    attention_pos_encoding: Optional[PositionalEncodingSpec] = None
    attention_activation: Optional[AttentionActivationSpec] = None
    norm_pre_attention: Optional[NormSpec] = None
    norm_pre_ffn: Optional[NormSpec] = None
    ffn_activation: Optional[FFNActivationSpec] = None


@dataclass
class CompilerIR:
    """A symbolic-and-lowerable program fragment."""

    layers: List[LayerSpec] = field(default_factory=list)
    embeddings: List[TokenEmbeddingRule] = field(default_factory=list)
    architecture: ModelArchitectureSpec = field(
        default_factory=ModelArchitectureSpec
    )

    def layer(self, index: int) -> LayerSpec:
        while len(self.layers) <= index:
            self.layers.append(LayerSpec())
        return self.layers[index]

    # ------------------------------------------------------------------
    # Architecture-spec resolution helpers (V2 toggle IR — Phase 8.X).
    # ------------------------------------------------------------------

    def resolve_attention_pos_encoding(
        self, layer_idx: int
    ) -> PositionalEncodingSpec:
        """Per-layer override wins; else falls back to ``architecture``."""
        override = self.layer(layer_idx).attention_pos_encoding
        if override is not None:
            return override
        return self.architecture.positional_encoding

    def resolve_attention_activation(
        self, layer_idx: int
    ) -> AttentionActivationSpec:
        override = self.layer(layer_idx).attention_activation
        if override is not None:
            return override
        return self.architecture.attention_activation

    def resolve_norm_pre_attention(self, layer_idx: int) -> NormSpec:
        override = self.layer(layer_idx).norm_pre_attention
        if override is not None:
            return override
        return self.architecture.norm_pre_attention

    def resolve_norm_pre_ffn(self, layer_idx: int) -> NormSpec:
        override = self.layer(layer_idx).norm_pre_ffn
        if override is not None:
            return override
        return self.architecture.norm_pre_ffn

    def resolve_ffn_activation(self, layer_idx: int) -> FFNActivationSpec:
        override = self.layer(layer_idx).ffn_activation
        if override is not None:
            return override
        return self.architecture.ffn_activation

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

    def lower_multi_pass(
        self,
        multi_pass: "MultiPassOp",
        ffn_passes: Sequence[object],
        dim_positions: Mapping[str, int],
        *,
        S: float = 100.0,
        write_scale: float = 1.0,
    ) -> List[int]:
        """Lower a :class:`MultiPassOp` into one ``PureFFN`` per pass.

        ``ffn_passes[k]`` receives the rules of pass ``k`` (via the same
        per-layer ``lower_ffn`` used by the single-pass path). Returns the
        per-pass end-unit cursor list. The caller applies the passes in
        order over the running residual (each a stacked SwiGLU forward), so
        pass ``k`` reads the workspace band pass ``k-1`` wrote — the
        cross-pass carry chain. No new lowering backend: this is N stacked
        applications of the proven single-pass FFN lowering.
        """
        if len(ffn_passes) != multi_pass.num_passes:
            raise ValueError(
                "lower_multi_pass: got "
                f"{len(ffn_passes)} PureFFN targets for "
                f"{multi_pass.num_passes} passes"
            )
        flat = multi_pass.as_flat_ir()
        ends: List[int] = []
        for idx, ffn in enumerate(ffn_passes):
            end = flat.lower_ffn(
                ffn,
                dim_positions,
                layer_idx=idx,
                start_unit=0,
                S=S,
                write_scale=write_scale,
            )
            ends.append(end)
        return ends

    def lower_attention(
        self,
        attn,
        HD: int,
        *,
        layer_idx: int = 0,
        dim_positions: Optional[Mapping[str, int]] = None,
        S: float = 100.0,
    ) -> int:
        """Lower one layer's attention specs into attention projection weights.

        Emits the per-head :class:`DeclarativeAttentionHeadSpec` rules
        first, then runs each :class:`RuntimeAttentionFragment` whose
        (deprecated) predicate matches ``attn``. Fragments receive
        ``(attn, dim_positions, HD, S)`` so they have the same context
        the legacy imperative helpers consumed.

        DSL Wave W7: in-tree builders no longer set
        ``runtime_predicate``; shape selection happens at IR-build
        time so the loop below emits every attached fragment
        unconditionally. ``should_emit`` is kept for backward
        compatibility with any out-of-tree caller that still attaches
        a predicate.

        ``dim_positions`` is required by fragments that resolve dim
        names at bake time; pass-through callers without fragments can
        leave it as ``None``.
        """

        from .primitives import Primitives

        layer = self.layer(layer_idx)
        heads = layer.attention.rules
        # V1/V2 vision (per-head dynamic head_dim): delegate to
        # ``generate_attention_heads`` so any spec declaring a non-default
        # ``head_dim`` triggers the cumulative-sum row-base path. When
        # every spec leaves ``head_dim=None`` this falls through to the
        # legacy ``head_idx * HD`` per-head loop — byte-identical with
        # the prior per-head ``Primitives.generate_attention_head`` call.
        Primitives.generate_attention_heads(
            attn, [head.spec for head in heads], HD
        )

        fragments = layer.attention.fragments
        for fragment in fragments:
            if not fragment.should_emit(attn):
                continue
            fragment.bake_fn(attn, dim_positions, HD, S)
        return len(heads)

    def lower_structural_ops(
        self,
        block,
        dim_positions: Optional[Mapping[str, int]] = None,
        *,
        layer_idx: int = 0,
        S: float = 100.0,
    ) -> int:
        """Lower :class:`StructuralOp` entries against a transformer block.

        Phase 8.I primitive: runs each structural op in declaration
        order against ``block`` (must expose ``block.attn``). Today only
        ``kind="attention_resize"`` is implemented; unknown kinds raise
        ``ValueError`` so a typo at the IR site is loud, not silent.

        For ``attention_resize`` the lowering:

        * Reads ``block._n_layers_hint`` to pick between
          :attr:`StructuralOp.target_num_heads` and
          :attr:`StructuralOp.small_num_heads`.
        * If ``attn.num_heads`` already matches or exceeds the chosen
          count, the resize is skipped (the ALiBi pin and follow-up
          still run -- this matches the legacy
          ``make_l15_attention_resize_op`` short-circuit so the same
          op invoked twice stays idempotent).
        * Otherwise re-allocates ``attn.W_q`` / ``W_k`` / ``W_v`` /
          ``W_o`` as fresh ``nn.Parameter`` tensors sized for
          ``target_num_heads * head_dim`` rows, copies the existing
          weights into the leading slice, and re-registers an
          ``alibi_slopes`` buffer with the standard
          ``2 ** (-8/N * (i+1))`` decay.
        * Pins the leading :attr:`StructuralOp.alibi_pin_count` ALiBi
          slots to :attr:`StructuralOp.alibi_pin_value` when both are
          set.
        * Invokes :attr:`StructuralOp.follow_up` (when set) as
          ``follow_up(block, dim_positions, S)``.

        Returns the number of structural ops lowered.
        """

        import torch
        from torch import nn

        layer = self.layer(layer_idx)
        applied = 0
        for sop in layer.structural_ops:
            if sop.kind != "attention_resize":
                raise ValueError(
                    f"StructuralOp.kind {sop.kind!r} not recognised by "
                    "lower_structural_ops"
                )
            attn = getattr(block, "attn", block)
            n_layers_hint = getattr(block, "_n_layers_hint", None)
            num_heads_new = sop.resolve_num_heads(n_layers_hint)
            d = attn.W_q.shape[1]
            head_dim_old = d // attn.num_heads

            if getattr(attn, "num_heads", 0) >= num_heads_new:
                # Resize is a no-op for the head COUNT (the module already has
                # >= num_heads_new heads), but the ALiBi slopes must still be
                # made DETERMINISTIC from the resized geometry rather than
                # inherited from construction time. Otherwise an
                # over-width-band widen that pushes the GLOBAL head count up to
                # (or past) ``num_heads_new`` -- e.g. ``C4_AX_BYTE1_FULL_WIDTH``
                # grows n_heads 10 -> 13 == this op's effective target -- skips
                # the resize and leaks the construction-time
                # ``alibi_base_heads`` slope geometry, diverging from the
                # golden build (which resized from a smaller count and
                # recomputed the slopes with N == this block's own head count).
                # Recompute with the block's CURRENT head count (== the count
                # the golden non-skip path resized to) so the resized block's
                # slopes are identical regardless of the widen.
                if (
                    hasattr(attn, "alibi_slopes")
                    and attn.alibi_slopes is not None
                ):
                    _n = int(attn.num_heads)
                    new_slopes = torch.tensor(
                        [2.0 ** (-8.0 / _n * (i + 1)) for i in range(_n)]
                    )
                    if sop.alibi_pin_value is not None:
                        new_slopes[: sop.alibi_pin_count] = sop.alibi_pin_value
                    attn.register_buffer("alibi_slopes", new_slopes)
                if sop.follow_up is not None:
                    sop.follow_up(block, dim_positions, S)
                applied += 1
                continue

            new_q_rows = num_heads_new * head_dim_old
            attn.num_heads = num_heads_new
            attn.head_dim = head_dim_old

            if (
                hasattr(attn, "alibi_slopes")
                and attn.alibi_slopes is not None
            ):
                new_slopes = torch.tensor(
                    [
                        2.0 ** (-8.0 / num_heads_new * (i + 1))
                        for i in range(num_heads_new)
                    ]
                )
                if sop.alibi_pin_value is not None:
                    new_slopes[: sop.alibi_pin_count] = sop.alibi_pin_value
                attn.register_buffer("alibi_slopes", new_slopes)

            old_W_q = attn.W_q.data
            old_W_k = attn.W_k.data
            old_W_v = attn.W_v.data
            attn.W_q = nn.Parameter(torch.zeros(new_q_rows, d))
            attn.W_k = nn.Parameter(torch.zeros(new_q_rows, d))
            attn.W_v = nn.Parameter(torch.zeros(new_q_rows, d))
            attn.W_q.data[:d, :] = old_W_q
            attn.W_k.data[:d, :] = old_W_k
            attn.W_v.data[:d, :] = old_W_v

            old_W_o = attn.W_o.data
            attn.W_o = nn.Parameter(torch.zeros(d, new_q_rows))
            attn.W_o.data[:, :d] = old_W_o

            if sop.follow_up is not None:
                sop.follow_up(block, dim_positions, S)
            applied += 1
        return applied

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
        from ..base_layers import PureFFN

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
        from ..base_layers import PureAttention

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
        # Phase 8.O.2 GQA: Q/O addressed by ``head_idx * HD``; K/V by
        # ``kv_head_idx * HD``. At ``group_size=1`` (the default)
        # the two coincide and this validator is byte-identical with
        # the pre-8.O.2 MHA path.
        kv_base = int(spec.kv_head_idx) * head_dim
        for role, writes, matrix_name, row_base in (
            ("q", spec.q, "W_q", base),
            ("k", spec.k, "W_k", kv_base),
            ("v", spec.v, "W_v", kv_base),
        ):
            matrix = getattr(attn, matrix_name).detach()
            for write in writes:
                observed = float(matrix[row_base + int(write.slot), int(write.dim)])
                expected = float(write.weight)
                if abs(observed - expected) > (atol + rtol * abs(expected)):
                    issues.append(AttentionComparisonIssue(
                        "lowering",
                        f"{label} {matrix_name}[{row_base + int(write.slot)}, "
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
    "AttentionActivationSpec",
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
    "FFNActivationSpec",
    "FFNComparisonIssue",
    "FFNComparisonReport",
    "FFNOp",
    "FFNRule",
    "LayerSpec",
    "ModelArchitectureSpec",
    "NormSpec",
    "PositionalEncodingSpec",
    "RuntimeAttentionFragment",
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
