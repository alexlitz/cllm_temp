"""DSL interpreter — execute Operations directly on a residual-state dict.

The compile pipeline (``compile_full_vm_dynamic``) lowers DSL Operations
into PyTorch weights and runs forward through a real transformer. This
module is the parallel path: it interprets the *same* Operations against
a symbolic residual-state dict without ever building neural weights.

Why this exists:

* **Byte-identity diagnostics**: compare what the rules say should happen
  vs. what the lowered FFN actually produces. The existing
  ``compare_symbolic_to_lowered_ffn`` does this for a single rule list;
  this interpreter extends it to the full op graph.
* **Debugging without compile cost**: a cold compile is ~1-2 minutes;
  interpreting a residual-state dict through 100 ops is sub-second.
* **Verifier substrate**: ``decl_verifier`` already runs partial symbolic
  evaluations; this gives them a uniform entry point.
* **Reference semantics**: pinning down what the DSL *means* lets us
  catch lowering bugs more directly.

What it does NOT do:

* Replace the compile pipeline. The deployed transformer is real PyTorch
  weights; this interpreter is a compile-time tool.
* Match neural numerics exactly. The IR's threshold is a hard cutoff;
  silu has soft transitions near threshold. For binary-valued
  one-hot inputs at ``S >> 0`` they agree to ``atol=1e-5`` — that's the
  validation gate. For other inputs they will diverge slightly.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .ir import (
    CompilerIR,
    DimRef,
    FFNOp,
    FFNRule,
    TokenEmbeddingRule,
    _state_value,
)


# ---------------------------------------------------------------------------
# Attention-gate audit constants
# ---------------------------------------------------------------------------

# Q-side dim-name prefixes that mark a write as a *condition gate* — these
# are the dims the doc enumerates as "tries to filter K positions". When
# the K-side at the same slot has no real discriminator (only CONST or
# nothing), the softmax-cancelling argument means the gate doesn't gate
# *under standard softmax*. Under softmax1 a uniform negative offset still
# suppresses the row (the zero-anchor sink wins). See
# ``docs/IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md`` §"Audit gate findings".
GATE_CONDITION_PREFIXES: Tuple[str, ...] = ("MARK_", "OP_", "HAS_", "IS_")

# Dim names that count as "uniform" on the K-side — they contribute the
# same value at every K row, so they cannot discriminate.
UNIFORM_K_DIM_NAMES: frozenset = frozenset({"CONST"})

# Score-magnitude threshold below which a softmax1 sink dominates a K row
# entirely (exp(-10) ≈ 4.5e-5; we use a conservative -8 floor so a slot
# contribution at least this negative is treated as a row-suppressor).
SOFTMAX1_SUPPRESS_THRESHOLD: float = -8.0

# Env flag: when set, the compile-time gate audit raises instead of warning.
GATE_AUDIT_STRICT_ENV = "C4_STRICT_GATE_CHECK"

# Env flag: when set, the compile-time gate audit is skipped entirely.
GATE_AUDIT_SKIP_ENV = "C4_SKIP_GATE_CHECK"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class InterpreterStep:
    """One step of the interpreter — what op ran and what state delta it
    produced. Useful for tracing through a sequence of ops."""

    op_name: str
    op_kind: str  # "ffn" / "attn" / "block" / "model"
    layer_idx: int
    rules_fired: int
    writes: List[Tuple[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class InterpreterResult:
    """Aggregate result of running a sequence of Operations."""

    state: Dict[str, float]
    steps: List[InterpreterStep] = field(default_factory=list)

    def writes_to(self, dim_name: str) -> List[Tuple[str, float]]:
        """Find every op that wrote to ``dim_name`` and its contribution."""

        out: List[Tuple[str, float]] = []
        for step in self.steps:
            for key, value in step.writes:
                if key == dim_name or key.startswith(f"{dim_name}+"):
                    out.append((step.op_name, value))
        return out


class DSLInterpreter:
    """Symbolic interpreter for DSL Operations.

    Holds a residual-state dict keyed by ``"DIM+offset"`` (or just
    ``"DIM"`` for offset 0). Operations are dispatched by ``kind``:

    * ``"ffn"``: applies ``CompilerIR.symbolic_ffn`` semantics — every
      rule whose condition sum >= threshold contributes ``gate_value *
      write_weight`` to each of its writes.
    * ``"attn"``: applies a symbolic attention model — each
      ``DeclarativeAttentionHeadSpec`` propagates the value-side dim
      to the output-side dim when the Q/K columns match. (Exact
      semantics in ``_apply_attention`` below.)
    * ``"block"``: ops with ``kind="block"`` typically wrap an FFN bake
      + side-effects. The interpreter applies any ``compiler_ir`` /
      ``compiler_ir_factory`` rules and otherwise treats ``bake_fn`` as
      opaque (records a note but does not execute imperative bakes).
    * ``"model"``: applies any ``TokenEmbeddingRule``s in the op (no-op
      otherwise — model-level bakes write to ``model.embed`` /
      ``model.head`` which aren't part of the residual state).

    Usage::

        interp = DSLInterpreter(initial_state={"MARK_AX": 1.0, "ALU_LO+5": 1.0})
        result = interp.run([op1, op2, op3])
        assert result.state["OUTPUT_LO+5"] == 1.0
    """

    def __init__(
        self,
        initial_state: Optional[Mapping[str, float]] = None,
        *,
        dim_positions: Optional[Mapping[str, int]] = None,
    ):
        self.state: Dict[str, float] = (
            dict(initial_state) if initial_state else {}
        )
        # Optional dim_positions — only needed if you want to check
        # against numerical position constraints (e.g. attention slot
        # indices). The interpreter is otherwise position-free.
        self.dim_positions: Optional[Dict[str, int]] = (
            dict(dim_positions) if dim_positions else None
        )

    # ----- per-op application ----------------------------------------------

    def apply_ffn_rules(
        self,
        rules: Sequence[FFNRule],
        *,
        op_name: str = "<rules>",
        layer_idx: int = 0,
    ) -> InterpreterStep:
        """Apply a list of ``FFNRule``s to the current state.

        Mirrors ``CompilerIR.symbolic_ffn`` but tracks per-rule firing
        and per-write contributions for tracing.
        """
        step = InterpreterStep(
            op_name=op_name, op_kind="ffn", layer_idx=layer_idx,
            rules_fired=0,
        )
        for rule in rules:
            score = sum(
                _state_value(self.state, term.dim) * term.weight
                for term in rule.conditions
            )
            if score < rule.threshold:
                continue
            step.rules_fired += 1
            gate_value = rule.gate_bias
            if rule.gate is not None:
                gate_value += (
                    _state_value(self.state, rule.gate) * rule.gate_weight
                )
            gate_value += sum(
                _state_value(self.state, term.dim) * term.weight
                for term in rule.gate_terms
            )
            for write in rule.writes:
                key = write.dim.key()
                contribution = gate_value * write.weight
                self.state[key] = self.state.get(key, 0.0) + contribution
                step.writes.append((key, contribution))
        return step

    def apply_attention_specs(
        self,
        specs,
        *,
        op_name: str = "<attn>",
        layer_idx: int = 0,
    ) -> InterpreterStep:
        """Apply attention specs symbolically.

        Symbolic model: each spec is interpreted as a value-propagation
        channel. For every ``AO`` (output write) in the spec, we set
        ``state[o.out_dim] += state.get(spec.value_dim_for(o.slot), 0)``.
        This abstracts away the Q/K matching (which depends on token
        positions / runtime activations) and just propagates whatever
        the spec's value-side dim currently holds to the output-side dim.

        This matches the "expected steady-state" semantics that
        ``decl_verifier`` and the byte-identity tests use — it's not the
        full softmax dynamics, but it captures the "this head moves data
        from V_dim to O_dim" intent.
        """
        step = InterpreterStep(
            op_name=op_name, op_kind="attn", layer_idx=layer_idx,
            rules_fired=0,
        )
        for spec in specs:
            step.rules_fired += 1
            # Build a slot-to-value-dim map from the V writes.
            v_by_slot: Dict[int, str] = {}
            for v in getattr(spec, "v", ()):
                v_by_slot[v.slot] = v.dim
            # For each O write, propagate the matching V dim's value.
            for o in getattr(spec, "o", ()):
                v_dim = v_by_slot.get(o.slot)
                if v_dim is None:
                    step.notes.append(
                        f"head_idx={getattr(spec, 'head_idx', '?')} "
                        f"O slot {o.slot} has no matching V dim — skipped"
                    )
                    continue
                v_value = self.state.get(v_dim, 0.0)
                contribution = v_value * o.weight
                # AttentionOutputWrite carries ``out_dim`` (an int residual
                # column); use it as the state key directly, mirroring the
                # V-side which also keys state by an int dim.
                key = o.out_dim
                self.state[key] = self.state.get(key, 0.0) + contribution
                step.writes.append((key, contribution))
        return step

    def apply_token_embedding_rules(
        self,
        rules: Sequence[TokenEmbeddingRule],
        *,
        op_name: str = "<embed>",
    ) -> InterpreterStep:
        """Apply ``TokenEmbeddingRule``s. Model-level bakes write into
        the embedding / head weight rows, not the residual state. The
        interpreter records them as notes without touching state.
        """
        step = InterpreterStep(
            op_name=op_name, op_kind="model", layer_idx=-1,
            rules_fired=len(rules),
        )
        for rule in rules:
            # TokenEmbeddingRule fields are target / token_ids / writes;
            # record a short summary without per-write expansion.
            step.notes.append(
                f"embed target={getattr(rule, 'target', '?')} "
                f"token_ids={getattr(rule, 'token_ids', ())} "
                f"writes={len(getattr(rule, 'writes', ()))}"
            )
        return step

    # ----- Operation dispatch ----------------------------------------------

    def apply_operation(self, op) -> InterpreterStep:
        """Dispatch an ``Operation`` to the right ``apply_*`` method.

        Tries ``op.compiler_ir`` first; falls back to
        ``op.compiler_ir_factory(dim_positions, HD=8)`` if available.
        Otherwise records a note that the op is opaque to symbolic
        interpretation.
        """
        op_name = getattr(op, "name", "<anon>")
        layer_idx = getattr(op, "layer_idx", 0) or 0
        kind = getattr(op, "kind", "ffn")

        # Try to extract an IR (which carries rules + specs + embedding rules)
        ir = self._extract_ir(op)
        if ir is None:
            return InterpreterStep(
                op_name=op_name, op_kind=kind, layer_idx=layer_idx,
                rules_fired=0,
                notes=["no compiler_ir / compiler_ir_factory; opaque to symbolic interpretation"],
            )

        layer = ir.layer(0)
        # Aggregate over the three rule sources the IR exposes
        step = InterpreterStep(
            op_name=op_name, op_kind=kind, layer_idx=layer_idx,
            rules_fired=0,
        )
        ffn_rules = layer.ffn.rules
        if ffn_rules:
            sub = self.apply_ffn_rules(
                ffn_rules, op_name=op_name, layer_idx=layer_idx,
            )
            step.rules_fired += sub.rules_fired
            step.writes.extend(sub.writes)
            step.notes.extend(sub.notes)
        attn_heads = getattr(layer.attention, "rules", ())
        if attn_heads:
            specs = [h.spec for h in attn_heads]
            sub = self.apply_attention_specs(
                specs, op_name=op_name, layer_idx=layer_idx,
            )
            step.rules_fired += sub.rules_fired
            step.writes.extend(sub.writes)
            step.notes.extend(sub.notes)
        embed_rules = getattr(ir, "embeddings", ())
        if embed_rules:
            sub = self.apply_token_embedding_rules(
                embed_rules, op_name=op_name,
            )
            step.rules_fired += sub.rules_fired
            step.notes.extend(sub.notes)
        return step

    def run(
        self,
        ops: Iterable,
    ) -> InterpreterResult:
        """Apply a sequence of Operations in order.

        Pure data — does not invoke the LayerCompiler's scheduler.
        Caller is responsible for passing ops in a meaningful order
        (typically the order ``LayerCompiler`` would produce after
        dep-graph resolution).
        """
        result = InterpreterResult(state=self.state)
        for op in ops:
            step = self.apply_operation(op)
            result.steps.append(step)
        return result

    # ----- helpers ---------------------------------------------------------

    def _extract_ir(self, op) -> Optional[CompilerIR]:
        """Get a ``CompilerIR`` from an Operation, trying the three
        common entry points in order.
        """
        # 1. Pre-built IR
        ir = getattr(op, "compiler_ir", None)
        if ir is not None:
            return ir
        # 2. Factory (needs dim_positions + HD)
        factory = getattr(op, "compiler_ir_factory", None)
        if factory is not None and self.dim_positions is not None:
            try:
                return factory(self.dim_positions, HD=8)
            except Exception:
                return None
        return None

    # ----- convenience for tests ------------------------------------------

    def get(self, dim_key: str) -> float:
        """Read a single dim value from the state."""
        return self.state.get(dim_key, 0.0)

    def set(self, dim_key: str, value: float) -> None:
        """Set a single dim value in the state."""
        self.state[dim_key] = value

    def reset(
        self, initial_state: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Reset the state to a new initial map (or empty)."""
        self.state = dict(initial_state) if initial_state else {}


# ---------------------------------------------------------------------------
# Attention-gate audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateAuditEntry:
    """One Q-side gate-effectivity classification.

    * ``slot`` — the head-local slot the Q-side write lands in.
    * ``q_dim_name`` — the (gate-condition) residual dim the Q-side
      writes (e.g. ``"MARK_PC"``, ``"OP_LEV"``).
    * ``kind`` — one of:
        - ``"safe"``: K-side at the slot writes at least one non-CONST
          dim (a real discriminator).
        - ``"no_op"``: K-side at the slot writes only CONST(s) AND the
          per-row score contribution is not uniformly suppressive — the
          "gate" does not filter K positions under either softmax mode.
        - ``"softmax1_suppress"``: K-side at the slot writes only CONST(s)
          but the per-Q-row offset is sufficiently negative that the
          softmax1 zero-anchor sink wins for rows where the condition
          is OFF. This is an *effective* gate under softmax1 (the
          deployed model uses softmax1; see
          ``docs/IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md``).
        - ``"q_only"``: K-side has no write at the slot at all. The
          Q-side write is literal dead weight (the doc calls this the
          "no K-side write at slot" case).
    * ``k_dim_names`` — the sorted tuple of K-side dim names at the
      same slot. Empty for ``q_only``; ``("CONST",)`` for the
      canonical CONST-only case.
    * ``head_idx``/``op_name`` — propagated by callers; left ``None``
      for the bare :func:`audit_attention_gates` entry point.
    * ``softmax_mode`` — the softmax mode the entry was classified
      under (``"softmax1"`` or ``"softmax"``). Default ``"softmax1"``
      matches the deployed model's ``attention_normalization``.
    """

    slot: int
    q_dim_name: str
    kind: str  # "safe" / "no_op" / "softmax1_suppress" / "q_only"
    k_dim_names: Tuple[str, ...] = ()
    head_idx: Optional[int] = None
    op_name: Optional[str] = None
    softmax_mode: str = "softmax1"


def build_dim_name_map(
    dim_positions: Optional[Mapping[str, int]] = None,
) -> Dict[int, Tuple[str, ...]]:
    """Build an int→sorted dim-name(s) reverse map.

    Multiple dim names can alias to the same position (e.g. ``H5+0`` and
    ``SP_BYTE0_IS_F8``). The reverse map records *all* names at a
    position so the audit can match on any of them.

    When ``dim_positions`` is omitted, falls back to the class-attribute
    table on ``vm_step._SetDim`` (the legacy / non-compiler-allocated
    layout). This is what the synthetic-spec tests use and lets the
    audit run without a live compiler.

    Names are filtered to uppercase identifiers binding to ints, which
    matches the convention used throughout ``_SetDim`` and the compiler's
    dim registry.
    """

    reverse: Dict[int, Set[str]] = defaultdict(set)
    if dim_positions is None:
        from ..vm_step import _SetDim

        for name, value in vars(_SetDim).items():
            if not name.isupper() or not isinstance(value, int):
                continue
            reverse[int(value)].add(name)
    else:
        for name, value in dim_positions.items():
            if not isinstance(value, int):
                continue
            if not name.isupper():
                continue
            reverse[int(value)].add(name)
    return {pos: tuple(sorted(names)) for pos, names in reverse.items()}


def _dim_names_at(
    position: int,
    dim_name_map: Mapping[int, Tuple[str, ...]],
) -> Tuple[str, ...]:
    return dim_name_map.get(int(position), ())


def _is_condition_dim_name(name: str) -> bool:
    return any(name.startswith(p) for p in GATE_CONDITION_PREFIXES)


def _classify_dim_names_as_condition(names: Sequence[str]) -> Optional[str]:
    """Return the first gate-condition-prefixed name in ``names``, or None."""
    for n in names:
        if _is_condition_dim_name(n):
            return n
    return None


def _classify_dim_names_as_uniform(names: Sequence[str]) -> bool:
    """K-side names count as uniform only if EVERY name at the position is
    in :data:`UNIFORM_K_DIM_NAMES` (i.e. ``CONST``). If any alias name at
    the position is non-CONST, the slot has a real discriminator there.
    """
    if not names:
        return False
    return all(n in UNIFORM_K_DIM_NAMES for n in names)


def _q_side_score_contributions_at_slot(
    spec,
    slot: int,
    cond_dim: int,
    k_const_weight: float,
    dim_name_map: Mapping[int, Tuple[str, ...]],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate the per-K-row score contribution from ``slot`` against a
    K row whose only non-trivial activation at the slot is ``CONST``
    (value 1), in two regimes:

    * ``off_score`` — gate condition is OFF (``cond_dim`` value 0). Only
      CONST-dim Q-side writes contribute. ``None`` when there is no
      such non-zero contribution (score is exactly 0).
    * ``on_score`` — gate condition is ON (``cond_dim`` value 1). Both
      the condition dim itself and CONST-dim Q-side writes contribute.
      ``None`` when there is no non-zero contribution.

    Static audit can't pin row-dependent non-CONST Q-side dims, so those
    contributions are treated as 0 in both regimes.
    """

    off_total = 0.0
    on_total = 0.0
    off_has_any = False
    on_has_any = False
    for qw in getattr(spec, "q", ()):
        if int(qw.slot) != slot:
            continue
        q_names = _dim_names_at(qw.dim, dim_name_map)
        if int(qw.dim) == int(cond_dim):
            # Gate condition contributes only in the ON regime.
            on_total += float(qw.weight) * float(k_const_weight)
            on_has_any = True
            continue
        if not q_names or "CONST" not in q_names:
            # Row-dependent (non-CONST) Q-side dim — undefined statically.
            continue
        contrib = float(qw.weight) * float(k_const_weight)
        off_total += contrib
        on_total += contrib
        off_has_any = True
        on_has_any = True
    off = off_total if off_has_any else None
    on = on_total if on_has_any else None
    return off, on


def audit_attention_gates(
    spec,
    dim_name_map: Optional[Mapping[int, Tuple[str, ...]]] = None,
    *,
    op_name: Optional[str] = None,
    softmax_mode: str = "softmax1",
) -> List[GateAuditEntry]:
    """Classify every Q-side condition gate in one ``DeclarativeAttentionHeadSpec``.

    Walks ``spec.q``; for each write whose dim resolves to a gate-condition
    name (``MARK_*`` / ``OP_*`` / ``HAS_*`` / ``IS_*``) it groups the
    K-side writes by slot and classifies the gate as:

    * ``"safe"`` when K-side writes at least one non-CONST dim at the
      same slot (real discriminator → gate actually filters K positions);
    * ``"softmax1_suppress"`` when ``softmax_mode="softmax1"`` and the
      K-side is CONST-only but the gate-OFF Q×K score contribution at
      the slot is uniformly more negative than
      :data:`SOFTMAX1_SUPPRESS_THRESHOLD` (the zero-anchor sink wins on
      every K row → row is suppressed → gate IS effective);
    * ``"no_op"`` when K-side writes only CONST(s) at the slot AND no
      softmax1 suppression applies (gate is ineffective under both
      softmax and softmax1);
    * ``"q_only"`` when K-side has no write at the slot at all (the
      Q-side write is literal dead weight).

    ``softmax_mode`` defaults to ``"softmax1"`` — the deployed model's
    ``attention_normalization``. Passing ``"softmax"`` reverts to the
    legacy "uniform-K → cancels" verdict and is useful for cross-checks.

    Returns the entries in spec-Q order. ``dim_name_map`` defaults to the
    ``_SetDim`` table; passing a compiler-allocated map handles
    ``pin_io_only=True`` layouts where ``CONST`` may live at a non-default
    position.
    """

    if dim_name_map is None:
        dim_name_map = build_dim_name_map(None)

    if softmax_mode not in ("softmax1", "softmax"):
        raise ValueError(
            f"audit_attention_gates: softmax_mode must be 'softmax1' or "
            f"'softmax' (got {softmax_mode!r})"
        )

    head_idx = getattr(spec, "head_idx", None)
    head_idx = int(head_idx) if head_idx is not None else None

    # Group K-side writes by slot — keep names *and* the CONST weight so
    # the softmax1 suppression check can do real arithmetic.
    k_by_slot: Dict[int, List[str]] = defaultdict(list)
    k_const_weight_by_slot: Dict[int, float] = defaultdict(float)
    for kw in getattr(spec, "k", ()):
        names = _dim_names_at(kw.dim, dim_name_map)
        slot = int(kw.slot)
        if not names:
            # Unknown dim — record by numeric value so we don't drop it.
            k_by_slot[slot].append(f"<unknown:{int(kw.dim)}>")
        else:
            k_by_slot[slot].extend(names)
            if "CONST" in names:
                k_const_weight_by_slot[slot] += float(kw.weight)

    out: List[GateAuditEntry] = []
    for qw in getattr(spec, "q", ()):
        q_names = _dim_names_at(qw.dim, dim_name_map)
        cond_name = _classify_dim_names_as_condition(q_names)
        if cond_name is None:
            continue
        slot = int(qw.slot)
        k_names_at_slot = tuple(sorted(set(k_by_slot.get(slot, []))))
        if not k_names_at_slot:
            kind = "q_only"
        elif _classify_dim_names_as_uniform(k_names_at_slot):
            kind = "no_op"
            if softmax_mode == "softmax1":
                k_const_w = k_const_weight_by_slot.get(slot, 0.0)
                off_score, on_score = _q_side_score_contributions_at_slot(
                    spec, slot, qw.dim, k_const_w, dim_name_map,
                )
                # The gate is effective under softmax1 if EITHER regime
                # (cond ON or cond OFF) drives the slot score below the
                # zero-anchor sink threshold — that regime's K rows are
                # suppressed relative to the sink, so the gate
                # discriminates between ON and OFF.
                if (
                    off_score is not None
                    and off_score <= SOFTMAX1_SUPPRESS_THRESHOLD
                ) or (
                    on_score is not None
                    and on_score <= SOFTMAX1_SUPPRESS_THRESHOLD
                ):
                    kind = "softmax1_suppress"
        else:
            kind = "safe"
        out.append(
            GateAuditEntry(
                slot=slot,
                q_dim_name=cond_name,
                kind=kind,
                k_dim_names=k_names_at_slot,
                head_idx=head_idx,
                op_name=op_name,
                softmax_mode=softmax_mode,
            )
        )
    return out


def _extract_attention_specs_from_op(
    op,
    dim_positions: Optional[Mapping[str, int]] = None,
) -> List[Any]:
    """Pull ``DeclarativeAttentionHeadSpec``s from an op.

    Tries ``op.compiler_ir`` first, falls back to
    ``op.compiler_ir_factory(dim_positions, HD)``. Returns ``[]`` for ops
    with no IR (e.g. legacy imperative bakes — those are covered by
    ``tools/q_side_gate_audit.py``).
    """

    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        factory = getattr(op, "compiler_ir_factory", None)
        if factory is None or dim_positions is None:
            return []
        try:
            ir = factory(dim_positions, 8)
        except Exception:
            return []
    if ir is None:
        return []
    specs: List[Any] = []
    layers = getattr(ir, "layers", None) or ()
    for layer in layers:
        attention = getattr(layer, "attention", None)
        if attention is None:
            continue
        for head in getattr(attention, "rules", ()) or ():
            spec = getattr(head, "spec", None)
            if spec is not None:
                specs.append(spec)
    return specs


def audit_compiler_attention_gates(
    compiler,
    dim_positions: Optional[Mapping[str, int]] = None,
    *,
    softmax_mode: str = "softmax1",
) -> List[GateAuditEntry]:
    """Run the gate audit across every declarative attention spec in a compiler.

    Iterates ``compiler.ops``, ``compiler.block_ops`` and
    ``compiler.model_ops`` (any may carry a ``compiler_ir`` or
    ``compiler_ir_factory``). For each spec, classifies its Q-side
    condition gates and tags entries with the originating ``op_name`` for
    cross-reference with the audit doc.

    ``dim_positions`` is the compiler-allocated position map. When
    available the audit uses it to resolve aliased dims (e.g. when
    ``pin_io_only=True`` moves CONST off its legacy slot); when omitted it
    falls back to the ``_SetDim`` table.

    ``softmax_mode`` defaults to ``"softmax1"`` (matches the deployed
    model). See :func:`audit_attention_gates` for the classifier rules.
    """

    dim_name_map = build_dim_name_map(dim_positions)
    entries: List[GateAuditEntry] = []
    for op_list_name in ("ops", "block_ops", "model_ops"):
        ops = getattr(compiler, op_list_name, None) or ()
        for op in ops:
            op_name = getattr(op, "name", "<anon>")
            specs = _extract_attention_specs_from_op(op, dim_positions)
            for spec in specs:
                entries.extend(
                    audit_attention_gates(
                        spec, dim_name_map,
                        op_name=op_name,
                        softmax_mode=softmax_mode,
                    )
                )
    return entries


def format_gate_audit_report(entries: Sequence[GateAuditEntry]) -> str:
    """Human-readable summary of the no-op / q-only entries.

    ``softmax1_suppress`` entries are counted separately and are not
    listed as flagged (they are effective gates under softmax1).
    """

    by_kind: Dict[str, List[GateAuditEntry]] = defaultdict(list)
    for e in entries:
        by_kind[e.kind].append(e)
    total = len(entries)
    safe = len(by_kind.get("safe", []))
    no_op = len(by_kind.get("no_op", []))
    q_only = len(by_kind.get("q_only", []))
    sm1_suppress = len(by_kind.get("softmax1_suppress", []))
    lines = [
        f"ATTENTION GATE AUDIT: {total} Q-side condition gate(s) — "
        f"{safe} safe, {sm1_suppress} softmax1_suppress (K=CONST but "
        f"row-suppressing under softmax1), {no_op} no-op (K=CONST), "
        f"{q_only} q-only (no K at slot)."
    ]
    flagged = list(by_kind.get("no_op", [])) + list(by_kind.get("q_only", []))
    by_op: Dict[str, List[GateAuditEntry]] = defaultdict(list)
    for e in flagged:
        by_op[e.op_name or "<anon>"].append(e)
    for op_name in sorted(by_op):
        op_entries = by_op[op_name]
        lines.append(f"  - {op_name}: {len(op_entries)} flagged")
        for e in op_entries[:8]:
            k_desc = (
                "(no K at slot)" if e.kind == "q_only"
                else f"K=[{', '.join(e.k_dim_names) or '-'}]"
            )
            head_desc = (
                f" head={e.head_idx}" if e.head_idx is not None else ""
            )
            lines.append(
                f"      slot={e.slot} Q={e.q_dim_name}{head_desc} "
                f"{k_desc} [{e.kind}]"
            )
        if len(op_entries) > 8:
            lines.append(f"      ... (+{len(op_entries) - 8} more)")
    return "\n".join(lines)


def run_attention_gate_audit(
    compiler,
    dim_positions: Optional[Mapping[str, int]] = None,
    *,
    softmax_mode: str = "softmax1",
) -> List[GateAuditEntry]:
    """Compile-time entry point: audit every attention spec and warn.

    * Honours ``C4_SKIP_GATE_CHECK=1`` (returns empty list, no warning).
    * Default: emits a single :func:`warnings.warn` summarising the
      no-op / q-only entries when the count is non-zero. Entries
      classified as ``softmax1_suppress`` are treated as effective and
      do *not* contribute to the flagged set.
    * When ``C4_STRICT_GATE_CHECK=1``: raises ``GateAuditError`` instead
      of warning, listing every flagged entry.

    ``softmax_mode`` defaults to ``"softmax1"`` (matches the deployed
    model). Pass ``"softmax"`` to recover the legacy (pre-softmax1-aware)
    verdict for cross-checks.

    Returns the *raw* audit-entry list (safe + suppress entries included)
    so callers can post-process or compare against a known baseline.
    """

    if os.environ.get(GATE_AUDIT_SKIP_ENV, "") == "1":
        return []
    entries = audit_compiler_attention_gates(
        compiler, dim_positions, softmax_mode=softmax_mode,
    )
    flagged = [
        e for e in entries
        if e.kind not in ("safe", "softmax1_suppress")
    ]
    if not flagged:
        return entries
    report = format_gate_audit_report(entries)
    if os.environ.get(GATE_AUDIT_STRICT_ENV, "") == "1":
        raise GateAuditError(report, flagged)
    warnings.warn(report, stacklevel=3)
    return entries


class GateAuditError(RuntimeError):
    """Raised by :func:`run_attention_gate_audit` under strict-check mode.

    Carries the flagged entries on ``self.flagged`` for programmatic
    inspection by callers / tests.
    """

    def __init__(self, message: str, flagged: Sequence[GateAuditEntry]):
        super().__init__(message)
        self.flagged: List[GateAuditEntry] = list(flagged)


__all__ = [
    "DSLInterpreter",
    "InterpreterStep",
    "InterpreterResult",
    "GateAuditEntry",
    "GateAuditError",
    "audit_attention_gates",
    "audit_compiler_attention_gates",
    "build_dim_name_map",
    "format_gate_audit_report",
    "run_attention_gate_audit",
    "GATE_AUDIT_SKIP_ENV",
    "GATE_AUDIT_STRICT_ENV",
    "GATE_CONDITION_PREFIXES",
    "SOFTMAX1_SUPPRESS_THRESHOLD",
    "UNIFORM_K_DIM_NAMES",
]
