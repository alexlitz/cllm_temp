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

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ir import (
    CompilerIR,
    DimRef,
    FFNOp,
    FFNRule,
    TokenEmbeddingRule,
    _state_value,
)


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
        ``state[o.dim] += state.get(spec.value_dim_for(o.slot), 0)``.
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
                key = o.dim
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
            step.notes.append(
                f"embed token={rule.token} table={rule.table} "
                f"dim={rule.target_dim} value={rule.value}"
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


__all__ = [
    "DSLInterpreter",
    "InterpreterStep",
    "InterpreterResult",
]
