"""Value-faithful, pure-IR forward simulator for the declarative VM.

The legacy :class:`~.dsl_interpreter.DSLInterpreter` is COVERAGE-faithful but
NOT VALUE-faithful: it is a bag-of-dims dict, models attention as a
context-free ``V -> O`` copy, applies every FFN rule whose linear condition
sum clears threshold (no SwiGLU nonlinearity), has no token positions, no
softmax, no argmax, and no cross-step carry. So it can tell you *whether* a
rule writes a cell, but not *which* one-hot wins the decode.

This module closes those four gaps and makes the interpreter a real
**pure-IR forward simulator**: it executes the SAME ``CompilerIR`` the
weight lowering consumes, but reproduces the deployed transformer's forward
math directly from the IR (no torch-lowering of fresh weight matrices, no
baked model). The output is a per-token residual stream + an argmax decode
that is byte-for-byte comparable with the real ``AutoregressiveVM``.

The four gaps, and how each is made faithful to the real op:

1. **Per-token positions.** The state is a real ``[S, d_model]`` residual
   tape over the 35-token VM-step window layout (PC@0+4, AX@5+4, SP@10+4,
   BP@15+4, STACK0@20+4, MEM@25 + 4 addr + 4 val, STEP_END@34), spanning
   *every* step in the context — not one bag per step. Each token has a real
   absolute position used by ALiBi.

2. **Real attention.** Each :class:`DeclarativeAttentionHeadSpec` is executed
   as true multi-head attention:
   ``softmax1((Q·Kᵀ)·scale + ALiBi_bias + causal) · V -> O`` with the
   per-head ALiBi slope (``spec.alibi_slope`` or the model default
   ``2**(-8/H*(head_idx+1))``), the ``head_idx*HD+slot`` row layout, and the
   ``W_o`` out-dim writes. This is the exact math of
   ``AutoregressiveAttention.forward`` (``use_softmax1=True`` + ALiBi),
   producing actual, context-dependent value routing.

3. **Real nonlinearity + argmax.** FFN ops apply the real SwiGLU at the
   declared scale ``S`` exactly as ``Primitives.lower_ffn`` /
   ``PureFFN.forward`` do: per rule, ``up = S*(Σ cond·w) - S*threshold``,
   ``gate = gate_bias + Σ gate·w``, ``hidden = silu(up)*gate``, and the
   write adds ``hidden * write.weight`` to the output dim. The token decode is
   **argmax** over the LM head (``head.weight·resid + head.bias``); the model
   has NO final norm, so this is exact.

4. **Cross-step state.** Because the residual tape spans the full multi-step
   context (autoregressive), attention at any token can read prior-step token
   positions, so cross-step carries (the ``*_PREV`` bands, the AX/STACK0
   relays) are modeled exactly as the real model models them — by attention
   reaching back across the 35-token step boundary.

Faithful-coverage boundary
---------------------------
A handful of ops are still imperative ``bake_fn``-only (the ALU divmod
compute being migrated to declarative IR in task #230, plus a few
model-level slope/seed ops). Those are NOT IR-executable; this interpreter
flags them as a coverage gap (``opaque_ops``) rather than faking them. Every
other op runs through the faithful IR forward.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .ir import CompilerIR
from .symbolic_forward import _OPCODE_NAMES, decode_instr  # noqa: F401


# ---------------------------------------------------------------------------
# 35-token VM-step window layout (mirrors dim_oracle / DraftVM.draft_tokens).
# ---------------------------------------------------------------------------

STEP_TOKENS = 35

POS_PC_MARKER = 0
POS_AX_MARKER = 5
POS_SP_MARKER = 10
POS_BP_MARKER = 15
POS_STACK0_MARKER = 20
POS_MEM_MARKER = 25
POS_STEP_END = 34


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class OpTrace:
    """One executed op + the dims/positions it touched (for attribution)."""

    name: str
    kind: str
    layer_idx: int
    rules_fired: int = 0
    heads_fired: int = 0
    note: str = ""


@dataclass
class FaithfulResult:
    """Result of a faithful forward over a token context.

    ``residual`` is the final ``[S, d_model]`` tape; ``logits`` is
    ``[S, vocab]``; ``decoded`` is the per-position argmax token id.
    """

    residual: torch.Tensor
    logits: torch.Tensor
    decoded: List[int]
    traces: List[OpTrace] = field(default_factory=list)
    opaque_skipped: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Op IR extraction (shared shape with dsl_interpreter / dsl_spec_gate).
# ---------------------------------------------------------------------------


def extract_op_ir(op, dim_positions: Mapping[str, int], HD: int = 8) -> Optional[CompilerIR]:
    """Return a ``CompilerIR`` for ``op`` (pre-built or factory), else None."""
    ir = getattr(op, "compiler_ir", None)
    if ir is not None:
        return ir
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is not None:
        try:
            return factory(dim_positions, HD)
        except Exception:
            return None
    return None


def op_is_ir_executable(op, dim_positions: Mapping[str, int], HD: int = 8) -> bool:
    return extract_op_ir(op, dim_positions, HD) is not None


# ---------------------------------------------------------------------------
# The faithful forward
# ---------------------------------------------------------------------------


class FaithfulInterpreter:
    """Pure-IR forward simulator over a per-token residual tape.

    Construct with the compiled layout's ``dim_positions`` + the per-physical-
    block op schedule (``ops_per_block``), the model dims, and the default head
    count. Then call :meth:`forward` with a seeded ``[S, d_model]`` residual
    (typically ``model.embed(token_ids)``) to run every block's attention + FFN
    faithfully and argmax-decode through the head.

    The interpreter never bakes weights: attention and FFN are computed
    directly from each op's ``CompilerIR`` (head specs / FFN rules), matching
    the lowering semantics term-for-term.
    """

    def __init__(
        self,
        *,
        dim_positions: Mapping[str, int],
        ops_per_block: Sequence[Sequence[Any]],
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        S: float = 100.0,
        use_softmax1: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.dim_positions = dict(dim_positions)
        self.ops_per_block = [list(b) for b in ops_per_block]
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.HD = int(head_dim) if head_dim is not None else self.d_model // self.num_heads
        self.S = float(S)
        self.use_softmax1 = bool(use_softmax1)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        # Reverse map: residual column -> dim name (for attribution).
        self._col_to_name: Dict[int, str] = {}
        for name, col in self.dim_positions.items():
            self._col_to_name.setdefault(int(col), name)

    # ----- per-op application over the full tape ---------------------------

    def _default_slope(self, head_idx: int) -> float:
        return 2.0 ** (-8.0 / self.num_heads * (head_idx + 1))

    def _apply_attention_op(
        self, ir: CompilerIR, x: torch.Tensor, layer_idx: int, trace: OpTrace,
    ) -> torch.Tensor:
        """Run one layer's attention specs as real softmax1+ALiBi MHA.

        ``x`` is ``[S, d_model]``. Returns the post-attention residual
        (with the residual add). Math matches
        ``AutoregressiveAttention.forward`` exactly for the ALiBi+softmax1
        path: per head, ``Q=resid·Wq^T`` over the spec's slot writes,
        ``scores=(Q·Kᵀ)*scale - slope*|i-j|`` (causal), softmax1, then
        ``W_o`` scatters the value slots back to residual out-dims.
        """
        layer = ir.layer(0)
        heads = getattr(layer.attention, "rules", ())
        if not heads:
            return x
        S = x.shape[0]
        scale = 1.0 / math.sqrt(float(self.HD))
        # Absolute positions for ALiBi distance.
        pos = torch.arange(S, device=x.device, dtype=x.dtype)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # [S, S]
        causal = torch.triu(
            torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        out_delta = torch.zeros_like(x)
        for head in heads:
            spec = head.spec
            hd = spec.effective_head_dim(self.HD)
            # Build Q/K/V projections [S, hd] from the slot/dim/weight writes.
            Q = torch.zeros(S, hd, device=x.device, dtype=x.dtype)
            K = torch.zeros(S, hd, device=x.device, dtype=x.dtype)
            V = torch.zeros(S, hd, device=x.device, dtype=x.dtype)
            for w in spec.q:
                Q[:, int(w.slot)] += x[:, int(w.dim)] * float(w.weight)
            for w in spec.k:
                K[:, int(w.slot)] += x[:, int(w.dim)] * float(w.weight)
            for w in spec.v:
                V[:, int(w.slot)] += x[:, int(w.dim)] * float(w.weight)
            slope = (
                float(spec.alibi_slope)
                if spec.alibi_slope is not None
                else self._default_slope(int(spec.head_idx))
            )
            scores = (Q @ K.t()) * scale  # [S, S]
            scores = scores - slope * dist
            scores = scores + causal
            if self.use_softmax1:
                # softmax1: anchor=0 sink in the denominator (ZFOD).
                anchor = torch.zeros((), device=x.device, dtype=x.dtype)
                max_val = torch.maximum(scores.amax(dim=-1, keepdim=True), anchor)
                exp_scores = torch.exp(scores - max_val)
                exp_anchor = torch.exp(anchor - max_val)
                attn = exp_scores / (exp_anchor + exp_scores.sum(dim=-1, keepdim=True))
            else:
                attn = torch.softmax(scores, dim=-1)
            head_out = attn @ V  # [S, hd]
            # W_o scatter: out_dim += head_out[:, slot] * weight.
            for w in spec.o:
                out_delta[:, int(w.out_dim)] += head_out[:, int(w.slot)] * float(w.weight)
            trace.heads_fired += 1
        return x + out_delta

    def _apply_ffn_op(
        self, ir: CompilerIR, x: torch.Tensor, layer_idx: int, trace: OpTrace,
    ) -> torch.Tensor:
        """Run one layer's FFN rules as real SwiGLU at scale S.

        Math matches ``Primitives.lower_ffn`` + ``PureFFN.forward``:
        per rule (one hidden unit),
            up   = S * Σ_cond (resid·w)  -  S*threshold
            gate = gate_bias + Σ_gate (resid·w)   (+ resid[gate]·gate_weight)
            hidden = silu(up) * gate
            resid[write.dim] += hidden * write.weight
        The residual add is applied once (delta accumulated across units).
        """
        rules = ir.layer(0).ffn.rules
        if not rules:
            return x
        S = x.shape[0]
        dp = self.dim_positions
        delta = torch.zeros_like(x)
        Sval = self.S
        for rule in rules:
            # up pre-activation: S * (Σ cond) - S*threshold
            up = torch.full((S,), -Sval * float(rule.threshold), device=x.device, dtype=x.dtype)
            for term in rule.conditions:
                col = term.dim.resolve(dp)
                up = up + Sval * float(term.weight) * x[:, col]
            # gate pre-activation.
            gate = torch.full((S,), float(rule.gate_bias), device=x.device, dtype=x.dtype)
            if rule.gate is not None:
                gate = gate + float(rule.gate_weight) * x[:, rule.gate.resolve(dp)]
            for term in rule.gate_terms:
                gate = gate + float(term.weight) * x[:, term.dim.resolve(dp)]
            hidden = torch.nn.functional.silu(up) * gate  # [S]
            if not torch.any(hidden != 0):
                continue
            for w in rule.writes:
                delta[:, w.dim.resolve(dp)] += hidden * float(w.weight)
            trace.rules_fired += 1
        return x + delta

    def _apply_op(self, op, x: torch.Tensor) -> Tuple[torch.Tensor, OpTrace, bool]:
        """Dispatch one op; returns (new_residual, trace, was_ir_executable)."""
        name = getattr(op, "name", "<anon>")
        kind = getattr(op, "kind", "ffn")
        layer_idx = getattr(op, "layer_idx", 0) or 0
        trace = OpTrace(name=name, kind=kind, layer_idx=layer_idx)
        ir = extract_op_ir(op, self.dim_positions, self.HD)
        if ir is None:
            trace.note = "opaque (imperative bake_fn; not IR-executable)"
            return x, trace, False
        # attention first, then ffn — within one op the IR carries both.
        x = self._apply_attention_op(ir, x, layer_idx, trace)
        x = self._apply_ffn_op(ir, x, layer_idx, trace)
        return x, trace, True

    # ----- public forward --------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        seed_residual: torch.Tensor,
        *,
        head_weight: Optional[torch.Tensor] = None,
        head_bias: Optional[torch.Tensor] = None,
    ) -> FaithfulResult:
        """Run every block's attention + FFN over the seeded residual tape.

        ``seed_residual`` is ``[S, d_model]`` — the embedding output for the
        token context (use ``model.embed(token_ids)[0]``). Blocks run in
        physical order. When ``head_weight``/``head_bias`` are supplied the
        decode is argmax over ``head_weight·resid + head_bias`` (the real LM
        head — no final norm).
        """
        x = seed_residual.to(self.device, self.dtype).clone()
        traces: List[OpTrace] = []
        opaque: List[str] = []
        for block_ops in self.ops_per_block:
            # Attention ops fire before FFN ops within a physical block,
            # matching TransformerBlock.forward (attn then ffn).
            attn_ops = [o for o in block_ops if getattr(o, "kind", "ffn") == "attn"]
            ffn_ops = [o for o in block_ops if getattr(o, "kind", "ffn") == "ffn"]
            other = [
                o for o in block_ops
                if getattr(o, "kind", "ffn") not in ("attn", "ffn")
            ]
            for op in attn_ops + ffn_ops + other:
                x, trace, ok = self._apply_op(op, x)
                traces.append(trace)
                if not ok:
                    opaque.append(trace.name)
        if head_weight is not None:
            logits = x @ head_weight.t().to(x.device, x.dtype)
            if head_bias is not None:
                logits = logits + head_bias.to(x.device, x.dtype)
        else:
            logits = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        decoded = logits.argmax(dim=-1).tolist()
        return FaithfulResult(
            residual=x, logits=logits, decoded=decoded,
            traces=traces, opaque_skipped=opaque,
        )

    # ----- attribution -----------------------------------------------------

    def name_for_col(self, col: int) -> str:
        return self._col_to_name.get(int(col), f"<col {col}>")

    def attribute_runtime_contribution(
        self,
        op_list: Sequence[Any],
        col: int,
        resid_at_pos: torch.Tensor,
    ) -> List[Tuple[str, str, float]]:
        """Rank declarative FFN rules by their RUNTIME SwiGLU contribution to
        ``col`` given the residual ``resid_at_pos`` (a ``[d_model]`` vector at
        one token position).

        This is the *structural* attribution: not "which rule could write the
        cell" (every static-default rule does) but "which rule's actual SwiGLU
        output dominates the wrong value at this position in the faithful
        forward". Returns ``[(op_name, rule_name, contribution)]`` sorted by
        ``|contribution|`` descending. Pairs with the dim_registry ownership
        so a wrong byte pins to the rule that produced it.
        """
        col = int(col)
        x = resid_at_pos.to(self.device, self.dtype)
        Sval = self.S
        dp = self.dim_positions

        def rd(ref) -> int:
            # Resolve a DimRef to a column; unknown dims (not allocated in
            # this layout, e.g. flag-gated bands) read as 0 — the same as the
            # residual stream treats an absent dim.
            try:
                return ref.resolve(dp)
            except KeyError:
                return -1

        def val(c: int) -> float:
            return float(x[c]) if 0 <= c < x.shape[0] else 0.0

        out: List[Tuple[str, str, float]] = []
        for op in op_list:
            ir = extract_op_ir(op, dp, self.HD)
            if ir is None:
                continue
            op_name = getattr(op, "name", "<anon>")
            for rule in ir.layer(0).ffn.rules:
                writes_col = [w for w in rule.writes if rd(w.dim) == col]
                if not writes_col:
                    continue
                up = -Sval * float(rule.threshold)
                for term in rule.conditions:
                    up += Sval * float(term.weight) * val(rd(term.dim))
                gate = float(rule.gate_bias)
                if rule.gate is not None:
                    gate += float(rule.gate_weight) * val(rd(rule.gate))
                for term in rule.gate_terms:
                    gate += float(term.weight) * val(rd(term.dim))
                hidden = float(torch.nn.functional.silu(torch.tensor(up)).item()) * gate
                for w in writes_col:
                    contrib = hidden * float(w.weight)
                    if abs(contrib) > 1e-9:
                        out.append((op_name, rule.name or "<rule>", contrib))
        out.sort(key=lambda t: abs(t[2]), reverse=True)
        return out

    def attribute_residual_dim(
        self,
        op_list: Sequence[Any],
        col: int,
    ) -> List[Tuple[str, str]]:
        """Return ``[(op_name, role)]`` for every op whose IR writes ``col``.

        ``role`` is ``"attn.o"`` (an attention W_o out-dim write) or
        ``"ffn.write"`` (an FFN rule write). This is the IR-level ownership
        attribution: given a residual column that holds a wrong value, which
        declarative op(s) produced it. Pairs with the dim_registry to name the
        owning rule.
        """
        out: List[Tuple[str, str]] = []
        col = int(col)
        for op in op_list:
            ir = extract_op_ir(op, self.dim_positions, self.HD)
            if ir is None:
                continue
            name = getattr(op, "name", "<anon>")
            layer = ir.layer(0)
            for head in getattr(layer.attention, "rules", ()):
                for w in head.spec.o:
                    if int(w.out_dim) == col:
                        out.append((name, f"attn.o head={head.spec.head_idx} slot={w.slot} w={w.weight}"))
            for rule in layer.ffn.rules:
                for w in rule.writes:
                    try:
                        wcol = w.dim.resolve(self.dim_positions)
                    except KeyError:
                        continue
                    if wcol == col:
                        rname = rule.name or "<rule>"
                        out.append((name, f"ffn.write rule={rname} w={w.weight}"))
        return out


# ---------------------------------------------------------------------------
# Convenience: token decode helpers (step-tape register readout).
# ---------------------------------------------------------------------------


def step_token_positions(n_steps: int, prompt_len: int = 0) -> Dict[int, str]:
    """Map absolute token positions to their step-tape role name.

    Returns ``{abs_pos: role}`` for ``n_steps`` step windows starting at
    ``prompt_len``. Used by the validator to find the AX/PC marker rows whose
    *next-token* argmax is the decoded register byte.
    """
    roles = {
        POS_PC_MARKER: "PC_MARK", POS_AX_MARKER: "AX_MARK",
        POS_SP_MARKER: "SP_MARK", POS_BP_MARKER: "BP_MARK",
        POS_STACK0_MARKER: "STACK0_MARK", POS_MEM_MARKER: "MEM_MARK",
        POS_STEP_END: "STEP_END",
    }
    out: Dict[int, str] = {}
    for step in range(n_steps):
        base = prompt_len + step * STEP_TOKENS
        for off in range(STEP_TOKENS):
            out[base + off] = roles.get(off, f"byte{off}")
    return out


__all__ = [
    "FaithfulInterpreter",
    "FaithfulResult",
    "OpTrace",
    "extract_op_ir",
    "op_is_ir_executable",
    "step_token_positions",
    "STEP_TOKENS",
]
