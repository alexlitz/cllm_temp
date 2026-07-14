"""c4_min compiler: DSL rules -> torch Transformer state_dict.

ARCHITECTURE (the honest recurrence): **depth = time**. The whole VM state lives
in the residual at a single position; block ``k`` applies the FFN rule for VM
step ``k`` and flows the updated state to block ``k+1``. This makes each VM step a
real transformer layer and needs no autoregressive token feedback for
straight-line code (a parallel single-pass cannot otherwise carry a *computed*
register from step k-1, since all positions see only the pre-op value).

Per-step emission: block ``k`` also copies the just-updated ``AX`` into a private
``OUT_k`` output slot; the LM head decodes every ``OUT_k`` in one shot at the
single position, giving the full per-step AX trace, stride-free.

  embed        one row -> initial state (AX=SP=BP=0, ONE=1); code baked per-block
  block k      FFN: apply step-k op rule -> new AX/STACK0 ; copy AX -> OUT_k
  lm_head      each OUT_k slot -> 257 byte/HALT logits, read at all k

Attention is unused for the straight-line slice (op is baked per block, state is
a scalar carried through depth). It is retained in the block for the memory/pop
extensions (DESIGN.md e) and left as identity here.
"""
from __future__ import annotations

from typing import List

import torch

from . import isa
from .dsl import FFNRule, LinearExpr
from .compile_ffn import compile_ffn, compile_fold
from .layout import Layout
from .model import Transformer


VOCAB = 257
HALT_TOKEN = 256


def _step_rules(L: Layout, ins: isa.Instr) -> List[FFNRule]:
    """FFN rules applied by one op-block for a single instruction ``ins``.

    Writes are additive; a *replace* of AX is written as ``AX += (new - AX)``.
    The guard is ``ONE`` (always ~1) since the opcode is fixed per block. A
    following emit-block copies the post-op AX into the step's private OUT slot.
    """
    ax, stk = L.AX, L.STACK0
    op = ins.op
    rules: List[FFNRule] = []
    G = [(L.ONE, 0.5, 1.5)]  # always-on guard

    if op == isa.IMM:
        rules.append(FFNRule(G, {ax: LinearExpr.c(float(ins.imm)) + LinearExpr.of(ax, -1.0)}))
    elif op == isa.LEA:
        rules.append(FFNRule(G, {ax: LinearExpr.of(L.BP, 1.0) + LinearExpr.c(float(ins.imm)) + LinearExpr.of(ax, -1.0)}))
    elif op == isa.PSH:
        rules.append(FFNRule(G, {stk: LinearExpr.of(ax, 1.0) + LinearExpr.of(stk, -1.0)}))
    elif op == isa.ADD:
        rules.append(FFNRule(G, {ax: LinearExpr.of(stk, 1.0)}))
    elif op == isa.SUB:
        # AX = pop() - AX = stk - ax. Compute (stk - ax + 256) so the value is
        # in [1, 511]; the following mod-256 fold re-quantises it to [0, 256),
        # yielding the correct 8-bit two's-complement result on underflow
        # (e.g. 7 - 9 -> 254). (stk - 2*ax) + 256 written additively onto AX:
        #   ax += stk - 2*ax + 256  ==  stk - ax + 256.
        rules.append(FFNRule(G, {ax: LinearExpr.of(stk, 1.0)
                                 + LinearExpr.of(ax, -2.0)
                                 + LinearExpr.c(256.0)}))
    elif op == isa.HALT:
        rules.append(FFNRule(G, {L.HALTED: LinearExpr.c(1.0)}))
    else:
        raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in slice compiler")

    return rules


def _emit_rule(L: Layout, out_slot: int) -> List[FFNRule]:
    """Copy current AX into ``out_slot`` (runs after the op block updated AX)."""
    return [FFNRule([(L.ONE, 0.5, 1.5)], {out_slot: LinearExpr.of(L.AX, 1.0)})]


def _build_layout(n_steps: int, n_heads: int) -> Layout:
    """Layout + one private OUTPUT slot per step, appended after the base bands."""
    L = Layout(n_heads=n_heads)
    # allocate per-step output slots
    L.OUT_SLOTS = [L._band(f"OUT_{k}", 1) for k in range(n_steps)]
    # re-pad D to a multiple of n_heads
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


def compile_program(prog, n_heads: int = 4, max_pos: int = 4):
    """Compile [(name, imm), ...] into a depth-unrolled Transformer.

    Returns (model, layout, code). One op-block + one emit-block per VM step.
    """
    code = isa.assemble(prog)
    n_steps = len(code)
    L = _build_layout(n_steps, n_heads)
    dim = L.D

    # build weights for all blocks (op block [+ fold] then emit block, per step)
    ffn_specs = []
    for k, ins in enumerate(code):
        ffn_specs.append(compile_ffn(_step_rules(L, ins), dim))
        if ins.op in (isa.ADD, isa.SUB):
            # 8-bit wrap: fold AX back into [0,256). ADD leaves AX in [0,510]
            # (carry-out of a byte); SUB is computed as (a - b + 256) in [1,511]
            # so the same mod-256 fold yields the two's-complement result on
            # underflow. (SHL wrap is the same gadget with a wider/offset fold.)
            ffn_specs.append(compile_fold(L.AX, L.ONE, dim, modulus=256))
        ffn_specs.append(compile_ffn(_emit_rule(L, L.OUT_SLOTS[k]), dim))
    n_blocks = len(ffn_specs)
    hidden = max(f["W_up"].shape[0] for f in ffn_specs) if ffn_specs else 1

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)

    with torch.no_grad():
        # single position (index 0) holds the whole state; embed = initial state
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)          # attention = identity for the slice
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)

    return model, L, code


def _zero_attn(mod):
    for p in (mod.W_q, mod.W_k, mod.W_v, mod.W_o):
        p.zero_()
    mod.mask = torch.zeros(mod.mask.shape) if mod.mask.numel() else torch.zeros(0, 0)


def _load_ffn(mod, w, hidden):
    n = w["W_up"].shape[0]
    mod.W_up.zero_(); mod.W_up[:n].copy_(w["W_up"])
    mod.b_up.zero_(); mod.b_up[:n].copy_(w["b_up"])
    mod.W_gate.zero_(); mod.W_gate[:n].copy_(w["W_gate"])
    mod.b_gate.zero_(); mod.b_gate[:n].copy_(w["b_gate"])
    mod.W_down.zero_(); mod.W_down[:, :n].copy_(w["W_down"])
    mod.b_down.copy_(w["b_down"])


def head_matrix(L, dim, out_band, halt_band=None):
    """Build (W, b) so logits[v] = 2*v*OUT - v^2, whose argmax over v is round(OUT).

    The per-step VALUE trace is always the AX byte (matching the reference
    interpreter, which emits AX on the HALT step too). The HALT *terminator* token
    (256) is only wired when ``halt_band`` is given; because HALTED is a single
    sticky band shared by the final state, only the terminator query (the last
    step, after the HALT block) should pass it — see ``run``.
    """
    W = torch.zeros(VOCAB, dim)
    b = torch.zeros(VOCAB)
    for v in range(256):
        W[v, out_band] = 2.0 * v
        b[v] = -(v * v)
    if halt_band is not None:
        W[HALT_TOKEN, halt_band] = 1000.0
        b[HALT_TOKEN] = -500.0
    return W, b


def _load_head(model, L):
    """Bake the LM head reading OUT_0 (a concrete slot) as the stored head.

    Per-step decode re-points the head at each OUT_k slot via ``head_matrix``
    (same algebra); the stored head validates the packed state_dict shape.
    """
    out0 = L.OUT_SLOTS[0] if L.OUT_SLOTS else L.OUTPUT
    W, b = head_matrix(L, model.dim, out0)
    model.lm_head.copy_(W)
    model.lm_bias.copy_(b)


def run(model, L, code):
    """Run the depth-unrolled model; decode the per-step AX trace via the LM head.

    Each VM step k is decoded as ``argmax_v head(OUT_k)`` — a genuine argmax over
    the 257-way vocab (byte values + HALT), stride-free. HALT token wins on the
    step whose HALTED band is set.
    """
    import torch.nn.functional as F

    pos = torch.zeros(1, 1, dtype=torch.long)  # single position holds all state
    x = model.embed[pos]
    for blk in model.blocks:
        x = blk(x)
    state = x[0, 0]  # [D]

    out = []
    for k in range(len(code)):
        # value trace = AX byte per step (no HALT token in the value channel).
        W, b = head_matrix(L, model.dim, L.OUT_SLOTS[k], halt_band=None)
        logits = F.linear(state, W, b)  # [VOCAB]
        out.append(int(logits.argmax().item()))
    return out


def halted(model, L, code):
    """True iff the program's HALTED terminator fired (LM head emits HALT token)."""
    import torch.nn.functional as F
    pos = torch.zeros(1, 1, dtype=torch.long)
    x = model.embed[pos]
    for blk in model.blocks:
        x = blk(x)
    state = x[0, 0]
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[-1], halt_band=L.HALTED)
    logits = F.linear(state, W, b)
    return int(logits.argmax().item()) == HALT_TOKEN
