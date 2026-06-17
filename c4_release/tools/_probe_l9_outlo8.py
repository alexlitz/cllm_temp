#!/usr/bin/env python3
"""Attribute OUTPUT_LO[8] at the L9 (block 13) input residual, step-9 ADJ AX row.

The block-13 forward adds +8.8 to OUTPUT_LO[8] (the dominant corruptor). This
reads the residual at the INPUT to block 13 (= output of block 12) and ranks the
L9 FFN rules' runtime SwiGLU contribution to OUTPUT_LO[8] on that residual, so
the owning rule is named. Also dumps the L9 attention O contribution.

Usage: python tools/_probe_l9_outlo8.py [id] [step] [nib] [blk]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.batched_pure_neural import Token

SE = int(Token.STEP_END); AX = int(Token.REG_AX); HALT = int(Token.HALT)


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    tstep = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    nib = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    blk = int(sys.argv[4]) if len(sys.argv) > 4 else 13
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    model = ctx.model
    dp = ctx.dim_positions
    LO = dp["OUTPUT_LO"]; col = LO + nib
    from neural_vm.unified_compiler.faithful_autoregressive import FaithfulAutoregressiveRunner
    with contextlib.redirect_stdout(io.StringIO()):
        ar = FaithfulAutoregressiveRunner(model=model, layout=ctx.layout)
    tape = list(ar._inner._serial._build_context(bc, [], []))
    pl = len(tape)
    for _ in range((tstep + 3) * 35 + 40):
        logits = ctx.fwd.forward(tape)
        nxt = int(logits[len(tape) - 1].argmax())
        tape.append(nxt)
        if nxt == HALT: break
    steps = []; cur = []
    for p in range(pl, len(tape)):
        cur.append(p)
        if tape[p] == SE: steps.append(cur); cur = []
    if tstep >= len(steps): tstep = len(steps) - 1
    axrow = None
    for p in steps[tstep]:
        if tape[p] == AX: axrow = p + 1; break
    trunc = tape[:axrow]
    padded = torch.tensor([trunc], dtype=torch.long, device=next(model.parameters()).device)
    row = len(trunc) - 1
    # residual at INPUT to block blk = output of block blk-1
    resid_in = model.forward(padded, stop_after_block=blk - 1)[0][row]
    resid_out = model.forward(padded, stop_after_block=blk)[0][row]
    print(f"id{pid} step{tstep} AX-b0 row={axrow} OUTPUT_LO[{nib}]: "
          f"in(blk{blk-1})={resid_in[col].item():+.3f} out(blk{blk})={resid_out[col].item():+.3f}")
    # which L9 op writes it: find the op(s) whose name maps to block blk.
    # Use the gate's flat_ffn_ops attribution on the INPUT residual.
    ranked = ctx.interp.attribute_runtime_contribution(ctx.flat_ffn_ops, col, resid_in)
    print(f"\n== rules writing OUTPUT_LO[{nib}] (runtime, on block-{blk} INPUT residual) ==")
    for opn, rn, c in ranked[:12]:
        print(f"   {c:+.4f}  {opn} :: {rn}")
    # static ownership of the col (which op/head), filter to attn.o and l9
    owners = ctx.interp.attribute_residual_dim(ctx.flat_ffn_ops, col)
    print(f"\n== L9 / attn.o static owners of OUTPUT_LO[{nib}] ==")
    seen = set()
    for opn, role in owners:
        if ("layer9" in opn or "l9" in opn.lower()) or "attn.o" in role:
            key = (opn, role.split(' w=')[0])
            if key in seen: continue
            seen.add(key)
            print(f"   {opn} :: {role}")


if __name__ == "__main__":
    main()
