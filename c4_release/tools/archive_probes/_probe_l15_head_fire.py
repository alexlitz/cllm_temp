#!/usr/bin/env python3
"""Which L15 head delivers OUTPUT_LO[8] at the step-9 ADJ AX row?

CPU faithful. Builds the real tape, finds the AX-b0 row at the given step,
runs the L15 block forward capturing per-head attention weights + O-output, and
reports which head writes OUTPUT_LO[8] (the value_scale=40 over-delivery).

Usage: python tools/_probe_l15_head_fire.py [id] [step] [nib]
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
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    model = ctx.model
    dp = ctx.dim_positions
    LO = dp["OUTPUT_LO"]
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
    # find the L15 physical block. layer15_memory_lookup ~ physical block 25-ish.
    # Hook every block's attn; capture per-head O contribution into OUTPUT_LO[nib].
    padded = torch.tensor([trunc], dtype=torch.long, device=next(model.parameters()).device)
    row = len(trunc) - 1
    col = LO + nib
    # locate L15 block by name
    target = None
    for bi, blk in enumerate(model.blocks):
        nm = type(blk.attn).__name__
        # check if this block's attn writes OUTPUT_LO heavily; just hook all and
        # record the per-block delta into col.
    print(f"id{pid} step{tstep} AX-b0 row={axrow} col=OUTPUT_LO[{nib}]={col}")
    prev = 0.0
    for bi in range(len(model.blocks)):
        out = model.forward(padded, stop_after_block=bi)[0][row]
        v = out[col].item()
        d = v - prev
        if abs(d) > 0.3:
            print(f"  block {bi:3d} {type(model.blocks[bi].attn).__name__:28s}: OUTPUT_LO[{nib}]={v:+.3f} (Δ{d:+.3f})")
        prev = v


if __name__ == "__main__":
    main()
