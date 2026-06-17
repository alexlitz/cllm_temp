#!/usr/bin/env python3
"""Diff the pre-head residual at the AX byte-0 row between a GOOD and BAD step.

CPU faithful. Builds the real autoregressive tape, then compares the full
[d_model] residual at step_good's AX-b0 row vs step_bad's, printing the dims that
differ most AND the top vocab logits at each (so we see which token the head
actually picks and which band drives it).

Usage: python tools/_probe_axb0_resid_diff.py [id] [good] [bad]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ["C4_TEST_SPEC_K"] = "0"; os.environ["C4_SMOKE_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.batched_pure_neural import Token

SE = int(Token.STEP_END); AX = int(Token.REG_AX); HALT = int(Token.HALT)


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    sg = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sb = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    inv = {v: k for k, v in dp.items()}
    from neural_vm.unified_compiler.faithful_autoregressive import FaithfulAutoregressiveRunner
    with contextlib.redirect_stdout(io.StringIO()):
        ar = FaithfulAutoregressiveRunner(model=ctx.model, layout=ctx.layout)
    tape = list(ar._inner._serial._build_context(bc, [], []))
    prompt_len = len(tape)
    for _ in range((max(sg, sb) + 3) * 35 + 40):
        logits = ctx.fwd.forward(tape)
        nxt = int(logits[len(tape) - 1].argmax())
        tape.append(nxt)
        if nxt == HALT:
            break
    steps = []; cur = []
    for p in range(prompt_len, len(tape)):
        cur.append(p)
        if tape[p] == SE: steps.append(cur); cur = []

    def axrow(s):
        for p in steps[s]:
            if tape[p] == AX: return p + 1
        return None
    rg, rb = axrow(sg), axrow(sb)
    R = ctx.fwd._residual_pre_head(tape)
    W = ctx.model.head.weight
    if W.is_sparse: W = W.to_dense()
    b = ctx.model.head.bias
    for nm, r, row in (("GOOD step%d" % sg, R[rg], rg), ("BAD step%d" % sb, R[rb], rb)):
        lg = W @ r + (b if b is not None else 0)
        top = torch.topk(lg, 6)
        print(f"\n[{nm} row={row} tok={tape[row]}] top logits: "
              + " ".join(f"{int(t)}:{v:.3f}" for t, v in zip(top.indices.tolist(), top.values.tolist())))
    # diff
    diff = (R[rb] - R[rg])
    order = torch.argsort(diff.abs(), descending=True)[:30]
    print(f"\n== dims where BAD differs most from GOOD ==")
    for d in order.tolist():
        if abs(diff[d].item()) < 1e-3: continue
        print(f"  dim {d:4d} {inv.get(d, str(d)):26s} good={R[rg][d].item():+.3f} bad={R[rb][d].item():+.3f} d={diff[d].item():+.3f}")


if __name__ == "__main__":
    main()
