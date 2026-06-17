#!/usr/bin/env python3
"""CPU faithful per-step AX + OUTPUT_LO attribution for test_simple_function.

Builds the model honoring env flags, decodes test_simple_function
(JSR 3; EXIT; NOP; ENT 0; IMM 42; LEV), prints per-step AX byte0, then attributes
OUTPUT_LO[wrong]/[right] at the step where AX first goes wrong.

Usage: python tools/_probe_simplefunc_ax.py [step] [right] [wrong]
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
from tests.test_smoke import _make_bytecode
from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import Token

SE = int(Token.STEP_END); AX = int(Token.REG_AX); HALT = int(Token.HALT)


def main():
    tstep = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    rn = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # 42=0x2a low nibble a
    wn = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 240=0xf0 low nibble 0
    bc = list(_make_bytecode([(Opcode.JSR, 3), Opcode.EXIT, Opcode.NOP,
                              (Opcode.ENT, 0), (Opcode.IMM, 42), Opcode.LEV]))
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions; LO = dp["OUTPUT_LO"]; HI = dp["OUTPUT_HI"]
    from neural_vm.unified_compiler.faithful_autoregressive import FaithfulAutoregressiveRunner
    with contextlib.redirect_stdout(io.StringIO()):
        ar = FaithfulAutoregressiveRunner(model=ctx.model, layout=ctx.layout)
    tape = list(ar._inner._serial._build_context(bc, [], []))
    pl = len(tape)
    for _ in range(12 * 35 + 40):
        logits = ctx.fwd.forward(tape)
        nxt = int(logits[len(tape) - 1].argmax())
        tape.append(nxt)
        if nxt == HALT: break
    steps = []; cur = []
    for p in range(pl, len(tape)):
        cur.append(p)
        if tape[p] == SE: steps.append(cur); cur = []
    print(f"test_simple_function decoded {len(steps)} steps")
    for si, st in enumerate(steps):
        for p in st:
            if tape[p] == AX:
                b0 = tape[p+1] if p+1 < len(tape) else None
                print(f"  step{si}: ax_b0={b0}" + (f" (0x{b0:02x})" if b0 is not None else ""))
                break
    if tstep < 0:
        tstep = len(steps) - 1
    axrow = None
    for p in steps[tstep]:
        if tape[p] == AX: axrow = p + 1; break
    if axrow is None:
        print("no AX row at step", tstep); return
    trunc = tape[:axrow]
    resid = ctx.fwd._residual_pre_head(trunc)[len(trunc) - 1]
    W = ctx.model.head.weight
    if W.is_sparse: W = W.to_dense()
    bvec = ctx.model.head.bias
    lg = W @ resid + (bvec if bvec is not None else 0)
    top = torch.topk(lg, 6)
    print(f"\nstep{tstep} AX-b0 row={axrow} argmax={int(top.indices[0])} "
          f"top: " + " ".join(f"{int(t)}:{v:.2f}" for t, v in zip(top.indices.tolist(), top.values.tolist())))
    print("OUTPUT_LO: " + " ".join(f"{i}:{resid[LO+i].item():+.2f}" for i in range(16)))
    print("OUTPUT_HI: " + " ".join(f"{i}:{resid[HI+i].item():+.2f}" for i in range(16)))
    for nm, nib in (("RIGHT", rn), ("WRONG", wn)):
        ranked = ctx.interp.attribute_runtime_contribution(ctx.flat_ffn_ops, LO + nib, resid)
        print(f"\n== OUTPUT_LO[{nib}] ({nm}) = {resid[LO+nib].item():+.3f} ==")
        for opn, rnme, c in ranked[:8]:
            print(f"   {c:+.4f}  {opn} :: {rnme}")
        if not ranked: print("   (no FFN writer)")


if __name__ == "__main__":
    main()
