#!/usr/bin/env python3
"""Attribute the step-9 AX byte-0 low-nibble->8 corruption to its owning rule.

CPU-only, faithful. Builds the production model once, runs the FAITHFUL
AUTOREGRESSIVE decode (which reproduces the real cross-step poisoned context),
finds the step-N AX byte-0 row, reads the pre-head residual THERE, and ranks the
declarative FFN rules by their runtime SwiGLU contribution to OUTPUT_LO[wrong]
and OUTPUT_LO[right]. Names the rule that writes the wrong nibble.

Usage: python tools/_probe_lev_ax_attrib.py [id] [step] [right_nib] [wrong_nib]
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

SE = int(Token.STEP_END); AX = int(Token.REG_AX)


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    tstep = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    rn = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    wn = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]

    # Build the gate context (model + interp + flat_ffn_ops + cached pre-head fwd)
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    LO = dp["OUTPUT_LO"]; HI = dp["OUTPUT_HI"]

    # Real autoregressive decoded tape (reproduces poisoning)
    from neural_vm.verification.faithful_autoregressive import FaithfulAutoregressiveRunner
    with contextlib.redirect_stdout(io.StringIO()):
        ar = FaithfulAutoregressiveRunner(model=ctx.model, layout=ctx.layout)
    # Build the full decoded context by replaying the spec_k=0 free run.
    # Reuse the simpler probe path: greedy argmax loop via the cached pre-head fwd.
    serial = ar._inner._serial
    tape = list(serial._build_context(bc, [], []))
    prompt_len = len(tape)
    STEP = 35
    for _ in range((tstep + 3) * STEP + 40):
        logits = ctx.fwd.forward(tape)  # [S, vocab]
        nxt = int(logits[len(tape) - 1].argmax())
        tape.append(nxt)
        if nxt == int(Token.HALT):
            break

    # find step tstep AX byte-0 row
    steps = []; cur = []
    for p in range(prompt_len, len(tape)):
        cur.append(p)
        if tape[p] == SE: steps.append(cur); cur = []
    # per-step AX byte0 dump
    print("per-step AX byte0:")
    for si, st in enumerate(steps):
        for p in st:
            if tape[p] == AX:
                b0 = tape[p + 1] if p + 1 < len(tape) else None
                print(f"  step{si}: ax_b0={b0}" + (f" (0x{b0:02x})" if b0 is not None else ""))
                break
    if tstep >= len(steps):
        print(f"only {len(steps)} steps decoded; using last step {len(steps)-1}")
        tstep = len(steps) - 1
    axrow = None
    for p in steps[tstep]:
        if tape[p] == AX: axrow = p + 1; break
    print(f"id{pid} {desc} exp={exp&0xff:#x} step{tstep} AX-b0 row={axrow} tok={tape[axrow]} "
          f"(decoded {len(steps)} steps)")

    # Residual must be read with the AX-b0 row as the LAST position (the
    # generation-time context), because the embedding's MEM-scan is position-
    # dependent — reading it as an interior row of the full tape gives a
    # DIFFERENT residual (and a different argmax) than what actually emitted.
    trunc = tape[:axrow]  # context ending right before the AX byte-0 token
    import torch as _t
    Rt = ctx.fwd._residual_pre_head(trunc)
    resid = Rt[len(trunc) - 1]  # the row whose next-token argmax = AX byte0
    W = ctx.model.head.weight
    if W.is_sparse: W = W.to_dense()
    bvec = ctx.model.head.bias
    lg = W @ resid + (bvec if bvec is not None else 0)
    top = _t.topk(lg, 6)
    print(f"\n[truncated-ctx residual] argmax={int(top.indices[0])} top: "
          + " ".join(f"{int(t)}:{v:.3f}" for t, v in zip(top.indices.tolist(), top.values.tolist())))
    print("OUTPUT_LO cells: " + " ".join(f"{i}:{resid[LO+i].item():+.3f}" for i in range(16)))
    print("OUTPUT_HI cells: " + " ".join(f"{i}:{resid[HI+i].item():+.3f}" for i in range(16)))
    # AX_CARRY band (the source the ADJ/LEV route reads). Read at L6-input (block
    # where the route fires) AND at pre-head, to see the contamination.
    for cn in ("AX_CARRY_LO", "AX_CARRY_HI", "SE_AX_CARRY_LO", "AX_FULL_LO"):
        cb = dp.get(cn)
        if cb is None: continue
        print(f"{cn} cells: " + " ".join(f"{i}:{resid[cb+i].item():+.2f}"
                                          for i in range(16) if abs(resid[cb+i].item()) > 0.05))
    for nm, nib in (("RIGHT", rn), ("WRONG", wn)):
        col = LO + nib
        ranked = ctx.interp.attribute_runtime_contribution(ctx.flat_ffn_ops, col, resid)
        print(f"\n== OUTPUT_LO[{nib}] ({nm}) value={resid[col].item():+.4f}  top rules ==")
        for op_name, rule_name, c in ranked[:8]:
            print(f"   {c:+.4f}  {op_name} :: {rule_name}")
        if not ranked:
            print("   (no runtime FFN writer — default/relay cell)")


if __name__ == "__main__":
    main()
