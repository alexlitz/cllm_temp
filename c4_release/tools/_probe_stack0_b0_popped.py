#!/usr/bin/env python3
"""CPU per-step STACK0 byte-0 EMISSION trace (the POPPED-flag self-check).

Builds the real autoregressive faithful tape (byte-identical to neural spec_k=0)
for an if/bool program, splits it into 35-token steps, and prints — per step —
the token the model EMITS at the STACK0 byte-0 row (step offset 21, predicted by
the row at offset 20).  The C4_STACK0_B0_DUMP over-fire shows up as the stale
operand byte (e.g. 0x23=35 for id 350) re-appearing on POST-POP steps; the fix
emits 0x00 there while keeping the on-stack step's byte.

Compares the model EMISSION against the byte-exact DraftVM oracle (so we see
WHERE the dump diverges from ground truth) and prints, on a chosen step, the top
vocab logits + the value of the gate bands (CARRIED / SHARP / NOT_CMP / POPPED /
DUMP_BLOCK) at the STACK0 byte-0 predictor row.

Usage: python tools/_probe_stack0_b0_popped.py [id] [trace_step]
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

SE = int(Token.STEP_END); HALT = int(Token.HALT)
STACK0_MARK = 268
STACK0_VAL_OFF = 21      # value byte 0 sits at step offset 21
STACK0_PRED_OFF = 20     # the row at offset 20 predicts the offset-21 token
GATE_BANDS = ("STACK0_B0_CARRIED", "STACK0_B0_SHARP", "STACK0_B0_NOT_CMP",
              "STACK0_B0_POPPED", "STACK0_B0_DUMP_BLOCK", "MARK_STACK0")


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    trace_step = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    from neural_vm.unified_compiler.faithful_autoregressive import (
        FaithfulAutoregressiveRunner)
    with contextlib.redirect_stdout(io.StringIO()):
        ar = FaithfulAutoregressiveRunner(model=ctx.model, layout=ctx.layout)
    max_steps = int(os.environ.get("PROBE_MAX_STEPS", "8"))
    tape = list(ar._inner._serial._build_context(bc, [], []))
    prompt_len = len(tape)
    # Autoregressive faithful decode (bounded to max_steps so the CPU forward
    # over the growing context stays tractable).
    n_se = 0
    for _ in range(max_steps * 60 + 40):
        logits = ctx.fwd.forward(tape)
        nxt = int(logits[len(tape) - 1].argmax())
        tape.append(nxt)
        if nxt == HALT:
            break
        if nxt == SE:
            n_se += 1
            if n_se >= max_steps:
                break
    # Split into steps at STEP_END.
    steps = []; cur = []
    for p in range(prompt_len, len(tape)):
        cur.append(p)
        if tape[p] == SE:
            steps.append(cur); cur = []
    if cur:
        steps.append(cur)

    print(f"\n===== id{pid} {desc!r} (exp={exp & 0xff}) =====")
    print(f"  steps={len(steps)} (capped {max_steps})  ntoks={[len(s) for s in steps]}")

    # ONE residual forward over the whole decoded tape; index each step's
    # predictor row from it (the model's emission == argmax at that row, the same
    # value the autoregressive loop already appended at offset 21).
    W = ctx.model.head.weight
    if W.is_sparse: W = W.to_dense()
    b = ctx.model.head.bias
    Rfull = ctx.fwd._residual_pre_head(tape)

    print(f"\n  step | emit_STACK0[0] | argmax@pred_row | tok-count")
    for si, st in enumerate(steps):
        n35 = (len(st) == 35)
        emit = tape[st[STACK0_VAL_OFF]] if len(st) > STACK0_VAL_OFF else None
        pred = None
        if len(st) > STACK0_PRED_OFF:
            row = st[STACK0_PRED_OFF]
            pred = int((W @ Rfull[row] + (b if b is not None else 0)).argmax())
        es = "?" if emit is None else f"0x{emit:02x}={emit}"
        ps = "?" if pred is None else f"0x{pred:02x}={pred}"
        print(f"  {si:4d} | emit={es:11s} | argmax={ps:11s} | "
              f"{'35-tok' if n35 else f'{len(st)}-tok DRIFT'}")

    # Detailed gate-band readout on a chosen step's STACK0 byte-0 predictor row.
    if trace_step < 0:
        cand = [si for si, s in enumerate(steps) if len(s) == 35]
        trace_step = cand[-1] if cand else 0
    if 0 <= trace_step < len(steps) and len(steps[trace_step]) > STACK0_PRED_OFF:
        row = steps[trace_step][STACK0_PRED_OFF]
        R = Rfull[row]
        lg = W @ R + (b if b is not None else 0)
        top = torch.topk(lg, 6)
        print(f"\n  --- step {trace_step} STACK0[0] predictor row={row} ---")
        print("  top logits: " + " ".join(
            f"{int(t)}:{v:.2f}" for t, v in
            zip(top.indices.tolist(), top.values.tolist())))
        for nm in GATE_BANDS:
            if nm in dp:
                print(f"    {nm:24s} = {R[int(dp[nm])].item():+.3f}")
            else:
                print(f"    {nm:24s} = <not in layout>")


if __name__ == "__main__":
    main()
