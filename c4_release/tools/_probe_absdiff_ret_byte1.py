#!/usr/bin/env python3
"""Attribute the absdiff LEV-return-step AX byte-1 0x01 leak (step 22).

Two-stage:
  (1) LM-head dim attribution: at the AX byte-1 predictor position (draft
      offset 6 of the leak step), rank residual dims by contribution to
      (logit[got_byte1] - logit[0]) => the residual cell the head reads.
  (2) Rule attribution: for the top OUTPUT_HI cell, name the declarative
      FFN rule whose runtime SwiGLU output dominates that cell.

Campaign 30-token config. Usage:
    CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 python tools/_probe_absdiff_ret_byte1.py <id> <leak_step> [--absdiff-fix]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
import warnings; warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from tools.interp_oracle_gate import (
    build_gate_context, build_code_prompt, oracle_tape_and_steps,
)
from neural_vm.verification.faithful_interpreter import STEP_TOKENS


def name_for(pos, items):
    for i, (nm, start) in enumerate(items):
        nxt = items[i + 1][1] if i + 1 < len(items) else start + 16
        if start <= pos < nxt:
            return f"{nm}+{pos - start}"
    return f"dim{pos}"


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); leak = int(sys.argv[2])
    if "--absdiff-fix" in sys.argv:
        os.environ["C4_ABSDIFF_FIX"] = "1"
    ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    items = sorted(dp.items(), key=lambda kv: kv[1])
    model = ctx.model
    W = model.head.weight
    if W.is_sparse: W = W.to_dense()
    W = W.float()
    b = model.head.bias.float()

    src, exp, desc = generate_test_programs()[pid]
    bc, data = compile_c(src)
    ot = oracle_tape_and_steps(bc, data, max_steps=48)
    prompt = build_code_prompt(bc, data)
    prefix = len(prompt)
    full = prompt + ot.draft_tokens

    # AX byte-1 predictor = draft offset 6 (byte-0 token) of the leak step.
    base = leak * STEP_TOKENS
    pred_off = base + 6            # predicts byte-1 at offset 7
    pred_pos = prefix + pred_off
    resid = ctx.fwd._residual_pre_head(full)[pred_pos].float()

    logits = resid @ W.t() + b
    got = int(logits.argmax())
    o_pc, o_ax = ot.steps[leak]
    print(f"id{pid} {desc} exp={exp} leak_step={leak} oracle_ax={o_ax} (0x{o_ax:08x})")
    print(f"  AX byte-1 predictor pos={pred_pos} (off {pred_off%STEP_TOKENS}) "
          f"got_byte1={got}  logit[{got}]={float(logits[got]):.3f} "
          f"logit[0]={float(logits[0]):.3f}")
    if got == 0:
        print("  (byte-1 is CLEAN 0 here — no leak at this step)")
    Wg = W[got]; Ww = W[0]
    dw = (Wg - Ww) * resid
    order = torch.argsort(dw.abs(), descending=True)
    print(f"  top residual dims driving (logit[{got}] - logit[0]):")
    top_output_cols = []
    for di in order[:20].tolist():
        if abs(float(dw[di])) < 0.05: break
        nm = name_for(di, items)
        print(f"    dim {di:4d} {nm:26s} res={float(resid[di]):9.3f} "
              f"dW={float(Wg[di]-Ww[di]):7.3f} contrib={float(dw[di]):8.3f}")
        if "OUTPUT" in nm:
            top_output_cols.append((di, nm))

    # Rule attribution for top OUTPUT cells.
    print("  --- rule attribution for top OUTPUT residual cells ---")
    for di, nm in top_output_cols[:4]:
        ranked = ctx.interp.attribute_runtime_contribution(
            ctx.flat_ffn_ops, di, resid)
        print(f"  cell dim{di} {nm} (res={float(resid[di]):.3f}):")
        for op_name, rule_name, contrib in ranked[:6]:
            print(f"      {contrib:+9.3f}  {op_name} :: {rule_name}")
        if not ranked:
            print("      (no runtime FFN writer — relayed/attention/default)")


if __name__ == "__main__":
    main()
