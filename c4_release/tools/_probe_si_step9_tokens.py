#!/usr/bin/env python3
"""Read the ACTUAL argmax tokens the model emits at var_three id300 SI-of-b
step-9 AX bytes (offsets 6..9), override OFF, to settle the AX=0 vs AX=6
question and confirm which nibble cell actually loses."""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE); _ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.interp_oracle_gate import build_gate_context, build_code_prompt, oracle_tape_and_steps
from neural_vm.verification.faithful_interpreter import STEP_TOKENS
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

def main():
    ctx = build_gate_context(verbose=True)
    src, exp, desc = generate_test_programs()[300]
    bc, data = compile_c(src)
    ot = oracle_tape_and_steps(bc, data, max_steps=30)
    prompt = build_code_prompt(bc, data); prefix = len(prompt)
    full_ctx = prompt + ot.draft_tokens
    logits = ctx.fwd.forward(full_ctx)
    fa = logits.argmax(dim=-1).tolist()
    def pred(t): return int(fa[prefix + t - 1])
    print(f"OVERRIDE={os.environ.get('C4_STORE_AX_B0_OVERRIDE')}", flush=True)
    for s in (9, 16, 19):
        base = s * STEP_TOKENS
        # AX marker at offset 5, bytes at offsets 6,7,8,9
        axmk = pred(base + 5)
        axbytes = [pred(base + 6 + j) for j in range(4)]
        o_ax = ot.steps[s][1]
        print(f"  step {s}: AX marker tok={axmk} bytes(lo->hi)={axbytes} -> AX={sum((b&0xFF)<<(8*j) for j,b in enumerate(axbytes))} | oracle AX={o_ax}", flush=True)
        # Also show top-3 vocab logits at the byte-0 predicting position
        pos = prefix + base + 5   # predicts token at offset 6 (AX byte0)
        top = torch.topk(logits[pos], 5)
        print(f"       byte0 top-5 vocab: {[ (int(i), round(float(v),2)) for v,i in zip(top.values.tolist(), top.indices.tolist()) ]}", flush=True)

if __name__ == "__main__":
    main()
