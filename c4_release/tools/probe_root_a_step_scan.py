#!/usr/bin/env python3
"""Scan ALL rows of the ENT step: which carry OP_ENT broadcast, which carry
MEM_VAL_B markers, MARK_*, IS_BYTE. Distinguishes the BP-register VALUE row
(dump MISFIRE) from the legit ENT MEM-store value row (intended dump fire).

CPU ONLY. Single forward up to (not including) the dump block.
Usage: CUDA_VISIBLE_DEVICES="" python tools/probe_root_a_step_scan.py 275 1
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

import sys
import argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from tools.interp_oracle_gate import (
    build_production_model, build_code_prompt, oracle_tape_and_steps,
    STEP_TOKENS,
)
from tools.faithful_interpreter_validate import (
    _attn_block_to_specs, _faithful_attn_forward, _faithful_ffn_forward,
    _COMPOSITE_FFN,
)

MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_SE",
         "MARK_STACK0"]


def main(idx, ent_step, crush_block=50):
    src, exp, desc = generate_test_programs()[idx]
    bc, data = compile_c(src)
    prompt = build_code_prompt(bc, data)
    tape = oracle_tape_and_steps(bc, data, max_steps=40)
    full_ctx = list(prompt) + list(tape.draft_tokens)
    prefix = len(prompt)

    model, layout = build_production_model("cpu")
    dp = dict(layout.dim_positions)

    token_ids = torch.tensor([full_ctx], dtype=torch.long)
    x = model.embed(token_ids)[0]
    d_model = model.d_model
    for bi, block in enumerate(model.blocks):
        if bi == crush_block:
            break
        attn = block.attn
        heads = _attn_block_to_specs(attn, d_model)
        x = _faithful_attn_forward(heads, x, attn.num_heads, attn.head_dim,
                                   getattr(attn, "use_softmax1", True))
        if type(block.ffn).__name__ in _COMPOSITE_FFN:
            x = block.ffn(x.unsqueeze(0))[0]
        else:
            x = _faithful_ffn_forward(block.ffn, x)

    base = prefix + ent_step * STEP_TOKENS
    print(f"=== id{idx} {desc} ENT step={ent_step} rows {base}..{base+STEP_TOKENS-1} ===")
    print(f"  {'row':>4} {'off':>3} {'OP_ENT':>8} {'IS_BYTE':>7} "
          f"{'MV0':>6} {'MV1':>6} {'MV2':>6} {'MV3':>6}  {'mark':>10}  "
          f"OLD(b1)  NEW(b1)")
    for off in range(STEP_TOKENS):
        pos = base + off
        r = x[pos]
        ent = float(r[dp["OP_ENT"]])
        isb = float(r[dp["IS_BYTE"]])
        mv = [float(r[dp[f"MEM_VAL_B{k}"]]) for k in range(4)]
        mk = ""
        for m in MARKS:
            if m in dp and abs(float(r[dp[m]])) > 0.3:
                mk += m.replace("MARK_", "") + " "
        # gate score for byte1 under OLD weights (e=1,m=8,T=14) and NEW
        # marker-required weights (e=1,m=20,T=23).
        blockers = 0.0
        for nm in MARKS:
            if nm in dp:
                blockers += -1000.0 * float(r[dp[nm]])
        up_old = 100.0 * (ent * 1.0 + mv[1] * 8.0 + blockers - 14.0)
        up_new = 100.0 * (ent * 1.0 + mv[1] * 20.0 + blockers - 23.0)
        old_t = ("FIRE" if up_old > 0 else "----")
        new_t = ("FIRE" if up_new > 0 else "----")
        # only show rows with OP_ENT > 4 or any MEM_VAL marker
        if ent > 4 or max(abs(v) for v in mv) > 0.3:
            print(f"  {pos:>4} {off:>3} {ent:>8.2f} {isb:>7.2f} "
                  f"{mv[0]:>6.2f} {mv[1]:>6.2f} {mv[2]:>6.2f} {mv[3]:>6.2f}  "
                  f"{mk:>10}  {old_t}({up_old:.0f})  {new_t}({up_new:.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=int, nargs="?", default=275)
    ap.add_argument("step", type=int, nargs="?", default=1)
    ap.add_argument("--crush", type=int, default=50)
    a = ap.parse_args()
    main(a.idx, a.step, a.crush)
