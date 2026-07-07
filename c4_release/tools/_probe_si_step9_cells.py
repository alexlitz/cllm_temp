#!/usr/bin/env python3
"""Measure OUTPUT_LO cell totals at var_three id300 SI-of-b step-9 AX marker,
override OFF, to size the store-AX-b0 fix (how strong the zero-default on
OUTPUT_LO+0 is vs the carried nibble on OUTPUT_LO+6)."""
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
from tools.interp_oracle_gate import build_gate_context, build_code_prompt, oracle_tape_and_steps
from neural_vm.verification.faithful_interpreter import STEP_TOKENS
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

def main():
    ctx = build_gate_context(verbose=True)
    dp = ctx.dim_positions
    OUT_LO = dp['OUTPUT_LO']
    src, exp, desc = generate_test_programs()[300]
    bc, data = compile_c(src)
    ot = oracle_tape_and_steps(bc, data, max_steps=30)
    prompt = build_code_prompt(bc, data); prefix = len(prompt)
    resid = ctx.fwd._residual_pre_head(prompt + ot.draft_tokens)
    pos = prefix + 9 * STEP_TOKENS + 5   # SI step-9 AX marker
    x = resid[pos]
    print(f"OVERRIDE={os.environ.get('C4_STORE_AX_B0_OVERRIDE')} SI step9 (b=6) OUTPUT_LO cell totals:", flush=True)
    for cell in range(10):
        ranked = ctx.interp.attribute_runtime_contribution(ctx.flat_ffn_ops, OUT_LO + cell, x)
        tot = sum(c for _, _, c in ranked)
        mark = "  <-- want winner (b low nibble)" if cell == 6 else ("  <-- zero-default" if cell == 0 else "")
        print(f"  OUTPUT_LO+{cell}: total={tot:+8.3f}{mark}", flush=True)

if __name__ == "__main__":
    main()
