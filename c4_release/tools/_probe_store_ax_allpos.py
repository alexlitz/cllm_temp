#!/usr/bin/env python3
"""Where does l16_store_ax_carry_lo (-W to OUTPUT_LO+0) leak? Scan ALL 30 token
positions of each VM step (not just the AX marker) for var_three + var_update +
add, override ON, and report every position with a nonzero store_lo->OUTPUT_LO+0
contribution. This finds the misfire rows behind the -24 arith regression.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ.setdefault("C4_STORE_AX_B0_OVERRIDE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE); _ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import warnings; warnings.filterwarnings("ignore")
import torch  # noqa
from tools.interp_oracle_gate import build_gate_context, build_code_prompt, oracle_tape_and_steps
from neural_vm.verification.faithful_interpreter import STEP_TOKENS
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

OP = {0:"LEA",1:"IMM",2:"JMP",3:"JSR",4:"BZ",5:"BNZ",6:"ENT",7:"ADJ",8:"LEV",
      9:"LI",10:"LC",11:"SI",12:"SC",13:"PSH",25:"ADD",26:"SUB",38:"EXIT"}
# 30-tok layout labels
def poslabel(k):
    if k==0: return "PC.mk"
    if 1<=k<=4: return f"PC.b{k-1}"
    if k==5: return "AX.mk"
    if 6<=k<=9: return f"AX.b{k-6}"
    if k==10: return "SP.mk"
    if 11<=k<=14: return f"SP.b{k-11}"
    if k==15: return "BP.mk"
    if 16<=k<=19: return f"BP.b{k-16}"
    if k==20: return "MEM.mk"
    if 21<=k<=24: return f"MEM.a{k-21}"
    if 25<=k<=28: return f"MEM.v{k-25}"
    if k==29: return "END"
    return str(k)

ctx = build_gate_context(verbose=True)
OUT_LO0 = ctx.dim_positions.get("OUTPUT_LO")

def store_lo_at(resid):
    ranked = ctx.interp.attribute_runtime_contribution(ctx.flat_ffn_ops, OUT_LO0, resid)
    store_lo = sum(c for on,rn,c in ranked if rn.startswith("l16_store_ax_carry_lo"))
    return store_lo

def probe(pid):
    src,exp,desc = generate_test_programs()[pid]
    bc,data = compile_c(src)
    ot = oracle_tape_and_steps(bc,data,max_steps=40)
    prompt = build_code_prompt(bc,data); prefix=len(prompt)
    resid = ctx.fwd._residual_pre_head(prompt+ot.draft_tokens)
    print(f"\n=== id{pid} {desc} ===")
    for s in range(len(ot.opcodes)):
        op = OP.get(ot.opcodes[s], ot.opcodes[s])
        hits=[]
        for k in range(STEP_TOKENS):
            pos = prefix + s*STEP_TOKENS + k
            v = store_lo_at(resid[pos])
            if abs(v) > 0.05:
                hits.append(f"{poslabel(k)}={v:+.1f}")
        if hits:
            tag = "[STORE]" if op in ("SI","SC") else "<<NONSTORE-LEAK"
            print(f"  step {s:3d} {str(op):4s} {tag}: " + " ".join(hits))

probe(300)   # var_three_0
probe(325)   # var_update_0
probe(0)     # add_0
probe(1)     # add_1
