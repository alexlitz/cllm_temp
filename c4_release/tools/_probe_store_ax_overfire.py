#!/usr/bin/env python3
"""Measure the l16_store_ax_carry_lo OVERRIDE over-fire — via the BUILT model's
own attribute_runtime_contribution (resolves dims off the built layout).

C4_STORE_AX_B0_OVERRIDE was defaulted OFF (7869e5c3) because ON regresses
ADD/SUB/cmp/mul (-24). The override adds a -30/S write to OUTPUT_LO+0 gated on
store_ax_conditions. This probe, with the override ON, ranks every rule's runtime
contribution to OUTPUT_LO+0 at each VM step's AX-marker predicting position, so
we can SEE the l16_store_ax_carry_lo_k (-W) misfire on NON-store (ADD/SUB) rows
and quantify it — the basis for a clean store-only discriminator.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ.setdefault("C4_STORE_AX_B0_OVERRIDE", "1")   # probe the ON build
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
      9:"LI",10:"LC",11:"SI",12:"SC",13:"PSH",14:"OR",15:"XOR",16:"AND",17:"EQ",
      18:"NE",19:"LT",20:"GT",21:"LE",22:"GE",23:"SHL",24:"SHR",25:"ADD",26:"SUB",
      27:"MUL",28:"DIV",29:"MOD",38:"EXIT"}

ctx = build_gate_context(verbose=True)
dp = ctx.dim_positions
OUT_LO0 = dp.get("OUTPUT_LO")
print(f"OUTPUT_LO base col = {OUT_LO0}")

def store_ax_contrib(resid):
    """Sum of contributions to OUTPUT_LO+0 from l16_store_ax_carry_lo_* rules
    (the -W over-fire) and total from all rules."""
    ranked = ctx.interp.attribute_runtime_contribution(ctx.flat_ffn_ops, OUT_LO0, resid)
    store_lo = sum(c for on, rn, c in ranked if rn.startswith("l16_store_ax_carry_lo"))
    total = sum(c for on, rn, c in ranked)
    top = ranked[:3]
    return store_lo, total, top

def probe_prog(pid):
    src,exp,desc = generate_test_programs()[pid]
    bc,data = compile_c(src)
    ot = oracle_tape_and_steps(bc,data,max_steps=40)
    prompt = build_code_prompt(bc,data); prefix=len(prompt)
    full_ctx = prompt + ot.draft_tokens
    resid_all = ctx.fwd._residual_pre_head(full_ctx)
    print(f"\n=== id{pid} {desc} ===")
    print("  step op   | store_lo->OUT_LO0  total->OUT_LO0 | top writers")
    for s in range(len(ot.opcodes)):
        op = ot.opcodes[s]
        pos = prefix + s*STEP_TOKENS + 5   # AX marker offset
        x = resid_all[pos]
        store_lo, total, top = store_ax_contrib(x)
        interesting = OP.get(op) in ("SI","SC","ADD","SUB","MUL","EQ","NE","LT","GT","LE","GE") or abs(store_lo) > 0.1
        if interesting:
            tag = "  [STORE]" if OP.get(op) in ("SI","SC") else "  <<NONSTORE" if abs(store_lo) > 0.1 else ""
            topstr = " ".join(f"{rn}={c:+.2f}" for on,rn,c in top)
            print(f"  {s:3d} {OP.get(op,op):4s} | store_lo={store_lo:+7.3f}  total={total:+7.3f} |{tag} {topstr}")

probe_prog(300)   # var_three_0 (SI stores of a/b/c + ADD)
probe_prog(0)     # add_0 (pure ADD)
probe_prog(325)   # var_update_0 (SI + ADD)
