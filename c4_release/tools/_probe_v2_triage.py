#!/usr/bin/env python3
"""Flat-interp triage for C4_STORE_AX_B0_OVERRIDE_V2 (the clean store-only
discriminator). Confirms the var_three id300 SI-of-b store step-9 AX byte-0
now = 6 (was 0), and that a pure ADD step (add_0 step-3) is UNTOUCHED (byte-0
still correct) — i.e. the ALU anti-condition prevents the -W bleed. Teacher-
forced single forward; DEFER the AR/1096 verdict to the main thread."""
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

OP = {9:"LI",11:"SI",13:"PSH",25:"ADD",26:"SUB",1:"IMM",0:"LEA"}

def run(pid, checks):
    ctx = build_gate_context(verbose=False)
    src, exp, desc = generate_test_programs()[pid]
    bc, data = compile_c(src)
    ot = oracle_tape_and_steps(bc, data, max_steps=40)
    prompt = build_code_prompt(bc, data); prefix = len(prompt)
    logits = ctx.fwd.forward(prompt + ot.draft_tokens)
    fa = logits.argmax(dim=-1).tolist()
    def pred(t): return int(fa[prefix + t - 1])
    print(f"=== id{pid} {desc} (V2={os.environ.get('C4_STORE_AX_B0_OVERRIDE_V2')}) ===", flush=True)
    ok = True
    for s in checks:
        base = s * STEP_TOKENS
        ax = sum((pred(base + 6 + j) & 0xFF) << (8 * j) for j in range(4))
        o_ax = ot.steps[s][1]
        op = OP.get(ot.opcodes[s], ot.opcodes[s])
        good = ax == o_ax
        ok = ok and good
        print(f"  step {s} {op}: AX got={ax} oracle={o_ax}  {'OK' if good else 'WRONG <<'}", flush=True)
    return ok

if __name__ == "__main__":
    a = run(300, [9, 16, 19, 23])   # var_three: SI-of-b + PSH-of-a + both ADDs
    print()
    b = run(0, [3])                 # add_0: pure ADD step must stay correct
    print()
    c = run(325, [5, 12, 13])       # var_update: SI + ADD + SI
    print(f"\nTRIAGE: var_three_step9_fixed={a}  add_untouched={b}  var_update={c}", flush=True)
