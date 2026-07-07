#!/usr/bin/env python3
"""Measure the l16_store_ax_carry_lo OVERRIDE over-fire.

C4_STORE_AX_B0_OVERRIDE was defaulted OFF (commit 7869e5c3) because ON regresses
ADD/SUB/cmp/mul (-24). The override adds a -30/S write to OUTPUT_LO+0 gated on
store_ax_conditions (OP_SI + OP_SC + MARK_AX ...). This probe evaluates, at the
AX-marker residual of each VM step, the store_ax rule's up / silu / gate to see
which NON-store (ADD/SUB) rows the -W misfires on, and by how much, so we can
author a CLEAN discriminator.
"""
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
S = ctx.interp.S

def resolve(name):
    from neural_vm.dim_registry import dim_ref
    # store_ax_conditions dims
    return name

# Resolve the store_ax condition/gate/write columns via dim_positions.
from neural_vm.dim_registry import dim_ref as DR
def col(ref):
    try:
        return ref.resolve(dp)
    except Exception:
        return -1

C_SI   = col(DR('opcode_flag','SI'))
C_SC   = col(DR('opcode_flag','SC'))
C_MAX  = col(DR('marker','AX'))
C_MPC  = col(DR('marker','PC'))
C_ISB  = dp.get("IS_BYTE", -1)
C_EXIT = col(DR('opcode_flag','EXIT'))
C_JMP  = col(DR('opcode_flag','JMP'))
C_ADD  = col(DR('opcode_flag','ADD'))
C_SUB  = col(DR('opcode_flag','SUB'))
C_OUT0 = col(DR('output_lo','nibble',0))
print(f"cols: SI={C_SI} SC={C_SC} MAX={C_MAX} MPC={C_MPC} ISB={C_ISB} EXIT={C_EXIT} JMP={C_JMP} ADD={C_ADD} SUB={C_SUB} OUT_LO0={C_OUT0}")

def store_ax_up(x):
    # up = -threshold*S + S*sum(w*cond)   (threshold=4.0)
    up = -4.0 * S
    for c, w in ((C_SI,1.0),(C_SC,1.0),(C_MAX,1.0),(C_MPC,-8.0),(C_ISB,-10.0),(C_EXIT,-20.0),(C_JMP,-20.0)):
        if c >= 0:
            up += S * w * float(x[c])
    return up

def probe_prog(pid, steps_of_interest):
    src,exp,desc = generate_test_programs()[pid]
    bc,data = compile_c(src)
    ot = oracle_tape_and_steps(bc,data,max_steps=40)
    prompt = build_code_prompt(bc,data); prefix=len(prompt)
    full_ctx = prompt + ot.draft_tokens
    resid_all = ctx.fwd._residual_pre_head(full_ctx)
    print(f"\n=== id{pid} {desc} ===")
    print("  step op   | SI SC MAX MPC ISB | up      silu(up)  | -> -W OUT_LO0 fires?")
    for s in range(len(ot.opcodes)):
        op = ot.opcodes[s]
        # AX marker predicting position for this step
        pos = prefix + s*STEP_TOKENS + 5  # AX marker offset=5
        x = resid_all[pos]
        up = store_ax_up(x)
        sil = float(torch.nn.functional.silu(torch.tensor(up)).item())
        vals = [float(x[c]) if c>=0 else 0.0 for c in (C_SI,C_SC,C_MAX,C_MPC,C_ISB)]
        flag = " <<FIRES" if abs(sil) > 0.05 else ""
        tag = "" if OP.get(op) not in ("SI","SC") else "  [genuine store]"
        if abs(sil) > 0.02 or OP.get(op) in ("SI","SC","ADD","SUB","MUL"):
            print(f"  {s:3d} {OP.get(op,op):4s} | {vals[0]:+.1f} {vals[1]:+.1f} {vals[2]:+.1f} {vals[3]:+.1f} {vals[4]:+.1f} | {up:+8.2f} {sil:+.4f} | {flag}{tag}")

# var_three (has genuine SI stores at steps 5,9,13) + arith
probe_prog(300, None)   # var_three_0
probe_prog(0, None)     # add_0
probe_prog(325, None)   # var_update_0
