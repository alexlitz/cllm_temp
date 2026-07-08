#!/usr/bin/env python3
"""Block-input attribution for the var_three id300 PSH-of-`a` store VALUE byte.

Brief: var_three step-19 first ADD returns 6 (=b) not 35 (=a+b). Hypothesis:
the PSH-of-`a` step (step 16, instr [18]) emits the WRONG store VALUE byte — b's
residual (6) leaks into a's PSH store value byte where 29 should be. This probe
runs the FAITHFUL forward over the production context and, per VM step, decodes
(from the model argmax) the whole step slice — PC/AX/SP/BP registers AND the MEM
addr/val block — vs the DraftVM oracle tape, so we can SEE which step's MEM value
byte first goes wrong and attribute it to the owning declarative rule at the
BLOCK-INPUT via attribute_runtime_contribution.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from tools.interp_oracle_gate import (  # noqa: E402
    build_gate_context, build_code_prompt, oracle_tape_and_steps,
)
from neural_vm.verification.faithful_interpreter import STEP_TOKENS  # noqa: E402

OP = {0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ", 6: "ENT",
      7: "ADJ", 8: "LEV", 9: "LI", 10: "LC", 11: "SI", 12: "SC", 13: "PSH",
      25: "ADD", 26: "SUB", 38: "EXIT"}

# 30-token layout offsets.
REG = {"PC": 0, "AX": 5, "SP": 10, "BP": 15}
MEM_MARKER = 20
MEM_ADDR = (21, 22, 23, 24)
MEM_VAL = (25, 26, 27, 28)


def decode4(slice_pred, base):
    v = 0
    for j in range(4):
        v |= (int(slice_pred[base + j]) & 0xFF) << (j * 8)
    return v & 0xFFFFFFFF


def main():
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c

    CID = 300
    tests = generate_test_programs()
    src, exp, desc = tests[CID]
    bc, data = compile_c(src)
    print(f"=== id{CID} {desc}  (STEP_TOKENS={STEP_TOKENS}) ===")
    print(f"SRC: {src}")

    ctx = build_gate_context(verbose=True)
    ot = oracle_tape_and_steps(bc, data, max_steps=30)
    prompt = build_code_prompt(bc, data)
    prefix = len(prompt)
    full_ctx = prompt + ot.draft_tokens
    n_steps = len(ot.steps)

    logits = ctx.fwd.forward(full_ctx)
    fa = logits.argmax(dim=-1).tolist()

    def pred_tok(t):
        return int(fa[prefix + t - 1])

    print("\nstep opcode | PC(o/g) AX(o/g) | MEMaddr(o/g) MEMval(o/g)")
    for s in range(n_steps):
        base = s * STEP_TOKENS
        sl = [pred_tok(base + k) for k in range(STEP_TOKENS)]
        o_pc, o_ax = ot.steps[s]
        g_pc = decode4(sl, REG["PC"] + 1)
        g_ax = decode4(sl, REG["AX"] + 1)
        g_maddr = decode4(sl, MEM_ADDR[0])
        g_mval = decode4(sl, MEM_VAL[0])
        # Oracle MEM addr/val is the DraftVM tape (trusted); read from tape.
        o_maddr = decode4(ot.draft_tokens[base:base + STEP_TOKENS], MEM_ADDR[0])
        o_mval = decode4(ot.draft_tokens[base:base + STEP_TOKENS], MEM_VAL[0])
        op = ot.opcodes[s] if s < len(ot.opcodes) else -1
        flag_ax = "" if g_ax == o_ax else "  <<AX"
        flag_mv = "" if g_mval == o_mval else "  <<MEMVAL"
        print(f"{s:3d} {OP.get(op, op):5s} | pc {o_pc}/{g_pc}  ax {o_ax}/{g_ax}"
              f" | maddr {o_maddr & 0xFFFF}/{g_maddr & 0xFFFF} "
              f"mval {o_mval}/{g_mval}{flag_ax}{flag_mv}")

    # Focused: step 16 = PSH-of-a; step 19 = first ADD.
    print("\n--- FOCUS: step 16 (PSH-of-a) MEM-value byte-0, step 19 (ADD) AX byte-0 ---")
    for (s, label) in [(16, "PSH-of-a"), (19, "ADD a+b")]:
        base = s * STEP_TOKENS
        sl = [pred_tok(base + k) for k in range(STEP_TOKENS)]
        o_mval = decode4(ot.draft_tokens[base:base + STEP_TOKENS], MEM_VAL[0])
        g_mval = decode4(sl, MEM_VAL[0])
        o_pc, o_ax = ot.steps[s]
        g_ax = decode4(sl, REG["AX"] + 1)
        print(f"  step {s} ({label}): "
              f"MEMval oracle={o_mval} got={g_mval} (byte0 o={o_mval & 0xFF} g={g_mval & 0xFF}) | "
              f"AX oracle={o_ax} got={g_ax}")

    return ctx, full_ctx, prefix, ot


if __name__ == "__main__":
    main()
