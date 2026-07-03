#!/usr/bin/env python3
"""Lookahead probe (#221): is op2's (consumer) opcode fetchable at the dump block?

The C4_STACK0_B0_DUMP over-fires on arithmetic-intermediate operand frames and
must fire on comparison-result frames. The ONLY separator is the CONSUMER opcode
(op2, the NEXT instruction). This probe asks:

  At the carried STACK0 operand frame (and nearby rows in op2's step), at the
  dump block (~41), is op2's opcode (OP_DIV/MUL/ADD vs OP_LT/GT/EQ) present in
  the residual on ANY band? And from WHERE could a lookahead head fetch it?

Strategy: for an expr_mul_div program (14*56/8 -> arith consumer) and an if_gt
program (comparison consumer), dump every OP_* + OPCODE_BYTE_* + FETCH_* band
at EVERY row of the LAST few steps at block 41, with the token id, so we can
see where op2's opcode lives relative to the dump's carried STACK0 row.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_lookahead_op2.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

STACK0_MARK = 268
ARITH = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
         "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR"]
CMP = ["OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_EQ", "OP_NE"]
CTRL = ["OP_JMP", "OP_BZ", "OP_BNZ", "OP_JSR", "OP_LEV", "OP_ENT"]
MEMOP = ["OP_LI", "OP_SI", "OP_LC", "OP_SC", "OP_LEA", "OP_PSH", "OP_IMM"]
ALLOPS = ARITH + CMP + CTRL + MEMOP

DUMP_BLK = 41


@torch.no_grad()
def residual_at_block(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return (r.to_dense() if r.is_sparse else r).float().cpu()


@torch.no_grad()
def full_residual_block(probe, ctx, block_idx):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0].float().cpu()   # [S, D]


def step_of(ctx, position):
    step = 0
    for i in range(position):
        if ctx[i] == int(Token.STEP_END):
            step += 1
    return step


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    P = {n: int(dp[n]) for n in ALLOPS if n in dp}
    # Also bands that might hold a fetchable address / opcode-byte
    band_names = ["OPCODE_BYTE_LO", "OPCODE_BYTE_HI", "FETCH_LO", "FETCH_HI",
                  "MARK_STACK0", "MARK_AX", "MARK_PC", "HAS_SE"]
    B = {n: int(dp[n]) for n in band_names if n in dp}
    probe = GroundTruthProbe.build()

    progs = {
        "expr_mul_div(14*56/8)": "int main() { return 14 * 56 / 8; }",
        "if_gt(28>9)":           "int main() { if (28 > 9) return 1; return 0; }",
        "expr_add_mul(3+4*5)":   "int main() { return 3 + 4 * 5; }",
        "expr_mod(46%6+6)":      "int main() { return 46 % 6 + 6; }",
    }

    for tag, src in progs.items():
        print(f"\n================ {tag} ================")
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=20)
        full = full_residual_block(probe, ctx, DUMP_BLK)  # [S,D]
        pl = len(probe._build_context(bc))
        # find STACK0 rows + their steps
        stack0_rows = [(step_of(ctx, i), i) for i in range(len(ctx))
                       if ctx[i] == STACK0_MARK]
        print("STACK0 marker rows (step, pos, tok-after):",
              [(s, p, ctx[p+1] if p+1 < len(ctx) else None) for s, p in stack0_rows])
        # For each STACK0 row, print the OP_* present at that row, and also at
        # the SAME step's other marker rows (where op2's opcode should be decoded)
        for s, p in stack0_rows:
            res = full[p]
            ops = {n: round(float(res[P[n]]), 3) for n in P
                   if abs(float(res[P[n]])) > 0.02}
            print(f"  [STACK0 step={s} pos={p}] OP_* present: {ops}")
        # Now scan WHOLE step window: for the last 3 steps, every row's dominant OP_*
        maxstep = max((s for s, _ in stack0_rows), default=0)
        print("  --- per-row dominant OP_* across the program (block 41) ---")
        cur_step = 0
        for i in range(pl, len(ctx)):
            if ctx[i] == int(Token.STEP_END):
                cur_step += 1
                continue
            res = full[i]
            ops = {n: round(float(res[P[n]]), 3) for n in P
                   if abs(float(res[P[n]])) > 0.05}
            if ops:
                marker = ""
                for mn, mp in B.items():
                    if mn.startswith("MARK") and abs(float(res[mp])) > 0.5:
                        marker += mn.replace("MARK_", "") + " "
                print(f"    step={cur_step} pos={i} tok={ctx[i]:>3} "
                      f"[{marker.strip()}] {ops}")


if __name__ == "__main__":
    main()
