#!/usr/bin/env python3
"""Consumer-opcode lookahead probe (#221).

For each STACK0-marker frame (the operand/result frames the dump can fire on),
identify the CONSUMER instruction (the NEXT step) and ask whether its opcode is
present-and-fetchable on ANY row of the consumer step at an EARLY block (so a
forward-propagating band could carry it back to the operand frame).

Key question: at the STACK0 operand frame of step N, is the opcode of step N+1
(the consumer) available somewhere causally REACHABLE (same step N+1 AX-marker
row, which the operand frame can attend FORWARD-in-position... no, attention is
causal). So instead: build the band AT the consumer step (where its opcode IS
present on its own AX row) and have it look BACKWARD to tag the most-recent
STACK0 operand frame. This probe confirms the consumer opcode is present at the
consumer step's AX-marker row at an early block, and prints, per STACK0 frame,
(this-step opcode, next-step/consumer opcode), so we can see the arith-vs-cmp
separation that the existing per-step STACK0 opcode CANNOT provide.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_consumer_opcode.py
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
AX_MARK = 258
ARITH = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
         "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR"]
CMP = ["OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_EQ", "OP_NE"]
CTRL = ["OP_JMP", "OP_BZ", "OP_BNZ", "OP_JSR", "OP_LEV", "OP_ENT"]
MEMOP = ["OP_LI", "OP_SI", "OP_LC", "OP_SC", "OP_LEA", "OP_PSH", "OP_IMM"]
ALLOPS = ARITH + CMP + CTRL + MEMOP


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


def dom_op(res, P):
    best, bv = None, 0.0
    for n in P:
        v = float(res[P[n]])
        if abs(v) > abs(bv):
            best, bv = n, v
    return best, round(bv, 3)


def main():
    for BLK in (8, 41):
        print(f"\n############## BLOCK {BLK} ##############")
        _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
        dp = layout.dim_positions
        P = {n: int(dp[n]) for n in ALLOPS if n in dp}
        probe = GroundTruthProbe.build()

        progs = {
            "expr_mul_div(14*56/8)": "int main() { return 14 * 56 / 8; }",
            "expr_add_mul(3+4*5)":   "int main() { return 3 + 4 * 5; }",
            "expr_mod(46%6+6)":      "int main() { return 46 % 6 + 6; }",
            "if_gt(28>9)":           "int main() { if (28 > 9) return 1; return 0; }",
            "if_lt(35<17)":          "int main() { if (35 < 17) return 1; return 0; }",
            "bool_and":              "int main() { return (17 > 3) && (5 < 9); }",
        }

        for tag, src in progs.items():
            bc = compile_c(src)[0]
            ctx = probe._final_context(bc, max_steps=20)
            full = full_residual_block(probe, ctx, BLK)
            # AX-marker row per step (carries the per-step opcode at +5)
            ax_rows = {}
            for i in range(len(ctx)):
                if ctx[i] == AX_MARK:
                    ax_rows[step_of(ctx, i)] = i
            # STACK0-marker rows
            stack0 = [(step_of(ctx, i), i) for i in range(len(ctx))
                      if ctx[i] == STACK0_MARK]
            print(f"\n== {tag} ==")
            for s, p in stack0:
                this_op, this_v = dom_op(full[p], P)
                # consumer step = s+1 (the NEXT VM step that reads top-of-stack)
                cons_row = ax_rows.get(s + 1)
                if cons_row is not None:
                    cons_op, cons_v = dom_op(full[cons_row], P)
                else:
                    cons_op, cons_v = None, None
                print(f"  STACK0 step={s} pos={p}: "
                      f"this_op={this_op}({this_v}) | "
                      f"consumer(step {s+1} AX)={cons_op}({cons_v})")


if __name__ == "__main__":
    main()
