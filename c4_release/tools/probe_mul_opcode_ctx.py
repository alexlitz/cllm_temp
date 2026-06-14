#!/usr/bin/env python3
"""Dump ALL opcode-band residuals at every carried STACK0 row for regressed-mul,
regressed-if, and ok-if programs to find a structural separator (the mul
operand-setup PSH row vs the if comparison-drift PSH row).

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_mul_opcode_ctx.py
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
ALLOPS = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD", "OP_AND", "OP_OR",
          "OP_XOR", "OP_SHL", "OP_SHR", "OP_JMP", "OP_BZ", "OP_BNZ", "OP_JSR",
          "OP_LEV", "OP_ENT", "OP_ADJ", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
          "OP_EQ", "OP_NE", "OP_LI", "OP_SI", "OP_LC", "OP_SC", "OP_LEA",
          "OP_PSH", "OP_IMM"]
LIVE_BLK = 44


@torch.no_grad()
def residual_at_block(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return (r.to_dense() if r.is_sparse else r).float().cpu()


def find_markers(ctx, prompt_len):
    rows, i, step = [], prompt_len, 0
    while i < len(ctx):
        if ctx[i] == STACK0_MARK:
            rows.append((step, i))
        if ctx[i] == int(Token.STEP_END):
            step += 1
        i += 1
    return rows


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    P = {n: int(dp[n]) for n in ALLOPS if n in dp}
    CARRIED = int(dp["STACK0_B0_CARRIED"])
    probe = GroundTruthProbe.build()

    progs = {
        "MUL_21x59":   "int main() { return 21 * 59; }",
        "MUL_97x94":   "int main() { return 97 * 94; }",
        "ifGT_50x89":  "int main() { if (50 > 89) return 1; return 0; }",
        "ifGT_28x9":   "int main() { if (28 > 9) return 1; return 0; }",
        "ifEQ_30x45":  "int main() { if (30 == 45) return 1; return 0; }",
        "boolAND":     "int main() { return (17 > 3) && (5 < 9); }",
        "boolOR":      "int main() { return (17 > 3) || (5 > 9); }",
        # arith result rows the dump must also stay dark on:
        "ADD16":       "int main() { return 100 + 200; }",
        "SUB16":       "int main() { return 300 - 100; }",
    }

    for tag, src in progs.items():
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=16)
        pl = len(probe._build_context(bc))
        rows = find_markers(ctx, pl)
        print(f"\n=== {tag}  (block {LIVE_BLK}) ===")
        for step, p in rows:
            res = residual_at_block(probe, ctx, LIVE_BLK, p)
            carr = round(float(res[CARRIED]), 1)
            if carr <= 0:
                continue
            nz = {n: round(float(res[P[n]]), 3) for n in P
                  if abs(float(res[P[n]])) > 0.004}
            print(f" step{step:>2} pos{p:>4} CARR={carr} | {nz}")


if __name__ == "__main__":
    main()
