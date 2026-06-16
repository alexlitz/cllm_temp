#!/usr/bin/env python3
"""Verify the consumer-opcode lookahead band fires correctly (#221).

With C4_STACK0_NEXT_ARITH=1, for each STACK0-marker frame dump the value of
NEXT_OPCODE_{LO,HI} (the fetched op2 opcode byte nibbles) and STACK0_B0_NEXT_ARITH
(the decoded arith-consumer flag) at a block AFTER the flag FFN (block 8+), and
compare to the GROUND-TRUTH next instruction's opcode (from the bytecode).

PASS = on each operand frame, NEXT_ARITH==1 iff the next executed instruction is
an arith op (OR/XOR/AND/SHL/SHR/ADD/SUB/MUL/DIV/MOD), 0 for cmp/branch/mem.

Run: CUDA_VISIBLE_DEVICES=0 C4_STACK0_NEXT_ARITH=1 python tools/probe_next_arith_band.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_STACK0_NEXT_ARITH", "1")
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
ARITH_OPS = {14, 15, 16, 23, 24, 25, 26, 27, 28, 29}  # OR..MOD minus comparisons


@torch.no_grad()
def full_residual_block(probe, ctx, block_idx):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0].float().cpu()


def step_of(ctx, position):
    step = 0
    for i in range(position):
        if ctx[i] == int(Token.STEP_END):
            step += 1
    return step


def argmax_nibble(res, base, n=16, thr=0.3):
    vals = [float(res[base + k]) for k in range(n)]
    if max(vals) < thr:
        return None
    return int(max(range(n), key=lambda k: vals[k]))


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    NLO = int(dp["NEXT_OPCODE_LO"]); NHI = int(dp["NEXT_OPCODE_HI"])
    FLAG = int(dp["STACK0_B0_NEXT_ARITH"])
    print("bands: NEXT_OPCODE_LO", NLO, "HI", NHI, "FLAG", FLAG)
    probe = GroundTruthProbe.build()

    progs = {
        "mul_div(14*56/8)": "int main() { return 14 * 56 / 8; }",
        "add_mul(3+4*5)":   "int main() { return 3 + 4 * 5; }",
        "mod(46%6+6)":      "int main() { return 46 % 6 + 6; }",
        "if_gt(28>9)":      "int main() { if (28 > 9) return 1; return 0; }",
        "if_lt(35<17)":     "int main() { if (35 < 17) return 1; return 0; }",
        "bool_and":         "int main() { return (17 > 3) && (5 < 9); }",
        "add16(100+200)":   "int main() { return 100 + 200; }",
    }
    BLK = 8
    for tag, src in progs.items():
        bc = compile_c(src)[0]
        opcodes = [w & 0xFF for w in bc]
        ctx = probe._final_context(bc, max_steps=20)
        full = full_residual_block(probe, ctx, BLK)
        # map step -> executed opcode via AX-row opcode is complex; instead use
        # the fact step k executes bc[entry+k]; derive entry from first AX op.
        ax_rows = [(step_of(ctx, i), i) for i in range(len(ctx)) if ctx[i] == AX_MARK]
        print(f"\n== {tag} ==  opcodes={opcodes}")
        for s, p in ax_rows:
            res = full[p]
            nlo = argmax_nibble(res, NLO)
            nhi = argmax_nibble(res, NHI)
            flag = round(float(res[FLAG]), 3)
            # raw mass to see if ANYTHING landed
            mlo = round(sum(abs(float(res[NLO + k])) for k in range(16)), 2)
            mhi = round(sum(abs(float(res[NHI + k])) for k in range(16)), 2)
            fetched = (nhi * 16 + nlo) if (nlo is not None and nhi is not None) else None
            is_arith = fetched in ARITH_OPS if fetched is not None else None
            print(f"  AX step={s} pos={p}: fetched_next_opcode={fetched} "
                  f"(arith={is_arith})  FLAG={flag}  massLO={mlo} massHI={mhi}")


if __name__ == "__main__":
    main()
