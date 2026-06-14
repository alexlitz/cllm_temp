#!/usr/bin/env python3
"""Find a batched-robust discriminator separating mul operand-SETUP PSH rows
from if comparison-RESULT PSH rows at the STACK0 byte-0 dump block input.

Dumps, for EVERY carried STACK0 row of a regressed mul program and a fixed if
program, the carry-band nibble one-hots (STACK0_B0_H1_PREV = low nibble,
STACK0_B0_H3_PREV = high nibble) plus the opcode sums and the current
NOT_CMP-rule firing. The boolean-magnitude hypothesis: the if comparison
result byte is always 0/1 (high nibble 0), so STACK0_B0_H3_PREV high-nibble
slots (hi>=1) are a clean MUL-operand separator.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_mul_operand_discriminator.py
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
CTRL = ["OP_JMP", "OP_BZ", "OP_BNZ"]
CMP = ["OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_EQ", "OP_NE"]
# The dump / not_cmp FFN run as L25-tail post_ops; probe near the final blocks.
DUMP_BLK = 41
W = 7


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
    P = {n: int(dp[n]) for n in (ARITH + CTRL + CMP) if n in dp}
    H1 = int(dp["STACK0_B0_H1_PREV"])
    H3 = int(dp["STACK0_B0_H3_PREV"])
    NOT_CMP = int(dp["STACK0_B0_NOT_CMP"])
    PREV_DOM = int(dp["STACK0_B0_PREV_DOM"])
    CARRIED = int(dp["STACK0_B0_CARRIED"])
    SHARP = int(dp["STACK0_B0_SHARP"])
    probe = GroundTruthProbe.build()

    progs = {
        # regressed mul (21*59=1239 -> got 61655): operand setup carries operands
        "mul_21x59": "int main() { return 21 * 59; }",
        "mul_26x20": "int main() { return 26 * 20; }",
        "mul_9x98":  "int main() { return 9 * 98; }",
        # fixed if (comparison result byte is 0 or 1)
        "if_gt":     "int main() { if (17 > 35) return 1; return 0; }",
        "if_lt":     "int main() { if (35 < 17) return 1; return 0; }",
        "bool_and":  "int main() { return (17 > 3) && (5 < 9); }",
    }

    def band_vec(res, base):
        return [round(float(res[base + j]), 2) for j in range(W)]

    def dom_slot(vec):
        # argmax slot if a clean one-hot (slot > sum of others), else None
        mx = max(range(W), key=lambda k: vec[k])
        if vec[mx] > sum(vec[k] for k in range(W) if k != mx) and vec[mx] > 0.5:
            return mx
        return None

    for tag, src in progs.items():
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=16)
        pl = len(probe._build_context(bc))
        rows = find_markers(ctx, pl)
        print(f"\n=== {tag}  ({len(rows)} STACK0 rows, block {DUMP_BLK}) ===")
        for step, p in rows:
            res = residual_at_block(probe, ctx, DUMP_BLK, p)
            sa = sum(float(res[P[n]]) for n in ARITH if n in P)
            sc = sum(float(res[P[n]]) for n in CTRL if n in P)
            scmp = sum(float(res[P[n]]) for n in CMP if n in P)
            h1 = band_vec(res, H1)
            h3 = band_vec(res, H3)
            d1, d3 = dom_slot(h1), dom_slot(h3)
            # high nibble = d3 - 4 (slot = hi+4); low nibble = d1 - 2 (slot=lo+2)
            hi = (d3 - 4) if d3 is not None else None
            lo = (d1 - 2) if d1 is not None else None
            carried = round(float(res[CARRIED]), 2)
            sharp = round(float(res[SHARP]), 2)
            not_cmp = round(float(res[NOT_CMP]), 2)
            prev_dom = round(float(res[PREV_DOM]), 2)
            print(f" step{step:>2} pos{p:>4} | Σari={sa:6.3f} Σctrl={sc:6.3f} "
                  f"Σcmp={scmp:6.3f} | H3dom={d3}(hi={hi}) H1dom={d1}(lo={lo}) "
                  f"| CARR={carried} SHARP={sharp} PDOM={prev_dom} NOT_CMP={not_cmp}")
            print(f"          H3_PREV={h3}")
            print(f"          H1_PREV={h1}")


if __name__ == "__main__":
    main()
