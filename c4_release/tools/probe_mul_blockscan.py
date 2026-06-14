#!/usr/bin/env python3
"""Scan late blocks to find where SHARP/PREV/NOT_CMP go live and where the dump
fires, for a regressed mul operand-setup row vs a fixed if comparison row.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_mul_blockscan.py
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
    H1 = int(dp["STACK0_B0_H1_PREV"])
    H3 = int(dp["STACK0_B0_H3_PREV"])
    NOT_CMP = int(dp["STACK0_B0_NOT_CMP"])
    PREV_DOM = int(dp["STACK0_B0_PREV_DOM"])
    CARRIED = int(dp["STACK0_B0_CARRIED"])
    SHARP = int(dp["STACK0_B0_SHARP"])
    probe = GroundTruthProbe.build()
    nblk = len(probe.model.blocks)

    # mul 21*59: the corrupted step-3 result row (pos chosen from prior probe).
    # if_gt: a healthy carried comparison drift row.
    progs = {
        # corpus if values that REGRESSED with the mass-blocker
        "if_gt_50x89": "int main() { if (50 > 89) return 1; return 0; }",
        "if_gt_28x9":  "int main() { if (28 > 9) return 1; return 0; }",
        "if_lt_50x44": "int main() { if (50 < 44) return 1; return 0; }",
        "if_eq_49x49": "int main() { if (49 == 49) return 1; return 0; }",
        "if_eq_30x45": "int main() { if (30 == 45) return 1; return 0; }",
        # corpus if values that STAYED OK
        "if_gt_17x35": "int main() { if (17 > 35) return 1; return 0; }",
        # mul over-fire references
        "mul_21x59":   "int main() { return 21 * 59; }",
        "mul_97x94":   "int main() { return 97 * 94; }",
    }

    def band_vec(res, base):
        return [round(float(res[base + j]), 1) for j in range(W)]

    # Probe EVERY STACK0 row at the LIVE post-op block (44: SHARP+NOT_CMP live).
    LIVE_BLK = 44
    for tag, src in progs.items():
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=16)
        pl = len(probe._build_context(bc))
        rows = find_markers(ctx, pl)
        print(f"\n=== {tag}  (block {LIVE_BLK}, {len(rows)} STACK0 rows) ===")
        print(f"{'step':>4} {'pos':>4} {'CARR':>6} {'SHARP':>7} {'PDOM':>7} "
              f"{'NOTCMP':>8} | FIRE? | H3_PREV / H1_PREV")
        for step, p in rows:
            res = residual_at_block(probe, ctx, LIVE_BLK, p)
            carr = round(float(res[CARRIED]), 1)
            sharp = round(float(res[SHARP]), 1)
            pdom = round(float(res[PREV_DOM]), 1)
            ncmp = round(float(res[NOT_CMP]), 1)
            h3 = band_vec(res, H3)
            h1 = band_vec(res, H1)
            # dump gate: 2 + 3*CARR + 3*SHARP - 1000*NOT_CMP > 7
            gate = 2 + 3 * carr + 3 * sharp - 1000 * ncmp
            fire = "FIRE" if gate > 7 else "dark"
            tag2 = ""
            if fire == "FIRE" and carr > 0 and sharp < 1:
                tag2 = "  <== OVER-FIRE (SHARP=0 but gate fires via CARRIED)"
            # band MASS (Σ slots) — the proposed discriminator.
            h1mass = sum(h1)
            h3mass = sum(h3)
            if carr <= 0:
                continue  # only carried rows can be over-fire victims
            blk_h1 = 2 - 0.05 * h1mass  # >1 -> block (weak low nibble)
            blk_h3 = 2 - 0.05 * h3mass
            newblock = (blk_h1 > 1) or (blk_h3 > 1)
            warn = "  <<< MASS-BLOCK" if newblock else ""
            print(f"{step:>4} {p:>4} {carr:>6} {sharp:>7} NOTCMP={ncmp:>7} "
                  f"| H1mass={h1mass:6.1f} H3mass={h3mass:6.1f}{warn}")


if __name__ == "__main__":
    main()
