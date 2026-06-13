#!/usr/bin/env python3
"""Compare carried STACK0 rows: framing-drift (if) vs healthy (add_16bit).

The re-point dump fires on EVERY carried STACK0 row. On the if/bool/expr
framing-drift row the byte's H1/H3 are nuked (-1e7) and the re-point correctly
re-supplies; but on a HEALTHY carried STACK0 row (e.g. add_16bit result) the
H1/H3 are already CORRECT, and the re-point must NOT corrupt them. This probe
dumps, for each carried STACK0-marker row of a chosen program, the residual
ENTERING the dump block (37): the H1/H3 emission cells (is the byte nuked?), the
STACK0_B0_*_PREV one-hot (what would the re-point write?), and CARRIED.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_add16.py
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
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

_REG = build_default_registry_dynamic()
STACK0_MARK = 268


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
    b_h1p = int(dp["STACK0_B0_H1_PREV"]); b_h3p = int(dp["STACK0_B0_H3_PREV"])
    b_carr = int(dp["STACK0_B0_CARRIED"])
    H1 = _REG.slots["H1"].start; H3 = _REG.slots["H3"].start
    probe = GroundTruthProbe.build()

    progs = {
        "if_drift": "int main() { if (17 > 35) return 1; return 0; }",
        "add_16bit": "int main() { return 100 + 200; }",
    }
    for tag, src in progs.items():
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=14)
        pl = len(probe._build_context(bc))
        rows = find_markers(ctx, pl)
        print(f"\n##### {tag}: {src} #####")
        print(f"  STACK0 markers (step,pos): {rows}")
        for step, pos in rows:
            res = residual_at_block(probe, ctx, 37, pos)  # entering dump-ish region
            h1 = [round(float(res[H1 + j]), 1) for j in range(7)]
            h3 = [round(float(res[H3 + j]), 1) for j in range(7)]
            carr = round(float(res[b_carr]), 1)
            h1p = [round(float(res[b_h1p + j]), 1) for j in range(7)]
            h3p = [round(float(res[b_h3p + j]), 1) for j in range(7)]
            h1nuke = min(h1); h3nuke = min(h3)
            # PREV sharpness: max / (sum of all). A clean one-hot ~ 1.0; a smear
            # ~ 1/W. Use H1_PREV (low nibble band).
            sm1 = sum(abs(v) for v in h1p); mx1 = max(abs(v) for v in h1p)
            sharp1 = round(mx1 / sm1, 2) if sm1 > 1e-3 else 0.0
            sm3 = sum(abs(v) for v in h3p); mx3 = max(abs(v) for v in h3p)
            sharp3 = round(mx3 / sm3, 2) if sm3 > 1e-3 else 0.0
            print(f"  step{step} pos{pos}: CARRIED={carr} "
                  f"H1min={h1nuke:.0f} H3min={h3nuke:.0f} "
                  f"PREV_sharp(H1={sharp1},H3={sharp3})")
            print(f"      H1={h1} H3={h3}")
            print(f"      H1_PREV={h1p} H3_PREV={h3p}")


if __name__ == "__main__":
    main()
