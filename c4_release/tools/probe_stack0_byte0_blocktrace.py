#!/usr/bin/env python3
"""Find the block that introduces the H1/H3 negative blowup at the STACK0
byte-0 predictor row on the carried CMP step (Root 2).

The STACK0 marker row (which predicts the byte-0 token) on the carried CMP step
carries an ENORMOUS negative residual in the H1/H3 nibble-emission bands
(~-10M), which the LM head reads at +5.0 and so suppresses the byte-0 token to
~-100M -> a [PC] marker wins. This walks the residual block-by-block to find
which block writes the blowup, and dumps the H0..H3 nibble bands per block.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_byte0_blocktrace.py
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

_REG = build_default_registry_dynamic()
STACK0_MARK = 268


@torch.no_grad()
def residual_at_block(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return (r.to_dense() if r.is_sparse else r).float().cpu()


def find_stack0_markers(ctx, prompt_len):
    rows = []
    i = prompt_len
    step = 0
    while i < len(ctx):
        if ctx[i] == STACK0_MARK:
            rows.append((step, i))
        if ctx[i] == int(Token.STEP_END):
            step += 1
        i += 1
    return rows


def band(res, name, width):
    base = _REG.slots[name].start
    return [round(float(res[base + j]), 1) for j in range(width)]


def main():
    probe = GroundTruthProbe.build()
    model = probe.model
    nblocks = len(model.blocks)

    src = "int main() { if (17 > 35) return 1; return 0; }"  # DRIFT 0x11
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=12)
    pl = len(probe._build_context(bc))
    rows = find_stack0_markers(ctx, pl)
    # step 2 = carried CMP step. Use the FIRST STACK0 marker in that step.
    cmp_marker = next(p for s, p in rows if s == 2)
    psh_marker = next(p for s, p in rows if s == 1)
    print(f"src={src!r}")
    print(f"PSH STACK0 marker pos={psh_marker} (step1)  "
          f"CMP STACK0 marker pos={cmp_marker} (step2)")

    for label, pos in (("PSH(step1,ok)", psh_marker), ("CMP(step2,bug)", cmp_marker)):
        print(f"\n##### {label} pos={pos} — H1/H3 per block #####")
        prev_h1sum = 0.0
        for blk in range(nblocks):
            res = residual_at_block(probe, ctx, blk, pos)
            h1 = band(res, "H1", 7)
            h3 = band(res, "H3", 7)
            h1sum = sum(abs(v) for v in h1)
            mark = ""
            if h1sum > 100 and prev_h1sum <= 100:
                mark = "  <<< H1 BLOWUP STARTS HERE"
            print(f"  block {blk:2d}: H1={h1}  H3={h3}{mark}")
            prev_h1sum = h1sum


if __name__ == "__main__":
    main()
