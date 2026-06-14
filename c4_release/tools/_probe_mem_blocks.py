#!/usr/bin/env python3
"""Trace OUTPUT_LO/HI (decoded byte) at each MEM-section row across ALL blocks
for a given step, to see WHERE the store value is written / corrupted.

Usage: python tools/_probe_mem_blocks.py [id] [step]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 268: "STACK0", 261: "MEM"}


def step_spans(ctx, pl):
    spans = []; i = pl; start = pl
    while i < len(ctx):
        if ctx[i] == SE:
            spans.append((start, i)); start = i + 1
        i += 1
    if start < len(ctx):
        spans.append((start, len(ctx)))
    return spans


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=20)
    pl = len(probe._build_context(bc))
    spans = step_spans(ctx, pl)
    a, b = spans[step]
    mem_pos = next((p for p in range(a, b) if ctx[p] == 261), None)
    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    print(f"id{pid} {desc} step={step} span[{a}:{b}] MEM@{mem_pos}")
    print(f"  addr bytes={ctx[mem_pos+1:mem_pos+5]} val bytes={ctx[mem_pos+5:mem_pos+9]}")

    def dbyte(row, lo, hi):
        ln = int(torch.argmax(row[lo:lo+16]).item())
        hn = int(torch.argmax(row[hi:hi+16]).item())
        return (hn << 4) | ln, round(float(row[lo+ln]),2), round(float(row[hi+hn]),2)

    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    # firing rows: addr predicted at mem_pos..mem_pos+3, val at mem_pos+4..mem_pos+7
    rows = {f"addr_b{k}": mem_pos + k for k in range(4)}
    rows.update({f"val_b{k}": mem_pos + 4 + k for k in range(4)})
    interest_blocks = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, nblk-1]
    cache = {}
    for blk in interest_blocks:
        if blk >= nblk: continue
        cache[blk] = probe.model.forward(padded, stop_after_block=blk)[0]
    for name, pos in rows.items():
        print(f"\n {name} (fire_row={pos}, predicts tok@{pos+1}={ctx[pos+1] if pos+1<len(ctx) else '?'}):")
        for blk in interest_blocks:
            if blk not in cache: continue
            r = cache[blk][pos]
            byte, lov, hiv = dbyte(r, OL, OH)
            print(f"    blk{blk:2d}: OUTPUT byte={byte:3d} (lo_w={lov} hi_w={hiv})")


if __name__ == "__main__":
    main()
