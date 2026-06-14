#!/usr/bin/env python3
"""Trace a single MEM byte-row's OUTPUT byte (and raw OUTPUT_LO/HI argmax mag)
across EVERY block to pinpoint the sentinel writer. Also reports the dim with
the max magnitude in OUTPUT region per block.

Usage: python tools/_probe_addr_trace.py [id] [step] [byteoffset]
  byteoffset: 0..3 addr bytes, 4..7 val bytes (row = mem_pos+byteoffset)
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
    boff = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    inv = {v: k for k, v in dp.items()}
    ctx = probe._final_context(bc, max_steps=20)
    pl = len(probe._build_context(bc))
    spans = step_spans(ctx, pl)
    a, b = spans[step]
    mem_pos = next((p for p in range(a, b) if ctx[p] == 261), None)
    pos = mem_pos + boff
    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    print(f"id{pid} {desc} step={step} byteoff={boff} row_pos={pos} (predicts tok@{pos+1}={ctx[pos+1]})")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    def dbyte(row):
        ln = int(torch.argmax(row[OL:OL+16]).item()); hn = int(torch.argmax(row[OH:OH+16]).item())
        return (hn << 4) | ln, round(float(row[OL+ln]),2), round(float(row[OH+hn]),2)

    prev = None
    for blk in range(nblk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
        byte, lov, hiv = dbyte(r)
        # max-magnitude dim in whole residual
        amax = int(torch.argmax(r.abs()).item()); amag = float(r[amax])
        mark = ""
        if prev is not None and byte != prev:
            mark = "  <<< CHANGED"
        print(f"  blk{blk:2d}: OUTPUT byte={byte:3d} (lo={lov} hi={hiv})  maxdim={inv.get(amax,amax)}={amag:.2e}{mark}")
        prev = byte


if __name__ == "__main__":
    main()
