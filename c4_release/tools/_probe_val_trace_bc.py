#!/usr/bin/env python3
"""Trace a MEM val-byte row's OUTPUT byte across all blocks for the SI smoke
store (working) so we can find WHERE the store value 42 enters. Compare with
ENT to localize the missing ENT value path.

Usage: python tools/_probe_val_trace_bc.py [step] [byteoff]
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
from neural_vm.embedding import Opcode  # noqa

SE = int(Token.STEP_END)


def make_bc(instrs):
    out = []
    for ins in instrs:
        op, imm = ins if isinstance(ins, tuple) else (ins, 0)
        out.append((int(op) & 0xFF) | (int(imm) << 8))
    return out


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
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    boff = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    bc = make_bc([(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42), Opcode.SI,
                  (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT])
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions; inv = {v: k for k, v in dp.items()}
    ctx = probe._final_context(bc, max_steps=30)
    pl = len(probe._build_context(bc))
    spans = step_spans(ctx, pl)
    a, b = spans[step]
    mem_pos = next((p for p in range(a, b) if ctx[p] == 261), None)
    pos = mem_pos + boff
    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    print(f"SI store step={step} byteoff={boff} row={pos} predicts tok@{pos+1}={ctx[pos+1]} (val bytes={ctx[mem_pos+5:mem_pos+9]})")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    def dbyte(row):
        ln = int(torch.argmax(row[OL:OL+16]).item()); hn = int(torch.argmax(row[OH:OH+16]).item())
        return (hn << 4) | ln, round(float(row[OL+ln]),2), round(float(row[OH+hn]),2)

    prev = None
    for blk in range(nblk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
        byte, lov, hiv = dbyte(r)
        amax = int(torch.argmax(r.abs()).item()); amag = float(r[amax])
        mark = "  <<<" if (prev is not None and byte != prev) else ""
        print(f"  blk{blk:2d}: byte={byte:3d} (lo={lov} hi={hiv}) maxdim={inv.get(amax,amax)}={amag:.2e}{mark}")
        prev = byte


if __name__ == "__main__":
    main()
