#!/usr/bin/env python3
"""Trace the store-VALUE relay dims at the MEM val_b0 prediction row across
blocks, for either a program-id (func) or the SI smoke bytecode. Shows the
STACK0_BYTE_VAL_* and STACK0_B0_* and OUTPUT bytes so we can see where the
value enters and how it relays to OUTPUT at block 30.

Usage: python tools/_probe_storeval_path.py {si|<id>} [step] [byteoff]
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
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

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
    which = sys.argv[1] if len(sys.argv) > 1 else "si"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else (3 if which == "si" else 1)
    boff = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if which == "si":
        bc = make_bc([(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42), Opcode.SI,
                      (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT]); ms = 30; desc = "SI"
    else:
        src, exp, desc = generate_test_programs()[int(which)]; bc = compile_c(src)[0]; ms = 20
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    spans = step_spans(ctx, pl)
    a, b = spans[step]
    mem_pos = next((p for p in range(a, b) if ctx[p] == 261), None)
    pos = mem_pos + boff
    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    SBV = {f"SBV{k}": dp.get(f"STACK0_BYTE_VAL_{k}_LO") for k in (1, 2, 3)}
    SB0 = {n: dp.get(n) for n in ("STACK0_BYTE0",)}
    print(f"{desc} step={step} byteoff={boff} row={pos} val_bytes={ctx[mem_pos+5:mem_pos+9]}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    def nib16(row, base):
        if base is None: return None
        n = int(torch.argmax(row[base:base+16]).item()); return n, round(float(row[base+n]),2)

    for blk in range(8, nblk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
        ob_l = nib16(r, OL); ob_h = nib16(r, OH)
        out_byte = (ob_h[0] << 4 | ob_l[0]) if ob_l and ob_h else None
        sbv = {k: nib16(r, v) for k, v in SBV.items() if v is not None}
        sbv = {k: v for k, v in sbv.items() if v and abs(v[1]) > 0.3}
        sb0v = round(float(r[dp["STACK0_BYTE0"]]), 2) if "STACK0_BYTE0" in dp else None
        # show STACK0_B0_H*_PREV bands raw max
        b0h1 = round(float(r[916:923].abs().max()), 1)
        b0h3 = round(float(r[923:930].abs().max()), 1)
        print(f"  blk{blk:2d}: OUT_byte={out_byte} OL_nib={ob_l} OH_nib={ob_h}  SBV={sbv}  STACK0_B0={sb0v} B0H1max={b0h1} B0H3max={b0h3}")


if __name__ == "__main__":
    main()
