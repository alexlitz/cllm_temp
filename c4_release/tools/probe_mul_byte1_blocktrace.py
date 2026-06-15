#!/usr/bin/env python3
"""Trace the byte-1 AX-emit token's OUTPUT_LO/HI across ALL blocks.

For a multi-byte MUL whose byte 1 is staged CORRECTLY (AX_FULL) but
emitted WRONG, walk every physical block and print OUTPUT_LO/HI argmax at
the byte-1 emit token row, so we can pinpoint the block that corrupts it.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/probe_mul_byte1_blocktrace.py A B
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def prog(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL, Opcode.EXIT])


def argmax_nib(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    cells = [float(row[base + i].item()) for i in range(width)]
    mx = max(range(width), key=lambda i: cells[i])
    return mx, round(cells[mx], 2)


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)

    bc = prog(a, b)
    ctx = probe._final_context(bc, max_steps=20)
    exp = a * b
    # last REG_AX token, byte-1 emit row = ax_pos+2
    ax_pos = None
    for i in range(len(ctx) - 1, -1, -1):
        if ctx[i] == int(Token.REG_AX):
            ax_pos = i
            break
    row0 = ax_pos + 1
    row1 = ax_pos + 2
    print(f"A={a} B={b} exp=0x{exp:04x} b1=0x{(exp>>8)&0xff:02x}  ax_pos={ax_pos} "
          f"byte0_row={row0} byte1_row={row1} nblocks={nblocks}")

    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    prev = None
    for blk in range(nblocks):
        with torch.no_grad():
            r = model.forward(toks, stop_after_block=blk)[0]
        lg = getattr(model.blocks[blk], "_logical_layer", blk)
        ol = argmax_nib(r[row1], dp, 'OUTPUT_LO')
        oh = argmax_nib(r[row1], dp, 'OUTPUT_HI')
        oht = argmax_nib(r[row1], dp, 'OUTPUT_HI_THIS_STEP')
        byte = (oh[0] << 4) | ol[0]
        cur = (ol, oh)
        marker = " <==CHANGED" if cur != prev else ""
        print(f"  blk{blk:2d}(L{lg:2d}) OUTPUT_LO={str(ol):14s} OUTPUT_HI={str(oh):14s} "
              f"OUTPUT_HI_TS={str(oht):14s} byte->0x{byte:02x}{marker}")
        prev = cur


if __name__ == "__main__":
    main()
