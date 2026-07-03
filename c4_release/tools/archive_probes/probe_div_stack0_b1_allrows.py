#!/usr/bin/env python3
"""Scan ALL rows / blocks for STACK0_BYTE_VAL_1 to settle (a)/(b)/(c).

For a multi-byte dividend (1162/37, high byte 0x04) we want to know whether
the PSH-stored high byte is present at ANY (row, block) in the residual.

Reads BUILT layout.dim_positions. spec_k=0, hook-free.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/probe_div_stack0_b1_allrows.py
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


PROGRAMS = {
    # 1162 = 0x048A : byte0=0x8A, byte1=0x04
    "div_1162_37": (_mk([(Opcode.IMM, 1162), Opcode.PSH, (Opcode.IMM, 37), Opcode.DIV, Opcode.EXIT]), 31),
    # 300 = 0x012C : byte0=0x2C, byte1=0x01
    "div_300_5": (_mk([(Opcode.IMM, 300), Opcode.PSH, (Opcode.IMM, 5), Opcode.DIV, Opcode.EXIT]), 60),
    # control single byte
    "div_84_2": (_mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 2), Opcode.DIV, Opcode.EXIT]), 42),
}


def nibval(cells):
    hot = [(i, round(v, 2)) for i, v in enumerate(cells) if v > 0.5]
    return hot


def band(row, base, width=16):
    return [round(float(row[base + i].item()), 3) for i in range(width)]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)

    s1lo = dp["STACK0_BYTE_VAL_1_LO"]
    s1hi = dp["STACK0_BYTE_VAL_1_HI"]
    clo = dp["CLEAN_EMBED_LO"]; chi = dp["CLEAN_EMBED_HI"]
    olo = dp["OUTPUT_LO"]; ohi = dp["OUTPUT_HI"]
    s0b1 = dp["STACK0_BYTE1"]
    mark_ax = dp["MARK_AX"]
    bi1 = dp.get("BYTE_INDEX_1")
    print(f"# dims: S0VAL1_LO={s1lo} S0VAL1_HI={s1hi} STACK0_BYTE1={s0b1} "
          f"CLEAN_LO={clo} OUTPUT_LO={olo} BYTE_INDEX_1={bi1}")

    for pname in selected:
        prog, expected = PROGRAMS[pname]
        ctx = probe._final_context(prog, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        out, code = probe.emitted_result(prog, max_steps=6)
        print(f"\n==== {pname} expected={expected} neural={code} "
              f"{'PASS' if code == expected else 'FAIL'} len(ctx)={len(ctx)} ====")

        # For each block, find ANY row where STACK0_BYTE_VAL_1 is hot.
        for blk in range(nblocks):
            resid = model.forward(padded, stop_after_block=blk)[0]
            for r in range(resid.shape[0]):
                lo = nibval(band(resid[r], s1lo))
                hi = nibval(band(resid[r], s1hi))
                if lo or hi:
                    # also show STACK0_BYTE1 flag + BYTE_INDEX_1 + CLEAN at this row
                    flag = round(float(resid[r, s0b1].item()), 2)
                    bi1v = round(float(resid[r, bi1].item()), 2) if bi1 is not None else None
                    print(f"  blk{blk:2d} row{r:3d}: S0VAL1_LO={lo} S0VAL1_HI={hi} "
                          f"STACK0_BYTE1={flag} BYTE_INDEX_1={bi1v}")

    print("\n(If NO rows printed for the multi-byte dividend: high byte stored nowhere => case (b).)")
    print("(If rows printed at the STACK0 byte-1 row but NOT at the DIV compute AX row: stored, not relayed => case (a).)")


if __name__ == "__main__":
    sel = sys.argv[1:] or ["div_1162_37", "div_300_5", "div_84_2"]
    main(sel)
