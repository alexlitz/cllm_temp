#!/usr/bin/env python3
"""Trace the BDToGEConverter multi-byte DIV fallback inputs at BUILT dims.

The converter (efficient_alu_neural.py:141-223) recovers operand-A byte 1
for DIV/MOD from the latest STACK0_BYTE1 row's CLEAN_EMBED_LO/HI via cummax.
We check, at the residual the L10 divmod post_op reads (just BEFORE the
divmod block), at the DIV compute AX row:
  - is AX_FULL_LO/HI staged? (-> position 2/3 path 1)
  - else: which STACK0_BYTE1 row does cummax pick, and does its
    CLEAN_EMBED_LO/HI carry the dividend high byte?

spec_k=0, hook-free, BUILT layout.dim_positions.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_bdge_fallback.py
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
    "div_1162_37": (_mk([(Opcode.IMM, 1162), Opcode.PSH, (Opcode.IMM, 37), Opcode.DIV, Opcode.EXIT]), 31),
    "div_300_5": (_mk([(Opcode.IMM, 300), Opcode.PSH, (Opcode.IMM, 5), Opcode.DIV, Opcode.EXIT]), 60),
    "div_84_2": (_mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 2), Opcode.DIV, Opcode.EXIT]), 42),
}


def onehot(row, base, w=16):
    return [i for i in range(w) if float(row[base + i].item()) > 0.5]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)

    AX_FULL_LO = dp.get("AX_FULL_LO"); AX_FULL_HI = dp.get("AX_FULL_HI")
    CLEAN_LO = dp["CLEAN_EMBED_LO"]; CLEAN_HI = dp["CLEAN_EMBED_HI"]
    STACK0_BYTE1 = dp["STACK0_BYTE1"]
    MARK_AX = dp["MARK_AX"]
    OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]
    ALU_LO = dp["ALU_LO"]; ALU_HI = dp["ALU_HI"]

    # Find the physical block that holds the divmod post_op (the FlattenedDivMod
    # / lookup install). We read the residual at the block JUST BEFORE it.
    print(f"# nblocks={nblocks}")
    for pname in selected:
        prog, expected = PROGRAMS[pname]
        ctx = probe._final_context(prog, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        out, code = probe.emitted_result(prog, max_steps=6)
        print(f"\n==== {pname} expected={expected} neural={code} "
              f"{'PASS' if code==expected else 'FAIL'} len={len(ctx)} ====")

        # Scan blocks 9..13 (the L10 region) for the input residual to divmod.
        for blk in (10, 11, 12, 13):
            if blk >= nblocks:
                continue
            resid = model.forward(padded, stop_after_block=blk)[0]
            # DIV/MOD compute rows
            div_rows = [r for r in range(resid.shape[0])
                        if float(resid[r, MARK_AX].item()) > 0.5
                        and (float(resid[r, OP_DIV].item()) > 0.5
                             or float(resid[r, OP_MOD].item()) > 0.5)]
            if not div_rows:
                continue
            row = div_rows[-1]
            r = resid[row]
            axf_lo = onehot(r, AX_FULL_LO) if AX_FULL_LO else None
            axf_hi = onehot(r, AX_FULL_HI) if AX_FULL_HI else None
            ax_full_present = bool(axf_lo or axf_hi)
            # Replicate cummax over STACK0_BYTE1 up to the div row
            stack1_rows = [r2 for r2 in range(row + 1)
                           if float(resid[r2, STACK0_BYTE1].item()) > 0.5]
            picked = stack1_rows[-1] if stack1_rows else None
            picked_clean_lo = onehot(resid[picked], CLEAN_LO) if picked is not None else None
            picked_clean_hi = onehot(resid[picked], CLEAN_HI) if picked is not None else None
            alu_lo_v = onehot(r, ALU_LO); alu_hi_v = onehot(r, ALU_HI)
            print(f"  blk{blk:2d} divrow={row}: ALU_LO={alu_lo_v} ALU_HI={alu_hi_v} "
                  f"| AX_FULL_present={ax_full_present} AX_FULL_LO={axf_lo} AX_FULL_HI={axf_hi}")
            print(f"           STACK0_BYTE1 rows<=divrow={stack1_rows} picked={picked} "
                  f"-> CLEAN_EMBED_LO={picked_clean_lo} CLEAN_EMBED_HI={picked_clean_hi}")


if __name__ == "__main__":
    sel = sys.argv[1:] or ["div_1162_37", "div_300_5", "div_84_2"]
    main(sel)
