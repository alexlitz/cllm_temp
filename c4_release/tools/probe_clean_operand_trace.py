#!/usr/bin/env python3
"""Trace the ALU_LO/HI operand band block-by-block at the ADD operand row.

Localizes WHERE the cell-8 (+0.45) and cell-0 (+/-0.5) bleed enters the
operand band the ALU reads, and dumps the CLEAN_EMBED at the attended
STACK0 carrier row (the CamValueBand source). This answers: is the dirty
hybrid intrinsic to the gather (softmax spread / dirty CLEAN_EMBED) or a
downstream additive write?

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_clean_operand_trace.py
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


# ADD a=0x02 (nibble2), b=0x00 -> want 0x02. Simple, single-byte, clean.
# Also 0x0C + 0x03 to exercise a mid nibble.
PROGRAMS = {
    "add_2_0": (_mk([(Opcode.IMM, 0x02), Opcode.PSH, (Opcode.IMM, 0x00), Opcode.ADD, Opcode.EXIT]), 0x02),
    "add_C_3": (_mk([(Opcode.IMM, 0x0C), Opcode.PSH, (Opcode.IMM, 0x03), Opcode.ADD, Opcode.EXIT]), 0x0F),
    "add_5_0": (_mk([(Opcode.IMM, 0x05), Opcode.PSH, (Opcode.IMM, 0x00), Opcode.ADD, Opcode.EXIT]), 0x05),
}


def band(row, dp, name, width=16):
    base = dp[name]
    return [round(float(row[base + i].item()), 3) for i in range(width)]


def hot(cells, thr=0.1):
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    st0_base = dp.get("STACK0_BYTE0")
    ce_lo = dp.get("CLEAN_EMBED_LO")

    for pname, (bc, expected) in PROGRAMS.items():
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        ax_row = ax_rows[-1]
        # find STACK0_BYTE0 carrier rows (the CamValueBand source)
        st0_rows = [r for r in range(S) if st0_base is not None and emb[r, st0_base].abs().item() > 0.5]
        print(f"=== {pname} want={hex(expected)} S={S} ax_row={ax_row} stack0_rows={st0_rows} ===")
        # CLEAN_EMBED at the carrier rows (embed-level, before any block)
        if ce_lo is not None:
            for r in st0_rows[-2:]:
                print(f"  embed CLEAN_EMBED_LO@row{r}: {hot(band(emb[r], dp, 'CLEAN_EMBED_LO'))}")
        # trace ALU_LO band at ax_row across blocks (0..12)
        for blk in range(0, 13):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            row = resid[ax_row]
            al = band(row, dp, "ALU_LO")
            print(f"  blk{blk:2d} ALU_LO@ax: {hot(al)}")
        print()


if __name__ == "__main__":
    main()
