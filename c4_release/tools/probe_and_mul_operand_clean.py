#!/usr/bin/env python3
"""Operand cleanliness probe for AND/MUL at the binop AX row.

Mirrors ``probe_div_operand_clean.py`` but dumps the full ALU_LO/HI and
AX_CARRY_LO/HI bands (operand A = ALU, operand B = AX_CARRY for the
bitwise/mul efficient-wrap lookups) at the AX row across several blocks,
so we can see exactly how the value maps to nibble cells and where the
cell-0 magnitude artifact (Wall-1, AX-row) sits.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_and_mul_operand_clean.py [and_basic|mul_basic|and_16bit]
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
    "and_basic": (_mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]), 0x2A),
    "mul_basic": (_mk([(Opcode.IMM, 6),    Opcode.PSH, (Opcode.IMM, 7),    Opcode.MUL, Opcode.EXIT]), 42),
    "and_16bit": (_mk([(Opcode.IMM, 0x0FFF), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.AND, Opcode.EXIT]), 0xFF),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [round(float(row[base + i].item()), 2) for i in range(width)]


def hot(cells, thr=0.3):
    if cells is None:
        return []
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    se_base = dp["MARK_SE_ONLY"]
    ax_base = dp["MARK_AX"]

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_row = se_rows[-1] if se_rows else None
        ax_row = max((r for r in ax_rows if (se_row is None or r < se_row)),
                     default=(ax_rows[-1] if ax_rows else None))
        print(f"=== {pname} expected={hex(expected)} S={S} ax_row={ax_row} se_row={se_row} ===")
        print(f"    ax_rows={ax_rows} se_rows={se_rows}")
        for blk in (8, 10, 13, 14, 15, 26, 27):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            row = resid[ax_row]
            print(f"  blk{blk:2d} AXrow(pos={ax_row}): "
                  f"ALU_LO={hot(band(row,dp,'ALU_LO'))} "
                  f"ALU_HI={hot(band(row,dp,'ALU_HI'))} "
                  f"CARRY_LO={hot(band(row,dp,'AX_CARRY_LO'))} "
                  f"CARRY_HI={hot(band(row,dp,'AX_CARRY_HI'))}")
        print()


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
