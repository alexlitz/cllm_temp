#!/usr/bin/env python3
"""Probe operand-band cleanliness at the AX row and SE row for div_basic.

Focused: at the L10 install block (~23) and just before it, dump ALU_LO/HI,
AX_CARRY_LO/HI at BOTH the AX marker row and SE row to determine whether the
@0 magnitude artifact (Wall 1) persists at the row the wide_div install FFN
gates on (MARK_AX).

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_div_operand_clean.py
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
    "div_basic": (_mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 2),  Opcode.DIV, Opcode.EXIT]), 42),
    "mod_basic": (_mk([(Opcode.IMM, 43), Opcode.PSH, (Opcode.IMM, 10), Opcode.MOD, Opcode.EXIT]), 3),
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
        print(f"=== {pname} expected={expected} S={S} ax_row={ax_row} se_row={se_row} ===")
        print(f"    ax_rows={ax_rows} se_rows={se_rows}")
        # probe at block 22 (just before install at 23) and 23
        for blk in (8, 22, 23):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            for label, row_idx in (("AX", ax_row), ("SE", se_row)):
                if row_idx is None:
                    continue
                row = resid[row_idx]
                print(f"  blk{blk:2d} {label}row(pos={row_idx}): "
                      f"ALU_LO={hot(band(row,dp,'ALU_LO'))} "
                      f"ALU_HI={hot(band(row,dp,'ALU_HI'))} "
                      f"CARRY_LO={hot(band(row,dp,'AX_CARRY_LO'))} "
                      f"CARRY_HI={hot(band(row,dp,'AX_CARRY_HI'))}")
        print()


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
