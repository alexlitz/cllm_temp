#!/usr/bin/env python3
"""Trace the 16-bit OR/XOR/AND multi-byte ALU at every MARK_AX row.

spec_k=0, hook-free. For a 16-bit bitwise op the operands span 2 bytes.
This dumps, for EACH MARK_AX row in the replay, the operand bands
(ALU_LO/HI = operand A nibbles, AX_CARRY_LO/HI = operand B nibbles),
the BYTE_INDEX one-hot at that row, and OUTPUT_LO/HI -- so we can see
which byte each AX row processes and whether byte-1 carries the same
{0,8,15} operand artifacts byte-0 had.
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
    "or_16bit":  (_mk([(Opcode.IMM, 0x0F00), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.OR, Opcode.EXIT]), 0x0FFF),
    "and_16bit": (_mk([(Opcode.IMM, 0x0FFF), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.AND, Opcode.EXIT]), 0x00FF),
    "xor_16bit": (_mk([(Opcode.IMM, 0x0F0F), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.XOR, Opcode.EXIT]), 0x0FF0),
    "or_basic":  (_mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x30), Opcode.OR, Opcode.EXIT]), 0x3F),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def argmax(cells):
    if not cells:
        return None
    return max(range(len(cells)), key=lambda i: cells[i])


def hot(cells, thr=0.2):
    if cells is None:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def main(selected, blk_arg):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    nblocks = len(model.blocks)

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        print(f"=== {pname} expected={hex(expected)} got={hex(got) if isinstance(got,int) else got} "
              f"ax_rows={ax_rows} ===")
        for blk in blk_arg:
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} --")
            for r in ax_rows:
                row = resid[r]
                bi = [round(float(row[dp[f'BYTE_INDEX_{k}']].item()), 2) for k in range(4)]
                alo = band(row, dp, "ALU_LO")
                ahi = band(row, dp, "ALU_HI")
                clo = band(row, dp, "AX_CARRY_LO")
                chi = band(row, dp, "AX_CARRY_HI")
                olo = band(row, dp, "OUTPUT_LO")
                ohi = band(row, dp, "OUTPUT_HI")
                print(f"  row{r:3d} BI={bi} | A_LO={hot(alo)} A_HI={hot(ahi)}")
                print(f"          | B_LO={hot(clo)} B_HI={hot(chi)}")
                print(f"          | O_LO={hot(olo)} O_HI={hot(ohi)}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")]
    sel = [a for a in args if not a.startswith("b=")] or list(PROGRAMS.keys())
    if not blks:
        blks = [11, 12]
    main(sel, blks)
