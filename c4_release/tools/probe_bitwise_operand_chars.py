#!/usr/bin/env python3
"""Characterize ALU_LO/HI + AX_CARRY operand bands at the AX row for a
sweep of AND operands, at the block right before the L10 bitwise post_op
(block 11, the L10 main FFN output) and after (block 12).

spec_k=0, hook-free. Goal: design the operand-cleanup stage (which cells
carry the artifact, what magnitude, vs the real answer-nibble cell).
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


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [round(float(row[base + i].item()), 2) for i in range(width)]


def hot(cells, thr=0.2):
    if cells is None:
        return []
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]

    # (A, B) operand pairs for AND
    pairs = [(0xFF, 0x2A), (0x70, 0x2A), (0x0F, 0x30), (0x3C, 0x0F),
             (0x80, 0x80), (0x08, 0x08), (0x12, 0x34)]
    for A, B in pairs:
        bc = _mk([(Opcode.IMM, A), Opcode.PSH, (Opcode.IMM, B), Opcode.AND, Opcode.EXIT])
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
        expect = A & B
        print(f"=== A={hex(A)} B={hex(B)} A&B={hex(expect)} "
              f"(Alo={A&0xF} Ahi={(A>>4)&0xF} Blo={B&0xF} Bhi={(B>>4)&0xF}) "
              f"ax_row={ax_row} ===")
        with torch.no_grad():
            resid = model.forward(toks, stop_after_block=11)[0]
        row = resid[ax_row]
        print(f"   blk11 ALU_LO={hot(band(row,dp,'ALU_LO'))} "
              f"ALU_HI={hot(band(row,dp,'ALU_HI'))}")
        print(f"   blk11 CARRY_LO={hot(band(row,dp,'AX_CARRY_LO'))} "
              f"CARRY_HI={hot(band(row,dp,'AX_CARRY_HI'))}")
    print()


if __name__ == "__main__":
    main()
