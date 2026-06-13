#!/usr/bin/env python3
"""Dump the width=2 MUL operand magnitudes at the MARK_AX MUL row.

The width=2 wide_mul 5-way AND reads, at the MARK_AX MUL row:
  - MARK_AX        (marker)
  - ALU_LO+a_lo    (operand A byte, low nibble one-hot)
  - ALU_LO+16+a_hi (= ALU_HI+a_hi; operand A byte, high nibble one-hot)
  - AX_CARRY_LO+b_lo    (operand B byte, low nibble one-hot)
  - AX_CARRY_LO+16+b_hi (= AX_CARRY_HI+b_hi; operand B byte, high nibble)

The input to the wide_mul FFN at logical L11 is the residual at the INPUT
of L11's physical block. We read it via stop_after_block at the block just
before L11's ffn fires. To capture the operand magnitudes the AND actually
sees we dump the residual at the input to the L11 block.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/probe_mul_width2_operands.py
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


def prog(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL, Opcode.EXIT])


def band(row, dp, name, width=16, thr=0.05):
    base = dp.get(name)
    if base is None:
        return None
    cells = [round(float(row[base + i].item()), 3) for i in range(width)]
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]

    # find L11 physical block
    l11_phys = None
    for phys, blk in enumerate(model.blocks):
        lg = getattr(blk, "_logical_layer", phys)
        if lg == 11:
            l11_phys = phys
            break
    print(f"L11 physical block = {l11_phys}")

    CASES = [(6, 7), (100, 5), (255, 255), (16, 16)]
    for a, b in CASES:
        bc = prog(a, b)
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        se_row = se_rows[-1] if se_rows else None
        ax_row = max((r for r in ax_rows if (se_row is None or r < se_row)),
                     default=(ax_rows[-1] if ax_rows else None))

        # residual at the INPUT to the L11 block = output of block l11_phys-1
        with torch.no_grad():
            r_in = model.forward(toks, stop_after_block=l11_phys - 1)[0]

        print(f"=== A={a}(0x{a:02x}) B={b}(0x{b:02x}) S={S} ax_row={ax_row} ===")
        print(f"    MARK_AX={round(float(r_in[ax_row, ax_base].item()),3)}")
        print(f"    ALU_LO     ={band(r_in[ax_row],dp,'ALU_LO')}")
        print(f"    ALU_HI     ={band(r_in[ax_row],dp,'ALU_HI')}")
        print(f"    AX_CARRY_LO={band(r_in[ax_row],dp,'AX_CARRY_LO')}")
        print(f"    AX_CARRY_HI={band(r_in[ax_row],dp,'AX_CARRY_HI')}")
        print(f"    OP_MUL={round(float(r_in[ax_row, dp['OP_MUL']].item()),3)}")
        print()


if __name__ == "__main__":
    main()
