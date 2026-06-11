#!/usr/bin/env python3
"""Pin WHY operand-A high nibbles mis-gather at the block-8 AX row.

For each test value pushed as operand A, dump:
  - the head-0 attended STACK0 row (argmax of the head-0 attention weights
    at the AX row), reconstructed from K/Q at spec_k=0 (hook-free)
  - CLEAN_EMBED_LO/HI at every STACK0-tagged row (is the token's nibble
    one-hot correct upstream?)
  - the resulting ALU_LO/HI at the AX row (the gather output)

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_operand_highnibble.py
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


# operand A = first value (pushed), operand B = 0x2A (clean), op = AND
def prog(a):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT])


VALUES = [0xFF, 0xF0, 0x0F, 0x2A, 0x08, 0x80, 0x88]


def band(row, dp, name, width=16, thr=0.3):
    base = dp.get(name)
    if base is None:
        return None
    cells = [round(float(row[base + i].item()), 2) for i in range(width)]
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]
    stack0_base = dp.get("MARK_STACK0")
    s0byte0 = dp.get("STACK0_BYTE0")

    for a in VALUES:
        bc = prog(a)
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

        # Residual just before block 8 (after block 7 -> stop_after_block=7)
        with torch.no_grad():
            r7 = model.forward(toks, stop_after_block=7)[0]
        # Residual after block 8 (the gather output)
        with torch.no_grad():
            r8 = model.forward(toks, stop_after_block=8)[0]

        # find STACK0-tagged rows (operand value carriers) at input to blk8
        s0_rows = []
        if stack0_base is not None:
            s0_rows = [r for r in range(S) if r7[r, stack0_base].abs().item() > 0.3]
        # also rows where STACK0_BYTE0 is hot (the K-selection target)
        s0b0_rows = []
        if s0byte0 is not None:
            s0b0_rows = [r for r in range(S) if abs(r7[r, s0byte0].item()) > 0.3]

        print(f"=== A={hex(a)} S={S} ax_row={ax_row} ===")
        print(f"    STACK0-tagged rows: {s0_rows}")
        print(f"    STACK0_BYTE0-hot rows: {[(r, round(float(r7[r, s0byte0].item()),2)) for r in s0b0_rows]}")
        # CLEAN_EMBED at each STACK0_BYTE0-hot row (pre-blk8 residual)
        for r in s0b0_rows:
            print(f"      row{r}: CLEAN_EMBED_LO={band(r7[r],dp,'CLEAN_EMBED_LO')} "
                  f"CLEAN_EMBED_HI={band(r7[r],dp,'CLEAN_EMBED_HI')}")
        # Output: ALU at AX row after blk8
        print(f"    blk8 AXrow(pos={ax_row}): "
              f"ALU_LO={band(r8[ax_row],dp,'ALU_LO')} "
              f"ALU_HI={band(r8[ax_row],dp,'ALU_HI')}")
        # also: what was CLEAN_EMBED at the AX row itself pre-blk8?
        print(f"    AXrow pre-blk8 CLEAN_EMBED_LO={band(r7[ax_row],dp,'CLEAN_EMBED_LO')} "
              f"CLEAN_EMBED_HI={band(r7[ax_row],dp,'CLEAN_EMBED_HI')}")
        print()


if __name__ == "__main__":
    main()
