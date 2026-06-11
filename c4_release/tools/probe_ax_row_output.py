#!/usr/bin/env python3
"""Trace OUTPUT_LO at the binop step's MARK_AX row (the real decode site).

The exit code is the prev-step OUTPUT_LO relayed into AX_FULL by L3 head 5,
read at the *AX marker row*. So the comparison/ALU result that reaches the
EXIT emission is the binop step's MARK_AX-row OUTPUT_LO -- NOT the SE row the
cascade probe traced. This probe reads OUTPUT_LO + CMP at the binop AX row
across blocks 8..36 so we can see WHERE the live result is written and why
eq_true lands on 0 while lt_true lands on 1.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_ax_row_output.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    "eq_true":  (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 5),  Opcode.EQ, Opcode.EXIT]), 1),
    "lt_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LT, Opcode.EXIT]), 1),
    "and_basic":(_mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]), 0x2A),
    "mul_basic":(_mk([(Opcode.IMM, 6),  Opcode.PSH, (Opcode.IMM, 7),  Opcode.MUL, Opcode.EXIT]), 42),
}


def onehot(probe, bc, block, pos, dp, base_name, width):
    base = dp.get(base_name)
    if base is None:
        return None
    dim_names = {f"{base_name}+{i}": base + i for i in range(width)}
    vals = probe.residual_at(bc, block_idx=block, position=pos, dim_names=dim_names)
    return [vals[f"{base_name}+{i}"] for i in range(width)]


def hot(cells, thr=0.3):
    if cells is None:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    se_base = dp["MARK_SE_ONLY"]
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
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_row = se_rows[-1]
        ax_row = max((r for r in ax_rows if r < se_row), default=ax_rows[-1])
        print(f"=== {pname} expected={expected} got={got} S={S} "
              f"ax_row={ax_row} se_row={se_row} nblocks={nblocks} ===")
        for blk in range(8, nblocks):
            olo = onehot(probe, bc, blk, ax_row, dp, "OUTPUT_LO", 16)
            ohi = onehot(probe, bc, blk, ax_row, dp, "OUTPUT_HI", 16)
            cmp = onehot(probe, bc, blk, ax_row, dp, "CMP", 4)
            olo_h = hot(olo)
            ohi_h = hot(ohi)
            cmp_h = hot(cmp)
            if olo_h or ohi_h or cmp_h:
                amax = max(range(16), key=lambda i: olo[i]) if olo else None
                print(f"  blk{blk:2d} AXrow: OLO_argmax={amax} OLO_hot={olo_h} "
                      f"OHI_hot={ohi_h} CMP={cmp_h}")
        print()


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
