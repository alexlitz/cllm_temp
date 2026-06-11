#!/usr/bin/env python3
"""Spec_k=0 probe of the L9 SE CMP cascade per-cell firing.

Reads, at the SE row (MARK_SE_ONLY) of the binop step:
  * SE_ALU_LO/HI and SE_AX_CARRY_LO/HI per-nibble one-hots (which index)
  * the 4 CMP cascade cells (CMP+0=hi_lt, +1=hi_eq, +2=lo_eq, +3=lo_lt)
  * the same SE_-tagged cascade if present

so we can see *exactly* which cascade rule fires wrongly under the
relayed 2-cell encoding (the Wall-3 question).

Usage:
    CUDA_VISIBLE_DEVICES="" L9_BLOCK=11 python tools/probe_l9_cmp_cascade.py
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
    "eq_false": (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.EQ, Opcode.EXIT]), 0),
    "lt_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LT, Opcode.EXIT]), 1),
    "le_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LE, Opcode.EXIT]), 1),
    "gt_true":  (_mk([(Opcode.IMM, 20), Opcode.PSH, (Opcode.IMM, 10), Opcode.GT, Opcode.EXIT]), 1),
    "ge_true":  (_mk([(Opcode.IMM, 20), Opcode.PSH, (Opcode.IMM, 10), Opcode.GE, Opcode.EXIT]), 1),
    "ne_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.NE, Opcode.EXIT]), 1),
}

L9_BLOCK = int(os.environ.get("L9_BLOCK", "11"))


def onehot(probe, bc, block, pos, dp, base_name, width):
    base = dp.get(base_name)
    if base is None:
        return None
    dim_names = {f"{base_name}+{i}": base + i for i in range(width)}
    vals = probe.residual_at(bc, block_idx=block, position=pos, dim_names=dim_names)
    cells = [vals[f"{base_name}+{i}"] for i in range(width)]
    return cells


def hot_idx(cells, thr=2.0):
    if cells is None:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def hot_idx_lo(cells, thr=0.4):
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
        if not se_rows:
            print(f"=== {pname}: NO SE rows; got={got} ===")
            continue
        se_row = se_rows[-1]

        cmp_cells = onehot(probe, bc, L9_BLOCK, se_row, dp, "CMP", 4)
        se_alu_lo = onehot(probe, bc, L9_BLOCK, se_row, dp, "SE_ALU_LO", 16)
        se_alu_hi = onehot(probe, bc, L9_BLOCK, se_row, dp, "SE_ALU_HI", 16)
        se_ax_lo = onehot(probe, bc, L9_BLOCK, se_row, dp, "SE_AX_CARRY_LO", 16)
        se_ax_hi = onehot(probe, bc, L9_BLOCK, se_row, dp, "SE_AX_CARRY_HI", 16)

        print(f"=== {pname} expected={expected} got={got} se_row={se_row} ===")
        print(f"  SE_ALU_LO hot(>2):     {hot_idx(se_alu_lo)}")
        print(f"  SE_AX_CARRY_LO hot(>.4):{hot_idx_lo(se_ax_lo)}")
        print(f"  SE_ALU_HI hot(>2):     {hot_idx(se_alu_hi)}")
        print(f"  SE_AX_CARRY_HI hot(>.4):{hot_idx_lo(se_ax_hi)}")
        cmp_names = ["CMP+0(hi_lt)", "CMP+1(hi_eq)", "CMP+2(lo_eq)", "CMP+3(lo_lt)"]
        if cmp_cells:
            print("  CMP cascade: " + " ".join(
                f"{n}={v:.2f}" for n, v in zip(cmp_names, cmp_cells)))
        # Trace OUTPUT_LO + CMP across blocks 11..36 at the SE row to see
        # where the comparison result is set / overwritten.
        if os.environ.get("TRACE_OUTPUT"):
            blkset = os.environ.get("TRACE_BLOCKS")
            blocks = ([int(x) for x in blkset.split(",")] if blkset
                      else range(11, len(model.blocks)))
            for blk in blocks:
                olo = onehot(probe, bc, blk, se_row, dp, "OUTPUT_LO", 16)
                cmpc = onehot(probe, bc, blk, se_row, dp, "CMP", 4)
                olo_hot = [(i, round(v, 2)) for i, v in enumerate(olo or []) if abs(v) > 0.3]
                cmp_hot = [(i, round(v, 2)) for i, v in enumerate(cmpc or []) if abs(v) > 0.3]
                if olo_hot or cmp_hot:
                    print(f"    blk{blk}: OUTPUT_LO={olo_hot} CMP={cmp_hot}")
        print()


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
