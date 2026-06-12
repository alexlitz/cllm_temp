#!/usr/bin/env python3
"""Wall-4 full diagnosis: trace CMP / OUTPUT_LO / OP_* / operands at BOTH the
binop AX row and the SE row across all blocks, plus the raw operand encoding
(ALU_LO/HI, AX_CARRY_LO/HI) at the AX row that feeds the L9 cmp cascade.

spec_k=0, hook-free. Cached build.
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
    "eq_false": (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 7),  Opcode.EQ, Opcode.EXIT]), 0),
    "lt_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LT, Opcode.EXIT]), 1),
    "le_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LE, Opcode.EXIT]), 1),
}


def band(probe, bc, block, pos, dp, base_name, width):
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
        print(f"\n=== {pname} expected={expected} got={got} S={S} "
              f"ax_row={ax_row} se_row={se_row} ===")
        # operand encoding at the AX row (block 10 = logical L9 input)
        for blk in (10, 11):
            alu_lo = band(probe, bc, blk, ax_row, dp, "ALU_LO", 16)
            alu_hi = band(probe, bc, blk, ax_row, dp, "ALU_HI", 16)
            axc_lo = band(probe, bc, blk, ax_row, dp, "AX_CARRY_LO", 16)
            axc_hi = band(probe, bc, blk, ax_row, dp, "AX_CARRY_HI", 16)
            print(f"  blk{blk} AXrow operands: ALU_LO={hot(alu_lo)} ALU_HI={hot(alu_hi)} "
                  f"AXC_LO={hot(axc_lo)} AXC_HI={hot(axc_hi)}")
            se_alu_lo = band(probe, bc, blk, se_row, dp, "SE_ALU_LO", 16)
            se_alu_hi = band(probe, bc, blk, se_row, dp, "SE_ALU_HI", 16)
            se_axc_lo = band(probe, bc, blk, se_row, dp, "SE_AX_CARRY_LO", 16)
            se_axc_hi = band(probe, bc, blk, se_row, dp, "SE_AX_CARRY_HI", 16)
            print(f"  blk{blk} SErow SE_operands: SE_ALU_LO={hot(se_alu_lo)} "
                  f"SE_ALU_HI={hot(se_alu_hi)} SE_AXC_LO={hot(se_axc_lo)} SE_AXC_HI={hot(se_axc_hi)}")
        # CMP + OUTPUT + OP at both rows across blocks
        for blk in range(10, nblocks):
            ax_cmp = band(probe, bc, blk, ax_row, dp, "CMP", 4)
            ax_olo = band(probe, bc, blk, ax_row, dp, "OUTPUT_LO", 16)
            se_cmp = band(probe, bc, blk, se_row, dp, "CMP", 4)
            se_olo = band(probe, bc, blk, se_row, dp, "OUTPUT_LO", 16)
            ax_cmp_h = hot(ax_cmp); se_cmp_h = hot(se_cmp)
            ax_olo_h = hot(ax_olo); se_olo_h = hot(se_olo)
            if ax_cmp_h or se_cmp_h or ax_olo_h or se_olo_h:
                ax_am = max(range(16), key=lambda i: ax_olo[i]) if ax_olo else None
                se_am = max(range(16), key=lambda i: se_olo[i]) if se_olo else None
                print(f"  blk{blk:2d} AX[cmp={ax_cmp_h} olo_am={ax_am} olo={ax_olo_h[:4]}] "
                      f"SE[cmp={se_cmp_h} olo_am={se_am} olo={se_olo_h[:4]}]")


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
