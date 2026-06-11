#!/usr/bin/env python3
"""Decode-row probe for DIV/MOD (Wall-4 check). Efficient single-context replay.

Traces OUTPUT_LO/OUTPUT_HI at BOTH the binop MARK_AX row and the
MARK_SE_ONLY row across blocks 8..36 for test_div_basic (84/2=42) and
test_mod_basic (43%10=3). Also dumps the operand bands (ALU_LO/ALU_HI,
AX_CARRY_LO/AX_CARRY_HI) at the AX row so we can confirm Wall-1
cleanliness of operands reaching the install FFN.

The question: is the DIV result decoded from the MARK_AX row (the
pre-Wave-B working path the wide_div_rules install writes to) or the
SE row (the CMP/ALU Wall-4 pattern)?

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_div_decode_row.py
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

BANDS = ["OUTPUT_LO", "OUTPUT_HI", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI"]


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def hot(cells, thr=0.3):
    if cells is None:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def argmax(cells):
    if not cells:
        return None
    return max(range(len(cells)), key=lambda i: cells[i])


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
        se_row = se_rows[-1] if se_rows else None
        ax_row = max((r for r in ax_rows if (se_row is None or r < se_row)),
                     default=(ax_rows[-1] if ax_rows else None))
        print(f"=== {pname} expected={expected} got={got} S={S} "
              f"ax_row={ax_row} se_row={se_row} nblocks={nblocks} ===")

        # one forward per block; read both rows + operand bands at once
        for blk in range(8, nblocks):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            for label, row_idx in (("AX", ax_row), ("SE", se_row)):
                if row_idx is None:
                    continue
                row = resid[row_idx]
                olo = band(row, dp, "OUTPUT_LO")
                ohi = band(row, dp, "OUTPUT_HI")
                alo = band(row, dp, "ALU_LO")
                ahi = band(row, dp, "ALU_HI")
                clo = band(row, dp, "AX_CARRY_LO")
                chi = band(row, dp, "AX_CARRY_HI")
                msgs = []
                if hot(olo) or hot(ohi):
                    msgs.append(f"OLO_amax={argmax(olo)} OLO={hot(olo)} "
                                f"OHI_amax={argmax(ohi)} OHI={hot(ohi)}")
                if label == "AX" and (hot(alo) or hot(ahi) or hot(clo) or hot(chi)):
                    msgs.append(f"ALU_LO={hot(alo)} ALU_HI={hot(ahi)} "
                                f"CARRY_LO={hot(clo)} CARRY_HI={hot(chi)}")
                if msgs:
                    print(f"  blk{blk:2d} {label}: " + " | ".join(msgs))
        print()


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
