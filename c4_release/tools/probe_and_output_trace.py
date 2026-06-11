#!/usr/bin/env python3
"""Trace AND OUTPUT_LO/HI raw values at the AX row block-by-block.

spec_k=0, hook-free. Shows whether the MARK_AX bitwise post_op (L10,
lookup-mode ``bitwise_rules``) writes the answer cells, where the
uniform floor + cell-0 spike appear, and where the final amplify flips
the argmax.
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
    "or_basic":  (_mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x30), Opcode.OR, Opcode.EXIT]), 0x3F),
    "and_clean": (_mk([(Opcode.IMM, 0x70), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]), 0x20),
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
        lo = expected & 0xF
        hi = (expected >> 4) & 0xF
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
        print(f"=== {pname} expected={hex(expected)} (lo={lo} hi={hi}) "
              f"got={got} ax_row={ax_row} ===")
        for blk in range(8, nblocks):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            row = resid[ax_row]
            olo = band(row, dp, "OUTPUT_LO")
            ohi = band(row, dp, "OUTPUT_HI")
            othis = band(row, dp, "OUTPUT_HI_THIS_STEP")
            a_lo = olo[argmax(olo)]
            a_hi = ohi[argmax(ohi)]
            print(f"  blk{blk:2d}: "
                  f"OLO[ans{lo}]={olo[lo]:+.2f} OLO[0]={olo[0]:+.2f} "
                  f"OLOamax={argmax(olo)}({a_lo:+.2f}) | "
                  f"OHI[ans{hi}]={ohi[hi]:+.2f} OHI[0]={ohi[0]:+.2f} "
                  f"OHIamax={argmax(ohi)}({a_hi:+.2f})"
                  + (f" | THIS[ans{lo}]={othis[lo]:+.2f}" if othis else ""))
        print()


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or list(PROGRAMS.keys())
    main(sel)
