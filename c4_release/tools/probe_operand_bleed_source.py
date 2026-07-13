#!/usr/bin/env python3
"""Localize the constant cell-8 (+0.454) / cell-0 (-0.519) ALU_LO bleed.

Prints the block map around L7/L8, and traces ALU_LO at the ax_row at
FINER granularity (which block first introduces cell-8 / cell-0). Also
dumps the CLEAN_EMBED_LO and STACK0_BYTE0 rows at the attended position to
confirm the CamValueBand source cleanliness.
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
    base = dp[name]
    return [round(float(row[base + i].item()), 3) for i in range(width)]


def hot(cells, thr=0.05):
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device

    blmap = probe.block_layer_map()
    print("=== BLOCK MAP (blocks 7..13) ===")
    for i in range(6, 14):
        if i < len(blmap):
            r = blmap[i]
            print(f"  block {i}: layer={r.get('layer')} ffn={r.get('ffn','')[:60]}")

    bc = _mk([(Opcode.IMM, 0x02), Opcode.PSH, (Opcode.IMM, 0x00), Opcode.ADD, Opcode.EXIT])
    ctx = probe._final_context(bc, max_steps=20)
    S = len(ctx)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(toks)[0]
    ax_base = dp["MARK_AX"]
    ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
    ax_row = ax_rows[-1]
    print(f"\n=== add_2_0 ax_row={ax_row} — ALU_LO fine block trace (want cell2=6, dirt cell0/cell8) ===")
    for blk in range(7, 13):
        with torch.no_grad():
            resid = model.forward(toks, stop_after_block=blk)[0]
        row = resid[ax_row]
        print(f"  blk{blk:2d} ALU_LO@ax: {hot(band(row,dp,'ALU_LO'))}  ALU_HI@ax: {hot(band(row,dp,'ALU_HI'))}")


if __name__ == "__main__":
    main()
