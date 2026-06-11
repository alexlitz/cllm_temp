#!/usr/bin/env python3
"""Trace CLEAN_EMBED_LO/HI at the operand-A carrier row block-by-block.

Finds the STACK0_BYTE0-hot operand-A carrier row, then dumps its
CLEAN_EMBED_LO/HI after each early block (0..8) to locate exactly which
block corrupts the high nibble (0xFF -> LO8/HI14 instead of LO15/HI15).
Also prints the raw token at that row.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_clean_embed_trace.py [VAL]
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


def band(row, dp, name, width=16, thr=0.3):
    base = dp.get(name)
    if base is None:
        return None
    cells = [round(float(row[base + i].item()), 2) for i in range(width)]
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    a = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0xFF
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    s0byte0 = dp.get("STACK0_BYTE0")

    bc = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT])
    ctx = probe._final_context(bc, max_steps=20)
    S = len(ctx)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)

    # find operand-A carrier: STACK0_BYTE0-hot row in the second-to-last
    # group (the AND step's operand). Use block 7 residual.
    with torch.no_grad():
        r7 = model.forward(toks, stop_after_block=7)[0]
    s0b0_rows = [r for r in range(S) if s0byte0 is not None and abs(r7[r, s0byte0].item()) > 0.3]
    print(f"A={hex(a)} S={S} STACK0_BYTE0-hot rows={s0b0_rows}")
    # token at each carrier
    for r in s0b0_rows:
        print(f"  row{r} token={ctx[r]} (={hex(ctx[r])})")

    # Trace CLEAN_EMBED at each carrier across blocks
    for r in s0b0_rows:
        print(f"\n--- carrier row {r} (token {hex(ctx[r])}) ---")
        # embedding (block -1)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        print(f"  embed : CLEAN_EMBED_LO={band(emb[r],dp,'CLEAN_EMBED_LO')} "
              f"CLEAN_EMBED_HI={band(emb[r],dp,'CLEAN_EMBED_HI')}")
        for blk in range(0, 9):
            with torch.no_grad():
                rb = model.forward(toks, stop_after_block=blk)[0]
            print(f"  blk{blk}: CLEAN_EMBED_LO={band(rb[r],dp,'CLEAN_EMBED_LO')} "
                  f"CLEAN_EMBED_HI={band(rb[r],dp,'CLEAN_EMBED_HI')}")


if __name__ == "__main__":
    main()
