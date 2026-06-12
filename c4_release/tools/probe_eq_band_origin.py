#!/usr/bin/env python3
"""Trace the OUTPUT_LO band at the byte0-emit row (reg_ax+0) block-by-block to
find WHICH block introduces the eq_false +238 uniform band on OLO[1..15] and
WHICH block sets eq_true's OLO[1] winner.

spec_k=0, hook-free, CACHED build.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe, Token


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
}


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    olo = dp.get("OUTPUT_LO")
    ohi = dp.get("OUTPUT_HI_THIS_STEP")
    REG_AX = int(Token.REG_AX)
    nblk = len(model.blocks)

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        ax_pos = max(i for i in range(S) if ctx[i] == REG_AX and i + 4 < S)
        emit = ax_pos  # logits[ax_pos] predicts the byte0 value token
        print(f"\n=== {pname} exp={expected} got={got} reg_ax@{ax_pos} emit_row={emit} ===")
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        prev = None
        for blk in range(0, nblk):
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=blk)
            band = [round(float(resid[0, emit, olo + k].item()), 1) for k in range(16)]
            am = max(range(16), key=lambda k: band[k])
            tail = band[1:]
            uniform = (max(tail) - min(tail) < 1.0) and (max(tail) > 5.0)
            sig = (band[0], band[1], am, uniform)
            if sig != prev:
                print(f"  blk{blk:2d} OLO[0]={band[0]:>9} OLO[1]={band[1]:>9} am={am} "
                      f"uniform_tail={uniform} band={band}")
                prev = sig


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
