#!/usr/bin/env python3
"""Campaign-config CMP operand survival probe (ALU vs SE_ALU), spec_k=0.

LIGHT version: ONE _final_context per program, ONE forward per block,
reads all bands from the same hidden state. Confirms whether operand A in
ALU_LO/HI is crushed all-negative by the L10 ALU-clear at the cmp AX row
while the SE_ALU mirror survives.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


PROGS = {
    "gt_5_52":   _mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 52),
                      Opcode.GT, Opcode.EXIT]),
    "gt_57_29":  _mk([(Opcode.IMM, 57), Opcode.PSH, (Opcode.IMM, 29),
                      Opcode.GT, Opcode.EXIT]),
    "eq_7_45":   _mk([(Opcode.IMM, 7),  Opcode.PSH, (Opcode.IMM, 45),
                      Opcode.EQ, Opcode.EXIT]),
}


def fmt(row, base, width=16, thr=0.5):
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in enumerate(vals)
                           if abs(v) > thr) + "]"


def main(blocks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    has_se = "SE_ALU_LO" in dp

    for pname, bc in PROGS.items():
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        ax_row = ax_rows[-1]
        print(f"=== {pname} got={got} ax_row={ax_row} has_se={has_se} ===",
              flush=True)
        for b in blocks:
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=b)[0]
            row = resid[ax_row]
            cmp = [float(row[dp["CMP"] + i].item()) for i in range(4)]
            line = (f"  blk{b:2d} CMP=[{cmp[0]:.1f},{cmp[1]:.1f},{cmp[2]:.1f},"
                    f"{cmp[3]:.1f}] ALU_LO={fmt(row, dp['ALU_LO'])} "
                    f"ALU_HI={fmt(row, dp['ALU_HI'])} "
                    f"CARRY_LO={fmt(row, dp['AX_CARRY_LO'])} "
                    f"CARRY_HI={fmt(row, dp['AX_CARRY_HI'])}")
            if has_se:
                line += (f" SE_ALU_LO={fmt(row, dp['SE_ALU_LO'])} "
                         f"SE_ALU_HI={fmt(row, dp['SE_ALU_HI'])}")
            print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [11, 12, 13, 14, 15, 22]
    main(blks)
