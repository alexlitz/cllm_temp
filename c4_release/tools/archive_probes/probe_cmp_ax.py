#!/usr/bin/env python3
"""AX-row CMP + raw operand probe across blocks, spec_k=0.

Tracks CMP[hi_lt,hi_eq,lo_eq,lo_lt] and raw ALU_LO/HI + AX_CARRY_LO/HI at
the binop AX (MARK_AX) row across a block range, for lt_true (now failing)
vs gt_true (passing). Finds where the comparison result is actually
decided and where clean operands changed it.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
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
    "lt_true_10_20": _mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20),
                          Opcode.LT, Opcode.EXIT]),
    "gt_true_20_10": _mk([(Opcode.IMM, 20), Opcode.PSH, (Opcode.IMM, 10),
                          Opcode.GT, Opcode.EXIT]),
}


def cells(probe, bc, block, pos, dp, name, width):
    base = dp[name]
    dn = {f"{name}+{i}": base + i for i in range(width)}
    vals = probe.residual_at(bc, block_idx=block, position=pos, dim_names=dn)
    return [vals[f"{name}+{i}"] for i in range(width)]


def fmt(vec, thr=0.5):
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in enumerate(vec)
                           if abs(v) > thr) + "]"


def main(blocks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]

    for pname, bc in PROGS.items():
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        ax_row = ax_rows[-1]
        print(f"=== {pname} got={got} ax_row={ax_row} ===")
        for b in blocks:
            cmp = cells(probe, bc, b, ax_row, dp, "CMP", 4)
            alo = cells(probe, bc, b, ax_row, dp, "ALU_LO", 16)
            ahi = cells(probe, bc, b, ax_row, dp, "ALU_HI", 16)
            clo = cells(probe, bc, b, ax_row, dp, "AX_CARRY_LO", 16)
            chi = cells(probe, bc, b, ax_row, dp, "AX_CARRY_HI", 16)
            print(f"  blk{b:2d} CMP=[{cmp[0]:.1f},{cmp[1]:.1f},{cmp[2]:.1f},"
                  f"{cmp[3]:.1f}] ALU_LO={fmt(alo)} ALU_HI={fmt(ahi)} "
                  f"CARRY_LO={fmt(clo)} CARRY_HI={fmt(chi)}")
        print()


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [8, 9, 10, 11, 12, 13]
    main(blks)
