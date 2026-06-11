#!/usr/bin/env python3
"""Spec_k=0 per-cell readout of block-8 operand-gather ALU_LO/HI one-hots.

Reads, at the AX marker row of the binop step (after physical block 8 =
operand-gather), the per-cell residual for ALU_LO, ALU_HI, AX_CARRY_LO,
AX_CARRY_HI (AX_CARRY is the CLEAN reference one-hot). Also dumps the
attended STACK0_BYTE0 source row's CLEAN_EMBED_LO/HI so we can see whether
the @0 magnitude artifact originates in the SOURCE residual or in the
head V/O projection.

Hook-free, spec_k=0, via tools.probe_groundtruth.residual_at.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_block8_operand_onehot.py [block]
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


# value 42 (eq_true operand), value 20 (the "nothing at true nibble" case).
PROGRAMS = {
    "eq_true_42": _mk([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42),
                       Opcode.EQ, Opcode.EXIT]),
    "eq_false_20": _mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20),
                        Opcode.EQ, Opcode.EXIT]),
    "and_ff_2a": _mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 0x2A),
                      Opcode.AND, Opcode.EXIT]),
}

BANDS = ["ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
         "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"]


def cells(probe, bc, block, pos, dp, name, width=16):
    base = dp[name]
    dim_names = {f"{name}+{i}": base + i for i in range(width)}
    vals = probe.residual_at(bc, block_idx=block, position=pos,
                             dim_names=dim_names)
    return [vals[f"{name}+{i}"] for i in range(width)]


def fmt(vec, thr=0.3):
    return "[" + ", ".join(f"{v:.2f}@{i}" for i, v in enumerate(vec)
                           if abs(v) > thr) + "]"


def main(block):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    s0_base = dp["STACK0_BYTE0"]

    for pname, bc in PROGRAMS.items():
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        # binop AX row = last AX marker before EXIT.
        ax_row = ax_rows[-1]
        # STACK0 source row (attended K): last STACK0_BYTE0 position before ax_row.
        s0_rows = [r for r in range(S)
                   if emb[r, s0_base].abs().item() > 0.5 and r < ax_row]
        s0_row = s0_rows[-1] if s0_rows else None

        print(f"=== {pname} got={got} S={S} ax_row={ax_row} "
              f"s0_row={s0_row} block={block} ===")
        for nm in BANDS:
            vec = cells(probe, bc, block, ax_row, dp, nm)
            print(f"  AX[{nm:14s}] = {fmt(vec)}")
        if s0_row is not None:
            for nm in ("CLEAN_EMBED_LO", "CLEAN_EMBED_HI"):
                vec = cells(probe, bc, block, s0_row, dp, nm)
                print(f"  S0[{nm:14s}] = {fmt(vec)}")
        print()


if __name__ == "__main__":
    blk = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(blk)
