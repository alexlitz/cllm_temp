#!/usr/bin/env python3
"""Trace ALU_LO/HI (operand A) + AX_CARRY_LO/HI (operand B) at the bitwise
MARK_AX compute row block-by-block, to locate the crush and the lookup read.

Also reports whether the BitwiseOperandSeRecoverFFN 'recover' branch would
FIRE (crush detected) at the lookup block, and what SE_ALU carries.

campaign config (C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1) via env.
spec_k=0, hook-free, tooling-only.
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
    "or_16bit":  (_mk([(Opcode.IMM, 0x0F00), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.OR,  Opcode.EXIT]), 0x0FFF),
    "and_16bit": (_mk([(Opcode.IMM, 0x0FFF), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.AND, Opcode.EXIT]), 0x00FF),
    "xor_16bit": (_mk([(Opcode.IMM, 0x0F0F), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.XOR, Opcode.EXIT]), 0x0FF0),
    "or_basic":  (_mk([(Opcode.IMM, 0x05), Opcode.PSH, (Opcode.IMM, 0x02), Opcode.OR,  Opcode.EXIT]), 0x07),
    "xor_basic": (_mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x09), Opcode.XOR, Opcode.EXIT]), 0x06),
    "and_basic": (_mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x0C), Opcode.AND, Opcode.EXIT]), 0x0C),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [round(float(row[base + i].item()), 2) for i in range(width)]


def hot(cells, thr=0.4):
    if not cells:
        return []
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    nblocks = len(model.blocks)
    print(f"nblocks={nblocks}  NO_STACK0_EMIT={os.environ.get('C4_NO_STACK0_EMIT')} "
          f"OPERAND_FROM_MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP')} "
          f"BITWISE_SE_RECOVER={os.environ.get('C4_BITWISE_BYTE0_SE_RECOVER','1')} "
          f"CLEAN_OPERAND_BITWISE={os.environ.get('C4_CLEAN_OPERAND_BITWISE','0')}")

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        ax_row = ax_rows[-1] if ax_rows else None
        print(f"\n=== {pname} expected={hex(expected)} got={got}={hex(got) if isinstance(got,int) else got} "
              f"AXrow={ax_row} PASS={got==expected} ===")
        for blk in range(8, nblocks):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            axr = resid[ax_row]
            alu_lo = hot(band(axr, dp, 'ALU_LO'))
            alu_hi = hot(band(axr, dp, 'ALU_HI'))
            se_lo = hot(band(axr, dp, 'SE_ALU_LO'))
            se_hi = hot(band(axr, dp, 'SE_ALU_HI'))
            out_lo = hot(band(axr, dp, 'OUTPUT_LO'))
            out_hi = hot(band(axr, dp, 'OUTPUT_HI'))
            # only print blocks around L8..lookup + when something changes
            print(f"  blk{blk:2d}: ALU_LO={alu_lo} ALU_HI={alu_hi} || "
                  f"SE_ALU_LO={se_lo} SE_ALU_HI={se_hi} || OUT_LO={out_lo} OUT_HI={out_hi}")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or ["or_16bit", "xor_16bit"]
    main(sel)
