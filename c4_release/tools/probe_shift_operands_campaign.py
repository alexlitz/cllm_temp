#!/usr/bin/env python3
"""Block-trace the L13 SHIFT operands/result at the MARK_AX compute row in
the CAMPAIGN config, for both the passing SHL (21<<1=42) and the failing
SHR (84>>1=42) smoke programs.

The L13 shift lookup (l13_ops._layer13_shifts_rules) computes
OUTPUT_LO/HI = shift(ALU, AX_CARRY_LO) at MARK_AX. We dump:
  - ALU_LO/HI       = operand A (value to shift)
  - AX_CARRY_LO/HI  = operand B (shift amount, in LO)
  - SE_ALU_LO/HI    = the L9 step_end mirror of operand A (survives the crush)
  - OUTPUT_LO/HI    = the shift result

Hypothesis (bitwise/cmp byte-0 SE-recover family): under the STACK0
emit-drop the L14 ALU-clear crushes operand A in ALU_LO/HI so the SHR
lookup never fires -> OUTPUT empty -> byte-0 decodes 0x00.

Set C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 for the campaign build.
spec_k=0, hook-free, tooling-only (model byte-identical).
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
    "shl": (_mk([(Opcode.IMM, 21), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHL, Opcode.EXIT]), 42),
    "shr": (_mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHR, Opcode.EXIT]), 42),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def hot(cells, thr=0.4):
    if not cells:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    nblocks = len(model.blocks)
    print(f"nblocks={nblocks}  campaign NO_STACK0_EMIT="
          f"{os.environ.get('C4_NO_STACK0_EMIT')} "
          f"OPERAND_FROM_MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP')}")
    print(f"SE_ALU_LO present={dp.get('SE_ALU_LO') is not None} "
          f"SE_ALU_HI present={dp.get('SE_ALU_HI') is not None}")

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
        print(f"\n=== {pname} expected={expected} got={got} AXrow={ax_row} ===")
        for blk in range(8, nblocks):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            axr = resid[ax_row]
            alu_lo = hot(band(axr, dp, 'ALU_LO'))
            alu_hi = hot(band(axr, dp, 'ALU_HI'))
            sealu_lo = hot(band(axr, dp, 'SE_ALU_LO') or [])
            sealu_hi = hot(band(axr, dp, 'SE_ALU_HI') or [])
            axc_lo = hot(band(axr, dp, 'AX_CARRY_LO'))
            out_lo = hot(band(axr, dp, 'OUTPUT_LO'))
            out_hi = hot(band(axr, dp, 'OUTPUT_HI'))
            print(f"  blk{blk:2d}: ALU[lo={alu_lo} hi={alu_hi}] "
                  f"SE_ALU[lo={sealu_lo} hi={sealu_hi}] "
                  f"AXC_LO={axc_lo} | OUT[lo={out_lo} hi={out_hi}]")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or list(PROGRAMS.keys())
    main(sel)
