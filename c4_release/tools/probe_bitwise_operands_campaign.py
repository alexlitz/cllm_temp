#!/usr/bin/env python3
"""Dump the two bitwise-lookup operands (ALU_LO/HI = operand A,
AX_CARRY_LO/HI = operand B) at the OR/XOR/AND MARK_AX compute row,
block-by-block, in the CAMPAIGN config.

The L10 bitwise lookup (alu_ops._l10_bitwise_lookup_rules) computes
result byte-0 = bitwise(ALU, AX_CARRY) at MARK_AX. Hypothesis: under
the STACK0 emit-drop, AX_CARRY (operand B) is clobbered with operand A
so A OP A is computed -> AND passes coincidentally, OR/XOR fail.

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


# The 3 smoke 16-bit bitwise programs (16-bit IMM PSH + 16-bit IMM op).
PROGRAMS = {
    "or_16bit":  (_mk([(Opcode.IMM, 0x0F00), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.OR,  Opcode.EXIT]), 0x0FFF),
    "and_16bit": (_mk([(Opcode.IMM, 0x0FFF), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.AND, Opcode.EXIT]), 0x00FF),
    "xor_16bit": (_mk([(Opcode.IMM, 0x0F0F), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.XOR, Opcode.EXIT]), 0x0FF0),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def amax(cells):
    if not cells:
        return None
    return max(range(len(cells)), key=lambda i: cells[i])


def hot(cells, thr=0.5):
    """Return list of (idx, val) cells above threshold (the 'lit' nibbles)."""
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
    # Probe blocks: just before and just after the L10 bitwise compute.
    PROBE_BLOCKS = [13, 14, 15, 16, 17, 20]
    print(f"nblocks={nblocks}  campaign NO_STACK0_EMIT="
          f"{os.environ.get('C4_NO_STACK0_EMIT')} "
          f"OPERAND_FROM_MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP')}")

    LAST_BLK = nblocks - 1
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
        se_base = dp.get("MARK_SE_ONLY")
        se_rows = [r for r in range(S)
                   if se_base is not None and emb[r, se_base].abs().item() > 0.5]
        # The SE row for this step: the LAST se row at/after ax_row.
        se_row = next((r for r in se_rows if r >= ax_row), se_rows[-1] if se_rows else None)
        print(f"\n=== {pname} expected={hex(expected)} got={got} "
              f"AXmarker={ax_row} SE_row={se_row} ===")
        # Trace OUTPUT at BOTH the SE (step-end) compute row and the REG_AX
        # marker row (byte-0 predictor). The byte-0 bitwise lookup gates on
        # MARK_SE_ONLY; a relay then copies SE->AX byte0.
        for blk in range(8, nblocks):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            axr = resid[ax_row]
            ser = resid[se_row] if se_row is not None else None
            ax_lo = hot(band(axr, dp, 'OUTPUT_LO'))
            ax_hi = hot(band(axr, dp, 'OUTPUT_HI'))
            se_lo = hot(band(ser, dp, 'OUTPUT_LO')) if ser is not None else []
            se_hi = hot(band(ser, dp, 'OUTPUT_HI')) if ser is not None else []
            se_hit = hot(band(ser, dp, 'OUTPUT_HI_THIS_STEP') or []) if ser is not None else []
            print(f"  blk{blk:2d}: AX[OUT_LO={ax_lo} HI={ax_hi}] | "
                  f"SE[OUT_LO={se_lo} HI={se_hi}"
                  + (f" HIthis={se_hit}" if se_hit else "") + "]")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or list(PROGRAMS.keys())
    main(sel)
