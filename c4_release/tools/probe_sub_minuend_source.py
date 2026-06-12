#!/usr/bin/env python3
"""Probe the SUB byte-1/2/3 cascade input: OUTPUT vs STACK0_BYTE_VAL_h.

For sub_16bit (0x100-1) vs sub_borrow (0-1) at the AX byte-1/2/3 EMIT rows,
across the L14 cascade blocks (15-25), dump:
  - OUTPUT_LO/HI decoded byte (the current cascade minuend source),
  - STACK0_BYTE_VAL_1/2/3 LO/HI decoded byte (the CORRECT minuend source),
  - CARRY band + TEMP+8/9 (ADD/SUB relay).

Goal: confirm whether STACK0_BYTE_VAL_h is PRESENT at the cascade input row.
If it is 0x00 at the SUB emit row, a same-row source swap is impossible and a
relay is required (informs the Phase 2 approach).
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
from neural_vm.batched_pure_neural import Token
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
    "sub_basic":  (_mk([(Opcode.IMM, 50), Opcode.PSH, (Opcode.IMM, 8), Opcode.SUB, Opcode.EXIT]), 42),
    "sub_16bit":  (_mk([(Opcode.IMM, 0x100), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFF),
    "sub_borrow": (_mk([(Opcode.IMM, 0), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFFFFFFFF),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def nibble(cells):
    if cells is None:
        return None
    mx = max(range(len(cells)), key=lambda i: cells[i])
    return mx if cells[mx] > 0.3 else None


def decode_byte(lo_cells, hi_cells):
    nl, nh = nibble(lo_cells), nibble(hi_cells)
    if nl is None and nh is None:
        return None
    return ((nh or 0) << 4) | (nl or 0)


def scalar(row, dp, name):
    base = dp.get(name)
    return round(float(row[base].item()), 2) if base is not None else None


def carryband(row, dp, width=4):
    base = dp.get("CARRY")
    if base is None:
        return None
    return [round(float(row[base + i].item()), 2) for i in range(width)]


def main(selected, blk_arg):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        ax_idx = None
        for i in range(S - 1, -1, -1):
            if ctx[i] == Token.REG_AX:
                ax_idx = i
                break
        b0, b1, b2, b3 = ax_idx + 1, ax_idx + 2, ax_idx + 3, ax_idx + 4
        print(f"=== {pname} expected={expected:#010x} got={got:#010x} S={S} "
              f"REG_AX@{ax_idx} b1@{b1} b2@{b2} b3@{b3} ===")
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        # Rows that PREDICT bytes 1..3 are b1-1..b3-1.
        watch = {b1 - 1: "->b1", b2 - 1: "->b2", b3 - 1: "->b3"}
        for blk in blk_arg:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (logical L{lg}) --")
            for r, which in watch.items():
                row = resid[r]
                ob = decode_byte(band(row, dp, "OUTPUT_LO"),
                                 band(row, dp, "OUTPUT_HI_THIS_STEP")
                                 or band(row, dp, "OUTPUT_HI"))
                sv1 = decode_byte(band(row, dp, "STACK0_BYTE_VAL_1_LO"),
                                  band(row, dp, "STACK0_BYTE_VAL_1_HI"))
                sv2 = decode_byte(band(row, dp, "STACK0_BYTE_VAL_2_LO"),
                                  band(row, dp, "STACK0_BYTE_VAL_2_HI"))
                sv3 = decode_byte(band(row, dp, "STACK0_BYTE_VAL_3_LO"),
                                  band(row, dp, "STACK0_BYTE_VAL_3_HI"))
                fmt = lambda v: f"{v:#04x}" if v is not None else "None"
                print(f"  row{r:3d}{which} O={fmt(ob)} SV1={fmt(sv1)} "
                      f"SV2={fmt(sv2)} SV3={fmt(sv3)} "
                      f"t8={scalar(row,dp,'TEMP+8')} t9={scalar(row,dp,'TEMP+9')} "
                      f"CARRY={carryband(row,dp)}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")]
    sel = [a for a in args if not a.startswith("b=")] or ["sub_16bit", "sub_borrow"]
    if not blks:
        blks = [14, 15, 16, 17, 18, 19, 20]
    main(sel, blks)
