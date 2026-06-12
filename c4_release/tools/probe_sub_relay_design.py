#!/usr/bin/env python3
"""Design probe for the SUB multi-byte minuend relay (Part 1).

For sub_16bit (0x100-1) vs sub_borrow (0-1) vs sub_basic (50-8):
  1. Dump the marker signature of the byte-1/2/3 EMIT rows (the cascade
     input rows) so we can find a Q-gate for a relay head.
  2. Dump STACK0_BYTE_VAL_1/2/3 across ALL rows so we find the populated
     PSH-frame rows (the relay K-target) and confirm 0x01 vs 0x00.

Runs at spec_k=0 (ground-truth, hook-free).
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

MARKERS = ["MARK_AX", "MARK_PC", "MARK_SP", "MARK_STACK0", "MARK_SE_ONLY",
           "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
           "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
           "OP_SUB", "TEMP+8", "TEMP+9", "H1+1"]


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


def decode_byte(lo, hi):
    nl, nh = nibble(lo), nibble(hi)
    if nl is None and nh is None:
        return None
    return ((nh or 0) << 4) | (nl or 0)


def scal(row, dp, name):
    base = dp.get(name)
    return round(float(row[base].item()), 2) if base is not None else None


def main(selected, blk_arg, mode):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        ax_idx = None
        for i in range(S - 1, -1, -1):
            if ctx[i] == Token.REG_AX:
                ax_idx = i
                break
        b1, b2, b3 = ax_idx + 2, ax_idx + 3, ax_idx + 4
        emit = {b1 - 1: "->b1", b2 - 1: "->b2", b3 - 1: "->b3"}
        print(f"=== {pname} exp={expected:#x} S={S} REG_AX@{ax_idx} "
              f"emit_rows={list(emit)} ===")
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        for blk in blk_arg:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (logical L{lg}) --")
            if mode == "emit":
                for r, which in emit.items():
                    row = resid[r]
                    sig = " ".join(f"{m}={scal(row,dp,m)}" for m in MARKERS
                                   if scal(row, dp, m) not in (None, 0.0, -0.0))
                    sv1 = decode_byte(band(row, dp, "STACK0_BYTE_VAL_1_LO"),
                                      band(row, dp, "STACK0_BYTE_VAL_1_HI"))
                    print(f"  row{r}{which} SV1={sv1} | {sig}")
            elif mode == "stack0":
                for r in range(S):
                    row = resid[r]
                    sv1 = decode_byte(band(row, dp, "STACK0_BYTE_VAL_1_LO"),
                                      band(row, dp, "STACK0_BYTE_VAL_1_HI"))
                    s0b1 = scal(row, dp, "STACK0_BYTE1")
                    if sv1 is not None or (s0b1 and s0b1 > 0.3):
                        print(f"  row{r} SV1={sv1} STACK0_BYTE1={s0b1} "
                              f"MARK_STACK0={scal(row,dp,'MARK_STACK0')} "
                              f"OP_SUB={scal(row,dp,'OP_SUB')} "
                              f"OP_PSH={scal(row,dp,'OP_PSH')}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")]
    mode = "emit"
    for a in args:
        if a.startswith("mode="):
            mode = a[5:]
    sel = [a for a in args if not a.startswith("b=") and not a.startswith("mode=")] \
        or ["sub_16bit", "sub_borrow"]
    if not blks:
        blks = [18]
    main(sel, blks, mode)
