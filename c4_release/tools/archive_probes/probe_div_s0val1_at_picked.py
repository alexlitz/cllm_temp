#!/usr/bin/env python3
"""Confirm: at the cummax-picked STACK0_BYTE1 row, the dividend high byte
lives in STACK0_BYTE_VAL_1_LO/HI (dim 602/618), NOT CLEAN_EMBED_LO/HI.

This is the BDToGEConverter fallback bug: it reads CLEAN_EMBED at that row
(=0x00) instead of STACK0_BYTE_VAL_1 (=correct high byte).

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_s0val1_at_picked.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            o, i = op; bc.append(o | (i << 8))
        else:
            bc.append(op)
    return bc


PROGRAMS = {
    "div_1162_37": (_mk([(Opcode.IMM, 1162), Opcode.PSH, (Opcode.IMM, 37), Opcode.DIV, Opcode.EXIT]), 31),  # hi=0x04
    "div_300_5": (_mk([(Opcode.IMM, 300), Opcode.PSH, (Opcode.IMM, 5), Opcode.DIV, Opcode.EXIT]), 60),       # hi=0x01
    "div_84_2": (_mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 2), Opcode.DIV, Opcode.EXIT]), 42),         # hi=0x00
}


def onehot(row, base, w=16):
    return [(i, round(float(row[base + i].item()), 2)) for i in range(w) if float(row[base + i].item()) > 0.5]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    S1LO = dp["STACK0_BYTE_VAL_1_LO"]; S1HI = dp["STACK0_BYTE_VAL_1_HI"]
    CLO = dp["CLEAN_EMBED_LO"]; CHI = dp["CLEAN_EMBED_HI"]
    STACK0_BYTE1 = dp["STACK0_BYTE1"]; MARK_AX = dp["MARK_AX"]
    OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]
    print(f"# S0VAL1_LO={S1LO} CLEAN_LO={CLO}")
    for pname in selected:
        prog, expected = PROGRAMS[pname]
        ctx = probe._final_context(prog, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        _, code = probe.emitted_result(prog, max_steps=6)
        resid = model.forward(padded, stop_after_block=11)[0]
        div_rows = [r for r in range(resid.shape[0])
                    if float(resid[r, MARK_AX].item()) > 0.5
                    and (float(resid[r, OP_DIV].item()) > 0.5 or float(resid[r, OP_MOD].item()) > 0.5)]
        row = div_rows[-1] if div_rows else None
        stack1_rows = [r2 for r2 in range(row + 1)
                       if float(resid[r2, STACK0_BYTE1].item()) > 0.5] if row is not None else []
        picked = stack1_rows[-1] if stack1_rows else None
        print(f"\n==== {pname} exp={expected} neural={code} divrow={row} picked_stack1_row={picked} ====")
        if picked is not None:
            pr = resid[picked]
            print(f"  @picked row {picked}:")
            print(f"    STACK0_BYTE_VAL_1_LO = {onehot(pr, S1LO)}   STACK0_BYTE_VAL_1_HI = {onehot(pr, S1HI)}")
            print(f"    CLEAN_EMBED_LO       = {onehot(pr, CLO)}   CLEAN_EMBED_HI       = {onehot(pr, CHI)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["div_1162_37", "div_300_5", "div_84_2"])
