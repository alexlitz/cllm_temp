#!/usr/bin/env python3
"""Trace the FULL STACK0 frame chain across every step of an expr program at
BUILT dim_positions (spec_k=0, hook-free). For each physical block of interest
and each STACK0_BYTE0 row, report the CLEAN_EMBED byte-0 value the frame holds.
Then, at the DIV/MOD/ADD (op2) AX marker, report which STACK0 frame the
operand-gather (L7 head 0) attends to and the byte-0 it reads.

The point: localize WHERE the stale `a` frame replaces the freshly-pushed
intermediate. Walk block-by-block on the SAME op2-relevant STACK0 rows.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_stack0_framechain.py 852 876 800
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.probe_groundtruth import build_groundtruth_probe


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def _byte0(resid, r, CLO, CHI):
    lo = [i for i in range(16) if float(resid[r, CLO + i].item()) > 0.5]
    hi = [i for i in range(16) if float(resid[r, CHI + i].item()) > 0.5]
    if len(lo) == 1 and len(hi) == 1:
        return hi[0] * 16 + lo[0]
    return f"lo={lo}/hi={hi}"


def main(ids):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    CLO = dp["CLEAN_EMBED_LO"]; CHI = dp["CLEAN_EMBED_HI"]
    STACK0_BYTE0 = dp["STACK0_BYTE0"]; MARK_AX = dp["MARK_AX"]
    MARK_STACK0 = dp["MARK_STACK0"]
    OP2 = {nm: dp["OP_" + nm] for nm in ("DIV", "MOD", "ADD")}
    OPS = {nm: dp["OP_" + nm] for nm in ("DIV", "MOD", "ADD", "MUL", "PSH", "IMM", "SUB")}
    ALU_LO = dp.get("ALU_LO"); ALU_HI = dp.get("ALU_HI")
    AX_FULL_LO = dp.get("AX_FULL_LO")

    # physical blocks of interest: L7 operand-gather (block 8), L10 persistence (block 11)
    # We probe a sweep so we can see WHERE the chain corrupts.
    blocks_of_interest = [5, 8, 11, 14]  # pre-L7, L7-out, L10-out, post

    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = probe._final_context(bc, max_steps=12)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        print(f"\n========== id={idx} {desc} exp={exp} ==========")
        # Use a mid block to identify STACK0 rows + op2 AX row
        ref = model.forward(padded, stop_after_block=11)[0]
        nrows = ref.shape[0]
        op2_row = max([r for r in range(nrows)
                       if float(ref[r, MARK_AX].item()) > 0.5
                       and any(float(ref[r, d].item()) > 0.5 for d in OP2.values())],
                      default=None)
        # all STACK0_BYTE0 rows (frame markers), with the opcode context of each
        s0_rows = [r for r in range(nrows) if float(ref[r, STACK0_BYTE0].item()) > 0.5]
        print(f"  op2 AX row = {op2_row}; STACK0_BYTE0 rows = {s0_rows}")
        # annotate each AX marker row with its opcode (to map steps)
        ax_rows = [r for r in range(nrows) if float(ref[r, MARK_AX].item()) > 0.5]
        for r in ax_rows:
            ops_here = [nm for nm, d in OPS.items() if float(ref[r, d].item()) > 0.5]
            print(f"    AX row {r}: op={ops_here}")

        for blk in blocks_of_interest:
            resid = model.forward(padded, stop_after_block=blk)[0]
            print(f"  --- block {blk} ---")
            for r in s0_rows:
                v = _byte0(resid, r, CLO, CHI)
                m0 = float(resid[r, MARK_STACK0].item())
                print(f"    STACK0_BYTE0 row {r}: byte0={v}  MARK_STACK0={m0:.2f}")
            if op2_row is not None and ALU_LO is not None:
                alu = _byte0(resid, op2_row, ALU_LO, ALU_HI)
                print(f"    op2 AX row {op2_row}: ALU byte0 = {alu}")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [852, 876, 800]
    main(ids)
