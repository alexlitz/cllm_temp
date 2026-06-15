#!/usr/bin/env python3
"""For single-byte expr_mul_div (10*9/1, 5*6/3), inspect the DIV step's
operand bands (ALU_LO/HI = operand-A byte0, AX_CARRY_LO/HI = operand-B byte0)
at the DIV MARK_AX row across blocks, and the STACK0_BYTE_VAL_0 the divmod
cummax would gather, to see why the dividend (the PSH'd MUL result) is lost.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_div_operand_select.py
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


def decode(row, lo, hi):
    los = [i for i in range(16) if float(row[lo + i].item()) > 0.5]
    his = [i for i in range(16) if float(row[hi + i].item()) > 0.5]
    if len(los) == 1 and len(his) == 1:
        return his[0]*16+los[0]
    if not los and not his:
        return "."
    return f"n{los}/{his}"


def main(ids):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    ALU_LO=dp["ALU_LO"]; ALU_HI=dp["ALU_HI"]
    AXC_LO=dp["AX_CARRY_LO"]; AXC_HI=dp["AX_CARRY_HI"]
    OUT_LO=dp["OUTPUT_LO"]; OUT_HI=dp["OUTPUT_HI"]
    MARK_AX=dp["MARK_AX"]; OP_DIV=dp["OP_DIV"]; OP_MOD=dp["OP_MOD"]
    STACK0_BYTE0=dp["STACK0_BYTE0"]
    S0V0_LO=dp.get("STACK0_BYTE_VAL_0_LO"); S0V0_HI=dp.get("STACK0_BYTE_VAL_0_HI")
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = probe._final_context(bc, max_steps=12)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        full = model.forward(padded, stop_after_block=nblocks-1)[0]
        nrows = full.shape[0]
        div_row = max([r for r in range(nrows) if float(full[r,MARK_AX].item())>0.5
                       and (float(full[r,OP_DIV].item())>0.5 or float(full[r,OP_MOD].item())>0.5)], default=None)
        s0_rows = [r for r in range(nrows) if float(full[r,STACK0_BYTE0].item())>0.5]
        s0_before_div = [r for r in s0_rows if div_row is None or r<=div_row]
        picked_s0 = s0_before_div[-1] if s0_before_div else None
        print(f"\n==== {desc} exp={exp} DIV_row={div_row} picked_s0(dividend low byte src)={picked_s0} ====")
        for b in (8, 11, 13, 14, 15):
            if b>=nblocks: continue
            resid = model.forward(padded, stop_after_block=b)[0]
            alu = decode(resid[div_row], ALU_LO, ALU_HI) if div_row is not None else "?"
            axc = decode(resid[div_row], AXC_LO, AXC_HI) if div_row is not None else "?"
            s0v = (decode(resid[picked_s0], S0V0_LO, S0V0_HI) if (picked_s0 is not None and S0V0_LO) else "?")
            print(f"  block {b}: @DIV ALU(opA_b0)={alu} AX_CARRY(opB_b0)={axc} | @picked_s0 STACK0_BYTE_VAL_0={s0v}")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [850, 856]
    main(ids)
