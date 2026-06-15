#!/usr/bin/env python3
"""Check the STACK0 byte-0 CLEAN_EMBED at the dividend frame for single-byte
expr_mul_div, to confirm whether the high nibble of byte 0 is lost (the
STACK0 byte-0 high-nibble framing-drift wall #221).

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_stack0_b0_nibble.py 850 851 856 853
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


def main(ids):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    CLO=dp["CLEAN_EMBED_LO"]; CHI=dp["CLEAN_EMBED_HI"]
    STACK0_BYTE0=dp["STACK0_BYTE0"]; MARK_AX=dp["MARK_AX"]
    OP_DIV=dp["OP_DIV"]; OP_MOD=dp["OP_MOD"]; OP_MUL=dp["OP_MUL"]
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = probe._final_context(bc, max_steps=12)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        resid = model.forward(padded, stop_after_block=11)[0]
        nrows = resid.shape[0]
        div_row = max([r for r in range(nrows) if float(resid[r,MARK_AX].item())>0.5
                       and (float(resid[r,OP_DIV].item())>0.5 or float(resid[r,OP_MOD].item())>0.5)], default=None)
        mul_row = max([r for r in range(nrows) if float(resid[r,MARK_AX].item())>0.5
                       and float(resid[r,OP_MUL].item())>0.5], default=None)
        s0_rows = [r for r in range(nrows) if float(resid[r,STACK0_BYTE0].item())>0.5]
        # dividend frame = the STACK0 frame after the MUL, before the DIV
        cand = [r for r in s0_rows if (mul_row is None or r>mul_row) and (div_row is None or r<=div_row)]
        print(f"\n==== {desc} exp={exp} MUL_row={mul_row} DIV_row={div_row} ====")
        for r in (cand[:3] if cand else []):
            lo=[i for i in range(16) if float(resid[r,CLO+i].item())>0.5]
            hi=[i for i in range(16) if float(resid[r,CHI+i].item())>0.5]
            val = (hi[0]*16+lo[0]) if len(lo)==1 and len(hi)==1 else f"lo={lo}/hi={hi}"
            print(f"  STACK0_BYTE0 row {r}: CLEAN_EMBED byte0 = {val}")


if __name__ == "__main__":
    ids=[int(x) for x in sys.argv[1:]] or [850,851,856,853]
    main(ids)
