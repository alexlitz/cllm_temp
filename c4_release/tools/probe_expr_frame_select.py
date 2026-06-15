#!/usr/bin/env python3
"""For every expr_mul_div / expr_add_mul program, report whether the STACK0
frame the 2nd op reads (the most-recent STACK0_BYTE0 row before the 2nd op's
MARK_AX) holds the INTERMEDIATE's byte 0 (correct) or a stale value (the
cross-step persistence wall #221). Classifies each fail.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_expr_frame_select.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import re
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
    OP2 = {nm: dp["OP_"+nm] for nm in ("DIV","MOD","ADD")}

    n_stale = 0; n_correct_b0 = 0; n_total = 0
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        # intermediate = first multi/add result
        m = re.search(r'(\d+)[*](\d+)', desc)  # a*b for mul_div; b*c for add_mul
        inter = None
        if 'mul_div' in desc:
            mm=re.search(r'(\d+)\*(\d+)/(\d+)',desc); inter=int(mm[1])*int(mm[2]) if mm else None
        elif 'add_mul' in desc:
            mm=re.search(r'(\d+)\+(\d+)\*(\d+)',desc); inter=int(mm[2])*int(mm[3]) if mm else None
        if inter is None: continue
        inter_b0 = inter & 0xFF
        ctx = probe._final_context(bc, max_steps=12)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        resid = model.forward(padded, stop_after_block=11)[0]
        nrows = resid.shape[0]
        op2_row = max([r for r in range(nrows) if float(resid[r,MARK_AX].item())>0.5
                       and any(float(resid[r,d].item())>0.5 for d in OP2.values())], default=None)
        if op2_row is None: continue
        s0 = [r for r in range(nrows) if float(resid[r,STACK0_BYTE0].item())>0.5 and r<=op2_row]
        if not s0: continue
        r = s0[-1]
        lo=[i for i in range(16) if float(resid[r,CLO+i].item())>0.5]
        hi=[i for i in range(16) if float(resid[r,CHI+i].item())>0.5]
        read_b0 = (hi[0]*16+lo[0]) if len(lo)==1 and len(hi)==1 else None
        n_total += 1
        ok = (read_b0 == inter_b0)
        if ok: n_correct_b0 += 1
        else: n_stale += 1
        tag = "OK_b0" if ok else "STALE/WRONG"
        print(f"  {idx} {desc}: inter={inter}(b0={inter_b0}) 2nd-op reads frame b0={read_b0} [{tag}]")
    print(f"\nSUMMARY: {n_correct_b0}/{n_total} 2nd-op reads CORRECT intermediate byte0; "
          f"{n_stale}/{n_total} read STALE/WRONG (cross-step persistence wall #221)")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or list(range(800,825))+list(range(850,875))
    main(ids)
