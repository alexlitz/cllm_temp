#!/usr/bin/env python3
"""Dump the L15 PSH-SP-byte rule discriminator dims at the var step-3 leak row
(pred_row=211) AND at the legit firing rows, reading the residual ENTERING
block 30 (after block 29). READ-ONLY, spec_k=0.

Usage: python tools/probe_var_l15_psh_disc.py [id] [block]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from neural_vm.batched_pure_neural import Token

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}

DIMS = ["PSH_AT_SP", "IS_BYTE", "MEM_STORE", "HAS_SE",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "H1+0", "H1+1", "H1+2", "H1+3", "H1+10", "H4+3",
        "MARK_BP", "MARK_SP", "MARK_AX", "MARK_STACK0", "MARK_MEM",
        "CMP+7", "OP_SI", "OP_PSH", "OP_LI"]


def getdim(dp, nm):
    if "+" in nm:
        base, off = nm.rsplit("+", 1)
        return dp[base] + int(off) if base in dp else None
    return dp.get(nm)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    blk = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device
    ctx = probe._final_context(bc)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        x_in = m.forward(padded, stop_after_block=blk - 1)[0].float()

    # find rows of interest: pred_row 211 and any SP-marker byte rows
    rows = [211]
    for i, t in enumerate(ctx):
        if MARKERS.get(t) == "SP":
            for j in range(1, 5):
                rows.append(i + j - 1)  # prediction rows for SP bytes
    rows = sorted(set([r for r in rows if 0 <= r < len(ctx)]))

    print(f"id={idx} {desc} block_in={blk-1} (entering block {blk})")
    hdr = "row   tok  " + " ".join(f"{d:>10}" for d in DIMS)
    print(hdr)
    for r in rows:
        vals = []
        for d in DIMS:
            di = getdim(dp, d)
            vals.append(f"{float(x_in[r][di]):>10.3f}" if di is not None else f"{'?':>10}")
        mark = " <== LEAK(211)" if r == 211 else ""
        print(f"{r:>4} {ctx[r]:>5}  " + " ".join(vals) + mark)


if __name__ == "__main__":
    main()
