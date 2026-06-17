#!/usr/bin/env python3
"""Dump all MEM_STORE / STACK0 rows in a window with their ADDR key (24-bit),
CLEAN_EMBED value, MEM_VAL bands and markers, at the L15 input. Lets us see
which store row the LI must read (mem[0xFFE8]=70) and whether ANY store row
carries value 70 at address 0xFFE8.

Usage: python tools/_probe_func_frame_stores.py <id> [lo] [hi] [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT)
NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}


def _find_l15(model, dimp):
    opli = dimp.get("OP_LI_RELAY"); cand = []
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        wq = wq.to_dense() if (wq.is_sparse or wq.is_sparse_csr) else wq
        if opli is not None and wq.shape[1] > opli and abs(float(wq[:, opli].abs().max())) > 1000:
            cand.append((bi, blk.attn.num_heads))
    cand.sort(key=lambda t: -t[1]); return cand[0][0] if cand else None


@torch.no_grad()
def main():
    pid = int(sys.argv[1])
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 230
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 290
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    L15 = _find_l15(model, dimp)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    cl_lo = dimp["CLEAN_EMBED_LO"]; cl_hi = dimp["CLEAN_EMBED_HI"]

    def hot16(p, base):
        return int(torch.argmax(resid[p, base:base + 16]).item())

    def addr24(p):
        b0 = hot16(p, dimp["ADDR_B0_LO"]) | (hot16(p, dimp["ADDR_B0_HI"]) << 4)
        b1 = hot16(p, dimp["ADDR_B1_LO"]) | (hot16(p, dimp["ADDR_B1_HI"]) << 4)
        b2 = hot16(p, dimp["ADDR_B2_LO"]) | (hot16(p, dimp["ADDR_B2_HI"]) << 4)
        return b0 | (b1 << 8) | (b2 << 16)

    print(f"id{pid} {desc} exp={exp}  L15={L15}  rows {lo}..{hi}")
    print(f"{'pos':>4} {'tok':>5} {'mark':<8} {'MSTORE':>6} {'CLEAN':>6} "
          f"{'ADDR24':>8} {'MV_B0':>6} {'MV_B1':>6} {'BIDX':>5} {'STK0':>6}")
    for p in range(lo, min(hi, resid.shape[0])):
        tok = ctx[p]
        marks = "+".join(n.replace("MARK_", "") for n in ("MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0")
                         if dimp.get(n) is not None and resid[p, dimp[n]].item() > 0.5)
        mstore = float(resid[p, dimp["MEM_STORE"]].item()) if "MEM_STORE" in dimp else 0.0
        clv = hot16(p, cl_lo) | (hot16(p, cl_hi) << 4)
        cl_mag = float(resid[p, cl_lo + hot16(p, cl_lo)].item())
        a24 = addr24(p)
        mvb0 = float(resid[p, dimp["MEM_VAL_B0"]].item()) if "MEM_VAL_B0" in dimp else 0.0
        mvb1 = float(resid[p, dimp["MEM_VAL_B1"]].item()) if "MEM_VAL_B1" in dimp else 0.0
        bidx = ""
        for bi in range(4):
            d = dimp.get(f"BYTE_INDEX_{bi}")
            if d is not None and resid[p, d].item() > 0.5:
                bidx = str(bi)
        stk0 = float(resid[p, dimp["STACK0_BYTE0"]].item()) if "STACK0_BYTE0" in dimp else 0.0
        # only show "interesting" rows
        if mstore > 0.3 or mvb0 > 0.3 or marks or (cl_mag > 0.3 and clv != 0) or abs(stk0) > 0.5:
            print(f"{p:>4} {tok:>5} {marks:<8} {mstore:>6.2f} {clv if cl_mag>0.3 else 0:>6} "
                  f"0x{a24:06x} {mvb0:>6.2f} {mvb1:>6.2f} {bidx:>5} {stk0:>6.1f}")


if __name__ == "__main__":
    main()
