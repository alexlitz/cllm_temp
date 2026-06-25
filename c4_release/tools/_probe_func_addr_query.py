#!/usr/bin/env python3
"""Trace ADDR_B0_LO/HI (the LI-query address key) and the LEA result (AX byte-0)
across the LEA and LI steps for a func program, to see what address the LI query
is content-addressing and whether the genuine value row carries that address.

Usage: python tools/_probe_func_addr_query.py <id> <lea_step> <li_step> [maxsteps] [val]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT); RAX = int(Token.REG_AX)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def _find_l15(model, dimp):
    opli = dimp.get("OP_LI_RELAY"); cand = []
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        wq = wq.to_dense() if (wq.is_sparse or wq.is_sparse_csr) else wq
        if opli is not None and wq.shape[1] > opli and abs(float(wq[:, opli].abs().max())) > 1000:
            cand.append((bi, blk.attn.num_heads))
    cand.sort(key=lambda t: -t[1]); return cand[0][0] if cand else None


def nz(sub):
    return [(k, round(float(sub[k]), 2)) for k in range(16) if abs(float(sub[k])) > 0.25]


@torch.no_grad()
def main():
    pid = int(sys.argv[1])
    steps_want = [int(x) for x in sys.argv[2:4]]
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 14
    val = int(sys.argv[5]) if len(sys.argv) > 5 else 57
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    steps = []; s = 0
    for i, t in enumerate(ctx):
        if t in (STEP_END, HALT):
            steps.append((s, i)); s = i + 1
    L15 = _find_l15(model, dimp)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    a0lo = dimp["ADDR_B0_LO"]; a0hi = dimp["ADDR_B0_HI"]
    cl_lo = dimp.get("CLEAN_EMBED_LO"); cl_hi = dimp.get("CLEAN_EMBED_HI")
    def clval(p):
        lo = int(torch.argmax(resid[p, cl_lo:cl_lo+16]).item())
        hi = int(torch.argmax(resid[p, cl_hi:cl_hi+16]).item())
        return lo | (hi << 4)
    print(f"id{pid} {desc} exp={exp}  L15={L15}")
    for stp in steps_want:
        st, en = steps[stp]
        axm = next((i for i in range(st, en+1) if ctx[i] == RAX), None)
        print(f"\n=== step {stp}  rows {st}..{en}  AX_marker={axm} ===")
        # AX-marker row ADDR_B0 (the LI query key) + AX clean value
        if axm is not None:
            print(f"  AX-row {axm}: clval={clval(axm)}  ADDR_B0_LO={nz(resid[axm, a0lo:a0lo+16])}  ADDR_B0_HI={nz(resid[axm, a0hi:a0hi+16])}")
        # every MEM marker + value rows in this step, with ADDR_B0 nibbles + clval
        for p in range(st, en+1):
            if ctx[p] == 261:  # MEM marker
                for q in range(p, min(p+6, en+1)):
                    cv = clval(q)
                    lo_nz = nz(resid[q, a0lo:a0lo+16]); hi_nz = nz(resid[q, a0hi:a0hi+16])
                    mb1 = float(resid[q, dimp["MEM_VAL_B1"]]) if "MEM_VAL_B1" in dimp else 0.0
                    if lo_nz or hi_nz or cv == val or abs(mb1) > 0.3:
                        print(f"    p{q}: tok={ctx[q]} clval={cv} B1={mb1:.1f} a0lo={lo_nz} a0hi={hi_nz}")


if __name__ == "__main__":
    main()
