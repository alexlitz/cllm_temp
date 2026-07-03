#!/usr/bin/env python3
"""Trace SE_AX_CARRY_HI / AX_CARRY_HI at the LEV step (and a clean step) across
blocks + rows to find where the spurious byte-1 leak value enters.
Usage: python tools/_probe_se_ax_carry.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_markers(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append((cur, i)); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append((cur, len(ctx)))
    return out


def band(r, dp, name, n=16):
    v = r[dp[name]:dp[name]+n]
    nz = [(i, round(float(v[i]),2)) for i in range(n) if abs(float(v[i]))>0.3]
    return nz


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    ax = sm[lev][0].get("AX")
    se_idx = sm[lev][1]  # STEP_END token index for the LEV step
    print(f"id{pid} {desc} lev_step={lev} AX_marker={ax} STEP_END={se_idx}")
    rows = {"AX_marker": ax, "AX+1(b0 tok)": ax+1, "AX+2(b1 tok)": ax+2,
            "STEP_END": se_idx}
    for label, pos in rows.items():
        print(f"\n== row {label} (pos={pos}, tok={ctx[pos]&0xFF}) ==")
        for blk in [8, 9, 10, 13, 26, 32, nblk-1]:
            if blk >= nblk: continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
            sehi = band(r, dp, "SE_AX_CARRY_HI"); selo = band(r, dp, "SE_AX_CARRY_LO")
            axhi = band(r, dp, "AX_CARRY_HI"); axlo = band(r, dp, "AX_CARRY_LO")
            print(f"  blk{blk:2d}: SE_AX_HI={sehi} SE_AX_LO={selo} | AX_HI={axhi} AX_LO={axlo}")


if __name__ == "__main__":
    main()
