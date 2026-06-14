#!/usr/bin/env python3
"""Probe 2: where does old_BP live cleanly at the BP register rows, and what
do the ENT-step MEM val rows carry across blocks? spec_k=0, hook-free.

Usage: python tools/_probe_bp_carry2.py <id> <maxsteps>
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_rows(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], []).append(i)
            i += 1; continue
        i += 1
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ent_step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    rows = step_rows(ctx, pl)
    nblk = len(probe.model.blocks)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    print(f"id{pid} {desc} exp={exp} n_blocks={nblk} ENT step={ent_step}")

    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    H1 = dp.get("H1"); H3 = dp.get("H3")

    def out_nibbles(resid):
        lo = int(torch.argmax(resid[OL:OL+16]))
        hi = int(torch.argmax(resid[OH:OH+16]))
        lo_m = round(float(resid[OL+lo]), 2)
        hi_m = round(float(resid[OH+hi]), 2)
        return lo, hi, lo_m, hi_m

    # 1) The PREV step's BP register byte rows: do they carry old_BP cleanly?
    prev = ent_step - 1
    print(f"\n=== PREV step {prev} BP register byte rows (old_BP source) ===")
    bp_idx = rows[prev]["BP"][0]
    for off in range(0, 5):
        pos = bp_idx + off
        rfin = probe.model.forward(padded, stop_after_block=nblk-1)[0][pos]
        lo, hi, lm, hm = out_nibbles(rfin)
        label = "marker" if off == 0 else f"byte{off-1}"
        h1v = round(float(rfin[H1:H1+7].abs().sum()), 1) if H1 else 0
        h3v = round(float(rfin[H3:H3+7].abs().sum()), 1) if H3 else 0
        print(f"  off{off} ({label:6s}) pos={pos} tok={ctx[pos]:3d} "
              f"OUTPUT byte=0x{(hi<<4)|lo:02x} (lo={lo}@{lm} hi={hi}@{hm}) "
              f"|H1|={h1v} |H3|={h3v}")

    # 2) Also check the prev step BP byte rows across a few blocks (to find the
    #    block where old_BP is cleanest / most causally usable).
    print(f"\n=== PREV step {prev} BP byte rows across blocks (OUTPUT decode) ===")
    for off in range(1, 5):
        pos = bp_idx + off
        line = [f"  byte{off-1} pos={pos} tok={ctx[pos]:3d}:"]
        for blk in [6, 10, 16, 26, 27, 32, 37, nblk-1]:
            if blk >= nblk:
                continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
            lo, hi, lm, hm = out_nibbles(r)
            line.append(f"b{blk}=0x{(hi<<4)|lo:02x}")
        print(" ".join(line))

    # 3) The ENT step's MEM val rows: what do they carry across blocks? Find
    #    where the garbage 0xFF gets written (which block corrupts val).
    print(f"\n=== ENT step {ent_step} MEM val rows across blocks (OUTPUT decode) ===")
    mem_idx = rows[ent_step]["MEM"][0]
    for valoff in range(0, 4):
        pos = mem_idx + 5 + valoff  # marker + 4 addr + valoff
        line = [f"  val{valoff} pos={pos} tok={ctx[pos]:3d}:"]
        for blk in [6, 10, 16, 22, 26, 27, 32, 37, nblk-1]:
            if blk >= nblk:
                continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
            lo, hi, lm, hm = out_nibbles(r)
            line.append(f"b{blk}=0x{(hi<<4)|lo:02x}")
        print(" ".join(line))

    # 4) Print the ACTUAL emitted MEM val tokens (the bug) and oracle for ref.
    print(f"\n=== ENT step {ent_step} emitted MEM section tokens ===")
    sec = [ctx[mem_idx + k] for k in range(9)]
    print(f"  raw tokens: {sec}")
    print(f"  addr bytes: {sec[1:5]}  val bytes: {sec[5:9]}")


if __name__ == "__main__":
    main()
