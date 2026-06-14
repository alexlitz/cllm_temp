#!/usr/bin/env python3
"""Probe the L15 memory_lookup output at the LI step's AX byte-0 prediction row.
Marker-parse to locate the row (authoritative decode). spec_k=0, hook-free.
Usage: python tools/_probe_li_l15.py <id> <maxsteps> <li_step>
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END); AXk = int(Token.REG_AX)
REGS = {int(Token.REG_PC): "PC", AXk: "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_markers(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]); li = int(sys.argv[3])
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    ax_marker = sm[li]["AX"]            # AX marker token index
    # AX value byte-0 is predicted at the AX marker row (model emits next token
    # = AX byte 0). So read residual AT ax_marker (predicts byte0) and ax_marker+1..
    print(f"id{pid} {desc} exp={exp} li_step={li} AX_marker_idx={ax_marker}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    bands = {k: dp[k] for k in ("OUTPUT_LO","OUTPUT_HI","ALU_LO","MEM_VAL_B0","MEM_STORE","AX_FULL_LO") if k in dp}
    # Probe rows: the AX marker row (predicts byte0) and the next 4 byte rows.
    nblk = len(probe.model.blocks)
    for rowoff in range(0, 5):
        pos = ax_marker + rowoff
        print(f"\n-- row ax_marker+{rowoff} (pos={pos}, tok={ctx[pos]}) --")
        for blk in [13, 18, 26, 27, 28, 32, 37, nblk-1]:
            if blk >= nblk: continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
            vals = {k: round(float(r[bands[k]]), 1) for k in bands}
            vals = {k: v for k, v in vals.items() if abs(v) > 0.3}
            print(f"   blk{blk:2d}: {vals}")


if __name__ == "__main__":
    main()
