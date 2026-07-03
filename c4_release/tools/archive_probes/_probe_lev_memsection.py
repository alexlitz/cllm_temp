#!/usr/bin/env python3
"""Dump the per-step token layout (markers + their value bytes) so we can see
whether the JSR return address (90) is materialized anywhere in the campaign
frame's MEM section or only on the (dropped) STACK0 push.

Usage: python tools/_probe_lev_memsection.py <id> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


@torch.no_grad()
def main():
    pid = int(sys.argv[1])
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    print(f"STEP_TOKENS={int(Token.STEP_TOKENS)} ctxlen={len(ctx)} prelude={pl} desc={desc}")
    # walk steps, print markers
    i = pl; step = 0; cur = []
    INV = {v: k for k, v in REGS.items()}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            toks = " | ".join(cur)
            print(f"step{step:>2}: {toks}")
            step += 1; cur = []; i += 1; continue
        if t in REGS:
            vals = ctx[i+1:i+5]
            v32 = 0
            for j, b in enumerate(vals):
                v32 |= (b & 0xFF) << (8*j)
            cur.append(f"{REGS[t]}@{i}={vals[0]}(0x{v32:x})")
            i += 5; continue
        cur.append(f"?{t}@{i}")
        i += 1
    if cur:
        print(f"step{step:>2}: {' | '.join(cur)}")


if __name__ == "__main__":
    main()
