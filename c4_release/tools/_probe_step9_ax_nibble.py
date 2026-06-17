#!/usr/bin/env python3
"""Trace step-9 AX byte-0 OUTPUT_LO/HI one-hot through blocks (spec_k=0).

Finds WHICH block tips the AX byte-0 low-nibble one-hot from the correct index
(6 for 0x46) to the wrong index (8 for 0x48) at the step AFTER the LEV epilogue.

Usage: python tools/_probe_step9_ax_nibble.py [id] [step]
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

SE = int(Token.STEP_END); AX = int(Token.REG_AX)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    tstep = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    LO = dp["OUTPUT_LO"]; HI = dp["OUTPUT_HI"]
    ctx = probe._final_context(bc, max_steps=14)
    pl = len(probe._build_context(bc))
    # find step tstep AX byte-0 row
    steps = []; cur = []
    for p in range(pl, len(ctx)):
        cur.append(p)
        if ctx[p] == SE: steps.append(cur); cur = []
    axrow = None
    for p in steps[tstep]:
        if ctx[p] == AX:
            axrow = p + 1; break  # byte-0 row
    print(f"id{pid} {desc} exp={exp&0xff:#x} step{tstep} AX-byte0 row={axrow} emitted_tok={ctx[axrow]}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    print("blk : OUTPUT_LO argmax(val)  OUTPUT_HI argmax(val)")
    prev_lo = None
    for blk in range(nblk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][axrow]
        lo = r[LO:LO+16]; hi = r[HI:HI+16]
        li = int(lo.argmax()); hival = int(hi.argmax())
        flag = ""
        if prev_lo is not None and li != prev_lo:
            flag = f"  <<< LO {prev_lo}->{li}"
        print(f"{blk:3d} : lo_idx={li}(max {lo[li].item():+.1f})  hi_idx={hival}(max {hi[hival].item():+.1f}){flag}")
        prev_lo = li


if __name__ == "__main__":
    main()
