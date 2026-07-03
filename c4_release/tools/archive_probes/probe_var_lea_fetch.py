#!/usr/bin/env python3
"""Compare the FETCH (immediate) and ALU (BP) bands at the step-2 LEA (imm=-8,
correct) vs the step-6 LEA (imm=-16, wrong) under FULL teacher-forcing.

LEA computes OUTPUT = ALU(BP) + FETCH(imm) nibble-wise (l9 lea_lo/hi rules).
If FETCH_LO/HI at step-6 reads the imm=-8 low byte (0xf8/0xff) instead of the
imm=-16 byte (0xf0/0xff), the immediate gather is the bug.

We read the bands at the AX-marker row of each LEA step, pre-L9-ALU and post.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, contextlib, io
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from neural_vm.batched_pure_neural import Token
from neural_vm.speculative import DraftVM

STEP = int(Token.STEP_TOKENS)


def argmax_nib(row, base):
    return int(torch.argmax(row[base:base + 16]).item())


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 275
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)

    from tools.probe_groundtruth import build_groundtruth_probe
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = build_groundtruth_probe()
        _m, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = layout.dim_positions
    FETCH_LO, FETCH_HI = dp["FETCH_LO"], dp["FETCH_HI"]
    ALU_LO, ALU_HI = dp["ALU_LO"], dp["ALU_HI"]
    OUT_LO, OUT_HI = dp["OUTPUT_LO"], dp["OUTPUT_HI_THIS_STEP"]
    dev = p._device

    # Build fully teacher-forced context.
    ctx = p._build_context(bc)
    plen = len(ctx)
    dv = DraftVM(list(bc))
    for _ in range(10):
        dv.step()
        ctx.extend(int(t) for t in dv.draft_tokens())
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    # find the physical block right after L9 ALU. Probe at several blocks.
    blm = p.block_layer_map()
    L9 = next(b["physical"] for b in blm if b["logical"] == 9)
    print(f"L9 phys block = {L9}  (probe pre={L9-1} post={L9})")

    # AX-marker row off = 5 within each step. The LEA result is emitted there.
    full = "--full" in sys.argv
    for step, immname in [(2, "imm=-8 (a=BP-8 correct=0xffe8)"),
                          (6, "imm=-16 (b=BP-16 correct=0xffe0)")]:
        axrow = plen + step * STEP + 5  # AX marker row
        print(f"\n--- step {step} {immname}  axrow={axrow} ---")
        blks = range(L9 - 1, len(p.model.blocks)) if full else (L9 - 1, L9, L9 + 1)
        prev = None
        for blk in blks:
            resid = p.model.forward(padded, stop_after_block=blk)[0]
            r = resid[axrow].float()
            ol, oh = argmax_nib(r, OUT_LO), argmax_nib(r, OUT_HI)
            lg = next((b["logical"] for b in blm if b["physical"] == blk), "?")
            flag = ""
            if prev is not None and (ol, oh) != prev:
                flag = f"  <-- CHANGED from {prev}"
            prev = (ol, oh)
            if not full:
                print(f"  blk{blk}: FETCH_LO={argmax_nib(r,FETCH_LO)} "
                      f"FETCH_HI={argmax_nib(r,FETCH_HI)}  "
                      f"ALU_LO={argmax_nib(r,ALU_LO)} ALU_HI={argmax_nib(r,ALU_HI)}  "
                      f"OUT_LO={ol} OUT_HI={oh}")
            else:
                byte0 = (oh << 4) | ol
                print(f"  blk{blk:2d}(L{lg}): OUT_LO={ol} OUT_HI={oh} "
                      f"-> byte0={byte0:#04x}{flag}")


if __name__ == "__main__":
    main()
