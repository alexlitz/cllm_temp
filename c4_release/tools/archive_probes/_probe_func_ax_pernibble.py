#!/usr/bin/env python3
"""Per-step AX byte-0 (low+high nibble) for a func_* program, spec_k=0.

Shows the emitted AX byte-0 token at every step so we can see WHICH step first
delivers the wrong low nibble (the exp-low-nibble -> 0x8 corruption).

Usage: python tools/_probe_func_ax_pernibble.py <id> [maxsteps]
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

SE = int(Token.STEP_END)
AX = int(Token.REG_AX)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    print(f"id{pid} {desc} exp={exp}(0x{exp&0xff:02x})")
    # split into steps; AX byte-0 = token right after REG_AX marker
    step = 0; i = pl
    while i < len(ctx):
        if ctx[i] == SE:
            step += 1; i += 1; continue
        if ctx[i] == AX:
            # AX register block: marker then 4 byte tokens
            b0 = ctx[i + 1] if i + 1 < len(ctx) else None
            b1 = ctx[i + 2] if i + 2 < len(ctx) else None
            print(f"  step {step:2d}: AX byte0={b0} (0x{b0:02x} lo={b0&0xf:x} hi={(b0>>4)&0xf:x})  byte1={b1}")
            i += 5; continue
        i += 1


if __name__ == "__main__":
    main()
