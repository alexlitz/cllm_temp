#!/usr/bin/env python3
"""Print the AR-EMITTED token sequence per step for a func program (the real
production decode), decoding PC/AX/SP/BP per step, and FLAG steps whose token
count != STEP_TOKENS (the frame desync). This is the authoritative AR view: it
reads the emitted tokens straight out of _final_context (the production decode).

Usage: python tools/_probe_func_aremit.py <id> [maxsteps]
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

SE = int(Token.STEP_END); HALT = int(Token.HALT)
RPC = int(Token.REG_PC); RAX = int(Token.REG_AX)
RSP = int(Token.REG_SP); RBP = int(Token.REG_BP)
MARK = {RPC: "PC", RAX: "AX", RSP: "SP", RBP: "BP", 261: "MEM", 268: "STK0"}
STEP_TOKENS = int(Token.STEP_TOKENS)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    print(f"id{pid} {desc} exp={exp}  STEP_TOKENS={STEP_TOKENS}  prompt_len={pl}")
    i = pl; step = 0; line = []; cur0 = pl
    def decode_reg(p):
        bs = ctx[p+1:p+5]
        return sum((b & 0xFF) << (8*j) for j, b in enumerate(bs))
    while i < len(ctx) and step < ms:
        nm = MARK.get(ctx[i])
        if ctx[i] == SE or ctx[i] == HALT:
            ntok = i - cur0 + 1
            flag = "" if ntok == STEP_TOKENS else f"  <<< ntok={ntok} (!= {STEP_TOKENS})"
            print(f"  step {step:2d}: " + "  ".join(line) + flag)
            line = []; step += 1; i += 1; cur0 = i
            if ctx[i-1] == HALT: break
            continue
        if nm in ("PC", "AX", "SP", "BP"):
            line.append(f"{nm}={decode_reg(i)}")
            i += 5; continue
        i += 1


if __name__ == "__main__":
    main()
