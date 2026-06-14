#!/usr/bin/env python3
"""Per-step neural register trace for func_* programs (LI-from-frame bug).
spec_k=0, hook-free.  Usage: python tools/_probe_func_li.py <id> [<id> ...]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARK = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", int(Token.STEP_END): "STEP_END",
        268: "STACK0", 261: "MEM"}


def trace(probe, pid, max_steps):
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    print(f"\n==== id{pid} {desc} exp={exp} ====")
    ctx = probe._final_context(bc, max_steps=max_steps)
    pl = len(probe._build_context(bc))
    i = pl; step = 0; line = []
    def flush(s, items):
        print(f"  step {s:2d}: " + "  ".join(items))
    while i < len(ctx):
        nm = MARK.get(ctx[i])
        if nm == "STEP_END":
            flush(step, line); line = []; step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP", "STACK0", "MEM"):
            bs = ctx[i+1:i+5]
            val = sum((b & 0xFF) << (8*j) for j, b in enumerate(bs))
            line.append(f"{nm}=0x{val:08x}({val})")
            i += 5; continue
        i += 1
    if line:
        flush(step, line)


def main():
    ids = [int(x) for x in sys.argv[1:]] or [550, 575]
    probe = build_groundtruth_probe()
    for pid in ids:
        trace(probe, pid, max_steps=20)


if __name__ == "__main__":
    main()
