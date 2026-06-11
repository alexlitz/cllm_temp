#!/usr/bin/env python3
"""Full emitted-token trace for id262 grouped by step (PC/AX/SP/BP/STACK0/MEM
registers per step), so we can see which step first diverges from the oracle.
spec_k=0, hook-free.
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


def main():
    probe = build_groundtruth_probe()
    src, exp, _ = generate_test_programs()[262]
    bc = compile_c(src)[0]
    print(f"id262 {src!r} exp={exp}")
    print("bytecode:", [hex(b) for b in bc])
    out, exit_code = probe.emitted_result(bc, max_steps=12)
    print(f"\nEMITTED exit_code = {exit_code} (want {exp})")
    ctx = probe._final_context(bc, max_steps=12)
    pl = len(probe._build_context(bc))
    i = pl; step = 0; cur = []
    def flush(s, items):
        print(f"  step {s}: " + "  ".join(items))
    line = []
    while i < len(ctx):
        nm = MARK.get(ctx[i])
        if nm == "STEP_END":
            flush(step, line); line = []; step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP", "STACK0", "MEM"):
            bs = ctx[i+1:i+5]
            val = sum((b & 0xFF) << (8*j) for j, b in enumerate(bs))
            line.append(f"{nm}=0x{val:08x}")
            i += 5; continue
        i += 1
    if line:
        flush(step, line)


if __name__ == "__main__":
    main()
