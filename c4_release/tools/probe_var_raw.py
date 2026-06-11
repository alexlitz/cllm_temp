#!/usr/bin/env python3
"""Raw token-stream dump for var_simple_12 at spec_k=0 + control compare.

Prints the prompt layout and every emitted step's marker+byte tokens so we can
verify the per-step register decode and pin exactly which marker/byte first
diverges. Also runs an IMM-only control (no stack store) for comparison.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from neural_vm.embedding import Opcode  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARK = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP", 260: "REG_BP",
        262: "STEP_END", 263: "HALT", 271: "TOOL_CALL"}


def dump(label, bc):
    print(f"\n########## {label} ##########")
    probe = build_groundtruth_probe()
    prompt = probe._build_context(bc)
    ctx = probe._final_context(bc)
    print(f"prompt_len={len(prompt)} total={len(ctx)}")
    print(f"PROMPT tokens: {prompt}")
    print("\n--- emitted token stream (per step) ---")
    i = len(prompt)
    step = 0
    while i < len(ctx):
        t = ctx[i]
        if t in MARK and MARK[t] in ("REG_PC", "REG_AX", "REG_SP", "REG_BP"):
            bs = ctx[i+1:i+5]
            val = sum((b & 0xFF) << (8*j) for j, b in enumerate(bs))
            print(f"  step{step} {MARK[t]:7s} bytes={[hex(b) for b in bs]} val={val}")
            i += 5
        elif t == 262:
            print(f"  step{step} STEP_END")
            step += 1
            i += 1
        elif t == 263:
            print(f"  HALT")
            i += 1
        else:
            print(f"  [pos{i}] raw token {t}")
            i += 1


def main():
    tests = generate_test_programs()
    src, exp, desc = tests[262]
    bc, _ = compile_c(src)
    dump(f"id=262 {desc} expected={exp}", bc)

    # Control 1: IMM-only, no stack store, no JSR/ENT.
    # int main()->just IMM 28 then HALT. Build raw bytecode.
    ctrl = [Opcode.IMM | (28 << 8), Opcode.EXIT]
    dump("CONTROL: IMM 28; EXIT (no var, no JSR/ENT)", ctrl)

    # Control 2: JSR/ENT prologue but NO store (just return literal).
    # int main(){ return 28; }
    src2 = "int main() { return 28; }"
    bc2, _ = compile_c(src2)
    dump("CONTROL: int main(){return 28;} (JSR/ENT, no var store)", bc2)


if __name__ == "__main__":
    main()
