#!/usr/bin/env python3
"""Print the full emitted per-step register trace for id 262 (spec_k=0)
and the oracle's expected register values, flagging the first divergence.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

TID = int(sys.argv[1]) if len(sys.argv) > 1 else 262
probe = build_groundtruth_probe(); m = probe.model
tests = generate_test_programs(); src, exp, _ = tests[TID]
bc, _ = compile_c(src)
print(f"id {TID} src={src!r} exp={exp}")
ctx = probe._final_context(bc); pl = len(probe._build_context(bc))

REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
        int(Token.REG_SP): "SP", int(Token.REG_BP): "BP"}
i = pl
step = -1
while i < len(ctx):
    t = ctx[i]
    if t == int(Token.STEP_END):
        step += 1
    if t in REGS and i + 4 < len(ctx):
        val = sum((ctx[i + 1 + j] & 0xFF) << (8 * j) for j in range(4))
        bytes_ = [hex(ctx[i + 1 + j] & 0xFF) for j in range(4)]
        print(f"  step{step if step >= 0 else 0} {REGS[t]:2s} = {val:#010x} "
              f"({val}) bytes={bytes_}")
    i += 1
out, code = probe.emitted_result(bc)
print(f"emitted exit_code = {code} (expected {exp})")
