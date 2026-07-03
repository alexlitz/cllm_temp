#!/usr/bin/env python3
"""Dump OUTPUT_HI_THIS_STEP+0 / +15 and OUTPUT_LO+0 at EVERY BP marker row,
on TEACHER-FORCED clean state, for the initial-only program (262) and the
recursive program (702), to find a discriminator that separates the genuine
nested ENT (BP byte0=0xd8) from initial ENT and from all other BP rows
(where l16_ent_nested_bp_byte0_d8 must NOT fire).

spec_k=0, hook-free.  Residual read AFTER block 28 (input to the L16/L20 rule).
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
from neural_vm.unified_compiler.symbolic_program import SymbolicDeclarativeProgramRunner  # noqa

probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device
OUT_LO = int(dp["OUTPUT_LO"]); OHTS = int(dp["OUTPUT_HI_THIS_STEP"])
OPENT = int(dp["OP_ENT"])
L16_IN = 28

def b4(v):
    v &= 0xFFFFFFFF
    return [(v >> (8 * j)) & 0xFF for j in range(4)]

def run(idx):
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    st = SymbolicDeclarativeProgramRunner().run(list(bc), data, max_steps=2000)
    ctx = list(probe._build_context(bc))
    bp_rows = []
    for tr in st.trace:
        for marker, val in ((Token.REG_PC, tr.pc_after), (Token.REG_AX, tr.ax_after),
                            (Token.REG_SP, tr.sp_after), (Token.REG_BP, tr.bp_after)):
            ctx.append(int(marker))
            if marker == Token.REG_BP:
                bp_rows.append((tr.step, tr.name, len(ctx) - 1, tr.bp_after))
            for byte in b4(val):
                ctx.append(byte)
        ctx.append(int(Token.STEP_END))
        if len(ctx) > 900:
            break
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=L16_IN)[0].float()
    print(f"\n### id={idx} {desc} (clean teacher-forced) — BP rows ###")
    print(f"{'ostep':>5} {'op':6s} {'bp0':>4} {'OP_ENT':>7} {'LO+0':>7} {'HI+0':>7} {'HI+15':>7}  need_d8")
    for ostep, op, mk, bp in bp_rows:
        row = r[mk]
        bp0 = b4(bp)[0]
        need = "D8" if bp0 == 0xd8 else ""
        print(f"{ostep:>5} {op:6s} 0x{bp0:02x} {float(row[OPENT]):>7.2f} "
              f"{float(row[OUT_LO]):>7.2f} {float(row[OHTS]):>7.2f} "
              f"{float(row[OHTS+15]):>7.2f}  {need}")

run(262)
run(702)
run(700)  # rec_factorial_0 (0!) — base case, fewer recursion levels
