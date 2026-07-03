#!/usr/bin/env python3
"""Read OUTPUT_HI_THIS_STEP+15 at the BP marker row for the GENUINE nested ENT
(id 702 oracle step 5, BP->0xffd8) using a TEACHER-FORCED clean context, so the
nested-vs-initial discriminator value is measured on correct state (not the
broken neural run).

spec_k=0, hook-free.  We build the context as the runner does, then append the
ORACLE-correct register tokens for each step up to and including the nested ENT
step.  The model is run truncated to L16's input block; we read OHTS+15 at the
nested BP marker row and at the initial BP marker row to confirm the
discriminator separation (initial positive, nested ~0).
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
OHTS15 = int(dp["OUTPUT_HI_THIS_STEP"]) + 15
OPENT = int(dp["OP_ENT"]); MARKBP = int(dp["MARK_BP"])
L16_BLOCK = 29

def b4(v):
    v &= 0xFFFFFFFF
    return [(v >> (8 * j)) & 0xFF for j in range(4)]

idx = 702
tests = generate_test_programs()
src, exp, desc = tests[idx]
bc, data = compile_c(src)
r = SymbolicDeclarativeProgramRunner()
st = r.run(list(bc), data, max_steps=2000)

# Build a teacher-forced context: prompt + per-step PC/AX/SP/BP + STEP_END.
ctx = list(probe._build_context(bc))
bp_rows = []  # (oracle_step, op, marker_idx, bp_after)
for tr in st.trace:
    pc, ax, sp, bp = tr.pc_after, tr.ax_after, tr.sp_after, tr.bp_after
    for marker, val in ((Token.REG_PC, pc), (Token.REG_AX, ax),
                        (Token.REG_SP, sp), (Token.REG_BP, bp)):
        ctx.append(int(marker))
        if marker == Token.REG_BP:
            bp_rows.append((tr.step, tr.name, len(ctx) - 1, bp))
        for byte in b4(val):
            ctx.append(byte)
    ctx.append(int(Token.STEP_END))
    if tr.name == "ENT" and tr.step >= 5:
        break  # stop just after the first genuine nested ENT (step 5)

padded = torch.tensor([ctx], dtype=torch.long, device=dev)
OUT_LO = int(dp["OUTPUT_LO"]); OUT_HI = int(dp["OUTPUT_HI"])

print(f"id {idx} {desc} (TEACHER-FORCED clean state)")
print(f"{'ostep':>5} {'op':6s} {'bp_byte0':>8} {'OP_ENT':>8} {'MARK_BP':>8} {'OHTS+15':>9}")
with torch.no_grad():
    res = m.forward(padded, stop_after_block=L16_BLOCK - 1)[0].float()
for ostep, op, mk, bp in bp_rows:
    row = res[mk]
    bp0 = b4(bp)[0]
    print(f"{ostep:>5} {op:6s} 0x{bp0:02x}     {float(row[OPENT]):>8.2f} "
          f"{float(row[MARKBP]):>8.2f} {float(row[OHTS15]):>9.3f}")

# BP byte0 OUTPUT band at L6 (phys 6) and just-before-L16 (phys 28) for the
# initial (ostep 1) and nested (ostep 5) ENT BP marker rows.
def band(row):
    parts = []
    for k in range(16):
        v = float(row[OUT_LO + k])
        if abs(v) > 0.5: parts.append(f"LO+{k}={v:+.1f}")
    for k in range(16):
        v = float(row[OUT_HI + k])
        if abs(v) > 0.5: parts.append(f"HI+{k}={v:+.1f}")
    return " ".join(parts)

want = {1: ("initial", 6), 5: ("nested", 6)}
for phys in (6, 28):
    with torch.no_grad():
        rr = m.forward(padded, stop_after_block=phys)[0].float()
    print(f"\n--- BP byte0 OUTPUT band after phys{phys} ---")
    for ostep, op, mk, bp in bp_rows:
        if ostep in (1, 5) and op == "ENT":
            print(f"  ostep{ostep} ({'initial' if ostep==1 else 'nested'}) "
                  f"bp_byte0=0x{b4(bp)[0]:02x}: {band(rr[mk])}")
