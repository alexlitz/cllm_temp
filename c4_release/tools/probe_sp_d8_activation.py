#!/usr/bin/env python3
"""Compute the EXACT l16_ent_nested_sp_byte0_d8 activation score at the
ENT-step SP byte0 marker row (block-28 input residual) for:
  - id262 var_simple_12 (initial ENT 8, want SP byte0=0xe8 -> d8 must NOT fire)
  - test_lea_basic (bootstrap ENT 0, want SP byte0=0xe0 -> d8 must NOT fire)
Read the rule's condition dims and reproduce the multi_way_and score so we can
calibrate a subtractive HI+15 discriminator. spec_k=0, hook-free.
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
from neural_vm.embedding import Opcode  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARK = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", int(Token.STEP_END): "|"}
probe = build_groundtruth_probe()
m = probe.model; dp = m.dim_positions; dev = next(m.parameters()).device
OHTS = int(dp["OUTPUT_HI_THIS_STEP"]); OUT_LO = int(dp["OUTPUT_LO"])
FH = int(dp["FETCH_HI"]); FL = int(dp["FETCH_LO"])

# Current l16_ent_nested_sp_byte0_d8 conditions (OP_ENT summed = 0.2+9.8=10.0):
COND = {
    "MARK_SP": (dp["MARK_SP"], 10.0),
    "HAS_SE": (dp["HAS_SE"], 1.0),
    "OP_ENT": (dp["OP_ENT"], 10.0),
    "IS_BYTE": (dp["IS_BYTE"], -1000.0),
    "MARK_AX": (dp["MARK_AX"], -1e6),
    "MARK_PC": (dp["MARK_PC"], -1e6),
    "MARK_BP": (dp["MARK_BP"], -1e6),
    "MARK_STACK0": (dp["MARK_STACK0"], -1e6),
    "MARK_MEM": (dp["MARK_MEM"], -1e6),
    "FETCH_LO+0": (FL + 0, 1.0),
    "FETCH_HI+0": (FH + 0, 1.0),
    "OUTPUT_HI_THIS_STEP+15": (OHTS + 15, 1.0),
}
THRESH = 70.0


def find_ent_steps(bc):
    from neural_vm.verification.symbolic_program import SymbolicDeclarativeProgramRunner
    st = SymbolicDeclarativeProgramRunner().run(list(bc), b"", max_steps=2000)
    return [tr.step for tr in st.trace if tr.name == "ENT"]


def find_ent_step(bc):
    s = find_ent_steps(bc)
    return s[0] if s else None


def analyze(bc, label, ent_step=None):
    if ent_step is None:
        ent_step = find_ent_step(bc)
    ctx = probe._final_context(bc)
    pl = len(probe._build_context(bc))
    step = 0; i = pl; sp = None
    while i < len(ctx):
        nm = MARK.get(ctx[i])
        if nm == "|": step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if step == ent_step and nm == "SP": sp = i
            i += 5; continue
        i += 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=28)[0].float()
    row = r[sp]
    bs = [ctx[sp + 1 + j] & 0xFF for j in range(4)]
    print(f"\n### {label} — ENT at step {ent_step}, SP byte0 emitted=0x{bs[0]:02x} ###")
    score = 0.0
    for nm, (d, w) in COND.items():
        v = float(row[int(d)])
        c = v * w
        score += c
        if abs(c) > 0.01 or nm in ("OUTPUT_HI_THIS_STEP+15", "FETCH_HI+0", "FETCH_LO+0"):
            print(f"  {nm:24s} val={v:+8.3f} w={w:+.1f} contrib={c:+10.2f}")
    print(f"  >>> SCORE={score:+.2f}  threshold={THRESH}  FIRES={score >= THRESH}")
    print(f"  [disc] HI+15={float(row[OHTS+15]):+.2f} HI+0={float(row[OHTS+0]):+.2f} "
          f"FETCH_HI+0={float(row[FH+0]):+.2f} FETCH_LO+8={float(row[FL+8]):+.2f}")


tests = generate_test_programs()
src, _, _ = tests[262]
analyze(compile_c(src)[0], "id262 var_simple_12 (initial ENT8, want 0xe8)")

# test_lea_basic: ENT 0, IMM 0, LEA 2, EXIT (bootstrap ENT at step0)
lea_bc = [int(Opcode.ENT) | (0 << 8), int(Opcode.IMM) | (0 << 8),
          int(Opcode.LEA) | (2 << 8), int(Opcode.EXIT)]
analyze(lea_bc, "test_lea_basic (bootstrap ENT0, want 0xe0)")

# GENUINELY NESTED: main calls helper f. f's ENT is entered with SP already
# lowered (pushed arg + return addr) -> SP saves at 0xffd8 -> d8 MUST fire.
nest_src = ("int f(int a){ int b; b=a; return b; } "
            "int main(){ int x; x=f(3); return x; }")
nest_bc = compile_c(nest_src)[0]
ent_steps = find_ent_steps(nest_bc)
print(f"\n=== nested program ENT steps (oracle): {ent_steps} ===")
for es in ent_steps:
    analyze(nest_bc, f"nested f/main ENT (step {es})", ent_step=es)
