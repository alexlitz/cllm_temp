#!/usr/bin/env python3
"""Trace ENT-step SP byte0 + byte1 emission for id 262 in the BROKEN neural run
AND a CLEAN teacher-forced run.  Reads FETCH_LO/HI (the relayed ENT immediate),
MARK_SP, OP_ENT, HAS_SE at the SP marker row (predicts byte0) and the SP byte0
row (predicts byte1), plus the OUTPUT band, to see whether the ENT SP frame
rules fire and where SP byte1=0xff is (or isn't) produced.

spec_k=0, hook-free.
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

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}
probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device
OUT_LO = int(dp["OUTPUT_LO"]); OHTS = int(dp["OUTPUT_HI_THIS_STEP"])
FL = int(dp["FETCH_LO"]); FH = int(dp["FETCH_HI"])
gates = {"MARK_SP": dp["MARK_SP"], "OP_ENT": dp["OP_ENT"], "HAS_SE": dp["HAS_SE"],
         "IS_BYTE": dp["IS_BYTE"], "BYTE_INDEX_0": dp.get("BYTE_INDEX_0")}

def b4(v):
    v &= 0xFFFFFFFF
    return [(v >> (8 * j)) & 0xFF for j in range(4)]

def band(row, base, lab):
    return " ".join(f"{lab}+{k}={float(row[base+k]):+.1f}"
                    for k in range(16) if abs(float(row[base+k])) > 0.5)

idx = 262
tests = generate_test_programs()
src, exp, _ = tests[idx]
bc, data = compile_c(src)
st = SymbolicDeclarativeProgramRunner().run(list(bc), data, max_steps=2000)

def find_sp(ctx, prompt_len, want_step):
    step = 0; i = prompt_len
    while i < len(ctx):
        nm = MARKERS.get(ctx[i])
        if nm == "STEP_END":
            step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if step == want_step and nm == "SP":
                return i
            i += 5; continue
        i += 1
    return None

def analyze(ctx, prompt_len, label):
    sp_mk = find_sp(ctx, prompt_len, 1)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    print(f"\n==== {label} ; step1 SP marker @ {sp_mk} ====")
    bs = [ctx[sp_mk + 1 + j] & 0xFF for j in range(4)]
    print(f"emitted step1 SP bytes = {[hex(b) for b in bs]}")
    for read_block in (6, 28):
        with torch.no_grad():
            r = m.forward(padded, stop_after_block=read_block)[0].float()
        # SP marker row (predicts byte0)
        rm = r[sp_mk]
        # SP byte0 row (predicts byte1)
        rb0 = r[sp_mk + 1]
        print(f"  -- after block {read_block} --")
        g = "  ".join(f"{k}={float(rm[int(d)]):+.2f}" for k, d in gates.items() if d is not None)
        print(f"   SP-marker-row gates: {g}")
        print(f"   SP-marker-row FETCH: {band(rm, FL, 'FL')}  {band(rm, FH, 'FH')}")
        print(f"   SP-marker-row OUT:   {band(rm, OUT_LO, 'LO')}  {band(rm, OHTS, 'HI')}")
        print(f"   SP-byte0-row OUT:    {band(rb0, OUT_LO, 'LO')}  {band(rb0, OHTS, 'HI')}")
        print(f"   SP-byte0-row FETCH:  {band(rb0, FL, 'FL')}  {band(rb0, FH, 'FH')}")

# BROKEN neural run
ctx_b = probe._final_context(bc)
pl = len(probe._build_context(bc))
analyze(ctx_b, pl, "BROKEN neural run")

# CLEAN teacher-forced run
ctx_c = list(probe._build_context(bc))
for tr in st.trace:
    for marker, val in ((Token.REG_PC, tr.pc_after), (Token.REG_AX, tr.ax_after),
                        (Token.REG_SP, tr.sp_after), (Token.REG_BP, tr.bp_after)):
        ctx_c.append(int(marker))
        for byte in b4(val):
            ctx_c.append(byte)
    ctx_c.append(int(Token.STEP_END))
analyze(ctx_c, pl, "CLEAN teacher-forced")
