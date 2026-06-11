#!/usr/bin/env python3
"""On the REAL broken run, read the candidate discriminator dims for the
l16_ent_nested_{sp,bp}_byte0_d8 rules at the ENT-step SP/BP byte0 marker rows,
for id 262 (var) AND test_lea_basic (a PASSING bootstrap-ENT smoke test that
must not regress).

We need a dim that is decisively DIFFERENT between:
  - id 262 step1 ENT (initial, want byte0 SP=0xe8 BP=0xf0 — d8 rule must NOT fire)
  - test_lea_basic step0 ENT (want SP/BP correct — d8 rule must NOT fire)
  - (the genuine nested case fires d8 — not directly measurable in a clean run)

spec_k=0, hook-free.  Read AFTER block 28 (input to the L20/block-29 rule).
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
read = {
    "OP_ENT": dp["OP_ENT"], "MARK_SP": dp["MARK_SP"], "MARK_BP": dp["MARK_BP"],
    "HI+0": OHTS + 0, "HI+13": OHTS + 13, "HI+15": OHTS + 15,
    "LO+0": OUT_LO + 0, "LO+8": OUT_LO + 8,
    "FETCH_LO+0": dp["FETCH_LO"] + 0, "FETCH_LO+8": dp["FETCH_LO"] + 8,
}

def find_ent_step(bc):
    # first step whose opcode is ENT, in oracle order
    from neural_vm.unified_compiler.symbolic_program import SymbolicDeclarativeProgramRunner
    st = SymbolicDeclarativeProgramRunner().run(list(bc), b"", max_steps=2000)
    for tr in st.trace:
        if tr.name == "ENT":
            return tr.step
    return None

def analyze(bc, label):
    ent_step = find_ent_step(bc)
    ctx = probe._final_context(bc)
    pl = len(probe._build_context(bc))
    step = 0; i = pl; sp = bp = None
    while i < len(ctx):
        nm = MARK.get(ctx[i])
        if nm == "|": step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if step == ent_step and nm == "SP": sp = i
            if step == ent_step and nm == "BP": bp = i
            i += 5; continue
        i += 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=28)[0].float()
    print(f"\n### {label} — ENT at step {ent_step} ###")
    for lab, row in (("SP byte0", sp), ("BP byte0", bp)):
        if row is None: print(f"  {lab}: not found"); continue
        bs = [ctx[row + 1 + j] & 0xFF for j in range(4)] if row + 4 < len(ctx) else []
        vals = "  ".join(f"{k}={float(r[row][int(d)]):+.2f}" for k, d in read.items())
        print(f"  {lab} emitted=0x{(bs[0] if bs else 0):02x}: {vals}")

tests = generate_test_programs()
src, _, _ = tests[262]
analyze(compile_c(src)[0], "id262 var_simple_12 (initial ENT, want SP=0xe8 BP=0xf0)")

# test_lea_basic: ENT, IMM 0, LEA 2, EXIT  (bootstrap ENT at step0)
lea_bc = [int(Opcode.ENT) | (0 << 8), int(Opcode.IMM) | (0 << 8),
          int(Opcode.LEA) | (2 << 8), int(Opcode.EXIT)]
analyze(lea_bc, "test_lea_basic (bootstrap ENT, PASSING)")
