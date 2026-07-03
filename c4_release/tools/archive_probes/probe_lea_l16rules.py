#!/usr/bin/env python3
"""Evaluate the l16 PSH-mem-addr0 rule conditions at the ENT BP byte0 row and
the LEA AX byte0 row for id 262, at the L20-input residual (block 29 output).

spec_k=0, hook-free. Identifies which L20 rule mis-fires the d8 / e0 / runaway.
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

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}
probe = build_groundtruth_probe(); m = probe.model; dp = m.dim_positions
dev = next(m.parameters()).device

idx = 262
tests = generate_test_programs(); src, exp, _ = tests[idx]
bc, _ = compile_c(src); ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
rows = {}; step = 0; i = pl
while i < len(ctx):
    t = ctx[i]; nm = MARKERS.get(t)
    if nm == "STEP_END": step += 1; i += 1; continue
    if nm in ("PC", "AX", "SP", "BP"): rows[(step, nm)] = i; i += 5; continue
    i += 1
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
with torch.no_grad():
    r29 = m.forward(padded, stop_after_block=29)[0].float()  # input to block 30 = L20

def d(name):
    if "+" in name:
        base, off = name.rsplit("+", 1); return dp[base] + int(off)
    return dp[name]

S = 100.0
shared = [
    ("PSH_AT_SP", 1.0), ("OP_JSR", -1000.0), ("OP_ENT", -1000.0),
    ("MARK_MEM", 1.0), ("MEM_STORE", 1.0), ("HAS_SE", 0.5),
    ("IS_BYTE", -1_000_000.0), ("MARK_PC", -1_000_000.0),
    ("MARK_AX", -1_000_000.0), ("MARK_SP", -1_000_000.0),
    ("MARK_BP", -1_000_000.0), ("MARK_STACK0", -1_000_000.0),
]
RULES = {
    "force_d8_from_l14_evidence": (shared + [
        ("H1+4", 1.0), ("OUTPUT_LO+8", 1.0), ("OUTPUT_HI_THIS_STEP+13", 1.0)], 8.0),
    "e0_from_addr_b0": (shared + [
        ("MEM_ADDR_SRC", 1.0), ("ADDR_B0_LO+0", 1.0), ("ADDR_B0_HI+14", 1.0)], 8.5),
    "e0_from_sp_no_addr_src": (shared + [
        ("MEM_ADDR_SRC", -1000.0), ("H1+4", 1.0), ("H1+11", 1.0), ("CMP+0", 1.0),
        ("ALU_LO+8", 2.0), ("ALU_LO+7", -10.0), ("ALU_LO+10", -10.0),
        ("ALU_LO+14", -10.0), ("OP_ENT", -1_000_000.0), ("OP_LEV", -1_000_000.0),
        ("OP_IMM", -1e9)], 8.5),
    "restore_lo_8": (shared + [("OUTPUT_LO+8", 1.0)], 5.9),
}

for label, prow in [("ENT BP byte0", rows[(1, "BP")]),
                    ("LEA AX byte0", rows[(2, "AX")])]:
    print(f"\n############ {label}  row={prow} ############")
    row = r29[prow]
    for rname, (conds, thr) in RULES.items():
        score = 0.0; big = []
        for nm, w in conds:
            v = float(row[int(d(nm))]); c = v * w; score += c
            if abs(c) > 1.0: big.append(f"{nm}={v:+.3f}(*{w:g}={c:+.1f})")
        fires = score >= thr
        flag = "  <<< FIRES" if fires else ""
        print(f"  {rname:28s} score={score:+12.2f} thr={thr} FIRES={fires}{flag}")
        if fires or abs(score) < 50:
            print(f"      {'  '.join(big[:8])}")
