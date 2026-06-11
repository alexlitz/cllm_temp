#!/usr/bin/env python3
"""Print l16_ent_nested_bp_byte0_d8 condition residuals at the ENT BP byte0 row
and the LEA AX byte0 row (block-29 output), to see why it fires at both."""
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

def d(n):
    if "+" in n:
        b, o = n.rsplit("+", 1); return dp[b] + int(o)
    return dp[n]

# current rule conditions (name, weight)
CONDS = [
    ("OP_ENT", 100.0), ("MARK_BP", 1.0), ("HAS_SE", 1.0),
    ("OUTPUT_HI_THIS_STEP+0", 100.0), ("IS_BYTE", -1e9),
    ("MARK_PC", -1e6), ("MARK_AX", -1e6), ("MARK_SP", -1e6),
    ("MARK_STACK0", -1e6), ("MARK_MEM", -1e6),
]
THR = 2000.0

def run(idx):
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
        r = m.forward(padded, stop_after_block=29)[0].float()
    print(f"\n### id={idx} {src[:40]!r} exp={exp} ###")
    tg = []
    if (1, "BP") in rows: tg.append(("step1 ENT BP byte0", rows[(1, "BP")]))
    if (2, "AX") in rows: tg.append(("step2 LEA AX byte0", rows[(2, "AX")]))
    for label, prow in tg:
        row = r[prow]; score = 0.0; parts = []
        for nm, w in CONDS:
            v = float(row[int(d(nm))]); c = v * w; score += c
            if abs(c) > 0.5 or nm in ("OP_ENT", "MARK_BP", "MARK_AX",
                                      "OUTPUT_HI_THIS_STEP+0"):
                parts.append(f"{nm}={v:+.3f}(={c:+.1f})")
        print(f"  {label}: score={score:+.2f} thr={THR} FIRES={score>=THR}")
        print(f"      {'  '.join(parts)}")

for i in [262]:
    run(i)
