#!/usr/bin/env python3
"""Read l16_ent_nested_bp_byte0_d8 conditions at EVERY ENT BP byte0 row of a
recursive program (genuine nested) vs id 262 (initial only). Find the real
nested-vs-initial discriminator. spec_k=0, hook-free."""
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

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}
probe = build_groundtruth_probe(); m = probe.model; dp = m.dim_positions
dev = next(m.parameters()).device
def d(n):
    if "+" in n:
        b, o = n.rsplit("+", 1); return dp[b] + int(o)
    return dp[n]
DIMS = ["OP_ENT", "OUTPUT_HI_THIS_STEP+0", "OUTPUT_LO+8",
        "OUTPUT_HI_THIS_STEP+13", "OUTPUT_LO+0", "OUTPUT_HI_THIS_STEP+15",
        "MARK_BP", "HAS_SE"]

def run(idx, maxsteps=30):
    tests = generate_test_programs(); src, exp, _ = tests[idx]
    bc, _ = compile_c(src); ctx = probe._final_context(bc)
    pl = len(probe._build_context(bc))
    rows = []; step = 0; i = pl
    while i < len(ctx):
        t = ctx[i]; nm = MARKERS.get(t)
        if nm == "STEP_END": step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if nm == "BP":
                bs = [ctx[i + 1 + j] & 0xFF for j in range(4)]
                rows.append((step, i, bs))
            i += 5; continue
        i += 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=29)[0].float()
    print(f"\n### id={idx} exp={exp} {src[:46]!r} ###")
    print(f"  {'step':>4} {'bp_b0':>6} " + " ".join(f"{n.replace('OUTPUT_','O').replace('_THIS_STEP',''):>10}" for n in DIMS))
    for step, mk, bs in rows[:maxsteps]:
        vals = [float(r[mk, int(d(n))]) for n in DIMS]
        print(f"  {step:>4}  0x{bs[0]:02x}  " + " ".join(f"{v:>10.2f}" for v in vals))

run(262)   # initial only (want 0xf0, d8 must NOT fire)
run(704, maxsteps=24)   # 4! recursive: step1 main ENT=0xfff0 (NO fire),
                        # step5 factorial ENT=0xffd8 (d8 MUST fire)
