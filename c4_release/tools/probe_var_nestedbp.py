#!/usr/bin/env python3
"""Read the exact condition dims of l16_ent_nested_bp_byte0_d8 at the BP byte0
row, for both the initial ENT (id 262) and a recursive program's ENT steps,
to find the nested-vs-initial discriminator.

spec_k=0, hook-free.  The rule (l16_ops.py:1447) reads:
  OP_ENT, MARK_BP, HAS_SE, OUTPUT_HI_THIS_STEP+15, IS_BYTE, MARK_{PC,AX,SP,STACK0,MEM}
and fires when score >= 20250, writing BP byte0 -> 0xd8.

The L16 ENT block is the physical block whose input we want (block 29 = L20
per probe; the rule lives in make_layer16_* which is L20 physically).  We read
the residual AFTER block 28 (input to block 29).
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

probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device

COND = {  # name -> (dim, weight)  for l16_ent_nested_bp_byte0_d8
    "OP_ENT": (dp["OP_ENT"], 100.0),
    "MARK_BP": (dp["MARK_BP"], 20000.0),
    "HAS_SE": (dp["HAS_SE"], 1.0),
    "OUTPUT_HI_THIS_STEP+15": (dp["OUTPUT_HI_THIS_STEP"] + 15, -50.0),
    "IS_BYTE": (dp["IS_BYTE"], -1_000_000_000.0),
    "MARK_PC": (dp["MARK_PC"], -1000.0),
    "MARK_AX": (dp["MARK_AX"], -1000.0),
    "MARK_SP": (dp["MARK_SP"], -1000.0),
    "MARK_STACK0": (dp["MARK_STACK0"], -1000.0),
    "MARK_MEM": (dp["MARK_MEM"], -1000.0),
}
THRESHOLD = 20250.0
L16_BLOCK = 29  # physical block of make_layer16_* (L20)

def analyze(idx, label):
    tests = generate_test_programs()
    src, exp, _ = tests[idx]
    bc, _ = compile_c(src)
    ctx = probe._final_context(bc)
    prompt_len = len(probe._build_context(bc))
    # find all BP markers per step
    step = 0; i = prompt_len; bp_rows = []
    while i < len(ctx):
        t = ctx[i]; nm = MARKERS.get(t)
        if nm == "STEP_END":
            step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if nm == "BP":
                bp_rows.append((step, i))
            i += 5; continue
        i += 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=L16_BLOCK - 1)[0].float()
    print(f"\n### id={idx} {label} exp={exp} bytecode_len={len(bc)} ###")
    for step, mk in bp_rows:
        row = r[mk]  # BP marker row (byte0 predicted from this row)
        bs = [ctx[mk + 1 + j] & 0xFF for j in range(4)]
        score = 0.0; parts = []
        for nm, (d, w) in COND.items():
            v = float(row[int(d)])
            c = v * w
            score += c
            if abs(c) > 0.5 or nm in ("MARK_BP", "OP_ENT", "OUTPUT_HI_THIS_STEP+15"):
                parts.append(f"{nm}={v:+.2f}(*{w:g}={c:+.1f})")
        fires = score >= THRESHOLD
        print(f"  step{step} BP marker@{mk} emitted_bp_byte0=0x{bs[0]:02x} "
              f"bp={bs} score={score:.1f} thr={THRESHOLD} FIRES={fires}")
        print(f"      {'  '.join(parts)}")

analyze(262, "var_simple_12 (initial ENT only)")
analyze(702, "rec_factorial_2 (5!, recursive - nested ENTs)")
