#!/usr/bin/env python3
"""Why does tail_sp_marker_byte0_f8_from_initial_stack_exact.lo.lane_0 (block-35
unit ~1110) fire at the id 262 ENT-step SP byte2/byte3 rows despite MARK_SP=10
+ OP_ENT=-1e6 blockers? Print the per-condition contribution at:
  - the SP MARKER row (intended firing site, byte0)
  - the SP byte1 row (predicts byte2 — the corrupted one)
  - the SP byte2 row (predicts byte3 — the corrupted one)
reading the residual at block 34 (the FFN input). spec_k=0, hook-free.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_CSR_INFERENCE"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END"}

# Conditions copied from tail_sp_marker_byte0_f8_from_initial_stack_exact.
COND = [
    ("MARK_SP", 10.0), ("H1+2", 0.01), ("H1+9", 0.01),
    ("H1+0", -1e9), ("H1+1", -1e9), ("H1+3", -1e9), ("H1+4", -1e9),
    ("CMP+4", 0.5), ("SP_BYTE0_IS_F8", 0.001), ("IN_STEP_FRESH", 0.001),
    ("OUTPUT_HI_THIS_STEP+14", -1000.0), ("ALU_LO+14", -1000.0),
    ("OP_IMM", -1e9), ("OP_ENT", -1e6),
    ("MARK_AX", -1e9), ("MARK_PC", -1e9), ("MARK_BP", -1e9),
    ("MARK_STACK0", -1e9), ("MARK_MEM", -1e9), ("OP_JSR", -1e6),
    ("IS_BYTE", -100.0), ("HAS_SE", 10.0),
    ("NEXT_PC", -1e6), ("NEXT_AX", -1000.0), ("NEXT_SP", -1e6),
    ("NEXT_BP", -1e6), ("NEXT_STACK0", -1e6), ("NEXT_MEM", -1e6),
    ("NEXT_SE", -1e6),
]
THRESHOLD = 20.04


def resolve(dp, name):
    if "+" in name:
        base, off = name.rsplit("+", 1)
        return int(dp[base]) + int(off)
    return int(dp[name])


def find_sp_marker(ctx, prompt_len, want_step):
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


def main():
    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device

    idx = 262
    tests = generate_test_programs()
    src, exp, _ = tests[idx]
    bc, data = compile_c(src)
    prompt_len = len(probe._build_context(bc))
    ctx = probe._final_context(bc, max_steps=9)
    sp = find_sp_marker(ctx, prompt_len, 1)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r34 = m.forward(padded, stop_after_block=34)[0].float()

    rows = {"SP-marker(b0)": sp, "SP-b1-row(pred b2)": sp + 2, "SP-b2-row(pred b3)": sp + 3}
    for label, rw in rows.items():
        row = r34[rw]
        score = 0.0
        terms = []
        for nm, w in COND:
            try:
                v = float(row[resolve(dp, nm)])
            except KeyError:
                v = 0.0
            c = w * v
            score += c
            if abs(c) > 0.5:
                terms.append((nm, v, w, c))
        print(f"\n=== {label}: total activation score = {score:+.3f}  (threshold {THRESHOLD}; "
              f"{'FIRES' if score >= THRESHOLD else 'no-fire'}) ===")
        for nm, v, w, c in sorted(terms, key=lambda t: -abs(t[3]))[:12]:
            print(f"   {nm:24s} val={v:+.4f} x w={w:+.1f} = {c:+.3f}")


if __name__ == "__main__":
    main()
