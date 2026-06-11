#!/usr/bin/env python3
"""Trace the step-1 ENT BP byte0 and step-2 LEA AX byte computation for id 262.

spec_k=0, hook-free. Reads:
  - the l16_ent_nested_bp_byte0_d8 discriminator (OUTPUT_HI_THIS_STEP+0) at the
    BP byte0 prediction row, AFTER block 28 (L20 input), and whether the rule
    would fire under its CURRENT conditions.
  - the OUTPUT_LO / OUTPUT_HI_THIS_STEP band across blocks at the BP byte0 and
    the LEA AX byte0/byte1 prediction rows.
  - the final LM-head argmax + top logits at each of those prediction rows.
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
bl = probe.block_layer_map()

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
tests = generate_test_programs()
src, exp, _ = tests[idx]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))

rows = {}
step = 0; i = pl
while i < len(ctx):
    t = ctx[i]; nm = MARKERS.get(t)
    if nm == "STEP_END":
        step += 1; i += 1; continue
    if nm in ("PC", "AX", "SP", "BP"):
        rows[(step, nm)] = i
        i += 5; continue
    i += 1

def emitted_bytes(mk):
    return [ctx[mk + 1 + j] & 0xFF for j in range(4)]

padded = torch.tensor([ctx], dtype=torch.long, device=dev)

targets = []
if (1, "BP") in rows:
    targets.append(("step1 ENT BP byte0", rows[(1, "BP")], 0xf0))
if (2, "AX") in rows:
    mk = rows[(2, "AX")]
    targets.append(("step2 LEA AX byte0", mk, 0xe8))
    targets.append(("step2 LEA AX byte1", mk + 1, 0xff))
    targets.append(("step2 LEA AX byte2", mk + 2, 0xff))
    targets.append(("step2 LEA AX byte3", mk + 3, 0xff))

print(f"=== id={idx} exp={exp} ===")
print(f"BP step1 emitted bytes = {emitted_bytes(rows[(1,'BP')])}")
print(f"AX step2 emitted bytes = {emitted_bytes(rows[(2,'AX')])}")

L20_INPUT_BLOCK = 28
with torch.no_grad():
    r28 = m.forward(padded, stop_after_block=L20_INPUT_BLOCK - 1)[0].float()
bp_row = rows[(1, "BP")]
print(f"\n=== l16_ent_nested_bp_byte0_d8 condition check @ BP byte0 row "
      f"{bp_row} (block {L20_INPUT_BLOCK-1} output) ===")
COND = [
    ("OP_ENT", dp["OP_ENT"], 100.0),
    ("MARK_BP", dp["MARK_BP"], 1.0),
    ("HAS_SE", dp["HAS_SE"], 1.0),
    ("OUTPUT_HI_THIS_STEP+0", dp["OUTPUT_HI_THIS_STEP"] + 0, 100.0),
    ("IS_BYTE", dp["IS_BYTE"], -1e9),
    ("MARK_PC", dp["MARK_PC"], -1e6),
    ("MARK_AX", dp["MARK_AX"], -1e6),
    ("MARK_SP", dp["MARK_SP"], -1e6),
    ("MARK_STACK0", dp["MARK_STACK0"], -1e6),
    ("MARK_MEM", dp["MARK_MEM"], -1e6),
]
THR = 2000.0
score = 0.0
for nm, d, w in COND:
    v = float(r28[bp_row, int(d)])
    c = v * w
    score += c
    print(f"  {nm:28s} resid={v:+10.4f} *{w:<10g} = {c:+.3f}")
print(f"  ----> score={score:.3f} thr={THR} FIRES={score >= THR}")

OUT_LO = dp["OUTPUT_LO"]
OUT_HI = dp["OUTPUT_HI_THIS_STEP"]
def band(row, base):
    return [float(row[base + k]) for k in range(16)]

nb = len(m.blocks)
print(f"dim_positions: OUTPUT_LO={OUT_LO} OUTPUT_HI_THIS_STEP={OUT_HI} "
      f"(overlap={OUT_LO==OUT_HI})")
with torch.no_grad():
    last_resid = m.forward(padded, stop_after_block=nb - 1)[0].float()
    logits_all = probe._forward_logits(list(ctx))  # [S, V]
print(f"\n=== OUTPUT bands (FINAL residual) + LM head logits at pred rows ===")
for label, prow, want in targets:
    lo = band(last_resid[prow], OUT_LO)
    hi = band(last_resid[prow], OUT_HI)
    lo_arg = max(range(16), key=lambda k: lo[k])
    hi_arg = max(range(16), key=lambda k: hi[k])
    print(f"\n--- {label}: pred_row={prow} want=0x{want:02x} ---")
    print(f"  OUTPUT_LO argmax nibble=0x{lo_arg:x}(val {lo[lo_arg]:+.1f}) "
          f"OUTPUT_HI argmax nibble=0x{hi_arg:x}(val {hi[hi_arg]:+.1f}) "
          f"=> 0x{(hi_arg<<4)|lo_arg:02x}")
    logits = logits_all[prow]
    top = torch.topk(logits, 6)
    pairs = [(int(top.indices[j]), float(top.values[j])) for j in range(6)]
    print(f"  LM head argmax tok={int(logits.argmax())}  top: "
          + "  ".join(f"t{t}={v:.0f}" for t, v in pairs))

print(f"\n=== block sweep of d8-defining dims @ BP byte0 row {bp_row} ===")
print(f"{'phys':>4} {'log':>3} {'LO+0':>8} {'LO+8':>8} {'HI+15':>8} {'HI+13':>8}")
with torch.no_grad():
    for phys in range(nb):
        r = m.forward(padded, stop_after_block=phys)[0].float()
        v0 = float(r[bp_row, OUT_LO + 0]); v8 = float(r[bp_row, OUT_LO + 8])
        h15 = float(r[bp_row, OUT_HI + 15]); h13 = float(r[bp_row, OUT_HI + 13])
        if abs(v0) + abs(v8) + abs(h15) + abs(h13) > 0.01:
            print(f"{phys:>4} {bl[phys]['logical']:>3} {v0:>8.3f} {v8:>8.3f} "
                  f"{h15:>8.3f} {h13:>8.3f}")
