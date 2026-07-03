#!/usr/bin/env python3
"""Block-by-block genesis of the LEA AX byte0 corruption + the BP byte0 0xd8.

For each physical block, read the FULL OUTPUT_LO/OUTPUT_HI nibble decode and a
few marker/ALU dims at the LEA AX byte0 row and the ENT BP byte0 row.
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

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}
probe = build_groundtruth_probe(); m = probe.model; dp = m.dim_positions
dev = next(m.parameters()).device; bl = probe.block_layer_map()
idx = 262; tests = generate_test_programs(); src, exp, _ = tests[idx]
bc, _ = compile_c(src); ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
rows = {}; step = 0; i = pl
while i < len(ctx):
    t = ctx[i]; nm = MARKERS.get(t)
    if nm == "STEP_END": step += 1; i += 1; continue
    if nm in ("PC", "AX", "SP", "BP"): rows[(step, nm)] = i; i += 5; continue
    i += 1
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
LO = dp["OUTPUT_LO"]; HI = dp["OUTPUT_HI_THIS_STEP"]
ALO = dp["ALU_LO"]; AHI = dp["ALU_HI"]
def decode(row):
    lo = [float(row[LO + k]) for k in range(16)]; hi = [float(row[HI + k]) for k in range(16)]
    la = max(range(16), key=lambda k: lo[k]); ha = max(range(16), key=lambda k: hi[k])
    return (ha << 4) | la, lo[la], hi[ha]
def adec(row):
    lo = [float(row[ALO + k]) for k in range(16)]; hi = [float(row[AHI + k]) for k in range(16)]
    la = max(range(16), key=lambda k: lo[k]); ha = max(range(16), key=lambda k: hi[k])
    return (ha << 4) | la, lo[la], hi[ha]
def mk(row):
    return (float(row[dp["MARK_AX"]]), float(row[dp["MARK_BP"]]),
            float(row[dp["OP_LEA"]]), float(row[dp["CMP"] + 7]))

for label, prow, want in [("ENT BP byte0", rows[(1, "BP")], 0xf0),
                          ("LEA AX byte0", rows[(2, "AX")], 0xe8)]:
    print(f"\n===== {label} row={prow} want=0x{want:02x} =====")
    print(f"{'phys':>4}{'log':>4} {'OUTdec':>7} {'LOval':>10} {'HIval':>10} "
          f"{'ALUdec':>7} {'mAX':>6} {'mBP':>6} {'LEA':>6} {'CMP7':>6}")
    with torch.no_grad():
        for phys in range(len(m.blocks)):
            r = m.forward(padded, stop_after_block=phys)[0].float()
            byte, lv, hv = decode(r[prow]); ab, _, _ = adec(r[prow])
            mAX, mBP, lea, cmp7 = mk(r[prow])
            print(f"{phys:>4}{bl[phys]['logical']:>4} 0x{byte:02x}    "
                  f"{lv:>10.1f} {hv:>10.1f} 0x{ab:02x}    "
                  f"{mAX:>6.2f} {mBP:>6.2f} {lea:>6.2f} {cmp7:>6.2f}")
