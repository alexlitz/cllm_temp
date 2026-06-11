#!/usr/bin/env python3
"""Pin the genesis discriminators for the two step-2 LEA divergences (id 262):
  (A) LEA PC byte0 (exp 0x2a, neu 0x00)  -- the desync trigger.
  (B) LEA AX byte1 (exp 0xff, neu 0x00)  -- sign-extension missing.

Reads the l16 LEA byte1 rule's gate dims (CMP+7, BYTE_INDEX_0/1, H1+1,
IS_BYTE, HAS_SE, OP_*) at the byte0 AND byte1 prediction rows, and the
L18/L19 OUTPUT_LO+0 flood that overwrites the PC byte0 0x2a.

spec_k=0, hook-free, READ-ONLY.  python tools/probe_var_lea_pc_disc.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTHONUNBUFFERED", "1")
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


def D(name):
    if "+" in name:
        b, o = name.rsplit("+", 1); return dp[b] + int(o)
    return dp[name]


DISC = ["CMP+7", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "H1+1", "IS_BYTE", "HAS_SE", "MARK_AX", "MARK_PC", "OP_LEA", "OP_ENT",
        "OP_JSR", "MARK_STACK0", "MARK_BP", "MARK_SP", "MARK_MEM"]

# Step-2 LEA: PC marker at rows[(2,'PC')], AX marker at rows[(2,'AX')].
pc_m = rows[(2, "PC")]; ax_m = rows[(2, "AX")]
targets = [
    ("LEA PC byte0 predrow", pc_m + 1 + 0 - 1),
    ("LEA AX byte0 predrow", ax_m + 1 + 0 - 1),
    ("LEA AX byte1 predrow", ax_m + 1 + 1 - 1),
]

print(f"=== id 262 step-2 LEA discriminators @ L16-input (block 27 out) ===")
with torch.no_grad():
    r27 = m.forward(padded, stop_after_block=27)[0].float()  # input to L16-ish
for label, prow in targets:
    print(f"\n-- {label} row={prow} --")
    vals = {nm: float(r27[prow][D(nm)]) for nm in DISC}
    print("  " + "  ".join(f"{nm}={vals[nm]:+.2f}" for nm in DISC))
    # Replicate the l16_lea_local_ax_byte1_ff gate score:
    # conditions: CMP+7*1 + HAS_SE*1 + H1+1*1 + IS_BYTE*1 + BYTE_INDEX_0*1
    #             + MARK_*(-10 each) ; threshold 4.5
    score = (vals["CMP+7"] + vals["HAS_SE"] + vals["H1+1"] + vals["IS_BYTE"]
             + vals["BYTE_INDEX_0"]
             - 10*(vals["MARK_PC"] + vals["MARK_AX"] + vals["MARK_SP"]
                   + vals["MARK_BP"] + vals["MARK_STACK0"] + vals["MARK_MEM"]))
    print(f"  l16_lea_local_ax_byte1_ff gate score={score:+.2f} thr=4.5 "
          f"FIRES={score >= 4.5}")

# PC byte0 flood: track OUTPUT_LO+0 and AX_LO+0 across blocks 26..30
print(f"\n=== PC byte0 OUTPUT_LO+0 / ALU_LO+0 flood (blocks 26..32) ===")
prow = pc_m + 1 + 0 - 1
LO0 = dp["OUTPUT_LO"] + 0; ALO0 = dp["ALU_LO"] + 0
HI0 = dp["OUTPUT_HI_THIS_STEP"] + 0
LO10 = dp["OUTPUT_LO"] + 10; HI2 = dp["OUTPUT_HI_THIS_STEP"] + 2
print(f"{'phys':>4}{'log':>4} {'OUT_LO0':>10}{'OUT_LO10':>10}{'OUT_HI0':>10}"
      f"{'OUT_HI2':>10}{'ALU_LO0':>10}{'OP_ENT':>8}{'OP_JSR':>8}")
with torch.no_grad():
    for phys in range(25, 33):
        r = m.forward(padded, stop_after_block=phys)[0].float()
        print(f"{phys:>4}{bl[phys]['logical']:>4} {float(r[prow][LO0]):>10.2f}"
              f"{float(r[prow][LO10]):>10.2f}{float(r[prow][HI0]):>10.2f}"
              f"{float(r[prow][HI2]):>10.2f}{float(r[prow][ALO0]):>10.2f}"
              f"{float(r[prow][dp['OP_ENT']]):>8.2f}"
              f"{float(r[prow][dp['OP_JSR']]):>8.2f}")
