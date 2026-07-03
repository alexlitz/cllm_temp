#!/usr/bin/env python3
"""Find which physical-block-29 (L20) FFN units write the giant d8 pattern
(OUTPUT_LO+8 +, OUTPUT_HI_THIS_STEP+13 +) at the step-1 ENT BP marker row of
id 262.  spec_k=0, hook-free.

We read the block-29 FFN hidden activations at the BP-marker prediction row by
running the model to block 28 (the FFN input residual) and applying the
block-29 FFN forward manually (the FFN is a pure function of its input row).
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
PHYS = 29
LO8 = int(dp["OUTPUT_LO"]) + 8
HI13 = int(dp["OUTPUT_HI_THIS_STEP"]) + 13

idx = 262
tests = generate_test_programs()
src, exp, _ = tests[idx]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
prompt_len = len(probe._build_context(bc))
# locate step1 BP marker
step = 0; i = prompt_len; bp_mk = None
while i < len(ctx):
    nm = MARKERS.get(ctx[i])
    if nm == "STEP_END":
        step += 1; i += 1; continue
    if nm in ("PC", "AX", "SP", "BP"):
        if step == 1 and nm == "BP":
            bp_mk = i; break
        i += 5; continue
    i += 1
print(f"step1 BP marker @ {bp_mk}")

padded = torch.tensor([ctx], dtype=torch.long, device=dev)
with torch.no_grad():
    pre = m.forward(padded, stop_after_block=PHYS - 1)[0]  # [S, D]
row = pre[bp_mk].float()  # input residual to block 29 FFN

blk = m.blocks[PHYS]
ffn = blk.ffn
# Apply layernorm if the block uses pre-norm before FFN (read what forward does)
def dense(w):
    return w.to_dense().float() if hasattr(w, "to_dense") else w.float()
Wg = dense(ffn.W_gate); Wu = dense(ffn.W_up); Wd = dense(ffn.W_down)
# Try with raw row (FFN likely operates on normed input; approximate by raw —
# we only need RELATIVE per-unit contribution to LO+8/HI+13).
x = row
gate = Wg @ x
up = Wu @ x
act = torch.nn.functional.silu(gate) * up if Wg.shape[0] == Wu.shape[0] else gate * up
# per-unit contribution to LO+8 and HI+13
contrib_lo8 = Wd[LO8] * act
contrib_hi13 = Wd[HI13] * act
for dimname, contrib in (("LO+8", contrib_lo8), ("HI+13", contrib_hi13)):
    order = torch.argsort(contrib.abs(), descending=True)[:8]
    print(f"\nTop block-{PHYS} units writing {dimname} at step1 BP marker row:")
    for u in order.tolist():
        c = float(contrib[u])
        if abs(c) < 1.0: continue
        print(f"  unit {u:5d} contrib={c:+.2f} act={float(act[u]):+.4f} "
              f"Wd[{dimname}]={float(Wd[LO8 if dimname=='LO+8' else HI13][u]):+.4f}")
