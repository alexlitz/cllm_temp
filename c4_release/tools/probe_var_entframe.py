#!/usr/bin/env python3
"""Trace the ENT-step (step 1) SP byte1 and BP byte emissions for id 262.

spec_k=0, hook-free. Reads dims via probe.model.dim_positions.

For var_simple_12 (id 262) the ENT step's frame setup is wrong:
  SP byte1 should be 0xff (SP stays in the 0xff_xx range) but emits 0x00.
  BP should hold old SP (0x0000fff0) but emits garbage (0x0001d8d8).
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
inv = {}
for k, v in dp.items():
    if int(v) not in inv and ".*." not in str(k):
        inv[int(v)] = k

def lbl(d):
    for bd in range(d, max(-1, d - 40), -1):
        if bd in inv:
            return f"{inv[bd]}+{d-bd}" if d != bd else inv[bd]
    return str(d)

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
tests = generate_test_programs()
src, exp, _ = tests[idx]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
prompt_len = len(probe._build_context(bc))
dev = next(m.parameters()).device

positions = {}
step = 0
i = prompt_len
while i < len(ctx):
    t = ctx[i]
    nm = MARKERS.get(t)
    if nm == "STEP_END":
        step += 1; i += 1; continue
    if nm in ("PC", "AX", "SP", "BP"):
        positions[(step, nm)] = i
        i += 5; continue
    i += 1

TGT_STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for reg in ("PC", "AX", "SP", "BP"):
    mk = positions.get((TGT_STEP, reg))
    if mk is None:
        print(f"step{TGT_STEP} {reg}: NOT FOUND"); continue
    bs = [ctx[mk + 1 + j] & 0xFF for j in range(4)]
    print(f"step{TGT_STEP} {reg} marker@{mk} bytes={[hex(b) for b in bs]}")

print()
@torch.no_grad()
def logits_at(pred_row):
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    lg = m.forward(padded)[0]
    return lg[pred_row]

targets = [("SP", 0), ("SP", 1), ("BP", 0), ("BP", 1), ("BP", 2), ("BP", 3)]
for reg, bi in targets:
    mk = positions.get((TGT_STEP, reg))
    if mk is None: continue
    byte_pos = mk + 1 + bi
    pred_row = byte_pos - 1
    lg = logits_at(pred_row)
    top = torch.topk(lg, 6)
    pairs = [(int(t), round(float(v), 2)) for v, t in zip(top.values.tolist(), top.indices.tolist())]
    emitted = ctx[byte_pos] & 0xFF
    print(f"step{TGT_STEP} {reg} byte{bi}: emitted=0x{emitted:02x} pred_row={pred_row} "
          f"top=[{', '.join(f'(0x{t:02x},{v})' for t,v in pairs)}]")

print()
OUT_LO = dp["OUTPUT_LO"]; OUT_HI = dp["OUTPUT_HI"]
watch = {f"LO+{k}": OUT_LO + k for k in range(16)}
watch.update({f"HI+{k}": OUT_HI + k for k in range(16)})
if "OUTPUT_HI_THIS_STEP" in dp:
    OUT_HTS = dp["OUTPUT_HI_THIS_STEP"]
    watch.update({f"HTS+{k}": OUT_HTS + k for k in range(16)})
bl = probe.block_layer_map()

# gate dims the nested-d8 rule reads
gate_dims = {}
for nm in ("MARK_BP", "MARK_SP", "OP_ENT", "HAS_SE", "IS_BYTE"):
    if nm in dp:
        gate_dims[nm] = dp[nm]
if "OUTPUT_HI_THIS_STEP" in dp:
    gate_dims["OUTPUT_HI_THIS_STEP+15"] = dp["OUTPUT_HI_THIS_STEP"] + 15

@torch.no_grad()
def sweep(pred_row, title):
    print(f"\n=== {title} (pred_row={pred_row}) OUTPUT band per block (|v|>0.3) ===")
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    prev = {}
    for phys in range(len(m.blocks)):
        r = m.forward(padded, stop_after_block=phys)[0][pred_row].float()
        hot = {n: float(r[d]) for n, d in watch.items() if abs(float(r[d])) > 0.3}
        changed = {n: v for n, v in hot.items() if abs(v - prev.get(n, 0.0)) > 0.3}
        gone = {n for n in prev if n not in hot and abs(prev[n]) > 0.3}
        if changed or gone:
            log = bl[phys]["logical"]
            cs = " ".join(f"{n}={v:+.2f}" for n, v in sorted(changed.items()))
            gs = (" gone:" + ",".join(sorted(gone))) if gone else ""
            print(f"  phys{phys:2d} L{log:<2} {cs}{gs}")
        prev = hot

sweep_targets = []
for a in sys.argv[3:]:
    reg, bi = a.split(":")
    sweep_targets.append((reg, int(bi)))
if not sweep_targets:
    sweep_targets = [("SP", 1), ("BP", 0)]
for reg, bi in sweep_targets:
    mk = positions.get((TGT_STEP, reg))
    if mk is None: continue
    pred_row = mk + 1 + bi - 1
    sweep(pred_row, f"step{TGT_STEP} {reg} byte{bi}")

# Gate-dim readout just before block 29 (input to L20 nested-d8 rule)
@torch.no_grad()
def gate_readout(pred_row, before_block, title):
    print(f"\n=== {title} gate dims after block {before_block-1} (pred_row={pred_row}) ===")
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    r = m.forward(padded, stop_after_block=before_block - 1)[0][pred_row].float()
    for nm, d in gate_dims.items():
        print(f"  {nm:24s} (dim {int(d)}) = {float(r[int(d)]):+.4f}")

for reg, bi in [("BP", 0)]:
    mk = positions.get((TGT_STEP, reg))
    if mk is None: continue
    pred_row = mk + 1 + bi - 1
    gate_readout(pred_row, 29, f"step{TGT_STEP} {reg} byte{bi}")
