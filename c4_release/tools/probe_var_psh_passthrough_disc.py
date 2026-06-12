#!/usr/bin/env python3
"""Discriminator probe for the L10 psh_stack0_passthrough head (block-12 head 3)
on var_simple_12 (id 262). Dump the head's Q-gate-relevant dims on the BUG row
(212, step-3 PSH SP byte3) vs the legit STACK0 byte rows + the GOOD row (245),
to find a dim that fires ONLY on the BUG row (so a NOT-blocker stays
byte-identical on the legit firing rows).

READ-ONLY. spec_k=0, hook-free."""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

ID = int(sys.argv[1]) if len(sys.argv) > 1 else 262
BLK = 12
probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device
bl = probe.block_layer_map()

tests = generate_test_programs()
src, exp, desc = tests[ID]
print(f"id={ID} {desc!r} exp={exp}")
bc, data = compile_c(src)
ctx = probe._final_context(bc)
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
S = len(ctx)

with torch.no_grad():
    x_in = m.forward(padded, stop_after_block=BLK - 1)[0].float()  # [S,D]

# The head's Q-gate dims (from _layer10_psh_stack0_passthrough_head_spec):
gate = ["IS_BYTE", "PSH_AT_SP", "MARK_STACK0", "MARK_SP", "MARK_AX", "MARK_BP",
        "MARK_PC", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
        "MEM_STORE", "CMP", "OP_PSH"]
# H1/H4 marker-proximity bands (used by Q slots 1/33 via H4+BP_IDX, H1+BP_IDX)
for base in ("H1", "H4"):
    for j in range(5):
        nm = f"{base}+{j}"
        if (base in dp):
            gate.append(nm)
gate_dims = {}
for nm in gate:
    if "+" in nm:
        b, off = nm.split("+");
        if b in dp:
            gate_dims[nm] = dp[b] + int(off)
    elif nm in dp:
        gate_dims[nm] = dp[nm]

# Find candidate rows: the BUG SP byte3 row, the GOOD IMM SP byte3 row, and all
# rows where MARK_STACK0/STACK0_BYTE* fire (the legit firing rows).
def val(row, nm):
    return float(x_in[row, gate_dims[nm]])

# enumerate STACK0 byte rows (legit firing) + SP byte rows (bug family)
print("\n=== rows where the head's Q gate could fire (PSH_AT_SP>0.5 or STACK0_BYTE*>0.5) ===")
hdr_dims = ["IS_BYTE", "PSH_AT_SP", "MARK_STACK0", "STACK0_BYTE0", "STACK0_BYTE1",
            "STACK0_BYTE2", "STACK0_BYTE3", "MARK_SP", "BYTE_INDEX_3", "MEM_STORE", "CMP"]
hdr_dims = [d for d in hdr_dims if d in gate_dims]
print("row  | " + " ".join(f"{d[:9]:>9}" for d in hdr_dims))
for row in range(S):
    psh = val(row, "PSH_AT_SP") if "PSH_AT_SP" in gate_dims else 0
    sb = max((val(row, f"STACK0_BYTE{j}") for j in range(4)
              if f"STACK0_BYTE{j}" in gate_dims), default=0)
    if psh > 0.5 or sb > 0.5:
        marks = " ".join(f"{val(row,d):9.2f}" for d in hdr_dims)
        tag = ""
        if row == 212: tag = " <== BUG (SP byte3)"
        if row == 245: tag = " <== GOOD"
        print(f"{row:>4} | {marks}{tag}")

# Explicit BUG vs GOOD full gate dump
print("\n=== full Q-gate dim dump: BUG(212) vs GOOD(245) ===")
for nm in sorted(gate_dims):
    b = float(x_in[212, gate_dims[nm]]); g = float(x_in[245, gate_dims[nm]])
    if abs(b) > 0.01 or abs(g) > 0.01:
        print(f"  {nm:>14}: BUG={b:8.3f}  GOOD={g:8.3f}  {'<-- DIFFERS' if abs(b-g)>0.5 else ''}")
