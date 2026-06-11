#!/usr/bin/env python3
"""Dump the W_up/W_gate source dims for the runaway block-6 hidden units
1393/1174 (the dim-79 driver) and identify the gate input that's hitting 67.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch, torch.nn.functional as F  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic  # noqa

reg = build_default_registry_dynamic()
def dimname(i):
    best = None
    for n, s in reg.slots.items():
        if s.start <= i < s.start + s.size:
            best = f"{n}+{i-s.start}" if s.size > 1 else n
            break
    return best or f"?{i}"

probe = build_groundtruth_probe()
tests = generate_test_programs()
src, exp, _ = tests[262]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
dev = next(probe.model.parameters()).device
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
POS = 92; BLK = 6
UNITS = [1393, 1174, 1401, 1182, 1368, 1149]

def _dense(t):
    if t.is_sparse or t.layout in (torch.sparse_csr, torch.sparse_coo):
        return t.to_dense().float()
    return t.float()

with torch.no_grad():
    r5 = probe.model.forward(padded, stop_after_block=BLK - 1)[0].float()
    blk = probe.model.blocks[BLK]
    a_out = blk.attn(r5.unsqueeze(0))[0]
    xin = a_out[POS]
    ffn = blk.ffn
    Wu = _dense(ffn.W_up.data); Wg = _dense(ffn.W_gate.data)
    bu = ffn.b_up.data.float(); bg = ffn.b_gate.data.float()
    for h in UNITS:
        urow = Wu[h]; grow = Wg[h]
        usrc = (urow.abs() > 1e-6).nonzero().flatten().tolist()
        gsrc = (grow.abs() > 1e-6).nonzero().flatten().tolist()
        print(f"\n--- hidden {h} ---", flush=True)
        print(f"  b_up={float(bu[h]):+.2f} b_gate={float(bg[h]):+.2f}", flush=True)
        print(f"  W_up src:", flush=True)
        for i in usrc:
            print(f"     {dimname(i):20s}(dim{i}) W={float(urow[i]):+9.2f}  "
                  f"x={float(xin[i]):+9.3f}  prod={float(urow[i]*xin[i]):+9.2f}", flush=True)
        print(f"  W_gate src:", flush=True)
        for i in gsrc:
            print(f"     {dimname(i):20s}(dim{i}) W={float(grow[i]):+9.2f}  "
                  f"x={float(xin[i]):+9.3f}  prod={float(grow[i]*xin[i]):+9.2f}", flush=True)
        print(f"  => up_total={float(F.linear(xin,urow.unsqueeze(0))+bu[h]):+.2f}  "
              f"gate_total={float(F.linear(xin,grow.unsqueeze(0))+bg[h]):+.2f}", flush=True)
