#!/usr/bin/env python3
"""Rank which block-6 FFN hidden units actually FIRE and drive dim 79 at pos 92.

spec_k=0, hook-free. Replays block 6's FFN forward manually on the real
post-block-5 residual and ranks hidden-unit contributions to dim 79 (and 85).
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

probe = build_groundtruth_probe()
tests = generate_test_programs()
src, exp, _ = tests[262]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
dev = next(probe.model.parameters()).device
padded = torch.tensor([ctx], dtype=torch.long, device=dev)

POS = 92; BLK = 6; DIMS = [79, 85]

def _dense(t):
    if t.is_sparse or t.layout in (torch.sparse_csr, torch.sparse_coo):
        return t.to_dense().float()
    return t.float()

with torch.no_grad():
    r5 = probe.model.forward(padded, stop_after_block=BLK - 1)[0].float()  # [S,D]
    x = r5[POS]  # [D] input to block 6 ffn (attn leaves dims 79/85 untouched)
    blk = probe.model.blocks[BLK]
    # attn first (residual stream into ffn)
    a_out = blk.attn(r5.unsqueeze(0))[0]  # [S,D]
    xin = a_out[POS]  # [D]
    ffn = blk.ffn
    Wu = _dense(ffn.W_up.data); bu = ffn.b_up.data.float()
    Wg = _dense(ffn.W_gate.data); bg = ffn.b_gate.data.float()
    Wd = _dense(ffn.W_down.data); bd = ffn.b_down.data.float()
    up = F.linear(xin, Wu) + bu          # [hidden]
    gate = F.linear(xin, Wg) + bg        # [hidden]
    hidden = F.silu(up) * gate           # [hidden]
    for d in DIMS:
        contrib = Wd[d] * hidden  # per-hidden contribution to dim d
        tot = float(contrib.sum() + bd[d])
        print(f"\n=== dim {d}: total ffn out (pre-resid) = {tot:+.3f}, "
              f"b_down={float(bd[d]):+.4f} ===", flush=True)
        order = contrib.abs().argsort(descending=True)
        for h in order[:20].tolist():
            print(f"  hidden {h:5d}: contrib={float(contrib[h]):+10.3f}  "
                  f"hidden_act={float(hidden[h]):+10.3f}  "
                  f"W_down={float(Wd[d,h]):+.4f}  "
                  f"up={float(up[h]):+8.2f} gate={float(gate[h]):+8.2f}", flush=True)
