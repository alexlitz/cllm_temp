#!/usr/bin/env python3
"""Inspect L5 (block 5) attention HD + which (head,slot) cells write W_o[397]
(CMP+1) and W_o[420..451] (FETCH). Determine head 3's actual write to CMP+1.
Also dump head-3 attention weights at pos 92 (where does it attend?).
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
    for n, s in reg.slots.items():
        if s.start <= i < s.start + s.size:
            return f"{n}+{i-s.start}" if s.size > 1 else n
    return f"?{i}"

probe = build_groundtruth_probe()
tests = generate_test_programs()
src, exp, _ = tests[262]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
dev = next(probe.model.parameters()).device
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
POS = pl  # 92
BLK = 5

def _dense(t):
    if t.is_sparse or t.layout in (torch.sparse_csr, torch.sparse_coo):
        return t.to_dense().float()
    return t.float()

attn = probe.model.blocks[BLK].attn
H = attn.num_heads
D = _dense(attn.W_o.data).shape[0]
HD = D // H
print(f"block {BLK}: num_heads={H} D={D} HD={HD}", flush=True)
Wo = _dense(attn.W_o.data)
# Which (head, slot) cells write CMP+1 (dim 397)?
row = Wo[397]
nz = (row.abs() > 1e-6).nonzero().flatten().tolist()
print(f"\nW_o[CMP+1=397] nonzero concat-slots:", flush=True)
for c in nz:
    h = c // HD; s = c % HD
    print(f"  concat {c}: head {h} slot {s}  W={float(row[c]):+.2f}", flush=True)

# Head 3 attention pattern at POS: where does it attend?
with torch.no_grad():
    r_in = probe.model.forward(padded, stop_after_block=BLK - 1)[0].float()
    x = r_in.unsqueeze(0)
    Wq = _dense(attn.W_q.data); Wk = _dense(attn.W_k.data); Wv = _dense(attn.W_v.data)
    B, S, _ = x.shape
    Q = F.linear(x, Wq).view(B, S, H, HD).transpose(1, 2)
    K = F.linear(x, Wk).view(B, S, H, HD).transpose(1, 2)
    V = F.linear(x, Wv).view(B, S, H, HD).transpose(1, 2)
    scale = 1.0 / (HD ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    alibi = getattr(attn, "alibi_slopes", None)
    if alibi is not None:
        sl = _dense(alibi.data) if hasattr(alibi, "data") else alibi
        pos = torch.arange(S, device=x.device)
        rel = (pos[None, :] - pos[:, None]).float()
        scores = scores + (sl.view(H, 1, 1) * rel[None])[None]
    w = torch.softmax(scores, dim=-1)
    for h in (3,):
        wr = w[0, h, POS]  # [S]
        top = wr.argsort(descending=True)[:6].tolist()
        print(f"\nhead {h} attention from POS={POS} (REG_PC):", flush=True)
        for j in top:
            print(f"  -> pos {j} (tok {ctx[j]}) weight={float(wr[j]):.4f}", flush=True)
        # output for head 3 at POS, and its W_o[397] projection
        oh = V[0, h, POS]  # NO -- need attended output
        out_h = (w[0, h, POS:POS+1] @ V[0, h])  # [1,HD]
        out_h = out_h[0]
        wo397 = Wo[397, h*HD:(h+1)*HD]
        print(f"  head{h} out·W_o[397] = {float((out_h*wo397).sum()):+.3f}", flush=True)
        nzslot = (wo397.abs()>1e-6).nonzero().flatten().tolist()
        for s in nzslot:
            print(f"    slot {s}: W_o={float(wo397[s]):+.2f}  out={float(out_h[s]):+.3f}", flush=True)
