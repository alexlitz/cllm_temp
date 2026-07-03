#!/usr/bin/env python3
"""Per-head attention contribution to CMP+1 (dim 397) at pos 92, blocks 4/5/6.

Manually replays each block's attention, isolating each head's W_o write to
dim 397, to find which head adds the big +40 at L5 (and the +25 at L6).
spec_k=0, hook-free.
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
POS = pl  # 92
DIM = 397  # CMP+1

def _dense(t):
    if t.is_sparse or t.layout in (torch.sparse_csr, torch.sparse_coo):
        return t.to_dense().float()
    return t.float()

# Replicate PureAttention.forward to get per-head output before W_o, then
# attribute W_o[DIM] per head.
def head_decomp(blk, x):  # x: [1,S,D]
    attn = blk.attn
    H = attn.num_heads
    D = x.shape[-1]
    HD = D // H
    Wq = _dense(attn.W_q.data); Wk = _dense(attn.W_k.data); Wv = _dense(attn.W_v.data)
    Wo = _dense(attn.W_o.data)
    B, S, _ = x.shape
    Q = F.linear(x, Wq).view(B, S, H, HD).transpose(1, 2)
    K = F.linear(x, Wk).view(B, S, H, HD).transpose(1, 2)
    V = F.linear(x, Wv).view(B, S, H, HD).transpose(1, 2)
    scale = 1.0 / (HD ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B,H,S,S]
    # causal mask + alibi if present
    mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    alibi = getattr(attn, "alibi_slopes", None)
    if alibi is not None:
        sl = _dense(alibi.data) if hasattr(alibi, "data") else alibi
        pos = torch.arange(S, device=x.device)
        rel = (pos[None, :] - pos[:, None]).float()  # [S,S]
        bias = sl.view(H, 1, 1) * rel[None]  # [H,S,S]
        scores = scores + bias[None]
    w = torch.softmax(scores, dim=-1)  # [B,H,S,S]
    out = torch.matmul(w, V)  # [B,H,S,HD]
    # per-head contribution to dim DIM at POS via W_o
    contribs = {}
    for h in range(H):
        oh = out[0, h, POS]  # [HD]
        wo_slice = Wo[DIM, h * HD:(h + 1) * HD]  # [HD]
        contribs[h] = float((oh * wo_slice).sum())
    return contribs

with torch.no_grad():
    for BLK in (4, 5, 6):
        r_in = probe.model.forward(padded, stop_after_block=BLK - 1)[0].float()
        c = head_decomp(probe.model.blocks[BLK], r_in.unsqueeze(0))
        print(f"\nblock {BLK} per-head W_o contribution to CMP+1 (dim 397) @pos{POS}:",
              flush=True)
        for h, v in c.items():
            if abs(v) > 1e-4:
                print(f"  head {h}: {v:+.4f}", flush=True)
        print(f"  sum={sum(c.values()):+.4f}", flush=True)
