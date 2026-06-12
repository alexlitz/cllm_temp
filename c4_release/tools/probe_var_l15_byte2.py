#!/usr/bin/env python3
"""For block-30 (logical L19 = L15 memory_lookup) heads 1-3, compute per-row
softmax1 wsum + OUTPUT-band contribution on var_simple_12 (id 262, with the
block-12 fix applied). Identify which rows the heads fire on, separate the
legit LI/LC load rows from the SP/BP register-byte rows that leak 0x0f, and
dump the H1 register-marker band so a NOT-blocker can target the register rows.

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
import torch.nn.functional as F  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

ID = int(sys.argv[1]) if len(sys.argv) > 1 else 262
BLK = int(sys.argv[2]) if len(sys.argv) > 2 else 30
HEADS = [1, 2, 3]
probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device
bl = probe.block_layer_map()
print(f"block {BLK} -> logical L{bl[BLK]['logical']}")

tests = generate_test_programs()
src, exp, desc = tests[ID]
print(f"id={ID} {desc!r} exp={exp}")
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
S = len(ctx)

attn = m.blocks[BLK].attn
H = attn.num_heads
HD = attn.head_dim
print(f"num_heads={H} head_dim={HD} softmax1={attn.use_softmax1}")
def dense(w):
    return w.to_dense().float() if (w.is_sparse_csr or w.is_sparse) else w.float()
Wq, Wk, Wv, Wo = dense(attn.W_q), dense(attn.W_k), dense(attn.W_v), dense(attn.W_o)
out_lo0, out_hi0 = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
out_dims = torch.tensor(list(range(out_lo0, out_lo0 + 16))
                        + list(range(out_hi0, out_hi0 + 16)), device=dev)

with torch.no_grad():
    x_in = m.forward(padded, stop_after_block=BLK - 1)[0].float()
    x = x_in.unsqueeze(0)
    Qa = F.linear(x, Wq).view(1, S, H, HD).transpose(1, 2)
    Ka = F.linear(x, Wk).view(1, S, H, HD).transpose(1, 2)
    Va = F.linear(x, Wv).view(1, S, H, HD).transpose(1, 2)
    causal = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)

def v(row, nm, off=0):
    base = dp.get(nm)
    return float('nan') if base is None else float(x_in[row, base + off])

for HEAD in HEADS:
    Q, K, V = Qa[:, HEAD], Ka[:, HEAD], Va[:, HEAD]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale + causal
    mx = scores.amax(-1, keepdim=True).clamp_min(0.0)
    ex = torch.exp(scores - mx)
    aw = ex / (ex.sum(-1, keepdim=True) + torch.exp(-mx))
    wsum = aw.sum(-1)[0]
    out_head = torch.matmul(aw, V)[0]
    contrib = out_head @ Wo[:, HEAD * HD:(HEAD + 1) * HD].T
    osig = contrib.index_select(1, out_dims).sum(-1)
    print(f"\n=== HEAD {HEAD} firing rows (wsum>0.05 or |OUT|>1) ===")
    print("row  | wsum | OUTsig | IS_B MAX MSP MSTK BI0 BI1 BI2 | H1+0 H1+1 H1+2 H1+3 H1+4 | OP_LI_R")
    for row in range(S):
        ws = float(wsum[row]); o = float(osig[row])
        if ws > 0.05 or abs(o) > 1.0:
            cells = [v(row,"IS_BYTE"), v(row,"MARK_AX"), v(row,"MARK_SP"),
                     v(row,"MARK_STACK0"), v(row,"BYTE_INDEX_0"), v(row,"BYTE_INDEX_1"),
                     v(row,"BYTE_INDEX_2")]
            h1 = [v(row,"H1",j) for j in range(5)]
            olir = v(row,"OP_LI_RELAY")
            print(f"{row:>4} | {ws:4.2f} | {o:6.1f} | " +
                  " ".join(f"{c:4.1f}" for c in cells) + " | " +
                  " ".join(f"{c:4.1f}" for c in h1) + f" | {olir:5.2f}")
