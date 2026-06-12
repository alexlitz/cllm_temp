#!/usr/bin/env python3
"""For the L10 psh_stack0_passthrough head (block-12 head 3), compute the per-row
softmax1 attention weight-SUM (how 'on' the head is) and OUTPUT-band contribution
magnitude across ALL rows of var_simple_12 (id 262). Identify the legit firing
rows (where it SHOULD write the pushed value) vs the BUG row (212) where it
crushes OUTPUT. Then dump candidate discriminator dims (H1/H4 +1/+3,
STACK0_BYTE*) so a NOT-blocker can be keyed to fire ONLY on the bug family.

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
BLK = 12
HEAD = 3
probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device

tests = generate_test_programs()
src, exp, desc = tests[ID]
print(f"id={ID} {desc!r} exp={exp}  HEAD={HEAD} BLK={BLK}")
bc, data = compile_c(src)
ctx = probe._final_context(bc)
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
S = len(ctx)

attn = m.blocks[BLK].attn
H = attn.num_heads
HD = attn.head_dim
def dense(w):
    return w.to_dense().float() if (w.is_sparse_csr or w.is_sparse) else w.float()
Wq, Wk, Wv, Wo = dense(attn.W_q), dense(attn.W_k), dense(attn.W_v), dense(attn.W_o)
sl = slice(HEAD * HD, (HEAD + 1) * HD)
out_lo0, out_hi0 = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
out_dims = torch.tensor(list(range(out_lo0, out_lo0 + 16))
                        + list(range(out_hi0, out_hi0 + 16)), device=dev)

with torch.no_grad():
    x_in = m.forward(padded, stop_after_block=BLK - 1)[0].float()
    x = x_in.unsqueeze(0)
    Q = F.linear(x, Wq).view(1, S, H, HD).transpose(1, 2)[:, HEAD]  # [1,S,HD]
    K = F.linear(x, Wk).view(1, S, H, HD).transpose(1, 2)[:, HEAD]
    V = F.linear(x, Wv).view(1, S, H, HD).transpose(1, 2)[:, HEAD]
    causal = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale + causal
    mx = scores.amax(dim=-1, keepdim=True).clamp_min(0.0)
    ex = torch.exp(scores - mx)
    denom = ex.sum(-1, keepdim=True) + torch.exp(-mx)
    aw = ex / denom                     # [1,S,S]
    wsum = aw.sum(-1)[0]                # [S]  weight-sum (softmax1 -> <1)
    out_head = torch.matmul(aw, V)[0]  # [S,HD]
    contrib = out_head @ Wo[:, sl].T   # [S,D]
    out_signed = contrib.index_select(1, out_dims).sum(-1)  # [S]

def v(row, nm, off=0):
    base = dp.get(nm)
    if base is None: return float('nan')
    return float(x_in[row, base + off])

disc = ["H1", "H4"]
print("\n=== per-row head firing (wsum>0.05 OR |OUT contrib|>1) ===")
print("row  | wsum  | OUTsig  | IS_B PSH MSTK SB0 SB1 SB2 SB3 BI3 | H1+1 H1+3 H4+1 H4+3 MEM CMP")
for row in range(S):
    ws = float(wsum[row]); osig = float(out_signed[row])
    if ws > 0.05 or abs(osig) > 1.0:
        cells = [v(row,"IS_BYTE"), v(row,"PSH_AT_SP"), v(row,"MARK_STACK0"),
                 v(row,"STACK0_BYTE0"), v(row,"STACK0_BYTE1"), v(row,"STACK0_BYTE2"),
                 v(row,"STACK0_BYTE3"), v(row,"BYTE_INDEX_3")]
        h = [v(row,"H1",1), v(row,"H1",3), v(row,"H4",1), v(row,"H4",3),
             v(row,"MEM_STORE"), v(row,"CMP")]
        tag = " <==BUG" if row==212 else (" <==GOOD" if row==245 else "")
        print(f"{row:>4} | {ws:5.2f} | {osig:7.1f} | " +
              " ".join(f"{c:4.1f}" for c in cells) + " | " +
              " ".join(f"{c:4.1f}" for c in h) + tag)
