#!/usr/bin/env python3
"""Identify the exact block-12 (logical L11) attention head that crushes OUTPUT
on the var_simple_12 (id 262) BUG row (step-3 PSH SP byte3, pred_row=212).

For each head: report its W_o OUTPUT_LO/HI column mass, its softmax attention
distribution on the BUG row, the position it attends, and that position's
OUTPUT band value. Compare against the GOOD row (step-4 IMM SP byte3, 245).

READ-ONLY. spec_k=0, hook-free. Dims via probe.model.dim_positions.
Usage: python tools/probe_var_block12_head.py [id] [block]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
import torch.nn.functional as F  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

ID = int(sys.argv[1]) if len(sys.argv) > 1 else 262
BLK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BUG = 212   # step3 SP byte3 -> emits BP marker 260 instead of 0x00
GOOD = 245  # step4 IMM SP byte3 -> correctly emits 0x00

probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
dev = next(m.parameters()).device
bl = probe.block_layer_map()
rev = {}
for n, d in dp.items():
    if isinstance(d, int):
        rev.setdefault(d, []).append(n)

tests = generate_test_programs()
src, exp, desc = tests[ID]
print(f"id={ID} {desc!r} exp={exp}")
bc, data = compile_c(src)
ctx = probe._final_context(bc)
padded = torch.tensor([ctx], dtype=torch.long, device=dev)

# OUTPUT band dims: OUTPUT_LO is 16-wide nibble band, OUTPUT_HI 16-wide.
out_lo0 = dp["OUTPUT_LO"]
out_hi0 = dp["OUTPUT_HI"]
out_dims = list(range(out_lo0, out_lo0 + 16)) + list(range(out_hi0, out_hi0 + 16))
out_dims = sorted(set(out_dims))
out_t = torch.tensor(out_dims, dtype=torch.long, device=dev)
print(f"OUTPUT band dims: {len(out_dims)} cols, range {min(out_dims)}-{max(out_dims)}")

blk = m.blocks[BLK]
attn = blk.attn
H = attn.num_heads
HD = attn.head_dim
print(f"block {BLK} / logical L{bl[BLK]['logical']}: {type(attn).__name__} "
      f"num_heads={H} head_dim={HD}")
print(f"  use_softmax1={getattr(attn,'use_softmax1',None)} "
      f"alibi={getattr(attn,'use_alibi',None)} rope={attn._rope_cos is not None}")

# residual entering block BLK
with torch.no_grad():
    x_in = m.forward(padded, stop_after_block=BLK - 1)[0].float()  # [S, D]

# Per-head W_o OUTPUT mass: which heads write the OUTPUT band?
Wo = attn.W_o
if Wo.is_sparse_csr or Wo.is_sparse:
    Wo = Wo.to_dense()
Wo = Wo.float()  # [D, D]   out = x + (head_out @ W_o.T) ; F.linear(out, W_o)
# F.linear(out, W_o) = out @ W_o.T ; column h of head-space is rows [h*HD:(h+1)*HD]
print("\n=== per-head W_o OUTPUT-band mass (|W_o[out_dim, head_slots]|) ===")
head_out_mass = []
for h in range(H):
    sl = slice(h * HD, (h + 1) * HD)
    sub = Wo.index_select(0, out_t)[:, sl]   # [n_out, HD]
    mass = float(sub.abs().sum())
    head_out_mass.append(mass)
    if mass > 0.5:
        print(f"  head {h:2d}: W_o OUTPUT-band |sum|={mass:9.2f}")

# discriminator dims on the two rows
disc = ["IS_BYTE", "MARK_SP", "MARK_PC", "MARK_AX", "MARK_BP", "MARK_SE_ONLY",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "PSH_AT_SP", "MEM_STORE", "CMP", "CONST", "OP_PSH", "OP_ENT", "OP_LEA",
        "OP_JSR", "OP_IMM", "OP_LI", "OP_SI"]
disc = [d for d in disc if d in dp]
print("\n=== discriminator dims (residual into block 12) BUG vs GOOD ===")
for d in disc:
    print(f"  {d:>14}: BUG={float(x_in[BUG, dp[d]]):8.3f}  GOOD={float(x_in[GOOD, dp[d]]):8.3f}")

def Wmat(w):
    if w.is_sparse_csr or w.is_sparse:
        return w.to_dense().float()
    return w.float()

# Now run the attention forward manually to get per-head attn weights + outputs
with torch.no_grad():
    x = x_in.unsqueeze(0)  # [1,S,D]
    B, S, D = x.shape
    Wq, Wk, Wv = Wmat(attn.W_q), Wmat(attn.W_k), Wmat(attn.W_v)
    Q = F.linear(x, Wq).view(B, S, H, HD).transpose(1, 2)
    K = F.linear(x, Wk).view(B, S, H, HD).transpose(1, 2)
    V = F.linear(x, Wv).view(B, S, H, HD).transpose(1, 2)
    # block 12: alibi=None, rope=False -> bias = causal mask only.
    causal = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
    scores = scores + causal.view(1, 1, S, S)
    if getattr(attn, "use_softmax1", False):
        # softmax1 / ZFOD: implicit anchor logit 0 in the denominator. weights
        # sum to <1 (head can output ~0 when all real scores are << 0).
        mx = scores.amax(dim=-1, keepdim=True).clamp_min(0.0)
        ex = torch.exp(scores - mx)
        denom = ex.sum(dim=-1, keepdim=True) + torch.exp(-mx)
        aw = ex / denom
    else:
        aw = F.softmax(scores, dim=-1)   # [1,H,S,S]
    out_heads = torch.matmul(aw, V)  # [1,H,S,HD]

for label, row in (("BUG", BUG), ("GOOD", GOOD)):
    print(f"\n=== {label} row {row} (IS_BYTE={float(x_in[row, dp['IS_BYTE']]):.2f}) ===")
    # decode OUTPUT band for x_in at this row
    for h in range(H):
        w = aw[0, h, row]            # [S]
        topp = torch.topk(w, 4)
        # this head's OUTPUT-band contribution to the residual (out @ W_o)
        ho = out_heads[0, h, row]    # [HD]
        contrib = ho @ Wo[:, h * HD:(h + 1) * HD].T  # [D]
        out_contrib = float(contrib.index_select(0, out_t).abs().sum())
        out_signed = float(contrib.index_select(0, out_t).sum())
        if head_out_mass[h] < 1.0 and out_contrib < 0.5:
            continue
        attended = ", ".join(
            f"{int(p)}({float(v):.2f})" for v, p in zip(topp.values, topp.indices))
        print(f"  head {h:2d} [Wo_out={head_out_mass[h]:7.1f}] "
              f"attends: {attended}  -> OUTPUT contrib |sum|={out_contrib:8.2f} "
              f"signed={out_signed:9.2f}")
