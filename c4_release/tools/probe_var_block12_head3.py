#!/usr/bin/env python3
"""Deep-dive block-12 (logical L11) HEAD 3 — the OUTPUT crusher on the var BUG
row. Dump its Q/K/V/O nonzero (dim, slot, weight) entries to identify which
declarative op owns it and whether a PSH_AT_SP/MEM_STORE-keyed Q NOT-blocker
slot is free.

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

BLK = int(sys.argv[1]) if len(sys.argv) > 1 else 12
HEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 3

probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
bl = probe.block_layer_map()
rev = {}
for n, d in dp.items():
    if isinstance(d, int):
        rev.setdefault(d, []).append(n)

def nm(d):
    cands = rev.get(d, [f"d{d}"])
    # prefer non-".*" names
    clean = [c for c in cands if ".*" not in c]
    return (clean or cands)[0]

attn = m.blocks[BLK].attn
H = attn.num_heads
HD = attn.head_dim
print(f"block {BLK} / logical L{bl[BLK]['logical']} head {HEAD}: "
      f"num_heads={H} head_dim={HD} softmax1={attn.use_softmax1}")

def dense(w):
    if w.is_sparse_csr or w.is_sparse:
        return w.to_dense()
    return w
Wq, Wk, Wv, Wo = (dense(attn.W_q).float(), dense(attn.W_k).float(),
                  dense(attn.W_v).float(), dense(attn.W_o).float())
sl = slice(HEAD * HD, (HEAD + 1) * HD)

# Q: W_q[head_slot, in_dim]  (Q = x @ W_q.T, so row=out slot, col=in dim)
print("\n=== HEAD Q (slot, in_dim, w) nonzero ===")
Wqh = Wq[sl]  # [HD, D]
nz = (Wqh.abs() > 1e-6).nonzero()
for s, d in nz.tolist():
    print(f"  Qslot {s:3d}  <- {nm(d):>16}(d{d})  w={float(Wqh[s,d]):+.3f}")
print("\n=== HEAD K (slot, in_dim, w) nonzero ===")
Wkh = Wk[sl]
nz = (Wkh.abs() > 1e-6).nonzero()
for s, d in nz.tolist():
    print(f"  Kslot {s:3d}  <- {nm(d):>16}(d{d})  w={float(Wkh[s,d]):+.3f}")
print("\n=== HEAD V (slot, in_dim, w) nonzero (first 40) ===")
Wvh = Wv[sl]
nz = (Wvh.abs() > 1e-6).nonzero()
for s, d in nz.tolist()[:40]:
    print(f"  Vslot {s:3d}  <- {nm(d):>16}(d{d})  w={float(Wvh[s,d]):+.3f}")
print(f"  ... total V nonzero: {nz.shape[0]}")
print("\n=== HEAD O (out_dim, slot, w) nonzero (first 40) ===")
Woh = Wo[:, sl]  # [D, HD]
nz = (Woh.abs() > 1e-6).nonzero()
for d, s in nz.tolist()[:40]:
    print(f"  Oslot {s:3d}  -> {nm(d):>16}(d{d})  w={float(Woh[d,s]):+.3f}")
print(f"  ... total O nonzero: {nz.shape[0]}")

# which Q slots are FREE (all-zero) for adding a NOT-blocker?
free = [s for s in range(HD) if float(Wqh[s].abs().sum()) == 0.0
        and float(Wkh[s].abs().sum()) == 0.0]
print(f"\n=== FREE Q/K slots (all-zero both sides): {len(free)} ===")
print(f"  {free[:20]} ...")
