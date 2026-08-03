"""Recompute cost of the single-launch 'full' form per block, to pick the hybrid threshold."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.fused_sparse_ffn import _dense_of

dev = torch.device("cuda:0")
torch.cuda.set_device(dev); torch.zeros(1).to(dev)
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev))

# For 'full': cost ~ sum over down-nnz (d,h) of (up_nnz[h]+gate_nnz[h]).
# vs 'upgate': cost ~ up_nnz + gate_nnz + down_nnz (each hidden computed once).
rows = []
for i, b in enumerate(model.blocks):
    ffn = b.ffn
    if getattr(ffn, "W_up", None) is None: continue
    Wu = _dense_of(ffn.W_up); Wg = _dense_of(ffn.W_gate); Wd = _dense_of(ffn.W_down)
    Dff = Wu.shape[0]
    up_nnz_per_h = (Wu != 0).sum(dim=1)   # [Dff]
    gt_nnz_per_h = (Wg != 0).sum(dim=1)
    hid_cost = (up_nnz_per_h + gt_nnz_per_h + 2)  # +2 for the two silu/mul + bias
    # for each down nnz (d,h) we recompute hid_cost[h]
    dn = (Wd != 0)   # [D, Dff]
    # sum over all down nonzeros of hid_cost[col]
    dn_cols = torch.nonzero(dn, as_tuple=False)[:,1]  # hidden indices read by down
    full_cost = int(hid_cost[dn_cols].sum()) if dn_cols.numel() else 0
    upgate_cost = int((up_nnz_per_h+gt_nnz_per_h).sum()) + int(dn.sum())
    rows.append((i, Dff, full_cost, upgate_cost, full_cost/max(upgate_cost,1)))

rows.sort(key=lambda r:-r[4])
print(f"{'blk':>4} {'Dff':>5} {'full_cost':>10} {'upgate_cost':>11} {'ratio':>6}")
for r in rows[:12]:
    print(f"{r[0]:>4} {r[1]:>5} {r[2]:>10} {r[3]:>11} {r[4]:>6.2f}")
print("...")
tot_full = sum(r[2] for r in rows); tot_ug = sum(r[3] for r in rows)
print(f"TOTAL full_cost={tot_full}  upgate_cost={tot_ug}  ratio={tot_full/tot_ug:.2f}")
# How many blocks have ratio > 1.5 (full loses)?
loses = [r for r in rows if r[4] > 1.5]
print(f"blocks where full recompute > 1.5x upgate: {len(loses)}  (Dffs: {sorted(set(r[1] for r in loses))})")
# hybrid: full for ratio<=1.5, upgate for the rest. count launches:
n = len(rows)
n_full = sum(1 for r in rows if r[4] <= 1.5)
n_ug = n - n_full
print(f"HYBRID launches: {n_full} full(1 each) + {n_ug} upgate(2 each) = {n_full + 2*n_ug} (vs upgate all={2*n}, full all={n})")
