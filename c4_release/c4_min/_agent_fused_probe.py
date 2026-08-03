"""Probe the FFN block structure to design the fused segmented kernel."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch
from collections import Counter
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.sparse_coo_spmm import _dense_of

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
torch.zeros(1).to(dev)
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev))

D = model.dim
print(f"model dim D={D}  blocks={len(model.blocks)}")

# Collect per-block FFN stats
n_ffn = 0
Dffs = []
nnz_up = []; nnz_gate = []; nnz_down = []
down_in_dims = set()
up_in_dims = set()
b_up_nz = 0; b_gate_nz = 0; b_down_nz = 0
routed = 0
for b in model.blocks:
    if getattr(b, "_routed", False):
        routed += 1
    ffn = b.ffn
    if getattr(ffn, "W_up", None) is None:
        continue
    Wu = _dense_of(ffn.W_up); Wg = _dense_of(ffn.W_gate); Wd = _dense_of(ffn.W_down)
    n_ffn += 1
    Dffs.append(Wu.shape[0])
    up_in_dims.add(Wu.shape[1]); up_in_dims.add(Wg.shape[1])
    down_in_dims.add(Wd.shape[1])
    nnz_up.append(int((Wu != 0).sum()))
    nnz_gate.append(int((Wg != 0).sum()))
    nnz_down.append(int((Wd != 0).sum()))
    b_up_nz += int((ffn.b_up != 0).sum())
    b_gate_nz += int((ffn.b_gate != 0).sum())
    b_down_nz += int((ffn.b_down != 0).sum())

print(f"n_ffn blocks (with W_up) = {n_ffn}  routed={routed}")
print(f"Dff: min={min(Dffs)} max={max(Dffs)} median={sorted(Dffs)[len(Dffs)//2]} sum={sum(Dffs)}")
print(f"up_in_dims (should be D)   = {sorted(up_in_dims)}")
print(f"down_in_dims (should=Dff)  = distinct-count {len(down_in_dims)} min={min(down_in_dims)} max={max(down_in_dims)}")
print(f"total nnz: up={sum(nnz_up)}  gate={sum(nnz_gate)}  down={sum(nnz_down)}  ALL={sum(nnz_up)+sum(nnz_gate)+sum(nnz_down)}")
print(f"bias nz: up={b_up_nz} gate={b_gate_nz} down={b_down_nz}")
# Dff distribution
c = Counter(Dffs)
print("Dff histogram (top 15):", sorted(c.items(), key=lambda x:-x[1])[:15])
# nnz per row stats for down (the expensive one)
allrows_down = []
for b in model.blocks:
    ffn = b.ffn
    if getattr(ffn, "W_up", None) is None: continue
    Wd = _dense_of(ffn.W_down)
    rc = (Wd != 0).sum(dim=1)
    allrows_down.append(rc)
rc = torch.cat(allrows_down)
print(f"W_down nnz/row: mean={rc.float().mean():.2f} max={rc.max()} median={rc.median()}")
# up/gate rows
allrows_up = []
for b in model.blocks:
    ffn = b.ffn
    if getattr(ffn, "W_up", None) is None: continue
    Wu = _dense_of(ffn.W_up)
    allrows_up.append((Wu != 0).sum(dim=1))
ru = torch.cat(allrows_up)
print(f"W_up nnz/row: mean={ru.float().mean():.2f} max={ru.max()} median={ru.median()}")

# Which blocks share the same ffn object (recurrent tie)?
ids = [id(b.ffn) for b in model.blocks if getattr(b.ffn,"W_up",None) is not None]
print(f"distinct ffn objects among {len(ids)} ffn blocks = {len(set(ids))}")

# down bias all zero?
print(f"b_down all-zero across blocks? {b_down_nz==0}")
