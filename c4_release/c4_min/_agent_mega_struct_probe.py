"""#809 structural probe — per-block (Dff, up/gate/down nnz, n_active) + the
delta-form launch structure, to design the persistent megakernel.

Prints the total nnz across all 242 blocks, the Dff distribution, and the
count of graph nodes (= launch boundaries) the delta form emits at K.
LEAN load (sparse_mm streaming, densify only touched weights)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.fused_sparse_ffn import _dense_of, FusedUpGateSiluDeltaFFN

dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.zeros(1).to(dev)
torch.backends.cuda.matmul.allow_tf32 = False
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev))
D = model.dim
print(f"dim={D} blocks={len(model.blocks)}")

n_ffn = 0
tot_up = tot_gt = tot_dn = 0
tot_units = 0
tot_active = 0
max_dff = 0
dff_hist = {}
routed = 0
for b in model.blocks:
    if getattr(b, "_routed", False):
        routed += 1
        continue
    if getattr(b.ffn, "W_up", None) is None:
        continue
    Wu = _dense_of(b.ffn.W_up); Wg = _dense_of(b.ffn.W_gate); Wd = _dense_of(b.ffn.W_down)
    dff = Wu.shape[0]
    nnz_u = int((Wu != 0).sum()); nnz_g = int((Wg != 0).sum()); nnz_d = int((Wd != 0).sum())
    n_active = int((Wd != 0).any(dim=1).sum())
    n_ffn += 1
    tot_up += nnz_u; tot_gt += nnz_g; tot_dn += nnz_d
    tot_units += dff; tot_active += n_active
    max_dff = max(max_dff, dff)
    bucket = 1 << (dff.bit_length())
    dff_hist[bucket] = dff_hist.get(bucket, 0) + 1

print(f"n_ffn_blocks={n_ffn} routed={routed}")
print(f"total hidden units (sum Dff)={tot_units:,}  max Dff={max_dff}")
print(f"total nnz up/gate/down = {tot_up:,}/{tot_gt:,}/{tot_dn:,}  (sum={tot_up+tot_gt+tot_dn:,})")
print(f"total down-active rows across blocks = {tot_active:,}")
print(f"Dff distribution (bucket->count): {dict(sorted(dff_hist.items()))}")
print(f"delta-form launches/step @K: 242 upgate + {n_ffn} down-delta = {2*n_ffn} FFN launches")
print(f"  (each is a graph node/launch boundary; + 3 live-attn blocks)")

# per-block delta n_active detail (first/last few)
K = 1024
mem_bytes = 0
specs = []
for i, b in enumerate(model.blocks):
    if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
        continue
    imp = FusedUpGateSiluDeltaFFN(b.ffn, dev)
    specs.append((i, imp.Dff, imp.n_active, imp.up_val.numel(), imp.gt_val.numel(), imp.dn_val.numel()))
print(f"\nsample blocks (idx, Dff, n_active, up_nnz, gt_nnz, dn_nnz):")
for s in specs[:3] + specs[len(specs)//2-1:len(specs)//2+1] + specs[-3:]:
    print(f"  blk {s[0]:3d} Dff={s[1]:5d} n_active={s[2]:5d} up_nnz={s[3]:6d} gt_nnz={s[4]:6d} dn_nnz={s[5]:6d}")
