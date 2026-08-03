"""Per-block correctness: FusedUpGateSiluFFN + FusedFullFFN vs dense SparseFFN.forward."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.fused_sparse_ffn import FusedUpGateSiluFFN, FusedFullFFN, _dense_of

dev = torch.device("cuda:0")
torch.cuda.set_device(dev); torch.zeros(1).to(dev)
torch.backends.cuda.matmul.allow_tf32 = False
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev))
K = 64

# Test on a range of block sizes: pick blocks by Dff (tiny, medium, big)
blocks = [b for b in model.blocks if getattr(b.ffn, "W_up", None) is not None]
dffs = [(_dense_of(b.ffn.W_up).shape[0], i) for i, b in enumerate(blocks)]
dffs.sort()
# sample: smallest, median, largest, and a few in between
sample_idx = [dffs[0][1], dffs[len(dffs)//4][1], dffs[len(dffs)//2][1],
              dffs[3*len(dffs)//4][1], dffs[-1][1]]
sample_idx = list(dict.fromkeys(sample_idx))

worst_ug = 0.0; worst_full = 0.0
for i in sample_idx:
    ffn = blocks[i].ffn
    Dff = _dense_of(ffn.W_up).shape[0]
    x = torch.randn(1, K, model.dim, device=dev) * 0.5
    ref = ffn.forward(x)                          # dense/sparse-mm reference
    ug = FusedUpGateSiluFFN(ffn, dev).forward(x)
    fl = FusedFullFFN(ffn, dev).forward(x)
    e_ug = (ug - ref).abs().max().item()
    e_fl = (fl - ref).abs().max().item()
    refmax = ref.abs().max().item()
    print(f"block {i:3d} Dff={Dff:5d}  |refmax|={refmax:7.3f}  "
          f"upgate L-inf={e_ug:.3e} ({e_ug/max(refmax,1e-9):.2e} rel)  "
          f"full L-inf={e_fl:.3e} ({e_fl/max(refmax,1e-9):.2e} rel)")
    worst_ug = max(worst_ug, e_ug/max(refmax,1e-9))
    worst_full = max(worst_full, e_fl/max(refmax,1e-9))
print(f"\nWORST rel L-inf: upgate={worst_ug:.3e}  full={worst_full:.3e}")
print("(both should be ~fp-accum-order, <<1 nibble margin; up/gate ~exact)")
