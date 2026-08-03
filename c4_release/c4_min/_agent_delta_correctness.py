import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.fused_sparse_ffn import FusedUpGateSiluDeltaFFN, _dense_of
dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.zeros(1).to(dev)
torch.backends.cuda.matmul.allow_tf32 = False
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev)); K=64
blocks=[b for b in model.blocks if getattr(b.ffn,"W_up",None) is not None]
dffs=sorted((_dense_of(b.ffn.W_up).shape[0],i) for i,b in enumerate(blocks))
sample=list(dict.fromkeys([dffs[0][1],dffs[len(dffs)//4][1],dffs[len(dffs)//2][1],dffs[3*len(dffs)//4][1],dffs[-1][1]]))
worst=0.0
for i in sample:
    ffn=blocks[i].ffn; Dff=_dense_of(ffn.W_up).shape[0]
    x=torch.randn(1,K,model.dim,device=dev)*0.5
    ref=ffn.forward(x); d=FusedUpGateSiluDeltaFFN(ffn,dev).forward(x)
    e=(d-ref).abs().max().item(); rm=ref.abs().max().item()
    print(f"block {i:3d} Dff={Dff:5d} n_active={FusedUpGateSiluDeltaFFN(ffn,dev).n_active:4d} L-inf={e:.3e} rel={e/max(rm,1e-9):.2e}")
    worst=max(worst,e/max(rm,1e-9))
print(f"WORST rel={worst:.3e}")
