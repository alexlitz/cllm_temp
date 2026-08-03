"""Attribute the fused upgate forward time: upgate-silu kernel vs down kernel,
per-block, and the pure per-launch floor."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch, triton, time
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.fused_sparse_ffn import (FusedUpGateSiluFFN, _dense_of,
    _fused_upgate_silu_kernel, _down_resid_kernel)

dev = torch.device("cuda:0")
torch.cuda.set_device(dev); torch.zeros(1).to(dev)
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
model.to(str(dev))
D = model.dim; K = 1024; BK = 256

# Build all fused FFNs
fused = []
for b in model.blocks:
    if getattr(b.ffn, "W_up", None) is None: continue
    fused.append(FusedUpGateSiluFFN(b.ffn, dev, block_k=BK))
print(f"{len(fused)} fused blocks, K={K}, BLOCK_K={BK}")

x = torch.randn(K, D, device=dev)*0.5
xk = x.transpose(0,1).contiguous()

def time_it(fn, n=30, w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1e6  # us

# time ONLY the upgate-silu kernels (all blocks) vs ONLY the down kernels
hbufs = [torch.empty(f.Dff, K, device=dev) for f in fused]
def run_upgate():
    for f,h in zip(fused,hbufs):
        g=(f.Dff, triton.cdiv(K,BK))
        _fused_upgate_silu_kernel[g](f.up_crow,f.up_col,f.up_val,f.gt_crow,f.gt_col,f.gt_val,
            xk,f.b_up,f.b_gate,h,K,xk.stride(0),xk.stride(1),h.stride(0),h.stride(1),BLOCK_K=BK)
ybuf = torch.empty(D,K,device=dev)
def run_down():
    for f,h in zip(fused,hbufs):
        g=(D, triton.cdiv(K,BK))
        _down_resid_kernel[g](f.dn_crow,f.dn_col,f.dn_val,h,xk,ybuf,K,
            h.stride(0),h.stride(1),xk.stride(0),xk.stride(1),ybuf.stride(0),ybuf.stride(1),BLOCK_K=BK)

t_ug = time_it(run_upgate)
t_dn = time_it(run_down)
print(f"upgate-silu (242 launches): {t_ug:8.1f} us  ({t_ug/242:.2f} us/launch)")
print(f"down        (242 launches): {t_dn:8.1f} us  ({t_dn/242:.2f} us/launch)")
print(f"sum = {t_ug+t_dn:.1f} us  (graph replay measured ~12300 us; eager overhead extra)")

# Pure launch floor: an empty-ish kernel launched 242x
@triton.jit
def _noop(y_ptr, K, BLOCK_K: tl.constexpr):
    pass
import triton.language as tl
# measure a trivial down-sized launch floor
def run_trivial():
    for _ in range(242):
        g=(D, triton.cdiv(K,BK))
        _down_resid_kernel[g](fused[0].dn_crow,fused[0].dn_col,fused[0].dn_val,hbufs[0],xk,ybuf,K,
            hbufs[0].stride(0),hbufs[0].stride(1),xk.stride(0),xk.stride(1),ybuf.stride(0),ybuf.stride(1),BLOCK_K=BK)
# per-block down grid is always (D=1679, K/BK) regardless of block -> occupancy is fine
# The upgate grid (Dff, K/BK) is tiny for small Dff. Count total upgate programs:
tot_ug_prog = sum(f.Dff * triton.cdiv(K,BK) for f in fused)
tot_dn_prog = len(fused) * D * triton.cdiv(K,BK)
print(f"total upgate programs across 242 launches: {tot_ug_prog:,}  (avg {tot_ug_prog/242:.0f}/launch)")
print(f"total down   programs across 242 launches: {tot_dn_prog:,}  (avg {tot_dn_prog/242:.0f}/launch)")
print(f"SMs=64. down avg programs/launch={tot_dn_prog/242:.0f} (good). upgate small blocks underfill.")
