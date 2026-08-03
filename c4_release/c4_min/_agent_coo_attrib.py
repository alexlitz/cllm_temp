"""Attribute the COO composed-forward time: FFN-COO kernels vs attention vs the
transpose/silu glue.  Answers WHY the full-step speedup (2.16x) is below the
per-block micro-speedup (7.5x)."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import time
import torch


def _time(fn, iters, warmup=5, dev=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / iters


def main():
    from .compact_alloc import build_compact_sparse_streaming
    from .sparse_coo_spmm import CooSpmmFFN
    from ._agent_flash_roofline_largek import (
        make_live_blocks_capturable, count_eager_launches)
    from .live_head_attention import (install_live_head_attention,
                                      install_dead_block_fusion)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")
    model.to(str(dev))
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)

    coos = []
    for b in model.blocks:
        if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
            continue
        coo = CooSpmmFFN(b.ffn, dev)
        b.ffn._coo = coo; b.ffn.forward = coo.forward
        coos.append((b, coo))
    make_live_blocks_capturable(model, dev)

    launch = count_eager_launches(model)
    K = 1024
    qpos = torch.arange(K, device=dev, dtype=torch.long)
    x = torch.randn(1, K, model.dim, device=dev) * 0.5
    D = model.dim

    # (1) full composed step
    def full():
        h = x
        for blk in model.blocks:
            out = blk(h, past_kv=None, q_positions=qpos, use_cache=True)
            h = out[0] if isinstance(out, tuple) else out
        return h
    t_full = _time(full, 15, dev=dev)

    # (2) just the FFN-COO across all blocks (no attention) — the sparse GEMM cost
    def ffn_only():
        h = x
        for blk in model.blocks:
            h = blk.ffn.forward(h)
        return h
    t_ffn = _time(ffn_only, 15, dev=dev)

    # (3) just the 3 live-attention blocks' attention
    live_blocks = [b for b in model.blocks
                   if not getattr(b.attn, "_dead_block_fused", False)]
    def attn_only():
        for blk in live_blocks:
            blk.attn.forward(x, past_kv=None, q_positions=qpos, use_cache=True)
    t_attn = _time(attn_only, 15, dev=dev)

    # (4) one COO ffn block forward (median-ish and the big one)
    big = max(coos, key=lambda bc: bc[1].Dff)[1]
    def one_big():
        big.forward(x)
    t_one = _time(one_big, 30, dev=dev)

    # (5) the glue in one CooSpmmFFN.forward: transpose+silu+transpose WITHOUT kernels
    #     (measure a raw silu*mul on [Dff,K] to size the elementwise cost)
    Dff = big.Dff
    up = torch.randn(Dff, K, device=dev); gate = torch.randn(Dff, K, device=dev)
    def glue():
        _ = torch.nn.functional.silu(up) * gate
    t_glue = _time(glue, 50, dev=dev)

    n_ffn = len(coos)
    print(f"K={K}  blocks={len(model.blocks)}  FFN blocks={n_ffn}  "
          f"eager launches~{launch['eager_launches']}")
    print(f"  full composed step         : {t_full*1e3:8.3f} ms")
    print(f"  FFN-COO only (all blocks)  : {t_ffn*1e3:8.3f} ms "
          f"({t_ffn/t_full*100:.0f}% of full)")
    print(f"  live-attention only (3 blk): {t_attn*1e3:8.3f} ms "
          f"({t_attn/t_full*100:.0f}% of full)")
    print(f"  one big COO FFN (Dff={Dff}) : {t_one*1e6:8.1f} us  "
          f"x{n_ffn} = {t_one*n_ffn*1e3:.2f} ms if all this size")
    print(f"  silu*mul glue [Dff={Dff},K]  : {t_glue*1e6:8.1f} us")
    print(f"  --> per-FFN-block avg       : {t_ffn/n_ffn*1e6:8.1f} us "
          f"(3 kernels + 2 transposes + silu each)")
    # kernels launched per step: 3 spmm * n_ffn + attention
    print(f"  --> ~{3*n_ffn} COO kernel launches/step (3 per FFN block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
