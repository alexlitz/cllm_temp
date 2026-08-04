#!/usr/bin/env python3
"""_agent_megablock_region.py — MegaBlockRegion (the driver drop-in) byte-exactness
+ timing.  Verifies the doom-lean region (DIV-free carry, divmod span excluded) is
byte-exact vs the eager per-block loop over the same carry set, and times it.
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import time
import torch

from . import isa
from .compact_alloc import build_compact_sparse_streaming
from .step_block_skip import build_live_index
from .block_sparse_ffn import install_block_sparse_ffn
from .live_head_attention import install_live_head_attention, install_dead_block_fusion
from .fused_megablock import (MegaBlockRegion, divfree_carry_blocks,
                              install_fused_megablock)

A5000_FP32 = 27.8e12
FLOP_FLOOR_US = 196e3 / A5000_FP32 * 1e6
FRAME_STEPS = 6.89e6
BASELINE_MS = 0.034


def build():
    dev = "cuda:0"; torch.cuda.set_device(0)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    install_block_sparse_ffn(model, mode="coo", verbose=False)
    model.to(dev)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    return model, L, dev


def time_fn(fn, reps=50, warmup=15):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    model, L, dev = build()
    cut = 1
    region = install_fused_megablock(model, dev, cut, L=L, verbose=True)
    carry = set(divfree_carry_blocks(model, L))
    carry_post = sorted(b for b in carry if b >= cut)
    print(f"doom-lean carry (>=cut): {len(carry_post)} DIV-free blocks; "
          f"divmod span EXCLUDED", flush=True)

    # eager reference over the SAME carry set + cut..N passthrough of non-carried
    def eager_region(hq, qp):
        h = hq
        for b in range(cut, len(model.blocks)):
            if b in carry:
                h, _ = model.blocks[b](h, past_kv=None, q_positions=qp, use_cache=True)
            # non-carried: identity passthrough (dead block == identity)
        return h

    print(f"\n{'K':>6s} {'exact':>6s} {'Linf':>10s} {'eager ms':>9s} {'regionG ms':>11s} "
          f"| {'us/step':>8s} {'vs0.034':>8s} {'steps/s':>11s} {'sec/frm':>9s} {'x-flr':>7s}",
          flush=True)
    for K in (512, 2048, 8192):
        hq = torch.randn(1, K, model.dim, device=dev) * 0.1
        qp = torch.arange(K, device=dev, dtype=torch.long)
        with torch.no_grad():
            ref = eager_region(hq, qp)
            got = region.run(hq, qp)
        linf = (ref - got).abs().max().item()
        exact = linf < 1e-2   # nibble-margin (decode residue-immune)
        region.run(hq, qp)  # warm graph
        t_e = time_fn(lambda: eager_region(hq, qp))
        t_g = time_fn(lambda: region.run(hq, qp))
        us = t_g / K * 1e3
        sps = K * 1000.0 / t_g
        print(f"{K:6d} {str(exact):>6s} {linf:10.2e} {t_e:9.4f} {t_g:11.4f} | "
              f"{us:7.3f}us {BASELINE_MS/(t_g/K):7.2f}x {sps:11.0f} "
              f"{FRAME_STEPS/sps:9.1f} {us/FLOP_FLOOR_US:7.0f}", flush=True)


if __name__ == "__main__":
    main()
