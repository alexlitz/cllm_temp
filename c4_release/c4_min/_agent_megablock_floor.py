#!/usr/bin/env python3
"""_agent_megablock_floor.py — the pure dead-FFN compute floor: ALL 49 DIV-free
dead-FFN blocks as ONE MegaBlockChain captured in ONE CUDA graph.  Isolates the
residual-traffic + launch floor from the 3 live CAM blocks (O(1) gathers, excluded).
Reports ms/step, steps/s, sec/frame, x-above the FLOP floor, vs the 0.034 baseline.
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
from .fused_megablock import MegaBlockChain

A5000_FP32 = 27.8e12
FLOP_PER_STEP = 196e3
FLOP_FLOOR_US = FLOP_PER_STEP / A5000_FP32 * 1e6
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-k", type=int, default=128)
    ap.add_argument("--Ks", default="512,2048,8192")
    a = ap.parse_args()
    model, L, dev = build()
    li = build_live_index(model, L)
    union = set()
    for op in li:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(li[op])
    dead = sorted(b for b in union if b >= 1
                  and getattr(model.blocks[b].attn, "_dead_block_fused", False))
    print(f"pure dead-FFN floor: {len(dead)} DIV-free dead-FFN blocks captured as ONE "
          f"megakernel graph  block_k={a.block_k}", flush=True)
    print(f"FLOP floor={FLOP_FLOOR_US:.5f} us/step ; task baseline={BASELINE_MS} ms/step",
          flush=True)
    chain = MegaBlockChain(model, dev, dead, a.block_k)

    print(f"\n{'K':>6s} {'eager ms':>9s} {'megaG ms':>9s} | {'us/step':>8s} "
          f"{'vs0.034':>8s} {'steps/s':>11s} {'sec/frm':>9s} {'x-floor':>9s}",
          flush=True)
    for K in [int(x) for x in a.Ks.split(",")]:
        hq = torch.randn(1, K, model.dim, device=dev) * 0.1

        def eager():
            h = hq.reshape(K, model.dim).transpose(0, 1).contiguous()
            for mf in chain._ffns:
                mf.run_inplace(h, K, h_scratch=chain._get_bufs(K)[1])
            return h
        # warm the graph
        chain.run_graphed(hq)
        t_e = time_fn(eager)
        t_g = time_fn(lambda: chain.run_graphed(hq))
        us = t_g / K * 1e3
        sps = K * 1000.0 / t_g
        secframe = FRAME_STEPS / sps
        xfloor = us / FLOP_FLOOR_US
        vs = BASELINE_MS / (t_g / K)
        print(f"{K:6d} {t_e:9.4f} {t_g:9.4f} | {us:7.3f}us {vs:7.2f}x "
              f"{sps:11.0f} {secframe:9.1f} {xfloor:9.0f}", flush=True)


if __name__ == "__main__":
    main()
