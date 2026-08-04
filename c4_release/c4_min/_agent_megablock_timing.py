#!/usr/bin/env python3
"""_agent_megablock_timing.py — ms/step of the fused megablock vs the baseline.

Times the DIV-free per-step floor three ways at K in {512,2048,8192}:
  (A) EAGER per-block loop over the DIV-free [cut=1,N) region (the baseline path).
  (B) MEGA  eager: dead-FFN contiguous segments -> MegaBlockChain.run (single-launch
            fused per block, residual reused in place), live CAM blocks eager.
  (C) MEGA-G: the whole dead-FFN chain captured in ONE CUDA graph per segment
            (MegaBlockChain.run_graphed) + live CAM eager -> collapses the 49x
            launch overhead to N_seg graph replays.
Reports ms/step, steps/sec, sec/frame (6.89M), and x-above the 0.0071 us FLOP floor,
vs the 0.034 ms baseline the task cites.

The 3 live CAM blocks (2,7,11) are direct-CAM O(1) gathers in the real driver; here
they are run via the installed live-head attention forward in ALL paths (identical),
so the delta isolates the dead-FFN fusion (the residual-traffic + launch lever).
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


def _mem_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return float(l.split()[1]) / 1e6
    return 1e9


def build():
    dev = "cuda:0"
    torch.cuda.set_device(0)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    install_block_sparse_ffn(model, mode="coo", verbose=False)
    model.to(dev)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    return model, L, dev


def divfree_region(model, L):
    live_index = build_live_index(model, L)
    union = set()
    for op in live_index:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(live_index[op])
    region = sorted(b for b in union if b >= 1)
    live = [b for b in region
            if not getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    return region, live


def build_items(model, dev, region, live, block_k):
    live_set = set(live); items = []; seg = []
    for b in region:
        if b in live_set:
            if seg:
                items.append(("mega", MegaBlockChain(model, dev, seg, block_k))); seg = []
            items.append(("live", b))
        else:
            seg.append(b)
    if seg:
        items.append(("mega", MegaBlockChain(model, dev, seg, block_k)))
    return items


def time_fn(fn, reps=40, warmup=12):
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
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--Ks", default="512,2048,8192")
    a = ap.parse_args()
    model, L, dev = build()
    region, live = divfree_region(model, L)
    items = build_items(model, dev, region, live, a.block_k)
    nseg = sum(1 for k, _ in items if k == "mega")
    print(f"dim={model.dim}  DIV-free region={len(region)} blocks "
          f"({len(region)-len(live)} dead-FFN in {nseg} segs, {len(live)} live CAM)  "
          f"block_k={a.block_k}  MemAvail={_mem_gb():.1f}GB", flush=True)
    print(f"FLOP floor={FLOP_FLOOR_US:.5f} us/step ; task baseline={BASELINE_MS} ms/step",
          flush=True)

    def eager(hq, qp):
        h = hq
        for b in region:
            h, _ = model.blocks[b](h, past_kv=None, q_positions=qp, use_cache=True)
        return h

    def mega(hq, qp, graphed):
        h = hq
        for kind, p in items:
            if kind == "live":
                h, _ = model.blocks[p](h, past_kv=None, q_positions=qp, use_cache=True)
            else:
                h = p.run_graphed(h) if graphed else p.run(h)
        return h

    print(f"\n{'K':>6s} {'eager ms':>9s} {'mega ms':>9s} {'megaG ms':>9s} | "
          f"{'best':>8s} {'vs0.034':>8s} {'steps/s':>10s} {'sec/frm':>9s} {'x-floor':>9s}",
          flush=True)
    for K in [int(x) for x in a.Ks.split(",")]:
        hq = torch.randn(1, K, model.dim, device=dev) * 0.1
        qp = torch.arange(K, device=dev, dtype=torch.long)
        t_e = time_fn(lambda: eager(hq, qp))
        t_m = time_fn(lambda: mega(hq, qp, False))
        t_g = time_fn(lambda: mega(hq, qp, True))
        best = min(t_m, t_g)
        # per-step: the whole region is ONE VM step over K rows, so ms/step is the
        # measured region time (K rows batched); ms per (step,row) = best/K.
        # steps/sec at this K = 1000/best * (throughput normalization: the batch
        # of K rows are K independent decode positions -> K steps done per launch).
        steps_per_launch = K
        sps = steps_per_launch * 1000.0 / best
        secframe = FRAME_STEPS / sps
        # x-above floor: per-step best = best/K ms; floor is FLOP_FLOOR_US us
        per_step_us = best / K * 1e3
        xfloor = per_step_us / FLOP_FLOOR_US
        vs = BASELINE_MS / (best / K)
        print(f"{K:6d} {t_e:9.4f} {t_m:9.4f} {t_g:9.4f} | "
              f"{best/K*1e3:7.2f}us {vs:7.2f}x {sps:10.0f} {secframe:9.1f} {xfloor:9.0f}",
              flush=True)


if __name__ == "__main__":
    main()
