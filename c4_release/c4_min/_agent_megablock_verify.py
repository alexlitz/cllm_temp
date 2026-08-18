#!/usr/bin/env python3
"""_agent_megablock_verify.py — byte-exactness + timing of the fused megablock.

Verifies the on-chip fused dead-FFN megakernel (fused_megablock.MegaBlockChain) is
byte-exact vs the eager per-block loop over the DIV-free dead-FFN blocks, then times
both at K in {512, 2048, 8192} vs the 0.034 ms/step baseline and the FLOP floor.

The 3 live CAM blocks (2,7,11) are run EAGERLY in BOTH paths (they are O(1) direct
gathers in the real driver; here we run their installed live-head attention forward
identically in both so the comparison isolates the dead-FFN fusion).  Measurement
only; byte-neutral (all flags default; the megakernel is opt-in).
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
from .live_head_attention import (install_live_head_attention,
                                  install_dead_block_fusion)
from .fused_megablock import MegaBlockChain

A5000_FP32 = 27.8e12
FLOP_PER_STEP = 196e3
FLOP_FLOOR_US = FLOP_PER_STEP / A5000_FP32 * 1e6
FRAME_STEPS = 6.89e6


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
    """Return the DIV-free live union blocks in [cut=1, N), and split into
    contiguous dead-FFN segments + the live CAM block indices."""
    live_index = build_live_index(model, L)
    union = set()
    for op in live_index:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(live_index[op])
    region = sorted(b for b in union if b >= 1)  # cut=1 -> block 0 excluded
    dead = [b for b in region
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    live = [b for b in region
            if not getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    return region, dead, live


def eager_region(model, region, hq, qp):
    """Eager per-block loop over the DIV-free region (the baseline path)."""
    h = hq
    for bi in region:
        h, _ = model.blocks[bi](h, past_kv=None, q_positions=qp, use_cache=True)
    return h


def build_segment_chains(model, dev, region, live, block_k):
    """Split `region` into (kind, payload) run-items: dead-FFN contiguous runs ->
    a MegaBlockChain; live CAM blocks -> the block index (eager)."""
    live_set = set(live)
    items = []
    seg = []
    for b in region:
        if b in live_set:
            if seg:
                items.append(("mega", MegaBlockChain(model, dev, seg, block_k)))
                seg = []
            items.append(("live", b))
        else:
            seg.append(b)
    if seg:
        items.append(("mega", MegaBlockChain(model, dev, seg, block_k)))
    return items


def run_composed(model, items, hq, qp, graphed=False):
    h = hq
    for kind, payload in items:
        if kind == "live":
            h, _ = model.blocks[payload](h, past_kv=None, q_positions=qp,
                                         use_cache=True)
        else:
            h = payload.run_graphed(h) if graphed else payload.run(h)
    return h


def time_fn(fn, reps=30, warmup=8):
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
    t0 = time.time()
    model, L, dev = build()
    region, dead, live = divfree_region(model, L)
    print(f"built {len(model.blocks)} blocks dim={model.dim} in {time.time()-t0:.1f}s "
          f"MemAvail={_mem_gb():.1f}GB", flush=True)
    print(f"DIV-free region [cut=1,N): {len(region)} blocks "
          f"({len(dead)} dead-FFN, {len(live)} live CAM {live})", flush=True)
    print(f"block_k={a.block_k}  FLOP floor={FLOP_FLOOR_US:.5f} us/step", flush=True)

    items = build_segment_chains(model, dev, region, live, a.block_k)
    nseg = sum(1 for k, _ in items if k == "mega")
    print(f"segments: {nseg} mega dead-FFN chains + {len(live)} eager live blocks",
          flush=True)

    Ks = [int(x) for x in a.Ks.split(",")]
    print(f"\n{'K':>6s} {'byte-exact':>11s} {'Linf':>10s} | "
          f"{'eager ms':>9s} {'mega ms':>9s} {'megaG ms':>9s} | "
          f"{'speedup':>8s} {'steps/s':>10s} {'sec/frame':>10s} {'x-floor':>9s}",
          flush=True)
    for K in Ks:
        hq = (torch.randn(1, K, model.dim, device=dev) * 0.1)
        qp = torch.arange(K, device=dev, dtype=torch.long)
        with torch.no_grad():
            ref = eager_region(model, region, hq, qp)
            got = run_composed(model, items, hq, qp, graphed=False)
            gotg = run_composed(model, items, hq, qp, graphed=True)
        linf = (ref - got).abs().max().item()
        linfg = (ref - gotg).abs().max().item()
        exact = (linf == 0.0) and (linfg == 0.0)

        t_eager = time_fn(lambda: eager_region(model, region, hq, qp))
        t_mega = time_fn(lambda: run_composed(model, items, hq, qp, graphed=False))
        t_megag = time_fn(lambda: run_composed(model, items, hq, qp, graphed=True))
        speed = t_eager / t_megag
        sps = 1000.0 / t_megag
        secframe = FRAME_STEPS / sps
        xfloor = (t_megag * 1e3) / FLOP_FLOOR_US
        print(f"{K:6d} {str(exact):>11s} {max(linf,linfg):10.2e} | "
              f"{t_eager:9.4f} {t_mega:9.4f} {t_megag:9.4f} | "
              f"{speed:7.2f}x {sps:10.0f} {secframe:10.1f} {xfloor:9.0f}",
              flush=True)


if __name__ == "__main__":
    main()
