#!/usr/bin/env python3
"""_agent_megachain_sweep.py — isolate the FULL 238-block dead-FFN mega chain (the
profiler's 69%, 1.77 us/step) and sweep block_k for the 2-kernel vs fused-hidden path
at the doom chunk (65536), to find whether the fused-hidden lever wins and at what tile.

Builds the SAME streaming model the dispatch profiler uses, extracts the DIV-free
dead-FFN blocks, builds ONE MegaBlockChain over the whole dead chain, and times its
run() (raw kernel stream, as precomputed_schedule drives it) at chunk=65536 for each
block_k, flag OFF (2-kernel) and ON (fused-hidden).  Reports us/step, achieved GB/s over
the profiler's structural hidden-bytes roofline, and byte-exact L-inf(fused vs 2k).

GPU 1.  Memory: builds the streaming model once (~same as the profiler, which fits 24 GB).
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
import torch

from . import isa
from .lib_neural import build_lib_model_streaming
from .tight_attn_compose import install_composed
from .step_block_skip import build_live_index
from . import fused_megablock as FM
from .fused_megablock import MegaBlockChain, _dense_of

HBM_BW_GBs = 768.0
RENDER_STEPS = 358_058

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _mem_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return float(l.split()[1]) / 1e6
    return 1e9


def dead_chain(model, L):
    """The DIV-free dead-FFN blocks in [cut=1,N) as ONE contiguous segment list (drop the
    live CAM blocks — we time the dead chain, the profiler's 69%)."""
    li = build_live_index(model, L)
    union = set()
    for op in li:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(li[op])
    region = sorted(b for b in union if b >= 1)
    dead = [b for b in region
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    return dead


def time_fn(fn, reps=30, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def hidden_bytes_per_kcol(model, dead):
    """The profiler's structural hidden-scratch bytes/K-col (2-kernel path): 2*Dff*4 per
    distinct dead block (write+read).  This is the 92%-of-bytes term the fused path drops."""
    seen = set(); tot = 0
    for b in dead:
        ffn = model.blocks[b].ffn
        if id(ffn) in seen:
            continue
        seen.add(id(ffn))
        tot += 2 * int(_dense_of(ffn.W_up).shape[0]) * 4
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--bks", default="16,32,64,128,256")
    a = ap.parse_args()
    if _mem_gb() < 25.0:
        raise SystemExit(f"[GUARD] MemAvail {_mem_gb():.1f}GB < 25 -> STOP")
    for f in COMPOSED:
        os.environ[f] = "1"
    dev = "cuda:0"
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    install_composed(model, verbose=False)
    dead = dead_chain(model, L)
    D = model.dim
    hbytes = hidden_bytes_per_kcol(model, dead)
    print(f"built n_blocks={len(model.blocks)} dim={D} in {time.time()-t0:.1f}s  "
          f"dead-FFN chain={len(dead)} blocks  hidden bytes/Kcol(2k)={hbytes}  "
          f"MemAvail={_mem_gb():.1f}GB", flush=True)

    K = a.chunk
    hq = torch.randn(1, K, D, device=dev) * 0.1
    print(f"\nchunk K={K}  (per-step = whole-dead-chain time / K)", flush=True)
    print(f"{'block_k':>7s} | {'2k us/st':>9s} {'F us/st':>9s} {'F/2k':>6s} | "
          f"{'2k GB/s':>8s} {'2k%pk':>6s} | {'Linf(FvH)':>9s}", flush=True)
    for bk in [int(x) for x in a.bks.split(",")]:
        os.environ["C4_MEGABLOCK_BLOCK_K"] = str(bk)
        os.environ["C4_FFN_FUSED_HIDDEN"] = "0"
        ch2 = MegaBlockChain(model, dev, dead)
        os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
        chf = MegaBlockChain(model, dev, dead)
        os.environ["C4_FFN_FUSED_HIDDEN"] = "0"
        with torch.no_grad():
            o2 = ch2.run(hq); of = chf.run(hq)
        linf = (o2 - of).abs().max().item()
        t2 = time_fn(lambda: ch2.run(hq))
        tf = time_fn(lambda: chf.run(hq))
        us2 = t2 / K * 1e3; usf = tf / K * 1e3
        # 2k achieved GB/s over the structural hidden bytes (92% of the profiler's total).
        gbs2 = (hbytes * K) / (t2 / 1e3) / 1e9
        print(f"{bk:7d} | {us2:8.3f} {usf:8.3f} {t2/tf:5.2f}x | "
              f"{gbs2:8.1f} {100*gbs2/HBM_BW_GBs:5.1f}% | {linf:9.2e}", flush=True)


if __name__ == "__main__":
    main()
