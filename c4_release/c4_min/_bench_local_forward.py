#!/usr/bin/env python3
"""STEP-4 forward MICROBENCH — the O(S^2)->O(S*W) attention win, isolated.

Builds the complete streaming model, constructs ONE big-K verify span (S tokens,
matching the nested-6x100 fast-path span ~1400-1900) against a large KV cache, and
TIMES the batched ``forward_hidden_cached`` (the fast-path forward) with GLOBAL
attention vs LOCAL (sliding-window) attention.  Reports ms/step-equiv, forward
wall, peak VRAM and GPU util for both — the pure forward speedup, no eviction /
draft / Python-overlay noise.

Run:  python -m c4_min._bench_local_forward --device cuda:1 --cache 1400 --span 1900
"""
from __future__ import annotations
import argparse
import sys
import time

import torch

from c4_min import blogspec_vocab as V
from c4_min.lib_neural import build_lib_model_streaming
from c4_min import local_attention as LA
from c4_min import sparse_forward as _SF


def _sample_gpu_util(device, stop_evt, out):
    import subprocess
    idx = device.split(":")[-1] if ":" in device else "0"
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "-i", idx, "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=2)
            out.append(int(r.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        stop_evt.wait(0.05)


def _time_forward(model, dev, S, cache, window, reps=6, gpu_util=False):
    """Time ``forward_hidden_cached`` over a span of ``S`` NEW rows against a
    ``cache``-row past KV.  Returns (ms_per_forward, peak_vram_gb, util_mean)."""
    if window is None:
        LA.uninstall_local_attention(model)
    else:
        LA.install_local_attention(model, window=window)
    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    D = model.dim
    # a synthetic past KV cache per block (random projected K/V at absolute
    # positions [0..cache-1]); the NEW span is at [cache..cache+S-1].
    torch.manual_seed(0)
    past = []
    for b in range(n_blocks):
        K = torch.randn(1, H, cache, HD, device=dev) * 0.05
        Vv = torch.randn(1, H, cache, HD, device=dev) * 0.05
        pos = torch.arange(cache, device=dev)
        past.append((K, Vv, pos))
    x = torch.randn(1, S, D, device=dev) * 0.05
    q_pos = torch.arange(cache, cache + S, device=dev)

    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
    # warmup
    with torch.no_grad():
        model.forward_hidden_cached(x, past_key_values=past, q_positions=q_pos,
                                    use_cache=True)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()

    util = []
    stop_evt = None
    th = None
    if gpu_util:
        import threading
        stop_evt = threading.Event()
        th = threading.Thread(target=_sample_gpu_util, args=(dev, stop_evt, util),
                              daemon=True)
        th.start()
    t = time.time()
    with torch.no_grad():
        for _ in range(reps):
            model.forward_hidden_cached(x, past_key_values=past, q_positions=q_pos,
                                        use_cache=True)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t
    if gpu_util and stop_evt is not None:
        stop_evt.set(); th.join(timeout=1)
    peak = (torch.cuda.max_memory_allocated(dev) / 1e9
            if dev.startswith("cuda") else 0.0)
    LA.uninstall_local_attention(model)
    del past, x
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    ms_per_fwd = wall / reps * 1000
    util_mean = (sum(util) / len(util)) if util else 0.0
    return ms_per_fwd, peak, util_mean


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--span", type=int, default=1900,
                    help="new query rows per forward (nested-6x100 span ~1400-1900)")
    ap.add_argument("--cache", type=int, default=1400,
                    help="past KV rows (the accumulated frozen frames)")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--k-forward-steps", type=int, default=None,
                    help="if set, span = k*FRAME_LEN (K VM steps per forward)")
    args = ap.parse_args(argv)

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    span = (args.k_forward_steps * V.FRAME_LEN if args.k_forward_steps
            else args.span)
    print(f"[build] streaming complete model on {dev} ...", flush=True)
    model, L, _ = build_lib_model_streaming(compute_mode="dense_kernel")
    model.to(dev)
    n_blocks = len(model.blocks)
    print(f"[build] n_blocks={n_blocks} n_heads={model.blocks[0].attn.n_heads} "
          f"dim={model.dim}", flush=True)
    print(f"[bench] span S={span}  cache={args.cache}  "
          f"total keys Sk={args.cache + span}  reps={args.reps}", flush=True)

    print("\n[global] timing full-causal attention forward ...", flush=True)
    g_ms, g_vram, g_util = _time_forward(model, dev, span, args.cache, None,
                                         reps=args.reps, gpu_util=True)
    print(f"  GLOBAL: {g_ms:.1f} ms/forward  ({g_ms/max(span//V.FRAME_LEN,1):.2f} "
          f"ms/step-equiv)  peak_vram={g_vram:.2f} GB  gpu_util={g_util:.0f}%",
          flush=True)

    print(f"\n[local ] timing sliding-window (W={args.window}) attention forward "
          f"...", flush=True)
    l_ms, l_vram, l_util = _time_forward(model, dev, span, args.cache, args.window,
                                         reps=args.reps, gpu_util=True)
    print(f"  LOCAL : {l_ms:.1f} ms/forward  ({l_ms/max(span//V.FRAME_LEN,1):.2f} "
          f"ms/step-equiv)  peak_vram={l_vram:.2f} GB  gpu_util={l_util:.0f}%",
          flush=True)

    steps = max(span // V.FRAME_LEN, 1)
    print("\n==================== STEP-4 FORWARD WIN ====================")
    print(f"  span S={span} ({steps} VM steps/forward), cache={args.cache}, "
          f"Sk={args.cache + span}")
    print(f"  GLOBAL forward: {g_ms:7.1f} ms  ({g_ms/steps:6.2f} ms/step)  "
          f"vram {g_vram:.2f} GB  util {g_util:.0f}%")
    print(f"  LOCAL  forward: {l_ms:7.1f} ms  ({l_ms/steps:6.2f} ms/step)  "
          f"vram {l_vram:.2f} GB  util {l_util:.0f}%")
    print(f"  FORWARD SPEEDUP: {g_ms/max(l_ms,1e-9):.2f}x   "
          f"VRAM: {g_vram/max(l_vram,1e-9):.2f}x lower")
    return 0


if __name__ == "__main__":
    sys.exit(main())
