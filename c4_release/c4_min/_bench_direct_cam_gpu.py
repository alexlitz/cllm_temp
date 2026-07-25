"""PART B GPU ms/step: the global CAM head O(K)/O(K^2) softmax score vs the O(1)
direct-index gather.

Two regimes:
  * DECODE (1 query row vs K cache rows): the per-step CAM score the token-by-token
    driver runs — O(K) per read.
  * BATCHED-VERIFY (Sq = block_steps*30 query rows vs Sk cache rows): the fused fast
    path — the score matrix ``[H, Sq, Sk]`` is O(Sq*Sk) ~ O(K^2), the exact cost the
    verify_blocks docstring calls the O(K^2) VRAM ceiling (``oom_backoff`` halves the
    block on OOM).  The direct gather reads ONE row per query — no score matrix, no
    O(K^2) VRAM.

Run:  python -m c4_min._bench_direct_cam_gpu
Needs a free CUDA card (>=18 GB).  Prints ms/step for softmax (3 heads = current,
2 heads = C4_UNIFY_CAM_HEAD) vs the direct gather, across cache depths.
"""
from __future__ import annotations

import time

import torch

from c4_min.blogspec_model import softmax1

HD = 51        # MEM_HEAD_CHANNELS — a CAM head's per-head dim


def _pick_device():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    # prefer the emptier card
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}"


def _softmax_score(n_global, Sq, Sk, dev, iters):
    Q = torch.randn(1, n_global, Sq, HD, device=dev)
    Kc = torch.randn(1, n_global, Sk, HD, device=dev)
    Vc = torch.randn(1, n_global, Sk, HD, device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        sc = torch.matmul(Q, Kc.transpose(-2, -1)) * (HD ** -0.5)   # [1,H,Sq,Sk] O(Sq*Sk)
        a = softmax1(sc, dim=-1)
        _ = torch.matmul(a, Vc)                                     # O(Sq*Sk)
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3


def _direct_gather(n_global, Sq, Sk, dev, iters):
    Vc = torch.randn(1, n_global, Sk, HD, device=dev)
    row = torch.randint(0, Sk, (1, n_global, Sq, 1), device=dev).expand(1, n_global, Sq, HD)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        _ = torch.gather(Vc, 2, row)                               # [1,H,Sq,HD] O(Sq)
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3


def main():
    dev = _pick_device()
    torch.cuda.set_device(dev)
    _softmax_score(3, 30, 128, dev, 5); _direct_gather(3, 30, 128, dev, 5)  # warmup
    print(f"device={dev} {torch.cuda.get_device_name(dev)}")

    print("\n== DECODE regime (1 query row vs K cache rows) — per-step O(K) score ==")
    print(f"{'K':>10} | {'softmax(3h) ms':>14} {'softmax(2h) ms':>14} {'direct ms':>10} | {'3h speedup':>10}")
    for K in (256, 4096, 65536, 250000):
        s3 = _softmax_score(3, 1, K, dev, 200)
        s2 = _softmax_score(2, 1, K, dev, 200)
        d = _direct_gather(3, 1, K, dev, 200)
        print(f"{K:>10} | {s3:>14.4f} {s2:>14.4f} {d:>10.4f} | {s3/max(d,1e-9):>9.1f}x", flush=True)

    print("\n== BATCHED-VERIFY regime (Sq = block_steps*30 vs Sk) — O(K^2) score matrix ==")
    print(f"{'block_steps':>11} {'Sk':>8} | {'softmax(3h) ms':>14} {'softmax(2h) ms':>14} {'direct ms':>10} | {'3h speedup':>10}")
    for bs, Sk in ((1, 256), (64, 4096), (256, 16384), (1000, 65536)):
        Sq = bs * 30
        try:
            s3 = _softmax_score(3, Sq, Sk, dev, 30)
            s2 = _softmax_score(2, Sq, Sk, dev, 30)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            d = _direct_gather(3, Sq, Sk, dev, 30)
            print(f"{bs:>11} {Sk:>8} | {'OOM (O(K^2) VRAM ceiling)':>31} {d:>10.4f} | direct has NO ceiling", flush=True)
            continue
        d = _direct_gather(3, Sq, Sk, dev, 30)
        print(f"{bs:>11} {Sk:>8} | {s3:>14.4f} {s2:>14.4f} {d:>10.4f} | {s3/max(d,1e-9):>9.1f}x", flush=True)


if __name__ == "__main__":
    main()
