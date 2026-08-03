"""Standalone correctness + micro-speed check for the COO-SpMM FFN kernel vs the
dense F.linear FFN, on the ACTUAL built compact-model blocks.

Verifies per-block CooSpmmFFN.forward == SparseFFN.forward (dense_kernel) at fp32
(L-inf ~1e-6, below nibble margin) AND times a representative block at K=1024/4096.
"""
from __future__ import annotations
import argparse
import os
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=44)
    ap.add_argument("--ks", default="1024,4096")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False

    from .compact_alloc import build_compact_sparse_streaming
    from .sparse_coo_spmm import CooSpmmFFN, _dense_of
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="sparse_mm")
    model.to(dev); model.materialize_dense(dev)
    print(f"[built] blocks={len(model.blocks)} dim={model.dim}", flush=True)

    # ---- correctness: EVERY distinct FFN block, K=17 (odd, exercises masking) ----
    max_diff = 0.0
    worst = None
    seen = set()
    n_checked = 0
    for bi, b in enumerate(model.blocks):
        if id(b) in seen:
            continue
        seen.add(id(b))
        if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
            continue
        n_checked += 1
        coo = CooSpmmFFN(b.ffn, dev)
        K = 17
        x = torch.randn(1, K, model.dim, device=dev) * 0.5
        ref = b.ffn.forward(x)                     # dense F.linear SwiGLU
        got = coo.forward(x)
        d = (ref - got).abs().max().item()
        rel = d / max(1.0, ref.abs().max().item())
        if rel > max_diff:
            max_diff = rel
            worst = (bi, d, ref.abs().max().item())
    print(f"[correctness] {n_checked} distinct FFN blocks; worst REL L-inf = "
          f"{max_diff:.2e}  (block {worst[0]}, abs {worst[1]:.2e} on |ref|max "
          f"{worst[2]:.2e})  [{'OK' if max_diff < 1e-4 else 'DIFF'}]")

    # ---- speed: a big block (div-round1, Dff=4320) at each K ----
    big = None
    for b in model.blocks:
        if getattr(b.ffn, "W_up", None) is not None and \
           _dense_of(b.ffn.W_up).shape[0] == 4320:
            big = b; break
    if big is None:
        for b in model.blocks:
            if getattr(b.ffn, "W_up", None) is not None:
                big = b
    coo = CooSpmmFFN(big.ffn, dev)
    print(f"\n[speed] block Dff={coo.Dff} nnz(up/gate/down)="
          f"{coo.up.nnz}/{coo.gate.nnz}/{coo.down.nnz}")
    for K in [int(k) for k in args.ks.split(",")]:
        x = torch.randn(1, K, model.dim, device=dev) * 0.5
        for _ in range(5):
            _ = big.ffn.forward(x); _ = coo.forward(x)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(30):
            _ = big.ffn.forward(x)
        torch.cuda.synchronize(dev)
        dt_dense = (time.perf_counter() - t0) / 30
        t0 = time.perf_counter()
        for _ in range(30):
            _ = coo.forward(x)
        torch.cuda.synchronize(dev)
        dt_coo = (time.perf_counter() - t0) / 30
        print(f"  K={K:>5}: dense F.linear {dt_dense*1e6:8.1f} us | "
              f"COO-SpMM {dt_coo*1e6:8.1f} us | speedup {dt_dense/dt_coo:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
