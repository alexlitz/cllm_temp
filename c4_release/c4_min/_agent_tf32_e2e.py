"""#751 — END-TO-END ms/step + util of forward_span, TF32 ON vs OFF (the unlock's
real-forward payoff), and the fused-concat-FFN grouped path measured in situ.

Times ``KBatchBoundedRunner.forward_span`` at K on a synthetic bounded stream for a
shallow op (ADD) and the deep DIV megablock, under:
  * TF32 OFF (the current #748 baseline path)
  * TF32 ON  (the #751 tensor-core unlock; byte-exact policy for shallow ops / K=1)
and reports ms/forward, ms/step_eff, executed-FLOP util, useful-FLOP util.  This is the
honest "what does the unlock buy on the WHOLE forward, not just an isolated FFN" number.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_tf32_e2e --device cuda:0 --K 128
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_POS_SPARSE"] = "1"

import torch

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from . import bench_pf_kbatch as B

A5000_FP32 = 27.8e12
A5000_TF32 = 55.6e12


def _time(fn, n=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--base-s", type=int, default=900)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=15.0)
    a = ap.parse_args(argv)
    B.runner_window = a.window

    device = a.device
    idx = int(device.split(":")[1]) if ":" in device else 0
    if not a.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    from .compact_alloc import build_compact_sparse_streaming
    model, L, _ = build_compact_sparse_streaming(code_size=a.code_size, compute_mode="dense_kernel")
    model.to(device); model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=a.window, selective_fp64=True)
    D = model.embed.shape[1]
    K = a.K
    print(f"[built] blocks={len(model.blocks)} dim={D} K={K}", flush=True)

    for opname in ("ADD", "DIV"):
        op = getattr(isa, opname)
        x, q_idxs, S = B._make_timing_stream(model, L, K, a.base_s, op, a.window)
        ops = [op] * len(q_idxs)
        live = runner.live_union(ops)
        n_store = int((x[0, :, int(L.IS_STORE)] != 0).sum())
        useful = B._count_flops_forward(model, L, runner, K, q_idxs, ops, D, a.window, n_store)

        def fwd():
            with torch.no_grad():
                return runner.forward_span(x, ops, q_idxs)

        print(f"\n{'='*80}\n[{opname}] S={S} live_blocks={len(live)} K={K} "
              f"useful={useful/1e9:.3f} GFLOP/fwd\n{'='*80}", flush=True)
        print(f"  {'tf32':6s} {'ms/fwd':>10} {'ms/step_eff':>12} {'useful-util%':>13} "
              f"{'speedup':>9}", flush=True)
        base_ms = None
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            ms = _time(fwd)
            if base_ms is None:
                base_ms = ms
            eff = ms / max(len(q_idxs), 1)
            util = 100.0 * (useful / (ms / 1e3)) / A5000_FP32
            print(f"  {'ON' if tf32 else 'OFF':6s} {ms:10.4f} {eff:12.5f} {util:13.4f} "
                  f"{base_ms/ms:8.2f}x", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
