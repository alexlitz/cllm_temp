"""_agent_dedup_fused_ksweep.py — the composed self-forward per-step us BY K (VRAM
per K) for the dedup-in-C + fused-MAC config, and the full self-forward wall
projected from the GROUNDED 23.2M step count.

The composed KBatchBoundedRunner forward cost per VM step is a property of the
batched attention+FFN forward at a given K (the dedup changes how a WEIGHT is
STORED/READ, not the per-step forward cost; the fused-MAC changes how many STEPS
the matmul portion needs).  We MEASURE the per-step us at K=128/256/512/1024/max
(VRAM peak each), then project the full self-forward wall using:

    baseline    total steps = 23,201,022  (grounded, WITH divmod, position-sparse)
    dedup+fused total steps  = matmul(1 step/MAC) + tail(unchanged)
                             = 164,654 + 10,627,826 = 10,792,480 (2.15x fewer)

so  wall = total_steps * (measured us/step).

CPU builds the model; GPU 0 runs the timed forwards.  Memory-safe: one lean build.
Run:  C4_POS_SPARSE=1 python -m c4_min._agent_dedup_fused_ksweep --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from . import isa

# grounded step counts (ground_true_self_emulation --with-divmod, position-sparse).
BASELINE_STEPS = 23_201_022
TAIL_STEPS = 10_627_826          # non-matmul tail (unchanged by weight-read / fused)
MACS = 164_654                   # nonzero weights = MACs at the S=1 decode row
DEDUP_FUSED_STEPS = MACS * 1 + TAIL_STEPS      # fused MAC = 1 step/MAC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--ks", default="128,256,512,1024")
    ap.add_argument("--max-k", type=int, default=2048,
                    help="the 'max' K to try (bounded so we don't OOM the box)")
    ap.add_argument("--base-s", type=int, default=900)
    a = ap.parse_args(argv)

    os.environ.setdefault("C4_POS_SPARSE", "1")
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[ksweep] CUDA unavailable -> cpu", file=sys.stderr)
        device = "cpu"
    cuda = device.startswith("cuda")

    from .compact_alloc import build_compact_sparse_streaming
    from .pf_kbatch import KBatchBoundedRunner
    from . import bench_pf_kbatch as B

    B.runner_window = a.window
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=a.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=a.window, selective_fp64=True)
    print("[built] n_blocks=%d dim=%d dev=%s build=%.1fs"
          % (len(model.blocks), model.embed.shape[1], device, time.time() - t0),
          flush=True)

    ks = [int(k) for k in a.ks.split(",") if k.strip()]
    if a.max_k not in ks:
        ks.append(a.max_k)

    print("\n" + "=" * 92, flush=True)
    print("COMPOSED-RUNNER per-step us BY K  (dedup-in-C + fused-MAC config; VRAM peak each)",
          flush=True)
    print("=" * 92, flush=True)
    print("  %6s  %10s  %11s  %9s  %9s" % ("K", "ms/forward", "us/step", "kv_rows", "peakGB"),
          flush=True)

    rows = []
    for K in ks:
        try:
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            x0t, q_idxs, S = B._make_timing_stream(model, L, K, a.base_s, isa.ADD, a.window)
            ops_t = [isa.ADD] * len(q_idxs)

            def _fwd():
                with torch.no_grad():
                    return runner.forward_span(x0t, ops_t, q_idxs)

            ms_fwd = B._time_fn(_fwd, 20, 6, cuda)
            us_step = 1000.0 * ms_fwd / max(len(q_idxs), 1)
            kv = runner.kv_rows_max(x0t, q_idxs, ops_t)
            peak = torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0.0
            rows.append((K, ms_fwd, us_step, kv, peak))
            print("  %6d  %10.3f  %11.4f  %9d  %9.2f" % (K, ms_fwd, us_step, kv, peak),
                  flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  %6d  OOM (%s)" % (K, str(e).split(chr(10))[0][:50]), flush=True)
                if cuda:
                    torch.cuda.empty_cache()
                break
            raise

    if not rows:
        print("no K completed", flush=True)
        return 1

    # best (lowest) us/step across the sweep -> the steady-state per-step.
    best_K, _bms, best_us, _kv, _pk = min(rows, key=lambda r: r[2])
    print("\n" + "-" * 92, flush=True)
    print("  FULL SELF-FORWARD WALL (from the grounded step count x the best measured us/step)",
          flush=True)
    print("-" * 92, flush=True)
    print("  best steady-state: %.4f us/step at K=%d" % (best_us, best_K), flush=True)

    def _wall(steps):
        return steps * best_us / 1e6      # us -> s

    wb = _wall(BASELINE_STEPS)
    wf = _wall(DEDUP_FUSED_STEPS)
    print("  baseline    (%d steps): %.0f s = %.1f min = %.2f hr"
          % (BASELINE_STEPS, wb, wb / 60, wb / 3600), flush=True)
    print("  dedup+fused (%d steps): %.0f s = %.1f min = %.2f hr   (%.2fx faster)"
          % (DEDUP_FUSED_STEPS, wf, wf / 60, wf / 3600, wb / wf), flush=True)
    print("  matmul portion at 1 step/MAC = %d steps (%.1f%% of dedup+fused total; the "
          "tail dominates)"
          % (MACS, 100.0 * MACS / DEDUP_FUSED_STEPS), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
