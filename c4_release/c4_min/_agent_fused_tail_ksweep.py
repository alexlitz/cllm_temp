"""_agent_fused_tail_ksweep.py — the composed self-forward per-step us BY K (VRAM per
K) for the dedup-in-C + fused-MAC + fused-TAIL config, and the full self-forward wall
from the GROUNDED composed step count.

Extends _agent_dedup_fused_ksweep with the FUSED-TAIL collapse: the non-matmul tail
(silu + element-wise adds) fuses to 1 step/element like the MAC, so the tail drops
10,627,826 -> 167,644 steps (63.4x) and the composed total drops to 332,298 steps.

    baseline           total = 23,201,022  (grounded, WITH divmod, position-sparse)
    dedup+fused-MAC    total = 164,654 + 10,627,826       = 10,792,480  (2.15x)
    + fused-TAIL       total = 164,654 + 167,644          =    332,298  (69.8x)

  fused tail = fused silu (42,457) + fused adds (40,559) + softmax kept un-fused (84,628).

so  wall = total_steps * (measured us/step).  The per-step us is a property of the
batched attention+FFN forward at a given K (the dedup/fuse change how many STEPS the
matmul/tail need, not the per-step forward cost), MEASURED here at K=128..1024.

CPU builds the model; GPU 0 runs the timed forwards.  Memory-safe: one lean build.
Run:  C4_POS_SPARSE=1 python -m c4_min._agent_fused_tail_ksweep --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from . import isa

# grounded step counts (ground_fused_tail_selfemu --with-divmod, position-sparse).
BASELINE_STEPS = 23_201_022
TAIL_STEPS = 10_627_826          # unfused non-matmul tail
MACS = 164_654                   # nonzero weights = MACs at the S=1 decode row
DEDUP_FUSED_MAC_STEPS = MACS + TAIL_STEPS        # fused MAC, tail un-fused = 10,792,480

# fused-TAIL: silu + adds -> 1 step/element, softmax kept un-fused.
SILU_FUSED = 42_457
ADD_FUSED = 40_559
SOFTMAX_KEPT = 84_628
TAIL_FUSED = SILU_FUSED + ADD_FUSED + SOFTMAX_KEPT          # 167,644
FUSED_TAIL_STEPS = MACS + TAIL_FUSED                        # 332,298


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--ks", default="128,256,512,1024")
    ap.add_argument("--max-k", type=int, default=2048)
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
    print("COMPOSED-RUNNER per-step us BY K  (dedup + fused-MAC + fused-TAIL; VRAM peak each)",
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

    best_K, _bms, best_us, _kv, _pk = min(rows, key=lambda r: r[2])
    print("\n" + "-" * 92, flush=True)
    print("  FULL SELF-FORWARD WALL (grounded step count x best measured us/step)",
          flush=True)
    print("-" * 92, flush=True)
    print("  best steady-state: %.4f us/step at K=%d" % (best_us, best_K), flush=True)

    def _wall(steps):
        return steps * best_us / 1e6      # us -> s

    def _fmt(s):
        if s < 60:
            return "%.0f s" % s
        if s < 3600:
            return "%.1f min" % (s / 60)
        return "%.2f hr" % (s / 3600)

    wb = _wall(BASELINE_STEPS)
    wm = _wall(DEDUP_FUSED_MAC_STEPS)
    wf = _wall(FUSED_TAIL_STEPS)
    print("  baseline        (%9d steps): %s"
          % (BASELINE_STEPS, _fmt(wb)), flush=True)
    print("  dedup+fused-MAC (%9d steps): %s   (%.2fx faster than baseline)"
          % (DEDUP_FUSED_MAC_STEPS, _fmt(wm), wb / wm), flush=True)
    print("  + fused-TAIL    (%9d steps): %s   (%.2fx faster than baseline, %.2fx vs fused-MAC)"
          % (FUSED_TAIL_STEPS, _fmt(wf), wb / wf, wm / wf), flush=True)
    print("  composition: matmul %d (fused MAC 1/MAC) + tail %d (fused silu %d + fused adds %d "
          "+ softmax %d kept)" % (MACS, TAIL_FUSED, SILU_FUSED, ADD_FUSED, SOFTMAX_KEPT),
          flush=True)
    print("  remaining floor: matmul is %.1f%% of the composed total; softmax (un-fused) is %.1f%%"
          % (100.0 * MACS / FUSED_TAIL_STEPS, 100.0 * SOFTMAX_KEPT / FUSED_TAIL_STEPS),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
