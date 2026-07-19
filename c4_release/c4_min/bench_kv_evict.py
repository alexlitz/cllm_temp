"""Prune wall-time: OLD per-entry-Python vs NEW vectorised keep-mask.

Measures ``prune_keep_mask_head`` (the per-head KV eviction decision) on the
deep-tail regime — a rec_fib(12)-scale ~250k-token stream whose live-heap cache
is a large near-duplicate-heavy set of register-marker / store keys.  Reports the
per-prune wall-time BEFORE (the O(cache^2) per-entry Python greedy loop from base
commit 4e0cb8fd) vs AFTER (the vectorised, on-device form), on CPU and on a GPU,
and confirms the GPU is BUSY (not idle) during the vectorised eviction.

    python -m c4_min.bench_kv_evict [--device cuda:1] [--metric cosine|exact]

The OLD implementation is loaded verbatim from the base commit so the comparison
is apples-to-apples (same policy, only the vectorisation differs).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import time
from typing import List

import torch

from c4_min.nibble_pure_forward_cached import (
    prune_keep_mask_head as new_prune,
)


# ---------------------------------------------------------------------------
# OLD (base-commit) per-entry Python keep-mask — loaded verbatim from git so the
# before/after comparison is the SAME policy with only the loop vectorised.
# ---------------------------------------------------------------------------
def _load_old_prune():
    src = subprocess.check_output(
        ["git", "show", "4e0cb8fd:c4_release/c4_min/nibble_pure_forward_cached.py"],
        cwd=_repo_root(), text=True)
    # slice out the function body (def prune_keep_mask_head ... return survivors)
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("def prune_keep_mask_head("))
    end = start
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("def ") or lines[i].startswith("# ====="):
            end = i
            break
    body = "\n".join(lines[start:end])
    ns = {"torch": torch, "math": math, "List": List, "Optional": __import__("typing").Optional}
    exec(body, ns)
    return ns["prune_keep_mask_head"]


def _repo_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_live_heap_cache(S, HD=8, n_groups=24, device="cpu", metric="cosine"):
    """A realistic deep-loop live-heap cache: a few distinct key groups (register
    markers / store addresses) repeated across many VM steps (near-duplicate, the
    supersession-heavy shape), plus freed/NULL zero rows."""
    g = torch.Generator().manual_seed(1234)
    proto = torch.randn(n_groups, HD, generator=g)
    idx = torch.randint(0, n_groups, (S,), generator=g)
    if metric == "exact":
        # content-addressed: verbatim repeats of a stored key (latest-write-wins)
        keys = proto[idx].clone()
    else:
        keys = proto[idx] * (1.0 + 0.0005 * torch.randn(S, 1, generator=g))
    vals = torch.randn(S, HD, generator=g)
    zmask = torch.rand(S, generator=g) < 0.15
    vals[zmask] = 0.0
    zk = zmask & (torch.rand(S, generator=g) < 0.5)
    keys[zk] = 0.0
    positions = torch.arange(S) * 30
    return (keys.to(device), vals.to(device), positions.to(device))


def timeit(fn, device, reps=3):
    fn()  # warmup (build kernels / caches)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def gpu_util_sample(device_idx, fn, stop_after=3.0):
    """Run ``fn`` in a TIGHT loop (no per-call sync — keep the launch queue full)
    for ~stop_after s while sampling nvidia-smi utilization; return the peak
    observed GPU utilization %.  A busy GPU during eviction reads well above 0%."""
    import threading
    stop = threading.Event()
    peak = [0]

    def sampler():
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "-i", str(device_idx),
                     "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True).strip()
                peak[0] = max(peak[0], int(out.split("\n")[0]))
            except Exception:
                pass
            time.sleep(0.005)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < stop_after:
        fn()             # keep launching kernels back-to-back (queue stays full)
        n += 1
    torch.cuda.synchronize()
    stop.set()
    th.join()
    return peak[0], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "exact"])
    ap.add_argument("--sizes", default="500,1000,2000,4000,8000")
    args = ap.parse_args()
    dev = args.device
    metric = args.metric
    ca = (metric == "exact")

    old_prune = _load_old_prune()
    slope, scale = 0.25, 8 ** -0.5
    cos_thr, zeps, reps = 0.99, 1e-9, 1e-6
    sizes = [int(s) for s in args.sizes.split(",")]

    # The OLD per-entry-Python keep-mask allocates CPU-side bool masks and cannot
    # run on a GPU tensor at all (RuntimeError: indices on cpu) — which is exactly
    # why the base ``evict_keep_index`` force-copied the whole cache to the CPU and
    # ran the eviction CPU-serial while the GPU sat idle.  So the real before/after
    # is: OLD = CPU-serial per-entry loop (the GPU-idle path), NEW = on ``dev``.
    print(f"NEW device={dev}  (OLD runs CPU-only)  metric={metric}  "
          f"content_addressed={ca}")
    print(f"{'S':>7} {'kept':>6} {'OLD-cpu ms':>11} {'NEW ms':>10} "
          f"{'speedup':>8} {'match':>6}")
    for S in sizes:
        keys_d, vals_d, positions_d = make_live_heap_cache(S, device=dev, metric=metric)
        keys_c, vals_c, positions_c = make_live_heap_cache(S, device="cpu", metric=metric)

        def run_old():
            return old_prune(keys_c, vals_c, positions_c, slope, scale,
                             cos_thr, zeps, reps, dup_metric=metric,
                             content_addressed=ca)

        def run_new():
            return new_prune(keys_d, vals_d, positions_d, slope, scale,
                             cos_thr, zeps, reps, dup_metric=metric,
                             content_addressed=ca)

        m_old = run_old().to("cpu")
        m_new = run_new().to("cpu")
        match = bool(torch.equal(m_old, m_new))
        t_old = timeit(run_old, "cpu")
        t_new = timeit(run_new, dev)
        print(f"{S:>7} {int(m_new.sum()):>6} {t_old*1e3:>11.2f} {t_new*1e3:>10.2f} "
              f"{t_old/t_new:>7.1f}x {str(match):>6}", flush=True)

    if dev.startswith("cuda"):
        # nvidia-smi -i indexes the VISIBLE ordinal (CUDA_VISIBLE_DEVICES remaps),
        # so query the physical id from the runtime.
        phys = torch.cuda.get_device_properties(
            int(dev.split(":")[1]) if ":" in dev else 0)
        idx = 0
        try:
            import os
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            idx = int(vis.split(",")[0]) if vis else 0
        except Exception:
            pass
        S = sizes[-1]
        keys, vals, positions = make_live_heap_cache(S, device=dev, metric=metric)
        util, n = gpu_util_sample(idx, lambda: new_prune(
            keys, vals, positions, slope, scale, cos_thr, zeps, reps,
            dup_metric=metric, content_addressed=ca))
        print(f"\nGPU-BUSY check (S={S}, {n} prunes in the sample window on "
              f"{phys.name}): peak GPU utilization during vectorised eviction = "
              f"{util}%  (the OLD CPU-serial per-entry Python loop leaves this GPU "
              f"at 0% — the whole prune ran on the CPU).")


if __name__ == "__main__":
    main()
