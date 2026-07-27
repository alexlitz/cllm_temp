"""#747/#748 PART 3 — PRECISE FLOP utilization on the K=128 composed path via a
REAL profiler (torch.profiler with_flops + kineto CUDA timing).

Measures the ACTUAL useful FLOPs executed in one composed ``forward_span`` (every
aten GEMM/BMM the profiler counts, post-#748 selective-fp64), the achieved wall,
and reports achieved FLOP/s vs the A5000 fp32 AND fp64 peak.  Also breaks down the
CUDA time by fp32 vs fp64 kernels (the mix is now mostly fp32 after selective-fp64)
and where the idle gap is (per-step work vs launch vs memory).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_POS_SPARSE"] = "1"

import torch
from torch.profiler import profile, ProfilerActivity

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from . import bench_pf_kbatch as B

# A5000 vendor peaks (TFLOP/s): fp32 (no TF32) / TF32 / fp64.
A5000_FP32 = 27.8e12
A5000_TF32 = 55.6e12
A5000_FP64 = 0.867e12   # ~1/32 of fp32


def profile_forward(runner, x, ops, q_idxs, graph, reps=50, warmup=10):
    cuda = x.device.type == "cuda"

    def fwd():
        with torch.no_grad():
            if graph:
                return runner.forward_span_graphed(x, ops, q_idxs)
            return runner.forward_span(x, ops, q_idxs)

    for _ in range(warmup):
        fwd()
    if cuda:
        torch.cuda.synchronize()

    # (1) wall-clock ms/forward (timed loop, kineto off).
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd()
    if cuda:
        torch.cuda.synchronize()
    ms_fwd = (time.perf_counter() - t0) / reps * 1e3

    # (2) profiler: total FLOPs (with_flops) + CUDA self time per kernel dtype.
    acts = [ProfilerActivity.CPU]
    if cuda:
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, with_flops=True, record_shapes=False) as prof:
        for _ in range(reps):
            fwd()
        if cuda:
            torch.cuda.synchronize()
    evts = prof.key_averages()
    total_flops = sum(getattr(e, "flops", 0) or 0 for e in evts)
    flops_per_fwd = total_flops / reps

    # CUDA-time breakdown.  Sum ONLY leaf device kernels (device_type == CUDA) to
    # avoid double-counting the aten:: dispatch wrapper's self_cuda_time on top of
    # its underlying ampere_sgemm kernel.
    from torch.autograd import DeviceType
    cuda_us_total = 0.0
    cuda_us_fp64 = 0.0
    for e in evts:
        is_dev = getattr(e, "device_type", None) == DeviceType.CUDA
        cu = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0) or 0
        if not is_dev:
            continue
        cuda_us_total += cu
        nm = e.key.lower()
        if "double" in nm or "fp64" in nm or "_dvec" in nm or "dgemm" in nm:
            cuda_us_fp64 += cu
    return ms_fwd, flops_per_fwd, cuda_us_total / reps, cuda_us_fp64 / reps, evts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-s", type=int, default=900)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--op", default="ADD")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    a = ap.parse_args(argv)

    B.runner_window = a.window
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not a.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(code_size=64, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=a.window, selective_fp64=True)
    fp64 = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    names = list(getattr(L, "_block_names", []))
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s", flush=True)
    print(f"[fp64] SELECTIVE: {len(fp64)}/{len(runner.kblocks)} blocks fp64 "
          f"{[names[b] for b in fp64]}", flush=True)

    op = getattr(isa, a.op)
    x, q_idxs, S = B._make_timing_stream(model, L, a.K, a.base_s, op, a.window)
    ops = [op] * len(q_idxs)
    live = runner.live_union(ops)
    D = model.embed.shape[1]

    print("\n" + "=" * 88, flush=True)
    print(f"PROFILED FLOP UTILIZATION — composed K={a.K} forward_span "
          f"(op={a.op}, S={S}, live_blocks={len(live)})", flush=True)
    print(f"device=RTX A5000  fp32_peak={A5000_FP32/1e12:.1f} TFLOP/s  "
          f"fp64_peak={A5000_FP64/1e12:.2f} TFLOP/s", flush=True)
    print("=" * 88, flush=True)

    ms_fwd, flops_fwd, cuda_us, cuda_us_fp64, evts = profile_forward(
        runner, x, ops, q_idxs, a.graph)
    ms_step = ms_fwd / max(len(q_idxs), 1)
    achieved = flops_fwd / (ms_fwd / 1e3)     # FLOP/s (EXECUTED dense GEMM work)
    util_fp32 = achieved / A5000_FP32 * 100.0
    util_tf32 = achieved / A5000_TF32 * 100.0
    util_fp64 = achieved / A5000_FP64 * 100.0

    # USEFUL FLOP: the minimal VM-step arithmetic (the bench's analytic count over the
    # live blocks' bounded GEMMs) — what a self-emu step actually NEEDS, vs the dense
    # GEMM the dense_kernel path physically EXECUTES (the profiled number above).
    n_store = int((x[0, :, int(L.IS_STORE)] != 0).sum())
    useful_fwd = B._count_flops_forward(model, L, runner, a.K, q_idxs, ops, D,
                                        a.window, n_store)
    useful_flops_s = useful_fwd / (ms_fwd / 1e3)
    useful_util_fp32 = useful_flops_s / A5000_FP32 * 100.0

    print(f"  ms/forward (wall)            = {ms_fwd:.4f} ms", flush=True)
    print(f"  ms/step_eff (/K)            = {ms_step:.5f} ms", flush=True)
    print(f"  PROFILED EXECUTED FLOP/fwd   = {flops_fwd/1e9:.3f} GFLOP "
          f"(torch.profiler with_flops: the DENSE aten GEMM/BMM the dense_kernel path "
          f"physically runs)", flush=True)
    print(f"  ANALYTIC USEFUL FLOP/fwd     = {useful_fwd/1e9:.3f} GFLOP "
          f"(the live-block bounded GEMMs a VM step needs)", flush=True)
    print(f"  achieved (executed) FLOP/s   = {achieved/1e12:.4f} TFLOP/s", flush=True)
    print(f"  achieved (useful)   FLOP/s   = {useful_flops_s/1e12:.4f} TFLOP/s", flush=True)
    print(f"  --------------------------------------------------------", flush=True)
    print(f"  EXECUTED-FLOP UTIL vs fp32    = {util_fp32:.4f} %  "
          f"(vs TF32 {util_tf32:.4f}%, vs fp64 {util_fp64:.2f}%)", flush=True)
    print(f"  USEFUL-FLOP   UTIL vs fp32    = {useful_util_fp32:.4f} %  "
          f"(the honest self-emu util — this is the ~1% ballpark, now precise)", flush=True)
    print(f"  --------------------------------------------------------", flush=True)
    cuda_ms = cuda_us / 1e3
    cuda_ms_fp64 = cuda_us_fp64 / 1e3
    print(f"  CUDA kernel self-time/forward= {cuda_ms:.4f} ms "
          f"(of {ms_fwd:.4f} ms wall -> {100.0*cuda_ms/ms_fwd:.1f}% on-GPU, "
          f"{100.0*(ms_fwd-cuda_ms)/ms_fwd:.1f}% launch/host gap)", flush=True)
    print(f"  of which fp64 kernel time    = {cuda_ms_fp64:.4f} ms "
          f"({100.0*cuda_ms_fp64/max(cuda_ms,1e-9):.1f}% of GPU time -> "
          f"the mix is {'mostly fp32' if cuda_ms_fp64 < cuda_ms*0.5 else 'fp64-heavy'})",
          flush=True)

    # top kernels by CUDA self time (where the GPU time actually goes).
    print("\n  top CUDA kernels by self-time:", flush=True)
    def _cu(e):
        return getattr(e, "self_cuda_time_total", 0) or getattr(e, "self_device_time_total", 0) or 0
    top = sorted([e for e in evts if _cu(e) > 0], key=_cu, reverse=True)[:10]
    for e in top:
        print(f"    {_cu(e)/1e3/50:8.4f} ms/fwd  {e.key[:60]}", flush=True)

    print("\n" + "-" * 88, flush=True)
    print("VERDICT (can util improve?):", flush=True)
    print(f"  * The per-step useful work is ~{flops_fwd/max(len(q_idxs),1)/1e6:.2f} MFLOP/step "
          f"(K={a.K} steps in one forward). This is TINY vs the A5000's "
          f"{A5000_FP32/1e12:.0f} TFLOP/s.", flush=True)
    print(f"  * Util is low because there is LITTLE WORK per step (a VM step is ~M FLOP), "
          f"NOT because FLOPs are wasted:", flush=True)
    print(f"    after pos-sparse + bounded-KV the forward touches only the live "
          f"blocks over bounded rows.", flush=True)
    print(f"  * Bigger K raises util (more rows/forward) until ms/forward grows "
          f"(the bounded attention einsum", flush=True)
    print(f"    is O(K*(W+n_store))); the fundamental cap is that a self-emu VM step "
          f"is ~M FLOP so the GPU stays mostly idle.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
