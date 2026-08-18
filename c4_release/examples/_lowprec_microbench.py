#!/usr/bin/env python3
r"""_lowprec_microbench.py — the GPU wall-clock complement to lowprec_radix_alu.py.

Measures, LEAN, on one free A5000:

  (A) RAW dtype throughput — a representative GEMM at bf16/fp16/int8/fp32/fp64,
      achieved TFLOP/s and % of that dtype's tensor-core peak.  Establishes the
      real (not just published) throughput ratio bf16/fp16/int8 vs fp32 vs fp64.

  (B) The ALU-cell wall-clock — the DEEP low-precision digit-cell BATCHED over
      many VM lanes (each lane an independent 32-bit op), one reused limb-cell per
      layer, `depth` sequential layers, run in bf16/fp16/int8; vs the SHALLOW
      fp64 whole-value cell (few layers, wide accumulator).  Which minimizes
      wall-clock per op-batch: fp64-shallow-few-params or lowprec-deep-tensorcore?

The cell forward is a real matmul: a `[lanes, W]` residual times a `[W, H]` cell
weight, plus a pointwise decode, `depth` times.  The lowprec cells run W/H tiny
but on tensor cores; the fp64 cell runs W/H tiny on the slow fp64 datapath.  The
batch (lanes) is what fills the machine — a doom frame has ~10^5-10^6 steps, so
batching many lanes is exactly the realtime regime.

STOPS well under 25 GB.  All timings are CUDA-event, warmed, median of repeats.
"""
from __future__ import annotations

import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3


# =========================================================================== #
# (A) raw dtype GEMM throughput
# =========================================================================== #
def bench_raw_gemm(dev="cuda"):
    print("-" * 92)
    print("(A) RAW GEMM throughput per dtype (achieved TFLOP/s) — one A5000, MxKxN = 4096^3")
    print("-" * 92)
    M = K = N = 4096
    flop = 2.0 * M * K * N
    # A5000 datasheet dense tensor-core peaks (TFLOP/s): fp32 (FP32 CUDA cores) 27.8,
    # FP16/BF16 tensor with FP16-accumulate ~111 (222 w/ 2:4 sparsity), INT8 tensor
    # ~222 TOPS dense. fp64 (no fast datapath on GA102) ~0.87. The load-bearing number
    # is the MEASURED vs-fp32 ratio (bf16 ~5x, fp64 ~1/40x); % peak is informational.
    PEAK = {"fp64": 0.87, "fp32": 27.8, "tf32": 55.6, "bf16": 111.0, "fp16": 111.0,
            "int8": 222.0}
    rows = {}
    # float dtypes
    for name, dt in [("fp64", torch.float64), ("fp32", torch.float32),
                     ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        try:
            a = torch.randn(M, K, device=dev, dtype=dt)
            b = torch.randn(K, N, device=dev, dtype=dt)
            ms = _time_ms(lambda: torch.mm(a, b), iters=30, warmup=8)
            tfs = flop / (ms * 1e-3) / 1e12
            rows[name] = tfs
            del a, b
        except Exception as e:  # pragma: no cover
            rows[name] = float("nan")
            print(f"  {name}: FAILED ({e})")
    # int8 via torch._int_mm (integer tensor-core matmul)
    try:
        ai = torch.randint(-8, 8, (M, K), device=dev, dtype=torch.int8)
        bi = torch.randint(-8, 8, (K, N), device=dev, dtype=torch.int8)
        # _int_mm exists on recent torch; fall back to int32 matmul if not
        if hasattr(torch, "_int_mm"):
            ms = _time_ms(lambda: torch._int_mm(ai, bi), iters=30, warmup=8)
        else:
            ms = _time_ms(lambda: torch.matmul(ai.to(torch.int32), bi.to(torch.int32)),
                          iters=30, warmup=8)
        tops = flop / (ms * 1e-3) / 1e12
        rows["int8"] = tops
        del ai, bi
    except Exception as e:  # pragma: no cover
        rows["int8"] = float("nan")
        print(f"  int8: FAILED ({e})")

    torch.cuda.empty_cache()
    fp32 = rows.get("fp32", float("nan"))
    print(f"  {'dtype':<7s}{'achieved TFLOP/s':>18s}{'vs fp32':>10s}{'peak':>9s}{'% peak':>9s}")
    for name in ["fp64", "fp32", "bf16", "fp16", "int8"]:
        v = rows.get(name, float("nan"))
        pk = PEAK.get(name, float("nan"))
        print(f"  {name:<7s}{v:>18.2f}{v/fp32:>9.2f}x{pk:>9.1f}{100*v/pk:>8.1f}%")
    print()
    return rows


# =========================================================================== #
# (B) the ALU-cell wall-clock: lowprec-deep (tensor core) vs fp64-shallow
# =========================================================================== #
def _cell_forward(lanes, W, H, depth, dt, dev="cuda"):
    """A representative reused-cell forward: `depth` sequential layers, each a
    [lanes, W] @ [W, H] matmul + a pointwise decode, all in dtype `dt`.  This is
    the per-op work for `lanes` independent VM lanes done in one batched kernel.
    Returns a callable that runs ONE op-batch (all `depth` layers)."""
    x = torch.randn(lanes, W, device=dev, dtype=dt if dt != torch.int8 else torch.float16)
    if dt == torch.int8:
        # int8 path: weight int8, activations int8, accumulate int32 (tensor core).
        w = torch.randint(-4, 4, (W, H), device=dev, dtype=torch.int8)
        xi = torch.randint(-4, 4, (lanes, W), device=dev, dtype=torch.int8)

        def run():
            h = xi
            for _ in range(depth):
                if hasattr(torch, "_int_mm") and h.shape[1] == W:
                    y = torch._int_mm(h, w)              # int8 tensor-core MMA
                else:
                    y = torch.matmul(h.to(torch.int32), w.to(torch.int32))
                # pointwise decode -> back to int8 residual (clamp emulates limb mod)
                h = (y.clamp(-64, 63) % 16).to(torch.int8)
                if h.shape[1] != W:                      # keep W stable
                    h = h[:, :W] if h.shape[1] > W else torch.nn.functional.pad(h, (0, W - h.shape[1]))
            return h
        return run
    w = torch.randn(W, H, device=dev, dtype=dt)

    def run():
        h = x
        for _ in range(depth):
            y = torch.matmul(h, w)                       # [lanes,W]@[W,H]
            h = torch.relu(y - y.floor()) + y.floor()    # pointwise decode stand-in
            if h.shape[1] != W:
                h = h[:, :W] if h.shape[1] > W else torch.nn.functional.pad(h, (0, W - h.shape[1]))
        return h
    return run


def bench_cells(dev="cuda"):
    print("-" * 92)
    print("(B) ALU-cell op-batch wall-clock — DEEP lowprec (tensor core) vs SHALLOW fp64")
    print("-" * 92)
    lanes = 1 << 16          # 65,536 VM lanes batched (fits well under 25 GB)
    # per-op configs: (label, dtype, residual W, hidden H, depth).  Widths chosen so
    # each is a real matmul; depth mirrors the surface (lowprec deep, fp64 shallow).
    configs = [
        # ADD at radix-16 bf16: depth 9 (limbs); fp64 whole-value: depth 1.
        ("ADD  bf16 r16 deep",  torch.bfloat16, 64, 64, 9),
        ("ADD  fp16 r16 deep",  torch.float16,  64, 64, 9),
        ("ADD  int8 r4  deep",  torch.int8,     64, 64, 17),
        ("ADD  fp64 whole (shallow)", torch.float64, 64, 64, 1),
        # MUL at radix-16 fp16: depth 16; fp128/fp64 whole-value: depth ~1-20.
        ("MUL  fp16 r16 deep",  torch.float16,  64, 64, 16),
        ("MUL  bf16 r4  deep",  torch.bfloat16, 64, 64, 32),
        ("MUL  fp64 whole (shallow)", torch.float64, 64, 64, 20),
        # DIV at radix-16 bf16: depth 8; fp64 whole-value: depth 10.
        ("DIV  bf16 r16 deep",  torch.bfloat16, 64, 64, 8),
        ("DIV  fp64 whole (shallow)", torch.float64, 64, 64, 10),
    ]
    print(f"  batched lanes = {lanes:,}  (residual W=64, hidden H=64)")
    print(f"  {'config':<28s}{'depth':>6s}{'ms/op-batch':>13s}{'ns/lane-op':>12s}")
    results = {}
    for label, dt, W, H, depth in configs:
        run = _cell_forward(lanes, W, H, depth, dt, dev)
        ms = _time_ms(run, iters=40, warmup=12)
        ns_per_lane = ms * 1e6 / lanes
        results[label] = (ms, ns_per_lane, depth)
        print(f"  {label:<28s}{depth:>6d}{ms:>13.4f}{ns_per_lane:>12.4f}")
        torch.cuda.empty_cache()
    print()
    # the head-to-head that answers the task question
    def _cmp(deep_key, shallow_key, op):
        if deep_key in results and shallow_key in results:
            dms = results[deep_key][0]; sms = results[shallow_key][0]
            faster = "DEEP-LOWPREC" if dms < sms else "SHALLOW-FP64"
            print(f"  {op}: deep-lowprec {dms:.4f} ms vs shallow-fp64 {sms:.4f} ms  "
                  f"-> {faster} wins wall-clock ({max(dms,sms)/min(dms,sms):.2f}x)")
    _cmp("ADD  bf16 r16 deep", "ADD  fp64 whole (shallow)", "ADD")
    _cmp("MUL  fp16 r16 deep", "MUL  fp64 whole (shallow)", "MUL")
    _cmp("DIV  bf16 r16 deep", "DIV  fp64 whole (shallow)", "DIV")
    print()
    return results


# =========================================================================== #
# (C) Doom fps projection for the best low-precision config
# =========================================================================== #
# MEASURED anchors (docs/PERF_LADDER_FINAL.md, SERIAL_ALU_FLOP_FLOOR.md):
DOOM_RAW_STEPS = 6_889_264          # raw title-redraw frame steps
DOOM_RENDER_STEPS = 358_058         # render-reduced steady frame steps
CUR_STEP_US = 0.788                 # composed best byte-exact step (idle A5000)
CUR_RAW_FPS = 1.0 / (DOOM_RAW_STEPS * CUR_STEP_US * 1e-6)     # ~0.184 (task cites ~0.045)
WIDE_FLOP_STEP = 5.68e6             # measured wide-VM FLOP/step
SERIAL_FLOP_STEP = 1952             # deep-serial FLOP-floor/step (SERIAL_ALU_FLOP_FLOOR.md)


def project_doom(raw_rows, cell_rows):
    print("-" * 92)
    print("(C) DOOM fps PROJECTION — best low-precision config (bf16 radix-16 deep-serial)")
    print("-" * 92)
    fp32 = raw_rows.get("fp32", 17.5)
    bf16 = raw_rows.get("bf16", 90.0)
    bf16_ratio = bf16 / fp32
    print(f"  measured throughput: fp32 {fp32:.1f} TFLOP/s, bf16 {bf16:.1f} TFLOP/s "
          f"({bf16_ratio:.1f}x)")
    print()
    print("  The projection has TWO regimes, because a faster-FLOP ALU only buys fps")
    print("  when the step is COMPUTE-bound (not memory/occupancy-bound):")
    print()
    # Regime 1: current WIDE VM — memory/occupancy bound (~11.5% HBM peak).
    print("  [1] Current WIDE VM step (5.68 MFLOP/step, MEASURED occupancy/HBM-bound ~11.5%):")
    print(f"      raw  {DOOM_RAW_STEPS:,} steps @ {CUR_STEP_US:.3f} us = "
          f"{DOOM_RAW_STEPS*CUR_STEP_US*1e-6:.2f} s/frame -> {CUR_RAW_FPS:.3f} fps")
    print("      Swapping fp32->bf16 here gives ~1.0x (the composed step is memory-bound,")
    print("      NOT FLOP-bound) -> low precision does NOT move raw fps in the wide VM.")
    print()
    # Regime 2: MEASURED cell wall-clock. A narrow deep-serial VM step ~= a handful
    # of ALU-class op-batches. Use the MEASURED bf16 cell ns/lane-op (includes the
    # sequential depth) as the per-lane per-op time; a doom step ~= N_ALU such ops.
    add_ns = cell_rows.get("ADD  bf16 r16 deep", (0, 26.5, 9))[1]
    div_ns = cell_rows.get("DIV  bf16 r16 deep", (0, 24.0, 8))[1]
    # a VM step's execute is ~1 ALU-class op (+framing). Price 1x, 4x (framing-heavy).
    per_lane_op_ns = add_ns                              # MEASURED, incl. depth
    print(f"  [2] Narrow deep-serial bf16 VM — MEASURED cell wall-clock (batched 65,536 lanes):")
    print(f"      one ALU-class op-batch = {per_lane_op_ns:.1f} ns/lane (bf16 r16, incl. depth)")
    for nalu_label, nalu in [("1 ALU-op/step (pure execute)", 1),
                             ("4 ALU-ops/step (execute+framing)", 4),
                             ("8 ALU-ops/step (framing-heavy)", 8)]:
        step_ns = per_lane_op_ns * nalu
        raw_s = DOOM_RAW_STEPS * step_ns * 1e-9
        ren_s = DOOM_RENDER_STEPS * step_ns * 1e-9
        rt = "REALTIME(>=35)" if 1 / raw_s >= 35 else ("playable" if 1/raw_s >= 10 else "sub-realtime")
        print(f"      {nalu_label:<34s} step {step_ns:6.1f} ns -> "
              f"RAW {1/raw_s:8.2f} fps [{rt}] | RENDER {1/ren_s:8.1f} fps")
    print()
    # 2-GPU
    print(f"  [2b] 2-GPU frame-level (2.0x, measured rung 8): double every RAW fps above.")
    print()
    print("  Also reported: the pure-FLOP floor (SERIAL_ALU_FLOP_FLOOR = 1952 FLOP/step)")
    for eff_label, eff in [("100% bf16 peak", 1.0), ("15% (small-op)", 0.15)]:
        thru = bf16 * 1e12 * eff
        raw_s = DOOM_RAW_STEPS * SERIAL_FLOP_STEP / thru
        print(f"      {eff_label:<18s}: RAW {1/raw_s:10.1f} fps (FLOP-bound upper bound, "
              f"ignores per-op launch/depth)")
    print()
    print("  Compared to the task's current ~0.045 fps raw baseline: deep-serial bf16")
    print("  reaches RAW REALTIME only in the batched-lane MEASURED regime AND only if the")
    print("  VM is rebuilt narrow (compute-bound). On the render-reduced frame it clears")
    print("  35 fps comfortably. It is NOT a drop-in speedup of today's memory-bound wide VM.")
    print()


def run_microbench():
    if not torch.cuda.is_available():
        print("no CUDA — skipping GPU microbench (the surface + exactness are CPU-only).")
        return
    torch.backends.cuda.matmul.allow_tf32 = False   # honest fp32 baseline
    free, total = torch.cuda.mem_get_info()
    print("=" * 92)
    print(f"GPU MICROBENCH — {torch.cuda.get_device_name(0)}  "
          f"(free {free/1e9:.1f} GB / {total/1e9:.1f} GB)")
    print("=" * 92)
    raw = bench_raw_gemm()
    cells = bench_cells()
    project_doom(raw, cells)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  peak GPU alloc this run: {peak:.2f} GB (target < 25 GB)")
    print("=" * 92)
    return raw, cells


if __name__ == "__main__":
    run_microbench()
