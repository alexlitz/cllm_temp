#!/usr/bin/env python3
"""_agent_continuous_frame.py — HONEST CONTINUOUS steady-state per-frame (capture amortized
out), in a FRESH CUDA context (no leftover graph pools from the build-comparison runs).

The whole-frame giant-K single-shot (4.11 s/frame @358 K) is dominated by the ONE-TIME
graph CAPTURE (+the build).  But a CONTINUOUS doom render keeps the captured graph resident
across frames — only the per-frame schedule BUILD + graph REPLAY recur.  This tool captures
the graph ONCE, then measures build+dispatch over several consecutive frames with the
capture amortized OUT — the real-time-relevant number.  It ALSO measures a double-buffer
PIPELINE (build(N+1) on a side stream || dispatch(N) graph replay) -> ~max(build,dispatch).

Uses C4_SCHED_GPU_BUILD (the deep-vectorized on-device build).  DIV-free whole frame.

Run: CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> \
     python -m c4_min._agent_continuous_frame --device cuda:0 --outer 120 --inner 255
"""
from __future__ import annotations
import argparse, os, time, gc, pickle

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested
from c4_min import precomputed_schedule as PS
from c4_min.precomputed_schedule import PipelinedScheduleBuilder  # noqa: F401

RENDER_REDUCED_FRAME = 358_058
RAW_FRAME = 6_889_264
REALTIME_S = 1.0
FPS35_S = 1.0 / 35.0

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]
LEVERS = ["C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH", "C4_PRECOMPUTED_SCHEDULE",
          "C4_SCHED_CHUNK", "C4_SCHED_FAST_BUILD", "C4_SCHED_GPU_BUILD"]


def _levers_on(chunk, gpu_build=True):
    for f in COMPOSED:
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1" if gpu_build else "0"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)


def _levers_off():
    for f in LEVERS:
        os.environ.pop(f, None)


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")


def _dispatch(sched, sg, dev, n):
    got_pc = torch.empty(n, dtype=torch.long, device=dev)
    got_sp = torch.empty(n, dtype=torch.long, device=dev)
    got_bp = torch.empty(n, dtype=torch.long, device=dev)
    got_ax = torch.empty(n, dtype=torch.long, device=dev)
    onchip_ = sched.onchip
    h0s = sched.h0_folded if onchip_ else sched.h0_table
    for lo in range(0, n, sg.chunk):
        hi = min(lo + sg.chunk, n)
        h0 = h0s[lo:hi].unsqueeze(0)
        if onchip_:
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, None, None, delta=delta, resident=False)
        else:
            ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
            cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0)
                   for b, t in sched.cam_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, resident=False)
        got_pc[lo:hi].copy_(pc_c); got_sp[lo:hi].copy_(sp_c)
        got_bp[lo:hi].copy_(bp_c); got_ax[lo:hi].copy_(ax_c)
    return got_pc, got_sp, got_bp, got_ax


def measure(model, L, code, draft, device, chunk, n_frames=5):
    dev = torch.device(device)
    n = draft.step_count
    mask = 0xFFFFFFFF
    _levers_on(chunk, gpu_build=True)

    # ---- capture the graph ONCE (build a schedule + one dispatch to capture). ----
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    torch.cuda.synchronize(dev)
    t_build0 = time.perf_counter() - t0
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    _dispatch(sched, sg, dev, n)             # producing replay = capture (one-time)
    torch.cuda.synchronize(dev)
    t_capture = time.perf_counter() - t0
    print(f"  [one-time] build0 {t_build0*1e3:.1f} ms  capture(incl 1 replay) "
          f"{t_capture*1e3:.1f} ms  chunk={sg.chunk}  n_chunks={(n+sg.chunk-1)//sg.chunk}",
          flush=True)
    fv, tv = torch.cuda.mem_get_info(dev)
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    print(f"  [VRAM] free {fv/1e9:.1f}/{tv/1e9:.1f} GB  peak_alloc {peak:.1f} GB", flush=True)
    del sched; gc.collect(); torch.cuda.empty_cache()

    # ---- SERIAL continuous: rebuild tables + replay the captured graph, N frames. ----
    build_s = []; disp_s = []
    s2 = None
    for _ in range(n_frames):
        del s2; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        s2 = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
        torch.cuda.synchronize(dev); build_s.append(time.perf_counter() - t0)
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        _dispatch(s2, sg, dev, n)
        torch.cuda.synchronize(dev); disp_s.append(time.perf_counter() - t0)
    del s2; gc.collect(); torch.cuda.empty_cache()
    b = sum(build_s) / len(build_s)
    d = sum(disp_s) / len(disp_s)
    pf_serial = b + d
    print(f"\n  SERIAL   continuous: build {b*1e3:8.2f} ms + dispatch {d*1e3:8.2f} ms "
          f"= {pf_serial*1e3:8.2f} ms/frame  ({b/n*1e6:.3f} + {d/n*1e6:.3f} us/step)",
          flush=True)

    # ---- NAIVE-PIPELINED continuous (the base harness's attempt): build(N+1) inside a
    #      ``with torch.cuda.stream(side)`` || dispatch(N).  This does NOT overlap because
    #      the build is ~90% CPU numpy — a cuda.stream context only redirects GPU kernel
    #      launches, so the CPU build fully completes on the main thread BEFORE dispatch is
    #      even issued.  Kept as the DEMONSTRATION of the serialization bug. ----
    side = torch.cuda.Stream(device=dev)
    cur = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
    torch.cuda.synchronize(dev)
    t_all0 = time.perf_counter()
    for i in range(n_frames):
        with torch.cuda.stream(side):
            nxt = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
        _dispatch(cur, sg, dev, n)
        torch.cuda.synchronize(dev)          # both streams done for this frame
        del cur; cur = nxt
    torch.cuda.synchronize(dev)
    pf_pipe = (time.perf_counter() - t_all0) / n_frames
    del cur; gc.collect(); torch.cuda.empty_cache()
    print(f"  NAIVE-PIPE continuous: build(N+1)@cuda.stream||dispatch(N) "
          f"= {pf_pipe*1e3:8.2f} ms/frame  (does NOT overlap — CPU build serializes)",
          flush=True)

    # ---- TRUE-PIPELINED continuous (C4_SCHED_PIPELINE): build(N+1) on a BACKGROUND THREAD
    #      + dedicated build stream || dispatch(N) graph replay on the default stream.  The
    #      numpy build releases the GIL so the main thread issues the replay concurrently;
    #      the build's GPU ops run on a separate stream; event-sync only at the swap.
    #
    #      HONEST STEADY-STATE: the first ~2 frames warm the pipeline (frame 0 has no
    #      overlap; frame 1 fills the double-buffer + first-replay allocator warmup), so a
    #      continuous render (thousands of frames) is dominated by the STEADY-STATE frame.
    #      We time each frame individually and report the steady-state MEDIAN (warmup
    #      discarded), the number the real-time doom loop actually sustains. ----
    os.environ["C4_SCHED_PIPELINE"] = "1"
    WARMUP = 2
    pb = PS.PipelinedScheduleBuilder(model, L, code, draft, dev, sg, mask=mask)
    cur = pb.build_blocking()                # frame 0 (no overlap possible yet)
    torch.cuda.synchronize(dev)
    pb.start()                               # kick off frame 1's build on the thread
    frame_walls = []
    total_frames = n_frames + WARMUP
    for i in range(total_frames):
        torch.cuda.synchronize(dev); t_f0 = time.perf_counter()
        _dispatch(cur, sg, dev, n)           # frame-N replay (overlaps the thread's build)
        nxt = pb.wait()                      # join build thread + event-sync the swap
        del cur; cur = nxt
        if i < total_frames - 1:
            pb.start()                       # kick off frame N+2's build
        torch.cuda.synchronize(dev); frame_walls.append(time.perf_counter() - t_f0)
    del cur; gc.collect(); torch.cuda.empty_cache()
    steady = sorted(frame_walls[WARMUP:])
    pf_thread = steady[len(steady) // 2]     # steady-state MEDIAN (warmup discarded)
    pf_thread_mean_all = sum(frame_walls) / len(frame_walls)
    print(f"  TRUE-PIPE  continuous: build(N+1)@thread+stream||dispatch(N) double-buffer "
          f"= {pf_thread*1e3:8.2f} ms/frame  (steady-state median; ~max(build,dispatch) "
          f"= {max(b,d)*1e3:.2f} ms)", flush=True)
    print(f"             per-frame walls (ms): {[round(x*1e3) for x in frame_walls]}"
          f"  [first {WARMUP} = warmup, discarded]  all-mean {pf_thread_mean_all*1e3:.1f}",
          flush=True)

    # ---- BYTE-EXACT: the thread-built schedule dispatch == the serial-built schedule
    #      dispatch (per-step AX/PC/SP/BP L-inf=0).  The overlap must NOT change results —
    #      the double-buffer swap must be correctly synchronized (no half-built read). ----
    s_ser = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
    torch.cuda.synchronize(dev)
    ser_pc, ser_sp, ser_bp, ser_ax = _dispatch(s_ser, sg, dev, n)
    torch.cuda.synchronize(dev)
    pb2 = PS.PipelinedScheduleBuilder(model, L, code, draft, dev, sg, mask=mask)
    pb2.start()
    s_thr = pb2.wait()                       # thread-built + event-synced schedule
    thr_pc, thr_sp, thr_bp, thr_ax = _dispatch(s_thr, sg, dev, n)
    torch.cuda.synchronize(dev)
    linf = max(int((ser_pc != thr_pc).sum()), int((ser_sp != thr_sp).sum()),
               int((ser_bp != thr_bp).sum()), int((ser_ax != thr_ax).sum()))
    print(f"  [BYTE-EXACT] thread-pipe vs serial decode mismatches (PC/SP/BP/AX) = {linf}"
          f"  ({'L-inf=0 OK' if linf == 0 else 'MISMATCH!'})", flush=True)
    del s_ser, s_thr, ser_pc, ser_sp, ser_bp, ser_ax, thr_pc, thr_sp, thr_bp, thr_ax
    gc.collect(); torch.cuda.empty_cache()
    os.environ.pop("C4_SCHED_PIPELINE", None)
    _levers_off()
    return {"build_s": b, "dispatch_s": d, "serial_s": pf_serial, "pipe_s": pf_pipe,
            "thread_s": pf_thread, "capture_s": t_capture, "peak_gb": peak,
            "chunk": int(sg.chunk), "n": n}


def _load_draft(outer, inner):
    cache = f"/tmp/_wf_draft_{outer}_{inner}.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            d = pickle.load(fh)
        print(f"  [draft] nested({outer},{inner}) steps={d.step_count} (cached)", flush=True)
        return build_nested(outer, inner)[0], d
    code = build_nested(outer, inner)[0]
    t0 = time.time()
    d = draft_pf_program(code, max_steps=3_000_000, mask=0xFFFFFFFF)
    print(f"  [draft] nested({outer},{inner}) steps={d.step_count} halted={d.halted} "
          f"{time.time()-t0:.0f}s", flush=True)
    if d.halted:
        with open(cache, "wb") as fh:
            pickle.dump(d, fh)
    return code, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outer", type=int, default=120)
    ap.add_argument("--inner", type=int, default=255)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--n-frames", type=int, default=5)
    args = ap.parse_args(argv)
    _guard()
    device = args.device
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"{time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()
    code, draft = _load_draft(args.outer, args.inner)
    n = draft.step_count
    install_composed(model, verbose=False)
    try:
        print(f"\n=== HONEST CONTINUOUS STEADY-STATE (capture amortized out) — n={n} steps ===",
              flush=True)
        r = measure(model, L, code, draft, device, args.chunk, n_frames=args.n_frames)
    finally:
        uninstall_composed(model)
    # ---- extrapolate to target step counts (build + dispatch scale ~linearly in n). ----
    print(f"\n  --- CONTINUOUS per-frame @ target step counts (capture amortized out) ---",
          flush=True)
    print(f"    single-shot baseline (capture NOT amortized) @358 K = 4.11 s/frame", flush=True)
    for label, fsteps in (("render-reduced", RENDER_REDUCED_FRAME), ("raw", RAW_FRAME)):
        ser = r["serial_s"] * (fsteps / n)
        pip = r["pipe_s"] * (fsteps / n)
        thr = r["thread_s"] * (fsteps / n)
        print(f"    {label:>15} ({fsteps:>9} steps):", flush=True)
        print(f"        SERIAL     {ser:8.4f} s/frame  {1.0/ser:7.2f} fps  "
              f"{ser/REALTIME_S:6.2f}x 1s  {ser/FPS35_S:7.1f}x 35fps", flush=True)
        print(f"        NAIVE-PIPE {pip:8.4f} s/frame  {1.0/pip:7.2f} fps  "
              f"{pip/REALTIME_S:6.2f}x 1s  {pip/FPS35_S:7.1f}x 35fps", flush=True)
        print(f"        TRUE-PIPE  {thr:8.4f} s/frame  {1.0/thr:7.2f} fps  "
              f"{thr/REALTIME_S:6.2f}x 1s  {thr/FPS35_S:7.1f}x 35fps"
              f"   {'>=1 fps!' if 1.0/thr >= 1.0 else '<1 fps'}", flush=True)
    print(f"\n  VRAM peak {r['peak_gb']:.1f} GB @ chunk {r['chunk']}  (24 GB ceiling)",
          flush=True)
    print("\n=== CONTINUOUS STEADY-STATE COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
