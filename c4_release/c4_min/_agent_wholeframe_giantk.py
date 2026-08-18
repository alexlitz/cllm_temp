#!/usr/bin/env python3
"""_agent_wholeframe_giantk.py — WHOLE-FRAME GIANT-K single-dispatch: run a whole
render-reduced doom frame as ONE precomputed-schedule single dispatch (the schedule build
paid ONCE, the graph replays over ALL steps) and MEASURE the honest end-to-end per-frame
time, broken into build vs dispatch vs decode.

THE FINDING to test (a4949ddf + a574c010): the single-dispatch STEADY-STATE dispatch is
~5 us/step; the ~333 us "wall" was ~91% the one-time schedule build + graph capture.  So
per-FRAME = (build + capture, once) + (n_steps * dispatch) + (decode + one DtoH).

TASKS:
 1. WHOLE-FRAME GIANT-K: run the largest DIV-free frame that fits VRAM as ONE
    precomputed-schedule single dispatch (resident, on-chip residual).
 2. MEASURE build vs dispatch vs decode; report sec/frame for 358,058 (render-reduced)
    and 6,889,264 (raw) step counts, x from 1 s/frame and 35 fps.
 3. MINIMIZE the one-time build (C4_SCHED_FAST_BUILD vectorizes the O(n_steps) cam-table
    build); report build us before/after.
 4. BYTE-EXACT: per-step AX/PC/SP/BP == K=1 reference AND == the fast-build == the
    loop-build (Linf=0), on the DIV-free battery + deep nested loop.

Run: CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> \
     python -m c4_min._agent_wholeframe_giantk --device cuda:0
"""
from __future__ import annotations
import argparse, os, time, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested, build_loop_countdown
from c4_min.bench_composed_fast_path import _battery
from c4_min import precomputed_schedule as PS

RENDER_REDUCED_FRAME = 358_058      # render-reduced doom frame (task-specified)
RAW_FRAME = 6_889_264               # raw doom frame (task-specified)
REALTIME_S = 1.0                    # 1 s/frame target
FPS35_S = 1.0 / 35.0                # 28.6 ms/frame

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]
LEVERS = ["C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH", "C4_PRECOMPUTED_SCHEDULE",
          "C4_SCHED_CHUNK", "C4_SCHED_FAST_BUILD", "C4_SCHED_GPU_BUILD"]


def _composed_on():
    for f in COMPOSED:
        os.environ[f] = "1"


def _lever(onchip, resident, chunk, fast_build, gpu_build=True):
    for f, v in (("C4_ONCHIP_RESIDUAL", onchip), ("C4_RESIDENT_BATCH", resident)):
        if v:
            os.environ[f] = "1"
        else:
            os.environ.pop(f, None)
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_SCHED_FAST_BUILD"] = "1" if fast_build else "0"
    # C4_SCHED_GPU_BUILD: the deep-vectorized on-device build (default ON here, the fast
    # path).  Only meaningful with fast_build + onchip (the compact-sparse -> W_o-delta path).
    os.environ["C4_SCHED_GPU_BUILD"] = "1" if (gpu_build and fast_build) else "0"


def _clear_levers():
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
    return a


def _vram_gb():
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


# ---------------------------------------------------------------------------
# instrumented single-dispatch: separate BUILD (schedule + graph capture) from
# DISPATCH (the replay loop) from DECODE (the compare + reduction + DtoH).
# ---------------------------------------------------------------------------
def _timed_run(model, L, code, draft, device, chunk, onchip, resident, fast_build,
               gpu_build=True):
    """Run the precomputed-schedule single dispatch, timing build / capture / dispatch /
    decode SEPARATELY.  Returns (result, timings_us_per_step_dict, wall_total_s)."""
    _composed_on()
    _lever(onchip, resident, chunk, fast_build, gpu_build=gpu_build)
    dev = torch.device(device)
    n = draft.step_count
    mask = 0xFFFFFFFF

    # -- (1) BUILD: precompute the whole-batch schedule tables + create the step graph obj
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    torch.cuda.synchronize(dev)
    t_build = time.perf_counter() - t0

    onchip_ = sched.onchip
    h0_src = sched.h0_folded if onchip_ else sched.h0_table
    resident_ = resident and n <= sg.chunk

    def _dispatch(collect=None):
        got_pc = torch.empty(n, dtype=torch.long, device=dev)
        got_sp = torch.empty(n, dtype=torch.long, device=dev)
        got_bp = torch.empty(n, dtype=torch.long, device=dev)
        got_ax = torch.empty(n, dtype=torch.long, device=dev)
        for lo in range(0, n, sg.chunk):
            hi = min(lo + sg.chunk, n)
            h0 = h0_src[lo:hi].unsqueeze(0)
            if onchip_:
                delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
                pc_c, sp_c, bp_c, ax_c = sg.replay(h0, None, None, delta=delta,
                                                   resident=resident_)
            else:
                ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
                cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0)
                       for b, t in sched.cam_tables.items()}
                pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, resident=resident_)
            got_pc[lo:hi].copy_(pc_c); got_sp[lo:hi].copy_(sp_c)
            got_bp[lo:hi].copy_(bp_c); got_ax[lo:hi].copy_(ax_c)
        return got_pc, got_sp, got_bp, got_ax

    # -- (2) CAPTURE: the FIRST dispatch captures the CUDA graph (one-time).  Time it.
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    got = _dispatch()
    torch.cuda.synchronize(dev)
    t_capture = time.perf_counter() - t0    # includes graph capture + 1 replay pass

    # -- (3) DISPATCH steady-state: replay the already-captured graph (no capture).
    reps = 3
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        got = _dispatch()
    torch.cuda.synchronize(dev)
    t_dispatch = (time.perf_counter() - t0) / reps

    # -- (4) DECODE: the one whole-batch compare + accepted-prefix reduction + DtoH sync.
    got_pc, got_sp, got_bp, got_ax = got
    got_ax = got_ax & mask
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    bad_normal = ((got_pc != sched.want_pc) | (got_ax != sched.want_ax)
                  | (got_sp != sched.want_sp) | (got_bp != sched.want_bp))
    bad = torch.where(sched.is_halt, got_ax != sched.want_ax, bad_normal)
    bad = bad & (~sched.is_file)
    any_bad = bad.any()
    first_bad = torch.argmax(bad.to(torch.uint8))
    n_ok = int(torch.where(any_bad, first_bad,
                           torch.tensor(n, device=dev)).item())     # the one host sync
    torch.cuda.synchronize(dev)
    t_decode = time.perf_counter() - t0

    all_matched = (n_ok == n)
    final_ax = int(got_ax[n - 1].item()) if all_matched and n > 0 else None
    res = {"accepted_steps": n_ok, "total_steps": n, "all_matched": all_matched,
           "final_ax": final_ax,
           "first_mismatch": None if all_matched else {
               "step": n_ok, "got_pc": int(got_pc[n_ok].item()),
               "want_pc": int(sched.want_pc[n_ok].item()),
               "got_ax": int(got_ax[n_ok].item()),
               "want_ax": int(sched.want_ax[n_ok].item())},
           "got_pc": got_pc, "got_sp": got_sp, "got_bp": got_bp, "got_ax": got_ax}
    # end-to-end wall for ONE frame = build + capture + decode.  t_capture ALREADY includes
    # the producing replay (the first _dispatch that captures the graph also computes the
    # result), so a single frame does NOT pay a second dispatch — t_dispatch is only the
    # steady-state per-step number for amortizing capture across many frames.
    wall_e2e = t_build + t_capture + t_decode
    tim = {
        "build_us": t_build / n * 1e6,
        "capture_us": t_capture / n * 1e6,
        "dispatch_us": t_dispatch / n * 1e6,
        "decode_us": t_decode / n * 1e6,
        "build_s": t_build, "capture_s": t_capture,
        "dispatch_s": t_dispatch, "decode_s": t_decode,
        "e2e_s": wall_e2e, "n": n,
    }
    _clear_levers()
    return res, tim


# ---------------------------------------------------------------------------
# CONTINUOUS steady-state: capture the graph ONCE, then run build+dispatch for SEVERAL
# consecutive frames back-to-back and report the honest per-frame time with the one-time
# capture amortized OUT.  This is the real-time-relevant number the 4.11 s single-shot
# obscures (a continuous doom render keeps the graph resident across frames — only the
# per-frame schedule BUILD + graph REPLAY recur).  Optionally PIPELINE build(N+1) with
# dispatch(N) via a double-buffer so continuous per-frame -> ~max(build, dispatch).
# ---------------------------------------------------------------------------
def _continuous_steady_state(model, L, code, draft, device, chunk, resident, gpu_build,
                             n_frames=4, pipeline=False):
    """Returns a dict: mean per-frame BUILD s, DISPATCH s, and continuous per-frame s
    (build+dispatch, capture amortized out) over ``n_frames`` consecutive frames on the
    SAME captured graph.  ``pipeline=True`` overlaps build(N+1) with dispatch(N) on a
    second CUDA stream (double-buffer) so continuous per-frame -> ~max(build, dispatch).

    Retries with a HALVED graph chunk on OOM (the earlier gpu/dict comparison runs leave
    CUDA-graph pools resident; a smaller graph chunk fits alongside them — byte-identical,
    the schedule is still built ONCE, the ONE graph replays over more chunks)."""
    from c4_min import precomputed_schedule as PS
    dev = torch.device(device)
    n = draft.step_count
    mask = 0xFFFFFFFF
    while True:
        try:
            return _continuous_steady_state_at(model, L, code, draft, device, chunk,
                                               resident and n <= chunk, gpu_build,
                                               n_frames, pipeline, dev, n, mask)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or chunk <= 8192:
                raise
            _clear_levers(); gc.collect(); torch.cuda.empty_cache()
            chunk //= 2
            print(f"      [continuous OOM -> retry chunk={chunk}]", flush=True)


def _continuous_steady_state_at(model, L, code, draft, device, chunk, resident, gpu_build,
                                n_frames, pipeline, dev, n, mask):
    from c4_min import precomputed_schedule as PS
    _composed_on()
    _lever(True, resident, chunk, True, gpu_build=gpu_build)

    # ---- capture the graph ONCE (build a schedule, do one dispatch to capture). ----
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    onchip_ = sched.onchip
    h0_src = sched.h0_folded if onchip_ else sched.h0_table
    resident_ = resident and n <= sg.chunk

    def _dispatch_once(sched_, sg_):
        got_pc = torch.empty(n, dtype=torch.long, device=dev)
        got_sp = torch.empty(n, dtype=torch.long, device=dev)
        got_bp = torch.empty(n, dtype=torch.long, device=dev)
        got_ax = torch.empty(n, dtype=torch.long, device=dev)
        h0s = sched_.h0_folded if onchip_ else sched_.h0_table
        for lo in range(0, n, sg_.chunk):
            hi = min(lo + sg_.chunk, n)
            h0 = h0s[lo:hi].unsqueeze(0)
            if onchip_:
                delta = {b: t[lo:hi] for b, t in sched_.cam_delta_tables.items()}
                pc_c, sp_c, bp_c, ax_c = sg_.replay(h0, None, None, delta=delta,
                                                    resident=resident_)
            else:
                ing = sched_.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
                cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0)
                       for b, t in sched_.cam_tables.items()}
                pc_c, sp_c, bp_c, ax_c = sg_.replay(h0, ing, cam, resident=resident_)
            got_pc[lo:hi].copy_(pc_c); got_sp[lo:hi].copy_(sp_c)
            got_bp[lo:hi].copy_(bp_c); got_ax[lo:hi].copy_(ax_c)
        return got_pc, got_sp, got_bp, got_ax

    # producing replay = graph capture (one-time, amortized out).  Then FREE the initial
    # schedule's tables (the captured graph ``sg`` has its OWN static buffers — resident=False
    # multi-chunk — so it no longer needs ``sched``'s tables; each continuous frame rebuilds
    # its own tables and reuses the SAME captured ``sg``).  This keeps VRAM at ONE schedule.
    torch.cuda.synchronize(dev)
    _dispatch_once(sched, sg)
    torch.cuda.synchronize(dev)
    del sched
    gc.collect(); torch.cuda.empty_cache()

    # ---- continuous frames: rebuild the schedule tables ONLY + replay the SAME captured
    # graph ``sg``.  ``_build_schedule_tables_only`` builds the O(n) tables WITHOUT
    # re-installing the megablock / re-creating (and re-capturing) the graph — the honest
    # per-frame BUILD.  Free each frame's tables before the next build (one schedule at a
    # time -> no 2x VRAM). ----
    if not pipeline:
        build_s = []; disp_s = []
        s2 = None
        for _ in range(n_frames):
            del s2; gc.collect(); torch.cuda.empty_cache()
            torch.cuda.synchronize(dev); t0 = time.perf_counter()
            s2 = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
            torch.cuda.synchronize(dev); build_s.append(time.perf_counter() - t0)
            torch.cuda.synchronize(dev); t0 = time.perf_counter()
            _dispatch_once(s2, sg)      # replay the ALREADY-captured graph over frame N tables
            torch.cuda.synchronize(dev); disp_s.append(time.perf_counter() - t0)
        del s2; gc.collect(); torch.cuda.empty_cache()
        b = sum(build_s) / len(build_s)
        d = sum(disp_s) / len(disp_s)
        per_frame = b + d
        mode = "serial"
    else:
        # DOUBLE-BUFFER PIPELINE: build(N+1) on a side stream while dispatch(N) replays the
        # graph.  The build is CPU/numpy + small device transfers (independent of the graph
        # replay's kernels), so overlapping them hides the smaller of the two.  We measure
        # the wall of n_frames of (issue build(N+1) async || dispatch(N)) -> ~max(build,disp).
        side = torch.cuda.Stream(device=dev)
        # prime: build frame 0's tables (reusing the captured graph sg; sched freed above).
        cur = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
        torch.cuda.synchronize(dev)
        build_s = []; disp_s = []
        t_all0 = time.perf_counter()
        nxt_holder = {}
        for i in range(n_frames):
            # start build(N+1) on the side stream (concurrent with dispatch(N)).
            tb0 = time.perf_counter()
            with torch.cuda.stream(side):
                nxt_holder["s"] = PS.build_schedule_tables_only(model, L, code, draft, dev,
                                                               sg, mask=mask)
            tb = time.perf_counter() - tb0
            # dispatch(N) on the default stream (the captured graph replay).
            td0 = time.perf_counter()
            _dispatch_once(cur, sg)
            td = time.perf_counter() - td0
            torch.cuda.synchronize(dev)      # both streams done for this frame
            build_s.append(tb); disp_s.append(td)
            cur = nxt_holder["s"]            # frame N's tables freed here (double-buffer)
        torch.cuda.synchronize(dev)
        per_frame = (time.perf_counter() - t_all0) / n_frames
        b = sum(build_s) / len(build_s)
        d = sum(disp_s) / len(disp_s)
        mode = "pipelined"
    _clear_levers()
    return {"build_s": b, "dispatch_s": d, "per_frame_s": per_frame,
            "n": n, "mode": mode, "n_frames": n_frames}


# ---------------------------------------------------------------------------
def find_big_divfree(model, L, device, target=RENDER_REDUCED_FRAME):
    """Find the largest DIV-free program that fully verifies (accepted==step_count),
    aiming near `target` steps.  Try nested loops (byte-safe) of increasing size.  The
    ~5-min CPU draft build for a whole frame is pickle-cached so re-runs are instant."""
    import pickle
    install_composed(model, verbose=False)
    best = None
    try:
        # nested(outer, inner) ~= outer*inner*5 + O(outer) steps, byte-safe.
        cands = [(120, 255), (180, 255), (255, 255), (255, 280), (280, 255), (255, 300)]
        for (o, i) in cands:
            code = build_nested(o, i)[0]
            cache = f"/tmp/_wf_draft_{o}_{i}.pkl"
            d = None
            if os.path.exists(cache):
                try:
                    with open(cache, "rb") as fh:
                        d = pickle.load(fh)
                    print(f"    nested({o},{i}): steps={d.step_count} (cached draft)",
                          flush=True)
                except Exception:
                    d = None
            if d is None:
                d = draft_pf_program(code, max_steps=3_000_000, mask=0xFFFFFFFF)
                if d.halted:
                    try:
                        with open(cache, "wb") as fh:
                            pickle.dump(d, fh)
                    except Exception:
                        pass
                print(f"    nested({o},{i}): steps={d.step_count} halted={d.halted}",
                      flush=True)
            if not d.halted:
                continue
            if any(d.frames[s].get("op") in ("DIV", "MOD") for s in range(d.step_count)):
                continue
            best = (f"nested_{o}_{i}", code, d)
            if d.step_count >= target:
                break
    finally:
        uninstall_composed(model)
    return best


# ---------------------------------------------------------------------------
def byte_exact(model, L, device):
    print("\n=== TASK 4: BYTE-EXACT (fast-build == loop-build == K=1 reference) ===",
          flush=True)
    progs = []
    for name, prog, seed in _battery():
        if seed:
            continue
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        d = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
        if not d.halted:
            continue
        if any(d.frames[s].get("op") in ("DIV", "MOD") for s in range(d.step_count)):
            continue
        progs.append((name, code))
    progs.append(("loop_countdown60", build_loop_countdown(60)[0]))
    progs.append(("nested_6_16", build_nested(6, 16)[0]))
    progs.append(("nested_10_24", build_nested(10, 24)[0]))
    progs.append(("nested_12_28", build_nested(12, 28)[0]))   # deep nested loop
    all_ok = True
    install_composed(model, verbose=False)
    print(f"{'prog':>18} {'steps':>7} {'gpu(acc,fin,m)':>20} {'loop(acc,fin,m)':>20} "
          f"{'Lgpu':>5} {'Ldict':>6} {'ok':>4}", flush=True)
    try:
        for name, code in progs:
            draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
            if not draft.halted:
                continue
            chunk = max(4096, draft.step_count)
            # GPU-BUILD (deep-vectorized), on-chip + resident.
            rg, _ = _timed_run(model, L, code, draft, device, chunk,
                               onchip=True, resident=True, fast_build=True, gpu_build=True)
            # DICT-FAST build (numpy-dict, gpu_build OFF), on-chip + resident.
            rf, _ = _timed_run(model, L, code, draft, device, chunk,
                               onchip=True, resident=True, fast_build=True, gpu_build=False)
            # LOOP build (original per-step loop), on-chip + resident.
            rl, _ = _timed_run(model, L, code, draft, device, chunk,
                               onchip=True, resident=True, fast_build=False, gpu_build=False)
            # per-step AX/PC/SP/BP L-inf: gpu-vs-dict AND dict-vs-loop (byte-exact chain).
            linf_gpu = 0    # gpu-build vs dict-build
            linf_dict = 0   # dict-build vs loop-build
            for k in ("got_pc", "got_sp", "got_bp", "got_ax"):
                linf_gpu = max(linf_gpu, (rg[k] != rf[k]).sum().item())
                linf_dict = max(linf_dict, (rf[k] != rl[k]).sum().item())
            g = (rg["accepted_steps"], rg["final_ax"], rg["all_matched"])
            lo = (rl["accepted_steps"], rl["final_ax"], rl["all_matched"])
            draft_exact = (rg["all_matched"] and rg["accepted_steps"] == draft.step_count
                           and rg["final_ax"] == draft.final_ax_masked)
            ok = (g == lo) and (linf_gpu == 0) and (linf_dict == 0) and draft_exact
            all_ok = all_ok and ok
            print(f"{name:>18} {draft.step_count:>7} {str(g):>20} {str(lo):>20} "
                  f"{linf_gpu:>5} {linf_dict:>6} {'OK' if ok else 'FAIL':>4}", flush=True)
            if not ok:
                print(f"    gpu first_mismatch={rg['first_mismatch']}", flush=True)
                print(f"    loop first_mismatch={rl['first_mismatch']}", flush=True)
            _guard()
    finally:
        uninstall_composed(model)
    print(f"\n  -> {'ALL BYTE-EXACT (Linf=0: gpu==dict==loop==K=1 ref)' if all_ok else 'DIVERGENCE FOUND'}",
          flush=True)
    return all_ok


# ---------------------------------------------------------------------------
def measure_wholeframe(model, L, device, name, code, draft, skip_loop_build=True):
    n = draft.step_count
    fv, tv = _vram_gb()
    print(f"\n=== TASK 1-3: WHOLE-FRAME GIANT-K single dispatch — '{name}', {n} DIV-free "
          f"steps ===", flush=True)
    print(f"  VRAM free {fv:.1f}/{tv:.1f} GB  memAvail {_mem_avail_gb():.1f}GB", flush=True)
    install_composed(model, verbose=False)
    try:
        # Try chunk >= n (whole frame = ONE resident single-chunk graph).  If that OOMs the
        # graph capture / FFN activation, fall back to the largest chunk that fits — the
        # single-dispatch model still holds (schedule built ONCE, graph replays per chunk).
        def _try(chunk, resident, fast_build, gpu_build=True):
            gc.collect(); torch.cuda.empty_cache()
            try:
                return _timed_run(model, L, code, draft, device, chunk,
                                  onchip=True, resident=resident, fast_build=fast_build,
                                  gpu_build=gpu_build)
            except torch.cuda.OutOfMemoryError as e:
                _clear_levers()
                gc.collect(); torch.cuda.empty_cache()
                return None
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _clear_levers()
                    gc.collect(); torch.cuda.empty_cache()
                    return None
                raise

        # A failed CUDA-graph capture (OOM mid-capture) corrupts the capture stream so ALL
        # later captures fail too — so we must NOT attempt a doomed single-chunk graph.  Pick
        # the chunk from measured VRAM scaling: peak ~= 14.7 GB @ chunk=131072, ~10 GB @
        # 65536 (probe).  Cap the resident single-chunk to what fits; for a larger frame go
        # straight to the largest fitting chunk (multi-chunk single-dispatch — the schedule
        # is STILL built ONCE, the ONE graph replays over ceil(n/chunk) chunks).
        _fv, _tv = _vram_gb()
        # largest safe graph chunk for this GPU (leave headroom for the resident tables).
        if _tv >= 40:
            max_chunk = 262144
        elif _tv >= 22:
            max_chunk = 131072
        else:
            max_chunk = 65536
        if n <= max_chunk:
            chunk = max(4096, n)
            resident = True
            probe = _try(chunk, resident=True, fast_build=True)
            print(f"  [fit] whole frame n={n} fits as ONE resident single-chunk graph "
                  f"(chunk={chunk})", flush=True)
        else:
            resident = False
            chunk = max_chunk
            print(f"  [n={n} > max graph chunk {max_chunk}] multi-chunk single-dispatch: "
                  f"{(n+chunk-1)//chunk} chunks of {chunk} (schedule built ONCE)", flush=True)
            probe = _try(chunk, resident=False, fast_build=True)
            while probe is None and chunk > 8192:
                chunk //= 2
                print(f"  [OOM] retry chunk={chunk} ({(n+chunk-1)//chunk} chunks)...",
                      flush=True)
                probe = _try(chunk, resident=False, fast_build=True)
        if probe is None:
            print("  [FATAL] no chunk fits VRAM -> STOP", flush=True)
            return None
        n_chunks = (n + chunk - 1) // chunk
        print(f"  [dispatch plan] chunk={chunk}  n_chunks={n_chunks}  resident={resident}",
              flush=True)

        # ---- BUILD MINIMIZATION: fast-build (vectorized) vs loop-build (per-step) ----
        # The SCHEDULE BUILD cost is independent of the graph chunk (only capture/dispatch
        # depend on chunk), so the loop-build can be measured at a SMALLER chunk if the dense
        # cam_tables path OOMs at the vec-build's chunk — a fair build-cost comparison.  (The
        # dense loop-build materialises 3x ~2.6 GB dense cam_tables + a [K,D]@[D,D] GEMM, so
        # the vec/sparse build ALSO cuts build VRAM, enabling the larger graph chunk.)
        print("\n  --- build minimization: GPU-build vs dict-build vs loop-build ---",
              flush=True)
        _guard()
        # GPU-BUILD (deep-vectorized, C4_SCHED_GPU_BUILD): the default fast path.
        rf, tf = _try(chunk, resident=resident, fast_build=True, gpu_build=True)
        print(f"    gpu-build  : build {tf['build_s']*1e3:9.1f} ms  "
              f"({tf['build_us']:8.2f} us/step)  chunk={chunk}  matched={rf['all_matched']}",
              flush=True)
        gc.collect(); torch.cuda.empty_cache(); _guard()
        # DICT-BUILD (numpy-sparse, gpu_build OFF): the prior fast path (baseline 5.05).
        _rd = _try(chunk, resident=resident, fast_build=True, gpu_build=False)
        rd, td = _rd if _rd is not None else (None, None)
        if td is not None:
            print(f"    dict-build : build {td['build_s']*1e3:9.1f} ms  "
                  f"({td['build_us']:8.2f} us/step)  chunk={chunk}  matched={rd['all_matched']}"
                  f"   -> GPU-build speedup {td['build_s']/max(tf['build_s'],1e-9):.2f}x "
                  f"({td['build_us']:.2f} -> {tf['build_us']:.2f} us/step)", flush=True)
        gc.collect(); torch.cuda.empty_cache(); _guard()
        # LOOP-BUILD (original per-step, dense cam_tables) — OOMs at whole-frame K; only
        # attempt when explicitly requested (slow OOM-retry cascade).
        rl = tl = None
        loop_chunk = chunk
        if not skip_loop_build:
            while rl is None and loop_chunk >= 8192:
                r = _try(loop_chunk, resident=(resident and loop_chunk >= n), fast_build=False,
                         gpu_build=False)
                if r is not None:
                    rl, tl = r
                    break
                print(f"    [loop-build OOM @ chunk={loop_chunk} — dense cam_tables path; "
                      f"retry smaller]", flush=True)
                loop_chunk //= 2
                gc.collect(); torch.cuda.empty_cache()
            if tl is not None:
                print(f"    loop-build : build {tl['build_s']*1e3:9.1f} ms  "
                      f"({tl['build_us']:8.2f} us/step)  chunk={loop_chunk}  "
                      f"matched={rl['all_matched']}", flush=True)
            else:
                print(f"    loop-build : OOM at every chunk (dense cam_tables too big) — the "
                      f"vec/sparse build is REQUIRED at this K", flush=True)
        else:
            print(f"    loop-build : [skipped — OOMs at whole-frame K, dense cam_tables "
                  f"3x2.6GB won't fit; --no-skip-loop-build to force]", flush=True)

        # ---- the honest per-frame breakdown (vectorized build, on-chip, resident) ----
        t = tf
        print("\n  --- HONEST END-TO-END PER-FRAME BREAKDOWN (measured, "
              f"{n} steps) ---", flush=True)
        print(f"    build (schedule)   : {t['build_s']*1e3:9.2f} ms   "
              f"{t['build_us']:8.3f} us/step", flush=True)
        print(f"    capture (1st disp) : {t['capture_s']*1e3:9.2f} ms   "
              f"{t['capture_us']:8.3f} us/step", flush=True)
        print(f"    dispatch (replay)  : {t['dispatch_s']*1e3:9.2f} ms   "
              f"{t['dispatch_us']:8.3f} us/step", flush=True)
        print(f"    decode (cmp+DtoH)  : {t['decode_s']*1e3:9.2f} ms   "
              f"{t['decode_us']:8.3f} us/step", flush=True)
        print(f"    ================================================", flush=True)
        print(f"    E2E single frame (build + capture[incl the producing replay] + decode)"
              f" : {t['e2e_s']*1e3:9.2f} ms  for {n} steps", flush=True)

        # ---- honest sec/frame for 358,058 and 6,889,264 steps ----
        # The measured frame (n steps) is our GROUND TRUTH.  Of the four costs: the graph
        # CAPTURE is a fixed one-time (a chunk-shaped compile, independent of n); BUILD,
        # DISPATCH and DECODE all scale ~linearly in n (schedule tables O(n), replay over
        # n/chunk chunks, compare/reduce over n).  So per_frame(f) = capture + (build +
        # dispatch + decode)*(f/n) — an interpolation for 358 K (n=463 K brackets it) and a
        # modest linear extrapolation for the 6.89 M raw frame.
        cap_fixed = t['capture_s'] - t['dispatch_s']    # capture-only (1st pass minus replay)
        cap_fixed = max(cap_fixed, 0.0)
        scale_s = t['build_s'] + t['dispatch_s'] + t['decode_s']   # the O(n) part
        disp_per_step = t['dispatch_s'] / n
        print(f"\n  --- steady-state dispatch = {disp_per_step*1e6:.3f} us/step "
              f"(the '~5us' finding); graph-capture fixed one-time = {cap_fixed*1e3:.1f} ms "
              f"---", flush=True)
        for label, fsteps in (("render-reduced", RENDER_REDUCED_FRAME),
                              ("raw", RAW_FRAME)):
            per_frame = cap_fixed + scale_s * (fsteps / n)
            b_ = t['build_s'] * fsteps / n
            d_ = t['dispatch_s'] * fsteps / n
            dc_ = t['decode_s'] * fsteps / n
            fps = 1.0 / per_frame
            print(f"    {label:>15} ({fsteps:>9} steps): "
                  f"{per_frame*1e3:9.2f} ms/frame = {per_frame:.4f} s/frame  "
                  f"{fps:8.2f} fps", flush=True)
            print(f"        vs 1 s/frame  : {per_frame/REALTIME_S:8.3f}x   "
                  f"vs 35 fps (28.6ms): {per_frame/FPS35_S:8.2f}x", flush=True)
            print(f"        breakdown: build={b_*1e3:.1f}ms capture={cap_fixed*1e3:.1f}ms "
                  f"dispatch={d_*1e3:.1f}ms decode={dc_*1e3:.1f}ms  "
                  f"(dispatch {100*d_/per_frame:.0f}%, build {100*b_/per_frame:.0f}%)",
                  flush=True)

        # ---- the MEASURED e2e at THIS n as the single-frame GROUND TRUTH ----
        print(f"\n  --- MEASURED single-frame e2e GROUND TRUTH at n={n} ---", flush=True)
        print(f"    {t['e2e_s']*1e3:.2f} ms/frame = {t['e2e_s']:.4f} s/frame  "
              f"{1.0/t['e2e_s']:.2f} fps  vs 1s={t['e2e_s']:.3f}x  "
              f"vs 35fps={t['e2e_s']/FPS35_S:.1f}x  (n={n} steps)", flush=True)

        # ================================================================
        # HONEST CONTINUOUS STEADY-STATE (capture amortized OUT): the real-time number.
        # Capture the graph ONCE, then measure build+dispatch over consecutive frames.
        # ================================================================
        print(f"\n  === HONEST CONTINUOUS STEADY-STATE (capture amortized out) ===",
              flush=True)
        gc.collect(); torch.cuda.empty_cache(); _guard()
        cs = _continuous_steady_state(model, L, code, draft, device, chunk, resident,
                                      gpu_build=True, n_frames=4, pipeline=False)
        b_ss, d_ss, pf_ss = cs["build_s"], cs["dispatch_s"], cs["per_frame_s"]
        print(f"    serial   : build {b_ss*1e3:8.2f} ms + dispatch {d_ss*1e3:8.2f} ms "
              f"= {pf_ss*1e3:8.2f} ms/frame  (n={n})", flush=True)
        gc.collect(); torch.cuda.empty_cache(); _guard()
        cp = _continuous_steady_state(model, L, code, draft, device, chunk, resident,
                                      gpu_build=True, n_frames=4, pipeline=True)
        pf_pipe = cp["per_frame_s"]
        print(f"    pipelined: build(N+1)||dispatch(N) double-buffer "
              f"= {pf_pipe*1e3:8.2f} ms/frame  (~max(build,dispatch); "
              f"build {cp['build_s']*1e3:.1f} disp {cp['dispatch_s']*1e3:.1f})", flush=True)
        # extrapolate the CONTINUOUS per-frame (build + dispatch scale ~linearly in n).
        print(f"\n    --- CONTINUOUS per-frame @ target step counts (capture amortized "
              f"out) ---", flush=True)
        for label, fsteps in (("render-reduced", RENDER_REDUCED_FRAME), ("raw", RAW_FRAME)):
            pf_ser = pf_ss * (fsteps / n)
            pf_pip = pf_pipe * (fsteps / n)
            print(f"    {label:>15} ({fsteps:>9} steps): "
                  f"serial {pf_ser:.4f} s/frame ({1.0/pf_ser:6.2f} fps, "
                  f"{pf_ser/REALTIME_S:.2f}x 1s, {pf_ser/FPS35_S:.1f}x 35fps)  |  "
                  f"pipelined {pf_pip:.4f} s/frame ({1.0/pf_pip:6.2f} fps, "
                  f"{pf_pip/REALTIME_S:.2f}x 1s)", flush=True)
        t["cont_serial_s"] = pf_ss; t["cont_pipe_s"] = pf_pipe
        t["cont_build_s"] = b_ss; t["cont_dispatch_s"] = d_ss
        return t
    finally:
        uninstall_composed(model)
        gc.collect(); torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-byte-exact", action="store_true")
    ap.add_argument("--target", type=int, default=RENDER_REDUCED_FRAME)
    ap.add_argument("--no-skip-loop-build", action="store_true",
                    help="force the (slow, OOMs @ whole-frame K) loop-build comparison")
    args = ap.parse_args(argv)
    _guard()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _clear_levers()
    _composed_on()
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()
    ok = True
    if not args.skip_byte_exact:
        ok = byte_exact(model, L, device)
    # find the largest DIV-free frame that fits (near the render-reduced target).
    print("\n=== finding a large DIV-free frame (near render-reduced 358,058) ===",
          flush=True)
    big = find_big_divfree(model, L, device, target=args.target)
    if big is None:
        print("  no large DIV-free program found", flush=True)
    else:
        name, code, draft = big
        measure_wholeframe(model, L, device, name, code, draft,
                           skip_loop_build=not args.no_skip_loop_build)
    print(f"\n{'=== WHOLE-FRAME GIANT-K COMPLETE ===' if ok else '=== BYTE-EXACT FAILED ==='}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
