#!/usr/bin/env python3
"""FAITHFUL SINGLE-DISPATCH build-reduction fps: does the genuine path reach ~1 fps?

Measures the REAL render-reduced doom frame (358,058 steps) TRUE-PIPE for FOUR configs on a
CLEAN idle GPU with a 120K draft:

  1. FAST  baseline (draft-trusted, no build-reduction)          -> the ~0.98 fps reference
  2. FAST  + C4_SCHED_CACHE_RESOLVED                             -> the ~1.29 fps reference
  3. FAITHFUL baseline (genuine, value-verify pipelined+vectorized) -> the 0.408 fps baseline
  4. FAITHFUL + BUILD-REDUCTION:
       C4_SCHED_CACHE_RESOLVED         (schedule build < dispatch, the fast-path lever)
       C4_FAITHFUL_PRECOMPUTE_CACHE    (value-precompute reused across unchanged store-log)
       C4_FAITHFUL_PRECOMPUTE_THREAD   (value-precompute overlaps the schedule build)

Reports per-step breakdown (dispatch vs build stages), s/frame, fps, TRUE-PIPE, and the
byte-exact + genuine (rejects wrong draft) verdicts.  Same measurement machinery as
_agent_faithful_sd_doom_fps.py; this one sweeps the build-reduction levers.

Run: CUDA_VISIBLE_DEVICES=0 python -m c4_min._agent_faithful_build_reduce_fps --draft-steps 120000
"""
from __future__ import annotations
import argparse, os, sys, time, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_CMP32", "1")
os.environ.setdefault("C4_CMP32_ORDER", "1")
os.environ.setdefault("C4_SHIFT32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "1")
os.environ.setdefault("C4_IMM_NIBS", "6")
os.environ.setdefault("C4_PC_WIDE", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DOOM = "/home/alexlitz/Documents/misc/c4_doom"
sys.path.insert(0, DOOM)

import numpy as np
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000

from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min import precomputed_schedule as PS
from c4_min.faithful_single_dispatch import (build_faithful_plan, verify_faithful,
                                             build_faithful_precompute, verify_faithful_fast)
from run_c4_min import (data_segment, tag_compiler_syscalls,
                        install_compiler_abi_file_dispatcher)

# reuse the byte-exact snapshot loader / pow2 / dispatch helpers from the sibling harness.
from c4_min._agent_faithful_sd_doom_fps import (
    RENDER_REDUCED_FRAME, COMPOSED, _levers_on, _guard, _apply_pow2, _load_snapshot,
    build_doom_draft, _dispatch, _dispatch_with_qaddr)

BUILD_REDUCE = ["C4_SCHED_CACHE_RESOLVED", "C4_FAITHFUL_PRECOMPUTE_CACHE",
                "C4_FAITHFUL_PRECOMPUTE_THREAD"]


def _clear_reduce():
    for f in BUILD_REDUCE:
        os.environ.pop(f, None)


def _measure_one(model, L, code, draft, device, chunk, n_frames, faithful, reduce):
    """Measure ONE (fast|faithful) x (baseline|build-reduced) path with only ITS graph
    resident.  Returns TRUE-PIPE per-frame median + the build/dispatch/value breakdown."""
    dev = torch.device(device); n = draft.step_count; mask = 0xFFFFFFFF
    _levers_on(chunk)
    _clear_reduce()
    if reduce:
        for f in BUILD_REDUCE:
            os.environ[f] = "1"
    # drop any cache left on the draft from a prior config (fresh, honest per-config warmup).
    for attr in ("_resolved_cache", "_faithful_precompute_cache"):
        if hasattr(draft, attr):
            delattr(draft, attr)
    if faithful:
        os.environ["C4_FAITHFUL_SINGLE_DISPATCH"] = "1"
    else:
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    torch.cuda.reset_peak_memory_stats(dev)
    plan = build_faithful_plan(model, L, code, draft) if faithful else None
    ws = np.asarray(draft.win_starts[:n], dtype=np.int64)
    # one-time build+capture (also byte-exact + genuine verdict below).
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    if faithful:
        _p, _s, _bp, _a, ma0 = _dispatch_with_qaddr(sched, sg, dev, n)
    else:
        _dispatch(sched, sg, dev, n)
    torch.cuda.synchronize(dev); t_build0 = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    del sched; gc.collect(); torch.cuda.empty_cache()
    tag = ("FAITHFUL" if faithful else "FAST") + ("+reduce" if reduce else "")
    print(f"  [{tag}] one-time build+capture {t_build0*1e3:.0f} ms  chunk={sg.chunk}  "
          f"peak_vram {peak:.1f} GB", flush=True)

    # serial breakdown medians (build vs dispatch vs value-verify) — for the per-step table.
    B = []; D = []; VER = []
    for _ in range(n_frames):
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.perf_counter()
        s2 = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
        pre = build_faithful_precompute(draft, plan, ws, n, mask=mask) if faithful else None
        torch.cuda.synchronize(dev); b = time.perf_counter() - t0
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        if faithful:
            _p, _s, _bp, _a, ma = _dispatch_with_qaddr(s2, sg, dev, n)
            torch.cuda.synchronize(dev); d = time.perf_counter() - t0
            t0 = time.perf_counter(); verify_faithful_fast(pre, ma); v = time.perf_counter() - t0
        else:
            _dispatch(s2, sg, dev, n); torch.cuda.synchronize(dev); d = time.perf_counter() - t0; v = 0.0
        B.append(b); D.append(d); VER.append(v)
        del s2; gc.collect()
    b = float(np.median(B)); d = float(np.median(D)); ver = float(np.median(VER))

    # byte-exact + genuine verdict on a fresh frame.
    gc.collect(); torch.cuda.empty_cache()
    s_ser = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
    torch.cuda.synchronize(dev)
    vd = None
    if faithful:
        pc, sp, bp2, ax, ma = _dispatch_with_qaddr(s_ser, sg, dev, n); ax = ax & mask
        pre_ser = build_faithful_precompute(draft, plan, ws, n, mask=mask)
        vd = verify_faithful_fast(pre_ser, ma)
        vd_ref = verify_faithful(draft, plan, ma, ws, mask=mask)
        assert (vd.ok == vd_ref.ok and vd.first_bad_step == vd_ref.first_bad_step
                and vd.kind == vd_ref.kind), f"pipelined verdict != verify_faithful: {vd} vs {vd_ref}"
    else:
        pc, sp, bp2, ax = _dispatch(s_ser, sg, dev, n); ax = ax & mask
    bad = (((pc != s_ser.want_pc) | (ax != (s_ser.want_ax & mask)) | (sp != s_ser.want_sp)
            | (bp2 != s_ser.want_bp)))
    bad = torch.where(s_ser.is_halt, ax != (s_ser.want_ax & mask), bad) & (~s_ser.is_file)
    nbad = int(bad.sum())
    del s_ser; gc.collect(); torch.cuda.empty_cache()

    # TRUE-PIPE (double-buffer): build (+ value-precompute) on the background thread overlaps
    # the prior frame's dispatch replay.  With the reduce flags on, the build thread's steady
    # state is a cache-hit copy (schedule) + a cache-hit value-precompute.
    os.environ["C4_SCHED_PIPELINE"] = "1"
    WARMUP = 3                       # warm the resolved + precompute caches before steady state
    pb = PS.PipelinedScheduleBuilder(model, L, code, draft, dev, sg, mask=mask, faithful=faithful)
    cur = pb.build_blocking(); pre_cur = pb.last_precompute(); torch.cuda.synchronize(dev); pb.start()
    walls = []; tot = n_frames + WARMUP
    for i2 in range(tot):
        torch.cuda.synchronize(dev); tf = time.perf_counter()
        if faithful:
            _p, _s, _b, _a, ma = _dispatch_with_qaddr(cur, sg, dev, n)
        else:
            _dispatch(cur, sg, dev, n)
        if faithful:
            verify_faithful_fast(pre_cur, ma)
        nxt = pb.wait(); del cur; cur = nxt; pre_cur = pb.last_precompute()
        if i2 < tot - 1: pb.start()
        torch.cuda.synchronize(dev); walls.append(time.perf_counter() - tf)
    del cur; gc.collect(); torch.cuda.empty_cache()
    os.environ.pop("C4_SCHED_PIPELINE", None)
    steady = sorted(walls[WARMUP:]); pipe = steady[len(steady) // 2]
    del sg; gc.collect(); torch.cuda.empty_cache()
    os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    _clear_reduce()
    serial = b + d + ver
    return dict(build=b, disp=d, ver=ver, serial=serial, pipe=pipe, peak=peak,
                nbad=nbad, ok=(vd.ok if vd else None), vd=vd, n=n)


def _row(label, r, n):
    ser = r["serial"]; pipe = r["pipe"]
    ss = ser * (RENDER_REDUCED_FRAME / n); sp = pipe * (RENDER_REDUCED_FRAME / n)
    return (f"    {label:>34}  {ss:8.3f}s {1.0/ss:6.3f}fps  {sp:8.3f}s {1.0/sp:6.3f}fps"
            f"  {'>=1!' if 1.0/sp >= 1.0 else '<1'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draft-steps", type=int, default=120000)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--n-frames", type=int, default=5)
    args = ap.parse_args(argv)
    _guard(); device = args.device
    d = build_doom_draft(args.draft_steps); _guard()
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=max(d.code_off + 2, 256),
                                            recurrent_divmod=True, addr32=True,
                                            compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} {time.time()-t0:.1f}s", flush=True)
    _guard()
    ops, imms, _data = _load_snapshot(); _apply_pow2(ops, imms)
    code = tag_compiler_syscalls([isa.Instr(int(o), int(i) & 0xFFFFFFFF) for o, i in zip(ops, imms)], isa)
    install_composed(model, verbose=False)
    n = d.step_count
    results = {}
    try:
        print(f"\n=== BUILD-REDUCTION fps ({n} DIV-free steps, {args.draft_steps} draft) ===", flush=True)
        print("\n--- 1. FAST baseline ---", flush=True)
        results["fast"] = _measure_one(model, L, code, d, device, args.chunk, args.n_frames, False, False)
        print("\n--- 2. FAST + C4_SCHED_CACHE_RESOLVED ---", flush=True)
        results["fast_reduce"] = _measure_one(model, L, code, d, device, args.chunk, args.n_frames, False, True)
        print("\n--- 3. FAITHFUL baseline (value-verify pipelined+vectorized) ---", flush=True)
        results["faith"] = _measure_one(model, L, code, d, device, args.chunk, args.n_frames, True, False)
        print("\n--- 4. FAITHFUL + BUILD-REDUCTION (sched-cache + precompute-cache + precompute-thread) ---", flush=True)
        results["faith_reduce"] = _measure_one(model, L, code, d, device, args.chunk, args.n_frames, True, True)
    finally:
        uninstall_composed(model)

    print(f"\n  === REAL-DOOM per-frame @ render-reduced ({RENDER_REDUCED_FRAME} steps) ===", flush=True)
    print(f"    {'path':>34}  {'SERIAL':>22}  {'TRUE-PIPE':>22}", flush=True)
    print(_row("fast (baseline)", results["fast"], n), flush=True)
    print(_row("fast + sched-cache", results["fast_reduce"], n), flush=True)
    print(_row("FAITHFUL (baseline)", results["faith"], n), flush=True)
    print(_row("FAITHFUL + build-reduction", results["faith_reduce"], n), flush=True)

    print("\n  COST BREAKDOWN (us/step):", flush=True)
    for key, lab in (("fast", "fast baseline"), ("fast_reduce", "fast+cache"),
                     ("faith", "FAITHFUL baseline"), ("faith_reduce", "FAITHFUL+reduce")):
        r = results[key]
        print(f"    {lab:>20}: build {r['build']/n*1e6:7.3f} | dispatch {r['disp']/n*1e6:7.3f} | "
              f"value-verify {r['ver']/n*1e6:6.4f}", flush=True)

    fr = results["faith_reduce"]; fb = results["faith"]; f0 = results["fast"]; fc = results["fast_reduce"]
    print(f"\n  faithful build: {fb['build']/n*1e6:.3f} -> {fr['build']/n*1e6:.3f} us/step "
          f"(build-reduction); faithful dispatch {fr['disp']/n*1e6:.3f} us/step", flush=True)
    print(f"  build-bound? faithful+reduce build {fr['build']/n*1e6:.3f} "
          f"{'<' if fr['build']<fr['disp'] else '>='} dispatch {fr['disp']/n*1e6:.3f} "
          f"-> {'DISPATCH-BOUND' if fr['build']<fr['disp'] else 'still BUILD-bound'}", flush=True)
    sp_fr = fr['pipe']*(RENDER_REDUCED_FRAME/n)
    print(f"\n  FAITHFUL+reduce TRUE-PIPE: {sp_fr:.3f} s/frame  {1.0/sp_fr:.3f} fps  "
          f"{'>= 1 fps!' if 1.0/sp_fr>=1.0 else '(< 1 fps)'}", flush=True)
    print(f"  vs FAITHFUL baseline {fb['pipe']*(RENDER_REDUCED_FRAME/n):.3f}s "
          f"({1.0/(fb['pipe']*(RENDER_REDUCED_FRAME/n)):.3f} fps)  "
          f"vs fast {1.0/(f0['pipe']*(RENDER_REDUCED_FRAME/n)):.3f} fps  "
          f"vs fast+cache {1.0/(fc['pipe']*(RENDER_REDUCED_FRAME/n)):.3f} fps", flush=True)
    print(f"  faithful/fast pipe slowdown: baseline {fb['pipe']/f0['pipe']:.2f}x -> "
          f"reduce {fr['pipe']/fc['pipe']:.2f}x (vs fast+cache)", flush=True)
    ok = (fr['nbad'] == 0 and fr['ok'])
    print(f"\n  BYTE-EXACT (faithful+reduce decode==draft AND verify accepts all): {ok} "
          f"(mismatches={fr['nbad']}, verify_ok={fr['ok']}, "
          f"addr_chk={fr['vd'].n_addr_checked} val_chk={fr['vd'].n_value_checked} "
          f"rt_chk={fr['vd'].n_routing_checked})", flush=True)
    print(f"  VRAM peak (faithful+reduce graph) {fr['peak']:.1f} GB", flush=True)
    print("\n=== COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
