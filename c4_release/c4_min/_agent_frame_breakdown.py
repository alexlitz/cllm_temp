#!/usr/bin/env python3
"""TASK 1 — per-frame FAITHFUL dispatch breakdown on the REAL doom frame.

Decompose the faithful per-frame wall (the ``_dispatch_with_qaddr`` loop) into:
  (A) pure GRAPH REPLAY   — the device compute (block0+mega+live+decode+in-graph W_q).
  (B) qaddr D2H copy      — the per-chunk ``last_qaddr().to("cpu")`` host round-trips.
  (C) reg-lane copies     — the got_* GPU->GPU slice copies.
  (D) sync + python loop   — the per-frame torch.cuda.synchronize + loop overhead.

Reconciles ab084eca's "~340 ms fixed per-frame overhead" (whole-frame wall level) vs the
FFN-GEMM-bound sub-graph profiler (isolated graph body).  Measures the SAME render-reduced
doom frame the fps harness uses, at the requested draft-steps, composed faithful config.

Run: CUDA_VISIBLE_DEVICES=0 python -m c4_min._agent_frame_breakdown --draft-steps 120000
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

RENDER_REDUCED_FRAME = 358_058

from c4_min._agent_faithful_sd_doom_fps import (
    build_doom_draft, _levers_on, _guard, _load_snapshot, _apply_pow2)
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min import precomputed_schedule as PS
from run_c4_min import tag_compiler_syscalls


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def profile(model, L, code, draft, device, chunk, n_frames):
    dev = torch.device(device); n = draft.step_count; mask = 0xFFFFFFFF
    _levers_on(chunk)
    os.environ["C4_FAITHFUL_SINGLE_DISPATCH"] = "1"
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    onchip = sched.onchip; h0s = sched.h0_folded if onchip else sched.h0_table
    got_pc = torch.empty(n, dtype=torch.long, device=dev)
    got_sp = torch.empty(n, dtype=torch.long, device=dev)
    got_bp = torch.empty(n, dtype=torch.long, device=dev)
    got_ax = torch.empty(n, dtype=torch.long, device=dev)

    def _one_full():
        per_head = {}
        for lo in range(0, n, sg.chunk):
            hi = min(lo + sg.chunk, n); h0 = h0s[lo:hi].unsqueeze(0)
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc, sp, bp, ax = sg.replay(h0, None, None, delta=delta, resident=False)
            got_pc[lo:hi].copy_(pc); got_sp[lo:hi].copy_(sp)
            got_bp[lo:hi].copy_(bp); got_ax[lo:hi].copy_(ax)
            for key, t in sg.last_qaddr(hi - lo).items():
                per_head.setdefault(key, []).append(
                    (lo, hi, t.detach().to("cpu").numpy().astype("int64")))
        return per_head

    _one_full(); torch.cuda.synchronize(dev)
    build_ms = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    nchunks = (n + sg.chunk - 1) // sg.chunk
    nheads = len(sg.last_qaddr(1))
    print(f"  built: chunk={sg.chunk} nchunks={nchunks} n_qaddr_heads={nheads} "
          f"build={build_ms:.0f}ms peak={peak:.1f}GB", flush=True)

    def _replay_only():
        for lo in range(0, n, sg.chunk):
            hi = min(lo + sg.chunk, n); h0 = h0s[lo:hi].unsqueeze(0)
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            sg.replay(h0, None, None, delta=delta, resident=False)

    def _replay_regs():
        for lo in range(0, n, sg.chunk):
            hi = min(lo + sg.chunk, n); h0 = h0s[lo:hi].unsqueeze(0)
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc, sp, bp, ax = sg.replay(h0, None, None, delta=delta, resident=False)
            got_pc[lo:hi].copy_(pc); got_sp[lo:hi].copy_(sp)
            got_bp[lo:hi].copy_(bp); got_ax[lo:hi].copy_(ax)

    def _qaddr_d2h_only():
        for lo in range(0, n, sg.chunk):
            hi = min(lo + sg.chunk, n)
            for key, t in sg.last_qaddr(hi - lo).items():
                t.detach().to("cpu").numpy().astype("int64")

    def _timed(fn, reps):
        walls = []
        for _ in range(reps):
            torch.cuda.synchronize(dev); t = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev); walls.append((time.perf_counter() - t) * 1e3)
        return _median(walls)

    for _ in range(2):
        _replay_only(); _replay_regs(); _one_full(); _qaddr_d2h_only()
    torch.cuda.synchronize(dev)

    reps = max(3, n_frames)
    t_full = _timed(_one_full, reps)
    t_replay = _timed(_replay_only, reps)
    t_regs = _timed(_replay_regs, reps)
    t_qaddr = _timed(_qaddr_d2h_only, reps)

    A = t_replay
    C = max(0.0, t_regs - t_replay)
    B = t_qaddr
    D = max(0.0, t_full - t_replay - C - B)
    scale = RENDER_REDUCED_FRAME / n

    def _row(name, ms):
        return (f"    {name:>26}: {ms:8.2f} ms/frame  {ms*1e3/n:7.3f} us/step  "
                f"{100*ms/t_full:5.1f}%  ->@358k {ms*scale/1e3:6.3f}s")

    print(f"\n  === FAITHFUL per-frame breakdown (n={n}, frame wall {t_full:.1f}ms) ===", flush=True)
    print(_row("(A) graph replay [compute]", A), flush=True)
    print(_row("(B) qaddr D2H copy", B), flush=True)
    print(_row("(C) reg-lane copies", C), flush=True)
    print(_row("(D) sync+python loop", D), flush=True)
    print(_row("FULL frame wall", t_full), flush=True)
    fps_full = 1.0 / (t_full * scale / 1e3)
    print(f"\n  frame wall @358k = {t_full*scale/1e3:.3f}s -> {fps_full:.3f} fps", flush=True)
    overhead = B + C + D
    print(f"\n  COMPUTE (A) = {100*A/t_full:.0f}%  |  OVERHEAD (B+C+D) = {100*overhead/t_full:.0f}%", flush=True)
    if A > overhead:
        print("  VERDICT: COMPUTE-BOUND (graph replay dominates the frame wall).", flush=True)
    else:
        print("  VERDICT: OVERHEAD-BOUND (D2H/sync/loop dominates).", flush=True)
    del sched, sg; gc.collect(); torch.cuda.empty_cache()
    os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    return dict(full=t_full, A=A, B=B, C=C, D=D, n=n, chunk=chunk, nchunks=nchunks,
                nheads=nheads, peak=peak)


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
    try:
        print(f"\n=== TASK 1 FRAME BREAKDOWN ({d.step_count} DIV-free steps) ===", flush=True)
        profile(model, L, code, d, device, args.chunk, args.n_frames)
    finally:
        uninstall_composed(model)
    print("\n=== BREAKDOWN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
