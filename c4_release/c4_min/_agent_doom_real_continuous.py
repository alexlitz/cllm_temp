#!/usr/bin/env python3
"""_agent_doom_real_continuous.py — the AIRTIGHT real-doom-trace composed continuous fps.

Runs the ACTUAL id-Doom bytecode (the byte-exact-verified _doom_bytecode_snapshot.npz,
the same code that runs byte-exact vs the independent aligned 32-bit oracle) through the
COMPOSED single-dispatch precomputed schedule — NOT the DIV-free nested(120,255) proxy.

Chain:
  1. draft a REAL DIV-free doom segment (draft_pf_program on the doom snapshot + the WAD
     data_seg + the compiler-ABI fio), skipping the one-time WAD/R_Init DIV cluster so the
     drafted window is genuinely DIV-free (asserts it, satisfying run_verify);
  2. feed that real doom draft into build_schedule + the single graph-per-chunk dispatch
     (the exact composed stack: dead-block-fusion + direct-CAM + fused-megablock +
     FFN-hidden-fusion + on-chip residual + resident batch), all composed flags ON;
  3. measure the continuous steady-state build + dispatch + TRUE-PIPE per-frame wall,
     scaled to the 358,058-step render-reduced frame, and report the fps;
  4. BYTE-EXACT: the single-dispatch decode of the real doom trace == the draft's per-step
     register targets (PC/AX/SP/BP L-inf=0) — the draft is == the transformer transition ==
     the independent aligned oracle (capstone C), so this is byte-exact real doom.

Requires ``<c4_release>/_doom_bytecode_snapshot.npz`` (the byte-exact-verified doom code
snapshot, ops/imms slot-re-encoded; gitignored — copy from the checkout that produced it,
committed inert in 923fb5e6) and the sibling ``/home/alexlitz/Documents/misc/c4_doom``
(``run_c4_min`` helpers).

Run: CUDA_VISIBLE_DEVICES=0 python -m c4_min._agent_doom_real_continuous --device cuda:0
     --draft-steps 120000 --image pow2 [--cache-resolved]
"""
from __future__ import annotations
import argparse, os, sys, time, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
# doom-proven config (memory note project_doom_port_native_c4_byte_exact): CFM +
# DRAFT_CMP32 + CMP32(+order) + MEM_ADDR_BITS=18 + EXACT_EVICT + MEM_EFF + wide PC/IMM +
# SHIFT32 (both draft & model — needed for the #829 pow2 DIV->SHR reduction to be exact).
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
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000   # doom compact stack base

from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, PFDraft
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min import precomputed_schedule as PS
from run_c4_min import (data_segment, tag_compiler_syscalls,
                        install_compiler_abi_file_dispatcher)

RENDER_REDUCED_FRAME = 358_058
RAW_FRAME = 6_889_264

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _levers_on(chunk):
    for f in COMPOSED:
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = "512"


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


def _slice_draft(d: PFDraft, lo: int, hi: int) -> PFDraft:
    """Extract a DIV-free sub-window [lo, hi) of a doom draft as a standalone PFDraft that
    the schedule can ingest.  The schedule reads .frames (decode targets), .store_log,
    .win_starts, .step_count, .prtf_steps, .load_log, .read_log, .code_off, .tokens.  We
    rebase the token/frame stream to start at the window's first query row so the schedule's
    per-position overlay math (frame-role tags / store addr-bits) lines up.

    Because the schedule's per-step verify is a per-QUERY-ROW independent check (each row's
    overlay + the frozen prefix cache is the driver's last-row read), a contiguous sub-window
    is byte-identical to the full run over those rows — the frozen prefix (code frames + all
    prior store frames) is IDENTICAL, and the window's own query rows carry the same targets.
    We keep the FULL token prefix (code frames + all store frames up to hi) so every gather /
    CAM read resolves to the exact store row it does in the full run, and restrict the decode
    to the [lo, hi) query rows.  This is the honest 'a real doom sub-frame' the composed
    schedule dispatches — same rows, same gathers, same decode."""
    # keep the full prefix; the schedule iterates draft.step_count query rows, so to time a
    # window we present a draft whose frames/win_starts are the [lo,hi) slice but whose
    # store_log/tokens/code_off carry the full context the CAM resolves against.
    sub = PFDraft(
        tokens=d.tokens,                       # full token stream (prefix intact)
        frames=d.frames[lo:hi],
        store_log=d.store_log,                 # full store_log (CAM resolves latest-write)
        step_count=hi - lo,
        halted=d.halted,
        final_ax_masked=d.frames[hi - 1]["ax"],
        win_starts=d.win_starts[lo:hi],
        out=[s for s in (d.out or [])],
        prtf_steps=[s - lo for s in (d.prtf_steps or []) if lo <= s < hi],
        load_log={k: v for k, v in (d.load_log or {}).items()},
        read_log={k: v for k, v in (d.read_log or {}).items()},
        code_off=d.code_off)
    return sub


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


IMM_OP, DIV_OP, MOD_OP, SHR_OP, AND_OP = 1, 28, 29, 24, 16


def _apply_pow2(ops, imms):
    """#829 pow2 strength-reduction on the (op, imm) stream: IMM 2^k ; DIV -> IMM k ; SHR
    and IMM 2^k ; MOD -> IMM 2^k-1 ; AND.  Byte-exact bytecode transform (same peephole
    pow2_strength_reduce applies to the c4img; 92 doom sites)."""
    n = len(ops)
    npow = 0
    for i in range(n - 1):
        o0, m0, o1 = int(ops[i]), int(imms[i]), int(ops[i + 1])
        if o0 == IMM_OP and m0 > 0 and (m0 & (m0 - 1)) == 0:
            if o1 == DIV_OP:
                imms[i] = m0.bit_length() - 1; ops[i + 1] = SHR_OP; npow += 1
            elif o1 == MOD_OP:
                imms[i] = m0 - 1; ops[i + 1] = AND_OP; npow += 1
    return npow


def _load_snapshot():
    snap = np.load(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "_doom_bytecode_snapshot.npz"))
    return list(snap["ops"]), list(snap["imms"]), snap["data"]


def build_doom_draft(image, draft_steps, skip_div_prefix):
    # the byte-exact snapshot (ops/imms already slot-re-encoded, verified vs aligned oracle)
    ops, imms, data = _load_snapshot()
    n = len(ops)
    if image == "pow2":
        npow = _apply_pow2(ops, imms)     # #829: 92 pow2 DIV/MOD -> SHR/AND (bytecode)
        print(f"[doom] #829 pow2 strength-reduction: {npow} DIV/MOD sites -> SHR/AND",
              flush=True)
    install_compiler_abi_file_dispatcher()
    code_isa = tag_compiler_syscalls(
        [isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF) for i in range(n)], isa)
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    print(f"[doom] drafting REAL doom bytecode ({n:,} instrs), max_steps={draft_steps:,} ...",
          flush=True)
    t0 = time.time()
    d = draft_pf_program(code_isa, max_steps=draft_steps, mask=0xFFFFFFFF,
                         data_seg=data_segment([int(b) for b in data]), fio=fio)
    print(f"[doom] drafted {d.step_count:,} steps halted={d.halted} in {time.time()-t0:.0f}s "
          f"({d.step_count/max(time.time()-t0,1e-9):.0f}/s)", flush=True)
    dm = [i for i in range(d.step_count) if d.frames[i].get("op") in ("DIV", "MOD")]
    print(f"[doom] generic DIV/MOD in drafted trace: {len(dm)}  {dm[:5]}", flush=True)
    # pow2 image is DIV-free from step 0 -> the WHOLE draft is the schedule input (no slice,
    # so the frame-counter reconstruction in the CAM builder stays consistent).
    lo, hi = 0, d.step_count
    win_dm = [i for i in range(lo, hi) if d.frames[i].get("op") in ("DIV", "MOD")]
    assert not win_dm, f"draft not DIV-free: {win_dm[:5]}"
    print(f"[doom] DIV-free real-doom trace: [{lo:,}, {hi:,}) = {hi-lo:,} steps "
          f"(0 generic DIV/MOD) — satisfies run_verify's DIV-free assertion", flush=True)
    return d, lo, hi


def measure(model, L, code, draft, device, chunk, n_frames):
    dev = torch.device(device)
    n = draft.step_count
    mask = 0xFFFFFFFF
    _levers_on(chunk)
    # ---- capture the graph ONCE. ----
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    torch.cuda.synchronize(dev); t_build0 = time.perf_counter() - t0
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    _dispatch(sched, sg, dev, n)
    torch.cuda.synchronize(dev); t_capture = time.perf_counter() - t0
    print(f"  [one-time] build0 {t_build0*1e3:.1f} ms  capture {t_capture*1e3:.1f} ms  "
          f"chunk={sg.chunk}  n_chunks={(n+sg.chunk-1)//sg.chunk}", flush=True)
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    fv, tv = torch.cuda.mem_get_info(dev)
    print(f"  [VRAM] free {fv/1e9:.1f}/{tv/1e9:.1f} GB  peak_alloc {peak:.1f} GB", flush=True)
    del sched; gc.collect(); torch.cuda.empty_cache()

    # ---- SERIAL continuous. ----
    build_s = []; disp_s = []; s2 = None
    for _ in range(n_frames):
        del s2; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        s2 = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
        torch.cuda.synchronize(dev); build_s.append(time.perf_counter() - t0)
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        _dispatch(s2, sg, dev, n)
        torch.cuda.synchronize(dev); disp_s.append(time.perf_counter() - t0)
    del s2; gc.collect(); torch.cuda.empty_cache()
    b = sum(build_s) / len(build_s); d = sum(disp_s) / len(disp_s)
    print(f"\n  SERIAL   continuous: build {b*1e3:8.2f} ms + dispatch {d*1e3:8.2f} ms "
          f"= {(b+d)*1e3:8.2f} ms/frame  ({b/n*1e6:.3f} + {d/n*1e6:.3f} us/step)", flush=True)

    # ---- TRUE-PIPE continuous. ----
    os.environ["C4_SCHED_PIPELINE"] = "1"
    WARMUP = 2
    pb = PS.PipelinedScheduleBuilder(model, L, code, draft, dev, sg, mask=mask)
    cur = pb.build_blocking(); torch.cuda.synchronize(dev); pb.start()
    frame_walls = []; total_frames = n_frames + WARMUP
    for i in range(total_frames):
        torch.cuda.synchronize(dev); t_f0 = time.perf_counter()
        _dispatch(cur, sg, dev, n)
        nxt = pb.wait(); del cur; cur = nxt
        if i < total_frames - 1:
            pb.start()
        torch.cuda.synchronize(dev); frame_walls.append(time.perf_counter() - t_f0)
    del cur; gc.collect(); torch.cuda.empty_cache()
    steady = sorted(frame_walls[WARMUP:])
    pf_thread = steady[len(steady) // 2]
    print(f"  TRUE-PIPE  continuous: double-buffer = {pf_thread*1e3:8.2f} ms/frame "
          f"(steady median; ~max(build,dispatch)={max(b,d)*1e3:.2f} ms)", flush=True)
    print(f"             per-frame walls (ms): {[round(x*1e3) for x in frame_walls]}"
          f"  [first {WARMUP} warmup]", flush=True)

    # ---- BYTE-EXACT: thread-pipe dispatch == serial dispatch == draft targets. ----
    s_ser = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
    torch.cuda.synchronize(dev)
    ser_pc, ser_sp, ser_bp, ser_ax = _dispatch(s_ser, sg, dev, n)
    ser_ax = ser_ax & mask
    torch.cuda.synchronize(dev)
    # compare to the draft's per-step targets (the aligned-oracle == transformer targets)
    want_pc = s_ser.want_pc; want_ax = s_ser.want_ax & mask
    want_sp = s_ser.want_sp; want_bp = s_ser.want_bp
    is_file = s_ser.is_file; is_halt = s_ser.is_halt
    bad_normal = ((ser_pc != want_pc) | (ser_ax != want_ax) | (ser_sp != want_sp)
                  | (ser_bp != want_bp))
    bad = torch.where(is_halt, ser_ax != want_ax, bad_normal) & (~is_file)
    n_bad_vs_draft = int(bad.sum())
    print(f"  [BYTE-EXACT] single-dispatch decode vs DRAFT targets (== aligned oracle) "
          f"mismatches = {n_bad_vs_draft}  ({'L-inf=0 OK' if n_bad_vs_draft==0 else 'MISMATCH!'})",
          flush=True)
    pb2 = PS.PipelinedScheduleBuilder(model, L, code, draft, dev, sg, mask=mask)
    pb2.start(); s_thr = pb2.wait()
    thr_pc, thr_sp, thr_bp, thr_ax = _dispatch(s_thr, sg, dev, n)
    torch.cuda.synchronize(dev)
    linf = max(int((ser_pc != thr_pc).sum()), int((ser_sp != thr_sp).sum()),
               int((ser_bp != thr_bp).sum()), int(((ser_ax) != (thr_ax & mask)).sum()))
    print(f"  [BYTE-EXACT] thread-pipe vs serial decode mismatches = {linf}  "
          f"({'L-inf=0 OK' if linf==0 else 'MISMATCH!'})", flush=True)
    os.environ.pop("C4_SCHED_PIPELINE", None)
    return {"build_s": b, "dispatch_s": d, "serial_s": b + d, "thread_s": pf_thread,
            "peak_gb": peak, "chunk": int(sg.chunk), "n": n,
            "byte_exact_vs_draft": n_bad_vs_draft == 0, "pipe_linf": linf}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image", default="pow2", choices=["pow2", "base"])
    ap.add_argument("--draft-steps", type=int, default=200000)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--no-skip-div", action="store_true")
    ap.add_argument("--cache-resolved", action="store_true",
                    help="Task 2 build-reduction: C4_SCHED_CACHE_RESOLVED (cache the resolved "
                         "gather arrays; per-frame build becomes a copy)")
    args = ap.parse_args(argv)
    if args.cache_resolved:
        os.environ["C4_SCHED_CACHE_RESOLVED"] = "1"
    _guard()
    device = args.device

    d, lo, hi = build_doom_draft(args.image, args.draft_steps, not args.no_skip_div)
    sub = d if (lo == 0 and hi == d.step_count) else _slice_draft(d, lo, hi)
    _guard()

    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=max(d.code_off + 2, 256),
                                            recurrent_divmod=True, addr32=True,
                                            compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"{time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()

    # the doom 'code' the schedule needs (isa list) — the SAME pow2-reduced code the trace
    # was drafted from, so the CFM code-fetch CAM resolves to the exact executed op/imm.
    ops, imms, _data = _load_snapshot()
    if args.image == "pow2":
        _apply_pow2(ops, imms)
    code = tag_compiler_syscalls(
        [isa.Instr(int(o), int(i) & 0xFFFFFFFF) for o, i in zip(ops, imms)], isa)

    install_composed(model, verbose=False)
    try:
        print(f"\n=== REAL-DOOM COMPOSED CONTINUOUS (n={sub.step_count} DIV-free doom steps) ===",
              flush=True)
        r = measure(model, L, code, sub, device, args.chunk, n_frames=args.n_frames)
    finally:
        uninstall_composed(model)

    n = r["n"]
    print(f"\n  --- REAL-DOOM continuous per-frame @ target step counts ---", flush=True)
    for label, fsteps in (("render-reduced", RENDER_REDUCED_FRAME), ("raw", RAW_FRAME)):
        ser = r["serial_s"] * (fsteps / n)
        thr = r["thread_s"] * (fsteps / n)
        print(f"    {label:>15} ({fsteps:>9} steps):", flush=True)
        print(f"        SERIAL     {ser:8.4f} s/frame  {1.0/ser:7.2f} fps", flush=True)
        print(f"        TRUE-PIPE  {thr:8.4f} s/frame  {1.0/thr:7.2f} fps"
              f"   {'>=1 fps!' if 1.0/thr >= 1.0 else '<1 fps'}", flush=True)
    print(f"\n  build {r['build_s']/n*1e6:.3f} us/step  dispatch {r['dispatch_s']/n*1e6:.3f} "
          f"us/step  ({'BUILD' if r['build_s']>r['dispatch_s'] else 'DISPATCH'}-bound)",
          flush=True)
    print(f"  BYTE-EXACT vs draft(==aligned oracle): {r['byte_exact_vs_draft']}  "
          f"pipe-vs-serial L-inf: {r['pipe_linf']}", flush=True)
    print(f"  VRAM peak {r['peak_gb']:.1f} GB @ chunk {r['chunk']}", flush=True)
    print("\n=== REAL-DOOM COMPOSED CONTINUOUS COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
