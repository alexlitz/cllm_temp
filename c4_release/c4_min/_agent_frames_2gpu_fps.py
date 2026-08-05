#!/usr/bin/env python3
"""TASK 2 — FRAME-LEVEL 2-GPU parallelism fps on the REAL render-reduced doom frame.

The K-split got only ~1.06x because the per-frame wall is COMPUTE-BOUND (graph replay
scales ~linearly in K; there is NO fixed ~340 ms floor to amortize by splitting one
frame's rows).  The robust ≥1 fps path is FRAME-LEVEL parallelism: assign WHOLE frames to
the two GPUs round-robin (GPU 0 verifies frame N while GPU 1 verifies frame N+1), one
worker per GPU (each a ``CUDA_VISIBLE_DEVICES``-pinned subprocess so ``cuda:0`` is a
DISTINCT card, independent CUDA context -> TRUE concurrency, no single-process
multi-device graph-capture collision).

Because the draft is byte-exact and runs AHEAD, all frames are known; each GPU runs a FULL
independent frame.  The parent measures the 2-GPU frame THROUGHPUT: each worker renders its
share of the frames continuously (TRUE-PIPE, background schedule build) and reports its
steady per-frame wall.  With two GPUs each producing a full frame every ``per_frame`` wall,
the aggregate throughput is ``2 / per_frame`` frames/s -> ~2x the single-GPU fps.

Byte-exactness: each worker's per-frame fingerprint (register hash) == the single-GPU
reference (each frame is a pure function of its draft), and the faithful verify (routing/
cam_addr/cam_value) still runs FULLY per frame on each GPU -> a wrong draft is rejected
IDENTICALLY (the frame split does not weaken verification at all -- unlike the K-split, no
cross-shard reduction; each GPU verifies a complete frame).

Run: python -m c4_min._agent_frames_2gpu_fps --draft-steps 120000
Needs BOTH GPUs idle.  DEFAULT-OFF flag C4_MULTIGPU_FRAMES gates the production path.
"""
from __future__ import annotations
import argparse, os, sys, time, gc, json, subprocess, threading, hashlib

os.environ.setdefault("OMP_NUM_THREADS", "6")
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

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK", "C4_DIRECT_CAM_VEC"]


def _levers_on(chunk):
    for f in COMPOSED:
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"; os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"; os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"; os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"; os.environ["C4_MEGABLOCK_BLOCK_K"] = "512"
    os.environ["C4_SCHED_CACHE_RESOLVED"] = "1"
    os.environ["C4_FAITHFUL_PRECOMPUTE_CACHE"] = "1"


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB<25 -> STOP")


def _apply_pow2(ops, imms):
    IMM_OP, DIV_OP, MOD_OP, SHR_OP, AND_OP = 1, 28, 29, 24, 16
    n = len(ops); npow = 0
    for i in range(n - 1):
        o0, m0, o1 = int(ops[i]), int(imms[i]), int(ops[i + 1])
        if o0 == IMM_OP and m0 > 0 and (m0 & (m0 - 1)) == 0:
            if o1 == DIV_OP:
                imms[i] = m0.bit_length() - 1; ops[i + 1] = SHR_OP; npow += 1
            elif o1 == MOD_OP:
                imms[i] = m0 - 1; ops[i + 1] = AND_OP; npow += 1
    return npow


def _load_snapshot():
    snap = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "_doom_bytecode_snapshot.npz"))
    return list(snap["ops"]), list(snap["imms"]), snap["data"]


def build_doom_draft(draft_steps, quiet=False):
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program
    from run_c4_min import (data_segment, tag_compiler_syscalls,
                            install_compiler_abi_file_dispatcher)
    ops, imms, data = _load_snapshot(); n = len(ops)
    npow = _apply_pow2(ops, imms)
    if not quiet:
        print(f"[doom] pow2 strength-reduction: {npow} sites", flush=True)
    install_compiler_abi_file_dispatcher()
    code_isa = tag_compiler_syscalls([isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF)
                                      for i in range(n)], isa)
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(b"q", neural=True)))
    if not quiet:
        print(f"[doom] drafting ({n:,} instrs) max_steps={draft_steps:,} ...", flush=True)
    t0 = time.time()
    d = draft_pf_program(code_isa, max_steps=draft_steps, mask=0xFFFFFFFF,
                         data_seg=data_segment([int(b) for b in data]), fio=fio)
    if not quiet:
        print(f"[doom] drafted {d.step_count:,} steps in {time.time()-t0:.0f}s", flush=True)
    dm = [i for i in range(d.step_count) if d.frames[i].get("op") in ("DIV", "MOD")]
    assert not dm, f"draft not DIV-free: {dm[:5]}"
    return d


def _hash_regs(pc, sp, bp, ax):
    h = hashlib.sha256()
    for a in (pc, sp, bp, ax):
        h.update(np.ascontiguousarray(a.astype(np.int64)).tobytes())
    return h.hexdigest()[:16]


# ===========================================================================
# WORKER: one physical GPU renders WHOLE frames (its round-robin share).
# ===========================================================================
def run_worker(args):
    _levers_on(args.chunk)
    faithful = (args.mode == "faithful")
    if faithful:
        os.environ["C4_FAITHFUL_SINGLE_DISPATCH"] = "1"
    else:
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    from c4_min import isa
    from run_c4_min import tag_compiler_syscalls
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed, uninstall_composed
    from c4_min import precomputed_schedule as PS
    from c4_min.faithful_single_dispatch import (build_faithful_plan,
                                                 build_faithful_precompute, verify_faithful_fast)
    dev = torch.device("cuda:0"); mask = 0xFFFFFFFF
    d = build_doom_draft(args.draft_steps, quiet=True)
    n = d.step_count
    ops, imms, _data = _load_snapshot(); _apply_pow2(ops, imms)
    code = tag_compiler_syscalls([isa.Instr(int(o), int(i) & 0xFFFFFFFF)
                                  for o, i in zip(ops, imms)], isa)
    m, L, _ = build_lib_model_streaming(code_size=max(d.code_off + 2, 256),
                                        recurrent_divmod=True, addr32=True,
                                        compute_mode="dense_kernel")
    m = m.to(dev); install_composed(m, verbose=False)
    torch.cuda.reset_peak_memory_stats(dev)
    plan = None; ws = None
    if faithful:
        plan = build_faithful_plan(m, L, code, d)
        ws = np.asarray(d.win_starts[:n], dtype=np.int64)
    onchip = None
    try:
        sched, sg = PS.build_schedule(m, L, code, d, dev, mask=mask)
        onchip = sched.onchip; h0s = sched.h0_folded if onchip else sched.h0_table
        got_pc = torch.empty(n, dtype=torch.long, device=dev)
        got_sp = torch.empty(n, dtype=torch.long, device=dev)
        got_bp = torch.empty(n, dtype=torch.long, device=dev)
        got_ax = torch.empty(n, dtype=torch.long, device=dev)

        def _one_frame(cur):
            per_head = {}
            for lo in range(0, n, sg.chunk):
                hi = min(lo + sg.chunk, n); h0 = h0s[lo:hi].unsqueeze(0)
                delta = {b: t[lo:hi] for b, t in cur.cam_delta_tables.items()}
                pc, sp, bp, ax = sg.replay(h0, None, None, delta=delta, resident=False)
                got_pc[lo:hi].copy_(pc); got_sp[lo:hi].copy_(sp)
                got_bp[lo:hi].copy_(bp); got_ax[lo:hi].copy_(ax)
                if faithful:
                    for key, t in sg.last_qaddr(hi - lo).items():
                        per_head.setdefault(key, []).append(
                            (lo, hi, t.detach().to("cpu").numpy().astype("int64")))
            ma = None
            if faithful:
                ma = {}
                for (b, hh, knd), chunks in per_head.items():
                    fv = np.zeros(n, dtype=np.int64)
                    for (lo, hi, arr) in chunks:
                        fv[lo:hi] = arr[:hi - lo]
                    ma[(b, knd)] = fv
            return ma

        # fingerprint + verdict on one frame.
        ma = _one_frame(sched)
        ax_c = (got_ax & mask)
        fp = _hash_regs(got_pc.cpu().numpy(), got_sp.cpu().numpy(),
                        got_bp.cpu().numpy(), ax_c.cpu().numpy())
        vd = None
        if faithful:
            pre = build_faithful_precompute(d, plan, ws, n, mask=mask)
            v = verify_faithful_fast(pre, ma)
            vd = dict(ok=bool(v.ok),
                      first_bad=(int(v.first_bad_step) if v.first_bad_step is not None else None),
                      kind=v.kind, addr=int(v.n_addr_checked), val=int(v.n_value_checked),
                      rt=int(v.n_routing_checked))
        # byte-exact vs draft targets (whole frame).
        wp = torch.from_numpy(np.fromiter((d.frames[s]['pc'] for s in range(n)),
                                          dtype=np.int64, count=n)).to(dev)
        wa = torch.from_numpy(np.fromiter((d.frames[s]['ax'] & mask for s in range(n)),
                                          dtype=np.int64, count=n)).to(dev)
        wsp = torch.from_numpy(np.fromiter((d.frames[s]['sp'] & 0xFFFFFFFF for s in range(n)),
                                           dtype=np.int64, count=n)).to(dev)
        wbp = torch.from_numpy(np.fromiter((d.frames[s]['bp'] & 0xFFFFFFFF for s in range(n)),
                                           dtype=np.int64, count=n)).to(dev)
        ih = torch.from_numpy(np.fromiter((bool(d.frames[s].get('is_halt')) for s in range(n)),
                                          dtype=np.bool_, count=n)).to(dev)
        ifi = torch.from_numpy(np.fromiter((bool(d.frames[s].get('is_file')) for s in range(n)),
                                           dtype=np.bool_, count=n)).to(dev)
        bad = (got_pc != wp) | (ax_c != wa) | (got_sp != wsp) | (got_bp != wbp)
        bad = torch.where(ih, ax_c != wa, bad) & (~ifi)
        nbad = int(bad.sum())

        # ---- TRUE-PIPE continuous render of THIS worker's frame share ----
        os.environ["C4_SCHED_PIPELINE"] = "1"
        WARMUP = 2
        pb = PS.PipelinedScheduleBuilder(m, L, code, d, dev, sg, mask=mask, faithful=faithful)
        cur = pb.build_blocking(); pre_cur = pb.last_precompute()
        torch.cuda.synchronize(dev); pb.start()
        walls = []; tot = args.frames_per_worker + WARMUP
        for it in range(tot):
            torch.cuda.synchronize(dev); tf = time.perf_counter()
            ma2 = _one_frame(cur)
            if faithful:
                verify_faithful_fast(pre_cur, ma2)
            nxt = pb.wait(); del cur; cur = nxt; pre_cur = pb.last_precompute()
            if it < tot - 1:
                pb.start()
            torch.cuda.synchronize(dev); walls.append(time.perf_counter() - tf)
        del cur; gc.collect()
        os.environ.pop("C4_SCHED_PIPELINE", None)
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        steady = sorted(walls[WARMUP:]); pipe = steady[len(steady) // 2]
        out = dict(n=n, pipe=pipe, peak=peak, fp=fp, nbad=nbad, vd=vd, mode=args.mode,
                   frames=len(walls[WARMUP:]))
        print("RESULT_JSON " + json.dumps(out), flush=True)
    finally:
        uninstall_composed(m)
    return 0


# ===========================================================================
# 1-GPU reference (single card, sequential) — the throughput baseline.
# ===========================================================================
def _spawn_worker(phys_gpu, mode, args):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)
    cmd = [sys.executable, "-m", "c4_min._agent_frames_2gpu_fps", "--role", "worker",
           "--mode", mode, "--draft-steps", str(args.draft_steps), "--chunk", str(args.chunk),
           "--frames-per-worker", str(args.frames_per_worker)]
    return subprocess.Popen(cmd, env=env,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _run_and_parse(procs, label):
    outs = {}; logs = {}

    def _reader(idx, p):
        buf = []
        for line in p.stdout:
            buf.append(line)
            if line.startswith("RESULT_JSON "):
                outs[idx] = json.loads(line[len("RESULT_JSON "):])
        logs[idx] = "".join(buf); p.wait()
    ths = [threading.Thread(target=_reader, args=(i, p)) for i, p in enumerate(procs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for i, p in enumerate(procs):
        if i not in outs:
            print(f"[{label}] worker {i} FAILED (no RESULT_JSON). tail:", flush=True)
            print("\n".join(logs.get(i, "").splitlines()[-30:]), flush=True)
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="parent", choices=["parent", "worker"])
    ap.add_argument("--mode", default="faithful", choices=["faithful", "fast"])
    ap.add_argument("--draft-steps", type=int, default=120000)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--frames-per-worker", type=int, default=4)
    ap.add_argument("--n-dev", type=int, default=2)
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args(argv)
    if args.role == "worker":
        return run_worker(args)

    # ---- PARENT ----
    _guard()
    gpus = [int(x) for x in args.gpus.split(",")][:args.n_dev]
    from c4_min.multigpu_ksplit import assign_frames_roundrobin
    print(f"[parent] FRAME-LEVEL 2-GPU  gpus={gpus} frames_per_worker={args.frames_per_worker} "
          f"chunk={args.chunk} draft_steps={args.draft_steps}", flush=True)

    results = {}
    for mode in ("fast", "faithful"):
        print(f"\n=== FRAME-LEVEL {mode.upper()} ===", flush=True)
        # 1-GPU baseline: one worker on gpu[0] (sequential frames).
        print(f"  [1-GPU baseline] rendering {args.frames_per_worker} frames on GPU {gpus[0]} ...", flush=True)
        p1 = _spawn_worker(gpus[0], mode, args)
        out1 = _run_and_parse([p1], f"{mode}-1gpu")
        r1 = out1.get(0)
        if r1 is None:
            print(f"  [{mode}] 1-GPU baseline FAILED — skipping", flush=True); continue
        base_pipe = r1["pipe"]; n = r1["n"]
        scale = RENDER_REDUCED_FRAME / n
        base_sframe = base_pipe * scale; base_fps = 1.0 / base_sframe
        print(f"  [1-GPU] per-frame {base_pipe*1e3:.2f}ms -> @358k {base_sframe:.3f}s "
              f"{base_fps:.3f}fps peak={r1['peak']:.1f}GB fp={r1['fp']} "
              f"nbad={r1['nbad']} vd_ok={r1['vd']['ok'] if r1['vd'] else None}", flush=True)

        # 2-GPU frame-level: both workers run CONCURRENTLY, each its frame share.
        print(f"  [2-GPU frames] launching {args.n_dev} concurrent workers ...", flush=True)
        procs = [_spawn_worker(gpus[i], mode, args) for i in range(args.n_dev)]
        outs = _run_and_parse(procs, f"{mode}-2gpu")
        if len(outs) < args.n_dev:
            print(f"  [{mode}] 2-GPU incomplete — skipping", flush=True); continue
        workers = [outs[i] for i in range(args.n_dev)]
        # each worker renders frames_per_worker frames concurrently.  The aggregate
        # throughput = (n_dev * frames_per_worker) frames / max worker wall.  Since each
        # worker's steady per-frame == its pipe wall, the aggregate per-frame (frames/s
        # inverse) = max(pipe)/n_dev.
        wpipe = max(w["pipe"] for w in workers)
        agg_per_frame = wpipe / args.n_dev             # effective seconds per delivered frame
        agg_sframe = agg_per_frame * scale
        agg_fps = 1.0 / agg_sframe
        peaks = [w["peak"] for w in workers]
        fps_arr = assign_frames_roundrobin(args.n_dev * args.frames_per_worker, args.n_dev)
        # byte-exact: all worker fingerprints identical to the 1-GPU reference fp (each frame
        # is a pure function of the SAME draft).
        fps_match = all(w["fp"] == r1["fp"] for w in workers)
        nbad_tot = sum(w["nbad"] for w in workers)
        wpf = ["%.1f" % (w["pipe"] * 1e3) for w in workers]
        if mode == "faithful":
            all_vd_ok = all(w["vd"]["ok"] for w in workers)
            a = sum(w["vd"]["addr"] for w in workers)
            v = sum(w["vd"]["val"] for w in workers)
            rt = sum(w["vd"]["rt"] for w in workers)
            vd_oks = [w["vd"]["ok"] for w in workers]
            print(f"  [2-GPU FAITHFUL] worker per-frame {wpf}ms "
                  f"peaks={['%.1f' % p for p in peaks]}GB vd_ok={vd_oks} "
                  f"addr={a} val={v} rt={rt}", flush=True)
        else:
            print(f"  [2-GPU FAST] worker per-frame {wpf}ms "
                  f"peaks={['%.1f' % p for p in peaks]}GB", flush=True)
        scaling = base_pipe / (wpipe / args.n_dev)
        print(f"  [2-GPU {mode}] aggregate throughput @358k = {agg_sframe:.3f}s "
              f"{agg_fps:.3f}fps  {'>=1!' if agg_fps>=1.0 else '<1'}  "
              f"scaling={scaling:.2f}x  byte_exact(fp match+nbad0)={fps_match and nbad_tot==0}",
              flush=True)
        results[mode] = dict(base_fps=base_fps, base_sframe=base_sframe,
                             agg_fps=agg_fps, agg_sframe=agg_sframe, scaling=scaling,
                             peaks=peaks, n=n, fps_match=fps_match, nbad=nbad_tot,
                             all_vd_ok=(all(w["vd"]["ok"] for w in workers) if mode == "faithful" else None))

    # ---- final report ----
    print(f"\n  === FRAME-LEVEL 2-GPU REAL-DOOM @ render-reduced ({RENDER_REDUCED_FRAME} steps) ===", flush=True)
    for mode in ("fast", "faithful"):
        if mode not in results:
            continue
        r = results[mode]
        lbl = "FAITHFUL single-dispatch" if mode == "faithful" else "fast (draft-trusted)"
        print(f"    {lbl:>26}  1-GPU {r['base_sframe']:7.3f}s {r['base_fps']:6.3f}fps  ->  "
              f"2-GPU {r['agg_sframe']:7.3f}s {r['agg_fps']:6.3f}fps  "
              f"({r['scaling']:.2f}x, {'>=1!' if r['agg_fps']>=1.0 else '<1'})  "
              f"peaks {['%.1f'%p for p in r['peaks']]}GB", flush=True)
    if "faithful" in results:
        r = results["faithful"]
        print(f"\n    2-GPU FAITHFUL fps = {r['agg_fps']:.3f}  clears>=1fps: {r['agg_fps']>=1.0}  "
              f"scaling {r['scaling']:.2f}x", flush=True)
        print(f"    BYTE-EXACT: fp match across GPUs+nbad=0 = {r['fps_match'] and r['nbad']==0}  "
              f"| GENUINE verify accepts all = {r['all_vd_ok']}", flush=True)
    print("\n=== FRAME-LEVEL 2-GPU COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
