#!/usr/bin/env python3
"""_agent_attnmega_frame_fps.py — honest per-frame fps at a 96K-step frame, C4_ATTN_MEGABLOCK
OFF vs ON, with the composed continuous stack (cache-resolved build + double-buffer pipeline).

Reports build ms, dispatch ms, and TRUE-PIPE steady-state ms/frame -> fps, so the dispatch
win translates to fps honestly (pipelined frame ~= max(build, dispatch)).
"""
from __future__ import annotations
import argparse, os, gc, time, pickle

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                if int(ln.split()[1]) / 1e6 < 25.0:
                    raise SystemExit("[GUARD] <25GB -> STOP")


def _levers_on(chunk, block_k, attn_mega):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
              "C4_DIRECT_CAM_VEC"):
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CACHE_RESOLVED"] = "1"      # build-reduction (continuous render)
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(block_k)
    os.environ["C4_ATTN_MEGABLOCK"] = "1" if attn_mega else "0"


def _dispatch(sched, sg, dev, n):
    got = [torch.empty(n, dtype=torch.long, device=dev) for _ in range(4)]
    onchip_ = sched.onchip
    h0s = sched.h0_folded if onchip_ else sched.h0_table
    for lo in range(0, n, sg.chunk):
        hi = min(lo + sg.chunk, n)
        h0 = h0s[lo:hi].unsqueeze(0)
        if onchip_:
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc, sp, bp, ax = sg.replay(h0, None, None, delta=delta, resident=False)
        else:
            ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
            cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0) for b, t in sched.cam_tables.items()}
            pc, sp, bp, ax = sg.replay(h0, ing, cam, resident=False)
        for g, v in zip(got, (pc, sp, bp, ax)):
            g[lo:hi].copy_(v)
    return got


def _measure(model, L, code, draft, dev, chunk, block_k, attn_mega, n_frames=6):
    from c4_min import precomputed_schedule as PS
    _levers_on(chunk, block_k, attn_mega)
    # drop any cached resolution from a prior config
    if hasattr(draft, "_resolved_cache"):
        del draft._resolved_cache
    n = draft.step_count
    mask = 0xFFFFFFFF
    torch.cuda.synchronize(dev)
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    _dispatch(sched, sg, dev, n)     # capture
    torch.cuda.synchronize(dev)
    del sched; gc.collect(); torch.cuda.empty_cache()
    # steady-state build + dispatch (cache-resolved -> build is a copy)
    bs, ds = [], []
    s2 = None
    for _ in range(n_frames):
        del s2; gc.collect()
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        s2 = PS.build_schedule_tables_only(model, L, code, draft, dev, sg, mask=mask)
        torch.cuda.synchronize(dev); bs.append(time.perf_counter() - t0)
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        out = _dispatch(s2, sg, dev, n)
        torch.cuda.synchronize(dev); ds.append(time.perf_counter() - t0)
    b = sorted(bs)[len(bs)//2]; d = sorted(ds)[len(ds)//2]
    del s2; gc.collect(); torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    os.environ.pop("C4_SCHED_CACHE_RESOLVED", None)
    return b, d, out, peak, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=96000)
    ap.add_argument("--block-k", type=int, default=256)
    args = ap.parse_args(argv)
    _guard()
    dev = torch.device(args.device)
    chunk = args.steps
    _levers_on(chunk, args.block_k, False)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.bench_fast_path import build_nested
    from c4_min.pf_speculative import draft_pf_program

    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(args.device)
    _guard()

    # a ~96K-step DIV-free draft (nested outer=25 inner=255 ~= 96K)
    cache = f"/tmp/_attnmega_draft_{args.steps}.pkl"
    code = build_nested(25, 255)[0]
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            draft = pickle.load(fh)
    else:
        draft = draft_pf_program(code, max_steps=3_000_000, mask=0xFFFFFFFF)
        if draft.halted:
            with open(cache, "wb") as fh:
                pickle.dump(draft, fh)
    print(f"[draft] steps={draft.step_count} (target ~{args.steps})", flush=True)
    install_composed(model, verbose=False)

    print(f"\n=== OFF ===", flush=True)
    b0, d0, out0, pk0, n = _measure(model, L, code, draft, dev, chunk, args.block_k, False)
    print(f"  build {b0*1e3:7.2f} ms  dispatch {d0*1e3:7.2f} ms  peak {pk0:.2f} GB", flush=True)
    print(f"\n=== ON (C4_ATTN_MEGABLOCK) ===", flush=True)
    b1, d1, out1, pk1, _ = _measure(model, L, code, draft, dev, chunk, args.block_k, True)
    print(f"  build {b1*1e3:7.2f} ms  dispatch {d1*1e3:7.2f} ms  peak {pk1:.2f} GB", flush=True)

    ok = all(int((a != b).sum()) == 0 for a, b in zip(out0, out1))
    print(f"\n  BYTE-EXACT (PC/SP/BP/AX) OFF vs ON: {'YES L-inf=0' if ok else 'NO'}", flush=True)

    def _fps(b, d):
        pipe = max(b, d)          # true-pipe ~= max(build, dispatch)
        ser = b + d
        return 1.0/pipe, 1.0/ser
    fp0, fs0 = _fps(b0, d0); fp1, fs1 = _fps(b1, d1)
    print(f"\n=== FRAME FPS @{n} steps, bk{args.block_k} ===", flush=True)
    print(f"  OFF: dispatch {d0*1e3:.1f}ms build {b0*1e3:.1f}ms -> "
          f"TRUE-PIPE {1000*max(b0,d0):.1f}ms {fp0:.2f} fps 1-GPU / {2*fp0:.2f} fps 2-GPU  "
          f"(serial {fs0:.2f} fps)", flush=True)
    print(f"  ON : dispatch {d1*1e3:.1f}ms build {b1*1e3:.1f}ms -> "
          f"TRUE-PIPE {1000*max(b1,d1):.1f}ms {fp1:.2f} fps 1-GPU / {2*fp1:.2f} fps 2-GPU  "
          f"(serial {fs1:.2f} fps)", flush=True)
    print(f"  dispatch speedup {d0/d1:.2f}x ; TRUE-PIPE fps {fp1/fp0:.2f}x", flush=True)


if __name__ == "__main__":
    main()
