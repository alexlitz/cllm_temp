#!/usr/bin/env python3
"""_agent_block0dk_verify.py — byte-exact + timing for C4_BLOCK0_DK (Task 1).

Compares three configs on the SAME doom chunk:
  * BASE  : C4_ATTN_MEGABLOCK OFF (per-live dense-FFN clone path)
  * MEGA  : C4_ATTN_MEGABLOCK ON  (fused [D,K] region; block-0 output transposed into it)
  * DK    : C4_ATTN_MEGABLOCK + C4_BLOCK0_DK ON (block-0 folded into [D,K]; NO transpose)

Asserts DK's decoded (PC,SP,BP,AX) lanes are L-inf=0 identical to MEGA (and BASE), and
times all three whole-step graph replays (us/step) at the doom chunk.  For DK the h0 is
passed in [D,C] layout (sched.h0_folded[:n].T, transposed ONCE outside the timed loop).
"""
from __future__ import annotations
import argparse, os, gc

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


def _levers_on(chunk, block_k, attn_mega, block0_dk):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
              "C4_DIRECT_CAM_VEC"):
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(block_k)
    os.environ["C4_ATTN_MEGABLOCK"] = "1" if attn_mega else "0"
    os.environ["C4_BLOCK0_DK"] = "1" if block0_dk else "0"


def _run_and_time(model, L, code, draft, dev, chunk, block_k, attn_mega, block0_dk, reps=40):
    from c4_min import precomputed_schedule as PS
    _levers_on(chunk, block_k, attn_mega, block0_dk)
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
    n = min(chunk, draft.step_count)
    onchip_ = sched.onchip
    if block0_dk:
        # [D,C] h0 (transposed ONCE, outside the timed loop) -> plain slice-copy fill.
        h0 = sched.h0_folded[:n].transpose(0, 1).contiguous()   # [D, n]
    else:
        h0 = (sched.h0_folded if onchip_ else sched.h0_table)[:n].unsqueeze(0)
    delta = {b: t[:n] for b, t in sched.cam_delta_tables.items()} if onchip_ else None
    ing = None if onchip_ else sched.ing_table[:n].permute(1, 0, 2).unsqueeze(0)
    cam = None if onchip_ else {b: t[:n].permute(1, 0, 2).unsqueeze(0)
                                for b, t in sched.cam_tables.items()}
    pc, sp, bp, ax = sg.replay(h0, ing, cam, delta=delta, resident=False)
    torch.cuda.synchronize(dev)
    out = (pc.clone(), sp.clone(), bp.clone(), ax.clone())
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(dev); e0.record()
    for _ in range(reps):
        sg.replay(h0, ing, cam, delta=delta, resident=False)
    e1.record(); torch.cuda.synchronize(dev)
    ms = e0.elapsed_time(e1) / reps
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    del sched, sg; gc.collect(); torch.cuda.empty_cache()
    return out, ms, ms * 1e3 / n, peak, n


def _cmp(a_lanes, b_lanes, names=("PC", "SP", "BP", "AX")):
    ok = True
    for nm, a, b in zip(names, a_lanes, b_lanes):
        nbad = int((a != b).sum())
        if nbad:
            ok = False
        print(f"    {nm}: mismatches={nbad}  Linf={int((a-b).abs().max()) if a.numel() else 0}",
              flush=True)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=96000)
    ap.add_argument("--block-k", type=int, default=256)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args(argv)
    _guard()
    dev = torch.device(args.device)
    _levers_on(args.chunk, args.block_k, False, False)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.bench_fast_path import build_nested

    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(args.device)
    _guard()
    import pickle
    with open("/tmp/_wf_draft_120_255.pkl", "rb") as fh:
        draft = pickle.load(fh)
    code = build_nested(120, 255)[0]
    install_composed(model, verbose=False)

    print(f"\n=== BASE (ATTN_MEGABLOCK OFF, per-live dense-FFN clone) ===", flush=True)
    base, ms_b, us_b, pk_b, n = _run_and_time(model, L, code, draft, dev,
                                              args.chunk, args.block_k, False, False, args.reps)
    print(f"  {ms_b:.3f} ms/chunk  {us_b:.4f} us/step  peak {pk_b:.2f} GB", flush=True)

    print(f"\n=== MEGA (ATTN_MEGABLOCK ON, block-0 output transpose in body) ===", flush=True)
    mega, ms_m, us_m, pk_m, _ = _run_and_time(model, L, code, draft, dev,
                                              args.chunk, args.block_k, True, False, args.reps)
    print(f"  {ms_m:.3f} ms/chunk  {us_m:.4f} us/step  peak {pk_m:.2f} GB", flush=True)

    print(f"\n=== DK (ATTN_MEGABLOCK + BLOCK0_DK ON, block-0 folded into [D,K]) ===", flush=True)
    dk, ms_d, us_d, pk_d, _ = _run_and_time(model, L, code, draft, dev,
                                            args.chunk, args.block_k, True, True, args.reps)
    print(f"  {ms_d:.3f} ms/chunk  {us_d:.4f} us/step  peak {pk_d:.2f} GB", flush=True)

    print(f"\n=== BYTE-EXACT (DK vs MEGA) ===", flush=True)
    ok_dm = _cmp(mega, dk)
    print(f"\n=== BYTE-EXACT (DK vs BASE) ===", flush=True)
    ok_db = _cmp(base, dk)
    print(f"\n  BYTE-EXACT: DK==MEGA {'YES' if ok_dm else 'NO'} | "
          f"DK==BASE {'YES' if ok_db else 'NO'}", flush=True)

    print(f"\n=== SPEEDUP @chunk={n}, bk{args.block_k} ===", flush=True)
    print(f"  BASE {us_b:.4f} -> MEGA {us_m:.4f} us/step  ({us_b/us_m:.3f}x)", flush=True)
    print(f"  MEGA {us_m:.4f} -> DK   {us_d:.4f} us/step  ({us_m/us_d:.3f}x)", flush=True)
    print(f"  BASE {us_b:.4f} -> DK   {us_d:.4f} us/step  ({us_b/us_d:.3f}x)", flush=True)
    print(f"  VRAM peak : BASE {pk_b:.2f} MEGA {pk_m:.2f} DK {pk_d:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
