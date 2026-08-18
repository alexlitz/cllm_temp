#!/usr/bin/env python3
"""_agent_block0dk_prof.py — kernel-level attribution of the whole-step graph body for
the DK config (C4_ATTN_MEGABLOCK + C4_BLOCK0_DK), to name the NEW dominant cost after the
block-0 input transpose is folded away.  Runs the body EAGERLY under torch.profiler.
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


def _levers_on(chunk, block_k, block0_dk):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
              "C4_DIRECT_CAM_VEC"):
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(block_k)
    os.environ["C4_ATTN_MEGABLOCK"] = "1"
    os.environ["C4_BLOCK0_DK"] = "1" if block0_dk else "0"


def _classify(name):
    n = name.lower()
    if any(t in n for t in ("sgemm", "gemm", "cutlass", "ampere", "cublas", "gemv")):
        return "GEMM(W_o/attn)"
    if any(t in n for t in ("triton", "megaffn", "up_gate", "upgate", "down_delta",
                            "fused_hidden", "wave")):
        return "FFN-mega(triton)"
    if "index" in n or "scatter" in n:
        return "FFN-scatter/index"
    if any(t in n for t in ("silu", "sigmoid")):
        return "silu"
    if "copy" in n or "clone" in n:
        return "copy/clone"
    if any(t in n for t in ("transpose", "permute", "contiguous", "cat")):
        return "transpose/cat"
    if any(t in n for t in ("add", "mul", "sub", "elementwise", "vectorized")):
        return "elementwise"
    if any(t in n for t in ("reduce", "argmax", "floor", "where", "arange", "clamp",
                            "round")):
        return "decode"
    return "other:" + n[:20]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=96000)
    ap.add_argument("--block-k", type=int, default=512)
    ap.add_argument("--no-dk", action="store_true")
    args = ap.parse_args(argv)
    _guard()
    dev = torch.device(args.device)
    block0_dk = not args.no_dk
    _levers_on(args.chunk, args.block_k, block0_dk)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.bench_fast_path import build_nested
    from c4_min import precomputed_schedule as PS

    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(args.device)
    _guard()
    import pickle
    with open("/tmp/_wf_draft_120_255.pkl", "rb") as fh:
        draft = pickle.load(fh)
    code = build_nested(120, 255)[0]
    install_composed(model, verbose=False)
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)

    n = min(args.chunk, draft.step_count)
    delta = {b: t[:n] for b, t in sched.cam_delta_tables.items()}
    if block0_dk:
        sg._s_h0 = sched.h0_folded[:n].transpose(0, 1).contiguous()   # [D, n]
    else:
        sg._s_h0 = sched.h0_folded[:n].unsqueeze(0)
    for b in sg.live_blocks:
        sg._s_delta[b] = delta[b]
    sg._alloc_out_lanes(n)
    sg.chunk = n

    for _ in range(3):
        with torch.no_grad():
            sg._body()
    torch.cuda.synchronize(dev)

    from torch.profiler import profile, ProfilerActivity
    reps = 20
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            with torch.no_grad():
                sg._body()
        torch.cuda.synchronize(dev)

    ka = prof.key_averages()
    buckets = {}
    tot = 0.0
    for e in ka:
        c = getattr(e, "self_device_time_total", 0) / 1e3
        if c <= 0:
            continue
        tot += c
        cls = _classify(e.key)
        buckets[cls] = buckets.get(cls, 0.0) + c
    tag = "DK" if block0_dk else "MEGA"
    print(f"\n=== {tag} WHOLE-STEP BODY kernel attribution @chunk={n} bk{args.block_k} ===",
          flush=True)
    print(f"  total device ms over {reps} reps = {tot:.2f}  ({tot/reps:.3f} ms/body, "
          f"{tot/reps*1e3/n:.4f} us/step)", flush=True)
    for cls, ms in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:22s} {ms:9.3f} ms  {100*ms/tot:5.1f}%", flush=True)
    print(f"\n  --- top 14 individual kernels ---", flush=True)
    rows = sorted(ka, key=lambda e: getattr(e, "self_device_time_total", 0), reverse=True)
    for e in rows[:14]:
        c = getattr(e, "self_device_time_total", 0) / 1e3
        if c <= 0:
            continue
        print(f"    {e.key[:52]:52s} n={e.count:4d}  {c:8.3f} ms  {100*c/tot:5.1f}%",
              flush=True)
    del sched; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
