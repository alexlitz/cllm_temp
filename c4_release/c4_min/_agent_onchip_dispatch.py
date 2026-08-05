#!/usr/bin/env python3
"""Measure DISPATCH-ONLY wall+device (schedule + graph built ONCE, then time the replay
loop) for baseline vs onchip vs onchip+resident, + confirm the W_o GEMM is gone from the
device kernel split.  This isolates the single-dispatch pass the task targets from the
one-time schedule build."""
from __future__ import annotations
import argparse, os, time
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

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]
def _on():
    for f in COMPOSED: os.environ[f] = "1"


def _dispatch_only(sched, sg, n, chunk, onchip, resident, dev):
    """The dispatch loop from run_verify, over an already-built schedule + graph."""
    got_pc = torch.empty(n, dtype=torch.long, device=dev)
    h0_src = sched.h0_folded if onchip else sched.h0_table
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        h0 = h0_src[lo:hi].unsqueeze(0)
        if onchip:
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, None, None, delta=delta, resident=resident)
        else:
            ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
            cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0) for b, t in sched.cam_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, resident=resident)
        got_pc[lo:hi].copy_(pc_c)
    return got_pc


def _build(model, L, code, draft, dev, chunk, onchip, resident):
    _on()
    if onchip: os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    else: os.environ.pop("C4_ONCHIP_RESIDUAL", None)
    if resident: os.environ["C4_RESIDENT_BATCH"] = "1"
    else: os.environ.pop("C4_RESIDENT_BATCH", None)
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
    n = sched.n_steps
    # capture the graph once (first dispatch)
    _dispatch_only(sched, sg, n, sg.chunk, onchip, resident, dev)
    os.environ.pop("C4_ONCHIP_RESIDUAL", None)
    os.environ.pop("C4_RESIDENT_BATCH", None)
    return sched, sg, n


def _tile_schedule(sched, K, dev):
    """Tile the resident schedule tables up to K rows (repeating rows) so the dispatch pass
    can be throughput-measured at a big K.  Device work per step is identical (same gathers,
    same block chain), so this is a faithful dispatch-throughput measurement at scale."""
    import copy
    n = sched.n_steps
    reps = (K + n - 1) // n
    def tile(t):
        return t.repeat((reps,) + (1,) * (t.dim() - 1))[:K].contiguous()
    s2 = copy.copy(sched)
    s2.n_steps = K
    s2.h0_table = tile(sched.h0_table)
    if sched.h0_folded is not None:
        s2.h0_folded = tile(sched.h0_folded)
    s2.ing_table = tile(sched.ing_table)
    s2.cam_tables = {b: tile(t) for b, t in sched.cam_tables.items()}
    if sched.cam_delta_tables is not None:
        s2.cam_delta_tables = {b: tile(t) for b, t in sched.cam_delta_tables.items()}
    return s2


def measure_scale(model, L, code, draft, dev):
    """Dispatch-pass wall+device at K in {8192,65536,262144} for onchip+resident=false
    (resident needs chunk>=K; use fixed chunk + the tiling)."""
    print("\n=== DISPATCH at K in {8192,65536,262144} (baseline vs onchip), chunk fixed ===",
          flush=True)
    from torch.profiler import profile, ProfilerActivity
    chunk = 8192
    for K in [8192, 65536, 262144]:
        for label, oc in [("baseline", False), ("onchip", True)]:
            _on()
            if oc: os.environ["C4_ONCHIP_RESIDUAL"] = "1"
            else: os.environ.pop("C4_ONCHIP_RESIDUAL", None)
            os.environ.pop("C4_RESIDENT_BATCH", None)
            os.environ["C4_SCHED_CHUNK"] = str(chunk)
            sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
            s2 = _tile_schedule(sched, K, dev)
            n = s2.n_steps
            _dispatch_only(s2, sg, n, sg.chunk, oc, False, dev)  # capture
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _dispatch_only(s2, sg, n, sg.chunk, oc, False, dev)
            torch.cuda.synchronize()
            wall = (time.perf_counter()-t0)/3*1e6/n
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _dispatch_only(s2, sg, n, sg.chunk, oc, False, dev); torch.cuda.synchronize()
            tot = sum(e.self_device_time_total for e in prof.key_averages()) or 1.0
            print(f"  K={K:>7} {label:>10}: dispatch wall {wall:6.2f} us/step  "
                  f"device {tot/n:6.2f} us/step  {n/(wall*n/1e6):10.0f} steps/s", flush=True)
            os.environ.pop("C4_ONCHIP_RESIDUAL", None)
            del sched, sg, s2; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--scale", action="store_true")
    args = ap.parse_args(); dev = args.device
    _on()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    code = build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    install_composed(model, verbose=False)
    steps = draft.step_count
    chunk = args.chunk
    print(f"\n=== DISPATCH-ONLY (schedule+graph built ONCE), chunk={chunk}, steps={steps} ===",
          flush=True)
    from torch.profiler import profile, ProfilerActivity
    for label, oc, res in [("baseline(dense W_o)", False, False),
                           ("onchip", True, False),
                           ("onchip+resident", True, True)]:
        sched, sg, n = _build(model, L, code, draft, dev, chunk, oc, res)
        # wall of dispatch-only
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            _dispatch_only(sched, sg, n, sg.chunk, oc, res, dev)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / iters * 1e6 / steps
        # device split
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _dispatch_only(sched, sg, n, sg.chunk, oc, res, dev)
            torch.cuda.synchronize()
        evs = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
        tot = sum(e.self_device_time_total for e in evs) or 1.0
        dev_us = tot / steps
        sgemm = sum(e.self_device_time_total for e in evs if "sgemm" in e.key.lower()) / steps
        print(f"\n  {label}: dispatch wall {wall:.2f} us/step  device {dev_us:.2f} us/step  "
              f"sgemm {sgemm:.2f} us/step", flush=True)
        for e in evs[:6]:
            print(f"      {e.self_device_time_total/tot*100:5.1f}%  {e.self_device_time_total/steps:7.3f}us/step  {e.key[:46]}", flush=True)
        del sched, sg
        torch.cuda.empty_cache()
    if args.scale:
        measure_scale(model, L, code, draft, dev)
    uninstall_composed(model)


if __name__ == "__main__":
    main()
