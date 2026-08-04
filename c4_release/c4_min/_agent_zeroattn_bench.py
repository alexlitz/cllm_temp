#!/usr/bin/env python3
"""_agent_zeroattn_bench.py — MEASURE ms/step of the composed verify forward with
vs without C4_DEAD_BLOCK_FUSION (the zero-attention-compute lever, #856).

Times the whole verify_blocks run (draft is free) and reports ms/step and
steps/sec for each config, and the implied frame time toward 1s/frame (6.89M/s).

LEAN STREAMING (C4_PF_CFM=1).  Run:
    cd c4_release
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_zeroattn_bench --device cuda:0
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import time
import torch


def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return 1e9


def _mem_guard(where=""):
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[MEM-GUARD] {a:.1f}GB < 25GB ({where}) STOP")


LEVER_FLAGS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_BANDED_LOCAL_ATTN",
               "C4_DIRECT_CAM_LIVE_LOCAL", "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION",
               "C4_BATCHED_BLOCK_SKIP"]


def _set_flags(**kw):
    for f in LEVER_FLAGS:
        os.environ.pop(f, None)
    for f, v in kw.items():
        os.environ[f] = v


def _fresh_model(dev):
    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    if dev != "cpu":
        model = model.to(dev)
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    return model, L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    _mem_guard("startup")
    dev = a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    cuda = dev.startswith("cuda")

    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc

    code = build_malloc(48)[0]
    draft = draft_pf_program(code, max_steps=40000, mask=0xFFFFFFFF)
    assert draft.halted
    n_steps = draft.step_count
    print(f"[bench] malloc draft steps={n_steps} K={a.K} dev={dev}", flush=True)

    def _time_verify(model, L):
        # warmup
        verify_blocks(model, L, code, draft, block_steps=a.K, device=dev,
                      evict=True, mask=0xFFFFFFFF, fast=True)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            vr = verify_blocks(model, L, code, draft, block_steps=a.K, device=dev,
                               evict=True, mask=0xFFFFFFFF, fast=True)
        if cuda:
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / a.reps
        return wall, vr

    configs = [
        ("lookup-no-fuse",
         dict(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
              C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
              C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1")),
        ("lookup+DEAD_FUSION",
         dict(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
              C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
              C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1",
              C4_DEAD_BLOCK_FUSION="1")),
    ]

    print(f"\n{'config':22s} {'ms/step':>10s} {'steps/sec':>12s} {'matched':>8s} "
          f"{'frame@6.89M':>12s}", flush=True)
    results = {}
    for name, flags in configs:
        _mem_guard(f"cfg {name}")
        _set_flags(**flags)
        model, L = _fresh_model(dev)
        wall, vr = _time_verify(model, L)
        del model
        if cuda:
            torch.cuda.empty_cache()
        ms_step = wall / n_steps * 1e3
        sps = n_steps / wall
        frame_s = 6.89e6 / sps          # sec to run 6.89M steps (1 doom frame)
        results[name] = (ms_step, sps)
        print(f"{name:22s} {ms_step:10.4f} {sps:12.0f} {str(vr.all_matched):>8s} "
              f"{frame_s:10.2f}s", flush=True)

    _set_flags()
    if len(results) == 2:
        base = results["lookup-no-fuse"][0]
        fuse = results["lookup+DEAD_FUSION"][0]
        print(f"\n  speedup from DEAD_BLOCK_FUSION: {base/fuse:.2f}x "
              f"({base:.4f} -> {fuse:.4f} ms/step)", flush=True)
        print(f"  steps/sec: {results['lookup-no-fuse'][1]:.0f} -> "
              f"{results['lookup+DEAD_FUSION'][1]:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
