#!/usr/bin/env python3
"""_agent_ab_timing.py — clean A/B ms/step timing of the host-sync-elimination
levers on top of the FULL dead-fusion + frozen-skip + lookup stack.

Configs (each a delta on the FULL stack):
  base          : the FULL stack as-of #856/#865 (per-row CAM loop)  [C4_DIRECT_CAM_VEC=0]
  +cam_vec      : + vectorized CAM-output gather                     [C4_DIRECT_CAM_VEC=1]
  +cam_vec+graph: + megakernel CUDA-graph of dead FFN segments       [+C4_GRAPH_MEGAKERNEL]

Reports ms/step, steps/sec, matched, final_ax, frame@6.89M for each.  Clean GPU
(the caller grabs an uncontended window).  malloc program; a separate harness runs
the real doom stream.
"""
from __future__ import annotations
import argparse, os, time
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

LEVERS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_BANDED_LOCAL_ATTN",
          "C4_DIRECT_CAM_LIVE_LOCAL", "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION",
          "C4_BATCHED_BLOCK_SKIP", "C4_OVERLAY_BATCHED", "C4_BATCHED_DECODE",
          "C4_EXACT_EVICT", "C4_GRAPH_MEGAKERNEL", "C4_DIRECT_CAM_VEC"]


def _set(**kw):
    for f in LEVERS:
        os.environ.pop(f, None)
    for f, v in kw.items():
        os.environ[f] = v


def _mem_avail_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return 1e9


def _fresh(dev, window=64):
    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                             addr32=True, compute_mode="dense_kernel")
    if dev != "cpu":
        model = model.to(dev)
    install_local_attention(model, window=window, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    return model, L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nmalloc", type=int, default=96)
    a = ap.parse_args(argv)
    if _mem_avail_gb() < 25.0:
        raise SystemExit(f"[MEM-GUARD] {_mem_avail_gb():.1f}GB<25GB STOP")
    dev = a.device
    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc
    code = build_malloc(a.nmalloc)[0]
    draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
    assert draft.halted
    n_steps = draft.step_count
    print(f"[ab] malloc({a.nmalloc}) steps={n_steps} K={a.K} dev={dev}", flush=True)

    FULL = dict(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
                C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
                C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1",
                C4_DEAD_BLOCK_FUSION="1", C4_OVERLAY_BATCHED="1",
                C4_BATCHED_DECODE="1", C4_EXACT_EVICT="1")

    configs = [
        ("base(loop)",       dict(FULL, C4_DIRECT_CAM_VEC="0")),
        ("+cam_vec",         dict(FULL, C4_DIRECT_CAM_VEC="1")),
        ("+cam_vec+graph",   dict(FULL, C4_DIRECT_CAM_VEC="1", C4_GRAPH_MEGAKERNEL="1")),
    ]

    print(f"\n{'config':18s} {'ms/step':>9s} {'steps/s':>9s} {'matched':>8s} "
          f"{'final_ax':>9s} {'frame@6.89M':>12s}", flush=True)
    res = {}
    for name, flags in configs:
        if _mem_avail_gb() < 25.0:
            raise SystemExit(f"[MEM-GUARD] {_mem_avail_gb():.1f}GB<25GB STOP@{name}")
        _set(**flags)
        m, L = _fresh(dev)
        # warmup
        verify_blocks(m, L, code, draft, block_steps=a.K, device=dev, evict=True,
                      mask=0xFFFFFFFF, fast=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            vr = verify_blocks(m, L, code, draft, block_steps=a.K, device=dev,
                               evict=True, mask=0xFFFFFFFF, fast=True)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / a.reps
        ms = wall / n_steps * 1e3
        sps = n_steps / wall
        res[name] = (ms, sps, vr.all_matched, vr.decoded_final_ax)
        print(f"{name:18s} {ms:9.3f} {sps:9.0f} {str(vr.all_matched):>8s} "
              f"{str(vr.decoded_final_ax):>9s} {6.89e6/sps:10.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()

    if "base(loop)" in res and "+cam_vec" in res:
        b, c = res["base(loop)"][0], res["+cam_vec"][0]
        print(f"\n[ab] cam_vec speedup: {b/c:.2f}x ({b:.3f} -> {c:.3f} ms/step)", flush=True)
        be = (res["base(loop)"][2:] == res["+cam_vec"][2:])
        print(f"[ab] byte-exact base==cam_vec: {be}", flush=True)
    _set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
