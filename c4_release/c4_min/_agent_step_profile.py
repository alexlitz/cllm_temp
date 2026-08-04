#!/usr/bin/env python3
"""_agent_step_profile.py — PROFILE the composed step to find the residual wall.

Runs the FULL lever stack (dead-fusion + frozen-skip + direct-CAM/local + batched
decode/overlay + exact-evict) and uses torch.profiler to count:
  * _local_scalar_dense (aten::_local_scalar_dense) host-syncs / step
  * DtoH / HtoD memcpys / step
  * the CUDA-time breakdown by top ops
so we can see whether the step is HOST-SYNC-bound, LAUNCH-bound, or COMPUTE-bound,
and what remains after C4_BATCHED_DECODE + C4_EXACT_EVICT + C4_OVERLAY_BATCHED.

Run:
    cd c4_release
    CUDA_VISIBLE_DEVICES=1 C4_PF_CFM=1 python -m c4_min._agent_step_profile --K 256
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from torch.profiler import profile, ProfilerActivity


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


LEVERS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_BANDED_LOCAL_ATTN",
          "C4_DIRECT_CAM_LIVE_LOCAL", "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION",
          "C4_BATCHED_BLOCK_SKIP", "C4_OVERLAY_BATCHED", "C4_BATCHED_DECODE",
          "C4_EXACT_EVICT", "C4_GRAPH_MEGAKERNEL"]


def _set(**kw):
    for f in LEVERS:
        os.environ.pop(f, None)
    for f, v in kw.items():
        os.environ[f] = v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--nmalloc", type=int, default=48)
    a = ap.parse_args(argv)
    dev = a.device
    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc
    code = build_malloc(a.nmalloc)[0]
    draft = draft_pf_program(code, max_steps=40000, mask=0xFFFFFFFF)
    n_steps = draft.step_count
    print(f"[prof] malloc({a.nmalloc}) steps={n_steps} K={a.K} graph={a.graph}", flush=True)

    FULL = dict(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
                C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
                C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1",
                C4_DEAD_BLOCK_FUSION="1", C4_OVERLAY_BATCHED="1",
                C4_BATCHED_DECODE="1", C4_EXACT_EVICT="1")
    if a.graph:
        FULL["C4_GRAPH_MEGAKERNEL"] = "1"
    _set(**FULL)
    m, L = _fresh(dev)

    # warmup
    verify_blocks(m, L, code, draft, block_steps=a.K, device=dev, evict=True,
                  mask=0xFFFFFFFF, fast=True)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        vr = verify_blocks(m, L, code, draft, block_steps=a.K, device=dev, evict=True,
                           mask=0xFFFFFFFF, fast=True)
        torch.cuda.synchronize()

    ka = prof.key_averages()
    # count the host-sync ops
    def _count(name):
        for e in ka:
            if e.key == name:
                return e.count
        return 0
    lsd = _count("aten::_local_scalar_dense")
    item = _count("aten::item")
    tolist = _count("aten::tolist")
    print(f"\n[prof] matched={vr.all_matched} ax={vr.decoded_final_ax} "
          f"forwards={vr.forwards}", flush=True)
    print(f"[prof] HOST SYNCS over the whole verify ({n_steps} steps):", flush=True)
    print(f"    _local_scalar_dense: {lsd}  ({lsd/max(n_steps,1):.1f}/step)", flush=True)
    print(f"    aten::item         : {item}  ({item/max(n_steps,1):.1f}/step)", flush=True)
    print(f"    aten::tolist       : {tolist}", flush=True)

    # top CUDA-time ops
    print("\n[prof] TOP ops by self CUDA time:", flush=True)
    try:
        rows = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
    except Exception:
        rows = sorted(ka, key=lambda e: getattr(e, "self_cuda_time_total", 0), reverse=True)
    for e in rows[:18]:
        cuda_us = getattr(e, "self_device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(e, "self_cuda_time_total", 0)
        cpu_us = e.self_cpu_time_total
        print(f"    {e.key[:42]:42s} cnt={e.count:6d} cuda={cuda_us/1e3:9.2f}ms "
              f"cpu={cpu_us/1e3:9.2f}ms", flush=True)

    # top CPU-time ops (launch/dispatch overhead)
    print("\n[prof] TOP ops by self CPU time:", flush=True)
    rows2 = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)
    for e in rows2[:15]:
        print(f"    {e.key[:42]:42s} cnt={e.count:6d} cpu={e.self_cpu_time_total/1e3:9.2f}ms",
              flush=True)
    _set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
