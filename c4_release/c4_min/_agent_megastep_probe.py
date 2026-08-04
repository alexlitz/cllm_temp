#!/usr/bin/env python3
"""_agent_megastep_probe.py — PER-STEP OVERHEAD COLLAPSE probe.

Now that C4_DEAD_BLOCK_FUSION makes the 238 dead-attention blocks pure identity
(x passthrough, no attn linears), the composed step is a Python loop of ~238 tiny
FFN launches + a handful of live-block lookups.  This probe answers:

  1. Does the whole-step forward CAPTURE into a CUDA graph now that attention=0
     (no dynamic ALiBi/causal masks)?  Test install_graphed_fused_forward
     (dead-FFN-segment graphs) composed with dead-block-fusion.
  2. What is the ms/step + steps/sec for: (a) dead-fusion eager, (b) dead-fusion
     + graphed megakernel, vs the (c) banded no-fuse baseline?
  3. BYTE-EXACT: does the graphed path give the SAME all_matched + decoded AX?

LEAN STREAMING (C4_PF_CFM=1).  malloc program (no c4_doom dependency) for fast
iteration; a separate harness runs the real doom stream at K=512.

Run:
    cd c4_release
    CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 python -m c4_min._agent_megastep_probe --K 256
"""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def _mem_avail_gb():
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


LEVERS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_BANDED_LOCAL_ATTN",
          "C4_DIRECT_CAM_LIVE_LOCAL", "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION",
          "C4_BATCHED_BLOCK_SKIP", "C4_OVERLAY_BATCHED", "C4_BATCHED_DECODE",
          "C4_EXACT_EVICT", "C4_GRAPH_MEGAKERNEL"]


def _set(**kw):
    for f in LEVERS:
        os.environ.pop(f, None)
    for f, v in kw.items():
        os.environ[f] = v


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
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--nmalloc", type=int, default=48)
    a = ap.parse_args(argv)
    _mem_guard("startup")
    dev = a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    cuda = dev.startswith("cuda")

    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc

    code = build_malloc(a.nmalloc)[0]
    draft = draft_pf_program(code, max_steps=40000, mask=0xFFFFFFFF)
    assert draft.halted
    n_steps = draft.step_count
    print(f"[probe] malloc({a.nmalloc}) steps={n_steps} K={a.K} dev={dev}", flush=True)

    # full lever stack (dead-fusion + frozen-skip + batched decode/overlay + lookups)
    FULL = dict(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
                C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
                C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1",
                C4_DEAD_BLOCK_FUSION="1", C4_OVERLAY_BATCHED="1",
                C4_BATCHED_DECODE="1", C4_EXACT_EVICT="1")

    def _time(model, L):
        stats = {}
        # warmup (also triggers graph capture on first span shape)
        vr0 = verify_blocks(model, L, code, draft, block_steps=a.K, device=dev,
                            evict=True, mask=0xFFFFFFFF, fast=True, stats=stats)
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

    results = {}
    capture_ok = True
    capture_err = None
    # (a) dead-fusion EAGER
    _mem_guard("eager")
    _set(**FULL)
    m, L = _fresh(dev)
    try:
        wall, vr = _time(m, L)
        results["deadfuse_eager"] = (wall / n_steps * 1e3, n_steps / wall, vr.all_matched,
                                     vr.decoded_final_ax)
    except Exception as e:
        results["deadfuse_eager"] = ("ERR:" + str(e)[:120],)
    del m
    if cuda:
        torch.cuda.empty_cache()

    # (b) dead-fusion + GRAPHED megakernel (C4_GRAPH_MEGAKERNEL)
    _mem_guard("graphed")
    _set(**dict(FULL, C4_GRAPH_MEGAKERNEL="1"))
    m, L = _fresh(dev)
    try:
        wall, vr = _time(m, L)
        results["deadfuse_graphed"] = (wall / n_steps * 1e3, n_steps / wall, vr.all_matched,
                                       vr.decoded_final_ax, 0)
    except Exception as e:
        capture_ok = False
        capture_err = str(e)
        results["deadfuse_graphed"] = ("ERR:" + str(e)[:200],)
    del m
    if cuda:
        torch.cuda.empty_cache()

    print("\n=== RESULTS (malloc, K={}) ===".format(a.K), flush=True)
    for name, r in results.items():
        if isinstance(r[0], str) and r[0].startswith("ERR"):
            print(f"  {name:22s} {r[0]}", flush=True)
        else:
            ms, sps, matched, ax = r[0], r[1], r[2], r[3]
            frame = 6.89e6 / sps
            extra = f" ngraph={r[4]}" if len(r) > 4 else ""
            print(f"  {name:22s} {ms:8.3f} ms/step  {sps:7.0f} steps/s  "
                  f"matched={matched} ax={ax} frame={frame:9.1f}s{extra}", flush=True)

    print(f"\n[probe] CUDA-graph capture_ok={capture_ok}", flush=True)
    if capture_err:
        print(f"[probe] capture_err: {capture_err[:300]}", flush=True)
    # byte-exact cross-check
    if "deadfuse_eager" in results and "deadfuse_graphed" in results:
        e, g = results["deadfuse_eager"], results["deadfuse_graphed"]
        if not (isinstance(e[0], str) and e[0].startswith("ERR")) and \
           not (isinstance(g[0], str) and g[0].startswith("ERR")):
            be = (e[2] == g[2] and e[3] == g[3])
            print(f"[probe] BYTE-EXACT eager==graphed: matched {e[2]}=={g[2]}, "
                  f"ax {e[3]}=={g[3]} -> {be}", flush=True)
            if not (isinstance(e[0], str)) and not (isinstance(g[0], str)):
                print(f"[probe] SPEEDUP graphed: {e[0]/g[0]:.2f}x "
                      f"({e[0]:.3f} -> {g[0]:.3f} ms/step)", flush=True)
    _set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
