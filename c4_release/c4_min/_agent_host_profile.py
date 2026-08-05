#!/usr/bin/env python3
"""_agent_host_profile.py — PROFILE THE HOST SIDE of the composed doom verify step.

TASK 1: rank the CPython dispatch cost across the composed ``verify_blocks`` step,
confirm ``direct_cam_batched._head_out_vec``'s ~3600 host calls/forward (the pin),
and get the top host consumers with their us.  Also measures the wall/device us/step
so a fix can be scored against the 847us wall / 289us device floor.

Mirrors ``_agent_composed_floor._levers_on`` EXACTLY (the harness that measured the
pin) so the profile is on the SAME code path.  Adds a ``C4_DIRECT_CAM_VEC`` toggle
(off = the pinned per-row loop path; on = the #871 vectorized gather) so we can
measure whether flipping VEC actually removes the ~3600 ``_head_out_vec`` calls.

Run:
    CUDA_VISIBLE_DEVICES=1 C4_PF_CFM=1 python -m c4_min._agent_host_profile \
        --device cuda:0 --K 8192 [--vec 0|1] [--cprofile]
"""
from __future__ import annotations

import argparse
import collections
import cProfile
import os
import pstats
import time
import traceback
import io

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks, set_gpu_verify
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested


DOOM_FRAME_INSTRS = 358_058          # render-REDUCED doom frame (per the task)


# --- host-sync tally (monkeypatch .item()) --------------------------------------
_tally = collections.Counter()
_call_tally = collections.Counter()      # a broader per-callsite counter (any func)
_orig_item = torch.Tensor.item
_tracing = [False]


def _traced_item(self):
    if _tracing[0]:
        st = traceback.extract_stack(limit=4)
        fr = st[-2]
        _tally[f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"] += 1
    return _orig_item(self)


torch.Tensor.item = _traced_item


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
    return a


ALL_LEVERS = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FROZEN_ROW_SKIP",
              "C4_CUT_SPAN_CHUNK", "C4_FUSED_MEGABLOCK", "C4_FUSED_DELTA_FFN",
              "C4_OVERLAY_BATCHED", "C4_STREAM_EMBED", "C4_GRAPH_BLOCK0",
              "C4_DIRECT_CAM_VEC", "C4_WHOLE_STEP_GRAPH", "C4_BLOCK0_FUSED_FFN",
              "C4_BLOCK0_DROP_DEAD_KV", "C4_QROW_CHUNK"]


def _levers_on(cut_chunk, vec):
    os.environ["C4_DEAD_BLOCK_FUSION"] = "1"
    os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
    os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
    os.environ["C4_FLASH_ATTN"] = "1"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "1"
    os.environ["C4_FROZEN_ROW_SKIP"] = "1"
    os.environ["C4_FUSED_MEGABLOCK"] = "1"
    os.environ["C4_FUSED_DELTA_FFN"] = "1"
    os.environ["C4_CUT_SPAN_CHUNK"] = str(cut_chunk)
    os.environ["C4_GRAPH_BLOCK0"] = "1"
    os.environ["C4_OVERLAY_BATCHED"] = "1"
    os.environ["C4_STREAM_EMBED"] = "1"
    os.environ["C4_BLOCK0_DROP_DEAD_KV"] = "1"
    os.environ["C4_WHOLE_STEP_GRAPH"] = "1"
    os.environ["C4_BLOCK0_FUSED_FFN"] = "1"
    if vec:
        os.environ["C4_DIRECT_CAM_VEC"] = "1"
    else:
        os.environ.pop("C4_DIRECT_CAM_VEC", None)


def _levers_off():
    for f in ALL_LEVERS:
        os.environ.pop(f, None)


def _run_verify(model, L, code, draft, K, device):
    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    return vr, stats


def profile_one(model, L, code, draft, K, device, vec, steps, do_cprofile):
    label = f"VEC={'ON' if vec else 'OFF'} K={K}"
    # WARMUP (build graphs, prime allocator) — levers on, gpu-verify on.
    _levers_on(256, vec); set_gpu_verify(True)
    _run_verify(model, L, code, draft, K, device)
    _levers_off()

    # cProfile pass (host-time ranking) — separate from the timed pass so the
    # profiler overhead doesn't pollute the wall.
    prof_txt = None
    if do_cprofile:
        _levers_on(256, vec); set_gpu_verify(True)
        torch.cuda.synchronize(device)
        pr = cProfile.Profile()
        pr.enable()
        _run_verify(model, L, code, draft, K, device)
        torch.cuda.synchronize(device)
        pr.disable()
        _levers_off()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(25)
        prof_txt = s.getvalue()

    # TIMED + .item() tally pass.
    _levers_on(256, vec); set_gpu_verify(True)
    torch.cuda.synchronize(device)
    _tally.clear(); _tracing[0] = True
    t0 = time.perf_counter()
    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    _tracing[0] = False
    set_gpu_verify(None); _levers_off()

    fwds = stats.get("forwards", vr.forwards)
    syncs = sum(_tally.values())
    us_step = wall * 1e6 / steps
    sps = steps / wall
    sec_frame = DOOM_FRAME_INSTRS / sps
    print(f"\n### {label}: {us_step:8.2f} us/step  {sps:10.0f} steps/s  "
          f"frame={sec_frame:6.2f}s  fwds={fwds}  matched={vr.all_matched}", flush=True)
    print(f"    .item() host syncs: {syncs} total ({syncs/max(fwds,1):.1f}/fwd)", flush=True)
    for site, cnt in _tally.most_common(12):
        print(f"        {cnt:8d}  {site}", flush=True)
    if prof_txt:
        print("    --- cProfile (tottime, top 25 host consumers) ---", flush=True)
        for ln in prof_txt.splitlines():
            print("    " + ln, flush=True)
    return dict(us_step=us_step, sps=sps, sec_frame=sec_frame, syncs=syncs,
                fwds=fwds, matched=vr.all_matched, final_ax=vr.decoded_final_ax)


def _timed(model, L, code, draft, K, device, steps):
    """One warmup + one timed verify at the CURRENT env flag state (no cProfile)."""
    _levers_on(256, os.environ.get("C4_DIRECT_CAM_VEC", "0") == "1"); set_gpu_verify(True)
    # honour whatever C4_DIRECT_CAM_VEC the caller pre-set (ablation controls it).
    _run_verify(model, L, code, draft, K, device)          # warmup
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    set_gpu_verify(None)
    return wall * 1e6 / steps, vr.decoded_final_ax, vr.all_matched


def ablate(model, L, code, draft, K, device, steps):
    """Isolate each host-fix's contribution: baseline (all my fixes OFF) -> +memo ->
    +idempotent-materialize -> +CAM-vec.  All measured on the SAME composed stack."""
    print(f"\n=== ABLATION @K={K} (each fix stacked) ===", flush=True)
    configs = [
        ("BASELINE (pin)",  dict(memo="0", idem="0", vec="0", ov="0")),
        ("+live_heads_memo", dict(memo="1", idem="0", vec="0", ov="0")),
        ("+materialize_idem", dict(memo="1", idem="1", vec="0", ov="0")),
        ("+cam_vec",        dict(memo="1", idem="1", vec="1", ov="0")),
        ("+overlay_precomp", dict(memo="1", idem="1", vec="1", ov="1")),
    ]
    ref_ax = None
    for name, cfg in configs:
        os.environ["C4_LIVE_HEADS_MEMO"] = cfg["memo"]
        os.environ["C4_MATERIALIZE_IDEMPOTENT"] = cfg["idem"]
        os.environ["C4_OVERLAY_PRECOMPUTE"] = cfg["ov"]
        if cfg["vec"] == "1":
            os.environ["C4_DIRECT_CAM_VEC"] = "1"
        else:
            os.environ.pop("C4_DIRECT_CAM_VEC", None)
        # clear any cached live-heads so a memo-OFF run truly recomputes.
        for blk in model.blocks:
            if hasattr(blk.attn, "_live_value_heads_cache"):
                delattr(blk.attn, "_live_value_heads_cache")
        us, ax, matched = _timed(model, L, code, draft, K, device, steps)
        _levers_off()
        if ref_ax is None:
            ref_ax = ax
        be = "BYTE-EXACT" if ax == ref_ax else f"DIVERGED({ax}!={ref_ax})"
        sps = steps / (us * steps / 1e6)
        print(f"  {name:20s} {us:8.2f} us/step  {sps:9.0f} steps/s  "
              f"frame={DOOM_FRAME_INSTRS/sps:7.2f}s  matched={matched}  {be}", flush=True)
    # restore defaults-on
    os.environ.pop("C4_LIVE_HEADS_MEMO", None)
    os.environ.pop("C4_MATERIALIZE_IDEMPOTENT", None)
    os.environ.pop("C4_OVERLAY_PRECOMPUTE", None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--vec", type=int, default=-1, help="0=off 1=on -1=both")
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--nested", type=int, nargs=2, default=[12, 28])
    args = ap.parse_args(argv)
    _guard()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    t0 = time.time()
    model, L, _ = build_lib_model_streaming(
        code_size=256, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB device={device}",
          flush=True)
    _guard()

    a, b = args.nested
    name, code = f"nested_{a}_{b}", build_nested(a, b)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    assert draft.halted, "draft did not halt"
    steps = draft.step_count
    print(f"[prog] {name}: {steps} DIV-free steps (deep nested loop)", flush=True)

    install_composed(model, verbose=False)
    results = {}
    try:
        if args.ablate:
            for K in ([args.K] if args.K else [8192]):
                _guard()
                ablate(model, L, code, draft, K, device, steps)
            uninstall_composed(model)
            return 0
        vecs = [0, 1] if args.vec < 0 else [args.vec]
        for vec in vecs:
            _guard()
            results[vec] = profile_one(model, L, code, draft, args.K, device,
                                       bool(vec), steps, args.cprofile)
    finally:
        uninstall_composed(model)

    if 0 in results and 1 in results:
        o, n = results[0], results[1]
        print(f"\n=== VEC OFF vs ON @K={args.K} ===", flush=True)
        print(f"  us/step : {o['us_step']:.2f} -> {n['us_step']:.2f}  "
              f"({o['us_step']/n['us_step']:.3f}x)", flush=True)
        print(f"  syncs   : {o['syncs']} -> {n['syncs']}", flush=True)
        print(f"  matched : {o['matched']} -> {n['matched']}  "
              f"final_ax {o['final_ax']}=={n['final_ax']} "
              f"{'BYTE-EXACT' if o['final_ax']==n['final_ax'] else 'DIVERGED'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
