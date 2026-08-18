#!/usr/bin/env python3
"""_agent_block0_graph_report.py — LEVER-1 (block-0 S-chunk CUDA graph) report.

Measures the composed doom step with C4_GRAPH_BLOCK0 ON vs OFF (all other composed
levers ON in both), on the real verify_blocks path at K in {8192, 65536, 262144}, and
byte-exact-checks the deep nested loop (the framing-drift-sensitive case) with the graph
ON.  Guards MemAvailable ONCE at start (25GB floor) so a transient dip mid-run from a
concurrent agent doesn't abort a valid measurement.

Run:
    CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 OMP_NUM_THREADS=4 \
        python -m c4_min._agent_block0_graph_report --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks, set_gpu_verify
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested

DOOM_FRAME_INSTRS = 6_890_000
FLOP_FLOOR_US = 0.0071


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _levers_on(cut_chunk, block0_graph):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FROZEN_ROW_SKIP",
              "C4_FUSED_MEGABLOCK", "C4_FUSED_DELTA_FFN", "C4_OVERLAY_BATCHED",
              "C4_STREAM_EMBED"):
        os.environ[f] = "1"
    os.environ["C4_CUT_SPAN_CHUNK"] = str(cut_chunk)
    os.environ["C4_BLOCK0_DROP_DEAD_KV"] = "1"
    os.environ["C4_GRAPH_BLOCK0"] = "1" if block0_graph else "0"


def _levers_off():
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FROZEN_ROW_SKIP",
              "C4_CUT_SPAN_CHUNK", "C4_FUSED_MEGABLOCK", "C4_FUSED_DELTA_FFN",
              "C4_OVERLAY_BATCHED", "C4_STREAM_EMBED", "C4_GRAPH_BLOCK0",
              "C4_BLOCK0_DROP_DEAD_KV", "C4_QROW_CHUNK"):
        os.environ.pop(f, None)


def _time_verify(model, L, code, draft, K, device, block0_graph, steps):
    _levers_on(256, block0_graph)
    set_gpu_verify(True)
    # warmup (build graphs + prime allocator).
    verify_blocks(model, L, code, draft, block_steps=K, device=device, evict=True,
                  mask=0xFFFFFFFF, fast=True, evict_interval_steps=8, exact_evict=True)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device, evict=True,
                       mask=0xFFFFFFFF, fast=True, evict_interval_steps=8,
                       exact_evict=True)
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    set_gpu_verify(None)
    _levers_off()
    ms_step = wall * 1e3 / steps
    sps = steps / wall
    return ms_step, sps, DOOM_FRAME_INSTRS / sps, vr.all_matched


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")
    device = args.device

    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.dim} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)

    name, code = "nested_12_28", build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    assert draft.halted
    steps = draft.step_count
    print(f"  program '{name}': {steps} DIV-free steps (deep nested loop)\n", flush=True)

    install_composed(model, verbose=False)
    try:
        # BYTE-EXACT: deep nested loop, graph ON, at a few K, both decode paths.
        print("=== BYTE-EXACT (deep nested loop, block-0 graph ON) ===", flush=True)
        allok = True
        for K in (512, 2048, 8192):
            _levers_on(256, True); set_gpu_verify(False)
            vs = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                               evict=True, mask=0xFFFFFFFF, fast=True,
                               evict_interval_steps=8, exact_evict=True)
            _levers_off()
            _levers_on(256, True); set_gpu_verify(True)
            vg = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                               evict=True, mask=0xFFFFFFFF, fast=True,
                               evict_interval_steps=8, exact_evict=True)
            set_gpu_verify(None); _levers_off()
            ok = (vs.all_matched and vg.all_matched
                  and vs.accepted_steps == draft.step_count
                  and vg.accepted_steps == draft.step_count
                  and vs.decoded_final_ax == draft.final_ax_masked
                  and vg.decoded_final_ax == draft.final_ax_masked)
            allok = allok and ok
            print(f"  K={K:>5} steps={steps} scalar=({vs.accepted_steps},"
                  f"{vs.decoded_final_ax},{vs.all_matched}) gpu=({vg.accepted_steps},"
                  f"{vg.decoded_final_ax},{vg.all_matched}) -> "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
        print(f"  -> {'BYTE-EXACT (nested loop)' if allok else 'DIVERGENCE'}\n",
              flush=True)

        # MEASURE: graph OFF vs ON at giant K.
        print("=== MEASURE: composed step, block-0 graph OFF vs ON ===", flush=True)
        print(f"  {'K':>7} {'OFF us/step':>12} {'ON us/step':>12} {'speedup':>8} "
              f"{'ON steps/s':>11} {'ON sec/frame':>13} {'ON x-floor':>11} "
              f"{'match':>6}", flush=True)
        for K in (8192, 65536, 262144):
            off_ms, _, _, off_m = _time_verify(model, L, code, draft, K, device,
                                               False, steps)
            on_ms, on_sps, on_frame, on_m = _time_verify(model, L, code, draft, K,
                                                         device, True, steps)
            print(f"  {K:>7} {off_ms*1e3:>12.2f} {on_ms*1e3:>12.2f} "
                  f"{off_ms/on_ms:>7.2f}x {on_sps:>11.0f} {on_frame:>12.1f}s "
                  f"{on_ms*1e3/FLOP_FLOOR_US:>10.0f}x {str(off_m and on_m):>6}",
                  flush=True)
    finally:
        uninstall_composed(model)
    print("\n=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
