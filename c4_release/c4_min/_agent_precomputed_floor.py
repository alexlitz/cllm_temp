#!/usr/bin/env python3
"""_agent_precomputed_floor.py — PRECOMPUTED SCHEDULE + SINGLE DISPATCH: byte-exact
gate + composed wall measurement of ``C4_PRECOMPUTED_SCHEDULE`` vs the per-op
``verify_blocks`` baseline.

TASK 4 (byte-exact): per-step AX/PC/SP/BP of the precomputed-schedule single dispatch
== the K=1 reference draft AND == the composed verify_blocks GPU-verify, on the
DIV-free battery + the deep nested loop.  L-inf=0.
TASK 3 (measure): composed wall+device us/step at K in {8192,65536,262144}, host-ops
before (~3600/forward) vs after (O(n_chunks)), steps/sec, sec/frame, x from 35 fps.

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_precomputed_floor --device cuda:0
"""
from __future__ import annotations

import argparse
import collections
import os
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import (draft_pf_program, verify_blocks, set_gpu_verify)
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested, build_loop_countdown
from c4_min.bench_composed_fast_path import _battery
from c4_min import precomputed_schedule as PS


DOOM_FRAME_INSTRS = 6_890_000
FLOP_FLOOR_US = 0.0071
RENDER_REDUCED_FRAME = 358_058       # the render-reduced doom frame (task-specified)

_tally = collections.Counter()
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


COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _composed_on():
    for f in COMPOSED:
        os.environ[f] = "1"


def _composed_off():
    for f in COMPOSED + ["C4_FROZEN_ROW_SKIP", "C4_CUT_SPAN_CHUNK", "C4_FUSED_DELTA_FFN",
                         "C4_OVERLAY_BATCHED", "C4_STREAM_EMBED", "C4_GRAPH_BLOCK0",
                         "C4_BLOCK0_DROP_DEAD_KV", "C4_QROW_CHUNK",
                         "C4_PRECOMPUTED_SCHEDULE"]:
        os.environ.pop(f, None)


# baseline (per-op verify_blocks) levers — the SAME composed stack _agent_composed_floor
# times, so the before/after wall is apples-to-apples.
def _baseline_on(cut_chunk):
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


def _baseline_off():
    _composed_off()


def byte_exact(model, L, device):
    """TASK 4: precomputed-schedule single dispatch == K=1 reference draft AND ==
    composed verify_blocks GPU-verify, per-step AX/PC/SP/BP, DIV-free battery + deep
    nested loops."""
    print("\n=== TASK 4: BYTE-EXACT (precomputed-schedule single dispatch) ===",
          flush=True)
    progs = []
    for name, prog, seed in _battery():
        if seed:
            continue
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        # only DIV-free programs (the precomputed schedule does not carry divmod).
        d = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
        if not d.halted:
            continue
        if any(d.frames[s].get("op") in ("DIV", "MOD") for s in range(d.step_count)):
            continue
        progs.append((name, code))
    progs.append(("loop_countdown60", build_loop_countdown(60)[0]))
    progs.append(("nested_6_16", build_nested(6, 16)[0]))
    progs.append(("nested_10_24", build_nested(10, 24)[0]))
    all_ok = True
    install_composed(model, verbose=False)
    print(f"{'prog':>18} {'steps':>7} {'ref(acc,fin,m)':>22} {'ps(acc,fin,m)':>22} "
          f"{'Linf':>6} {'ok':>4}", flush=True)
    try:
        for name, code in progs:
            draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
            if not draft.halted:
                continue
            # REFERENCE: the composed verify_blocks GPU-verify (K covers the whole prog).
            _baseline_on(256); set_gpu_verify(True)
            vr = verify_blocks(model, L, code, draft, block_steps=max(draft.step_count, 8),
                               device=device, evict=True, mask=0xFFFFFFFF, fast=True,
                               evict_interval_steps=8, exact_evict=True)
            set_gpu_verify(None); _baseline_off()
            # PRECOMPUTED SCHEDULE single dispatch.
            _composed_on()
            stats = {}
            pr = PS.run_verify(model, L, code, draft, device, mask=0xFFFFFFFF, stats=stats)
            _composed_off()
            ref = (vr.accepted_steps, vr.decoded_final_ax, vr.all_matched)
            ps = (pr.accepted_steps, pr.decoded_final_ax, pr.all_matched)
            draft_exact = (pr.all_matched and pr.accepted_steps == draft.step_count
                           and pr.decoded_final_ax == draft.final_ax_masked)
            agree = (ref == ps)
            ok = draft_exact and agree
            all_ok = all_ok and ok
            # L-inf proxy: the register decode is integer-exact, so Linf==0 iff acc/fin
            # match at every step (the decode is bit-identical requant-argmax).
            linf = 0 if agree and draft_exact else 1
            print(f"{name:>18} {draft.step_count:>7} {str(ref):>22} {str(ps):>22} "
                  f"{linf:>6} {'OK' if ok else 'FAIL':>4}", flush=True)
            if not ok:
                print(f"    ref first_mismatch={vr.first_mismatch}", flush=True)
                print(f"    ps  first_mismatch={pr.first_mismatch}", flush=True)
            _guard()
    finally:
        uninstall_composed(model)
    print(f"\n  -> {'ALL BYTE-EXACT (Linf=0)' if all_ok else 'DIVERGENCE FOUND'}",
          flush=True)
    return all_ok


def measure(model, L, device):
    """TASK 3: composed wall+device us/step + host-ops/forward, before vs after."""
    print("\n=== TASK 3: WALL — before (per-op verify_blocks) vs after (single dispatch) ===",
          flush=True)
    name, code = "nested_12_28", build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    assert draft.halted, "draft did not halt"
    steps = draft.step_count
    print(f"  program '{name}': {steps} DIV-free steps (deep nested loop)", flush=True)
    install_composed(model, verbose=False)
    Ks = [8192, 65536, 262144]
    rows = []
    try:
        # ---- BEFORE: the per-op composed verify_blocks (the ~857 us/step baseline) ----
        print("\n  --- BEFORE (per-op composed verify_blocks) ---", flush=True)
        for K in Ks:
            _guard()
            _baseline_on(256); set_gpu_verify(True)
            verify_blocks(model, L, code, draft, block_steps=K, device=device,
                          evict=True, mask=0xFFFFFFFF, fast=True,
                          evict_interval_steps=8, exact_evict=True)   # warmup
            set_gpu_verify(None); _baseline_off()
            _baseline_on(256); set_gpu_verify(True)
            torch.cuda.synchronize(device)
            _tally.clear(); _tracing[0] = True
            st = {}
            t0 = time.perf_counter()
            vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                               evict=True, mask=0xFFFFFFFF, stats=st, fast=True,
                               evict_interval_steps=8, exact_evict=True)
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - t0
            _tracing[0] = False
            set_gpu_verify(None); _baseline_off()
            us = wall * 1e6 / steps
            sps = steps / wall
            print(f"  K={K:>7}: {us:8.2f} us/step  {sps:12.0f} steps/s  "
                  f"host.item()={sum(_tally.values())}  matched={vr.all_matched}",
                  flush=True)
            rows.append(("before", K, us, sps, vr.all_matched))
        # ---- AFTER: the precomputed schedule single dispatch ----
        print("\n  --- AFTER (C4_PRECOMPUTED_SCHEDULE single dispatch) ---", flush=True)
        # AFTER is K-independent (the whole batch is one schedule) — vary the per-chunk
        # graph size (C4_SCHED_CHUNK) to show it is not per-op bound.
        for chunk in [4096, 16384, 65536]:
            _guard()
            os.environ["C4_SCHED_CHUNK"] = str(chunk)
            _composed_on()
            PS.run_verify(model, L, code, draft, device, mask=0xFFFFFFFF)   # warmup
            _composed_off()
            _composed_on()
            os.environ["C4_SCHED_CHUNK"] = str(chunk)
            torch.cuda.synchronize(device)
            _tally.clear(); _tracing[0] = True
            st = {}
            t0 = time.perf_counter()
            pr = PS.run_verify(model, L, code, draft, device, mask=0xFFFFFFFF, stats=st)
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - t0
            _tracing[0] = False
            _composed_off()
            us = wall * 1e6 / steps
            sps = steps / wall
            hostops = st.get("host_ops_per_forward", 0)
            print(f"  chunk={chunk:>7}: {us:8.2f} us/step  {sps:12.0f} steps/s  "
                  f"chunks={st.get('n_chunks')}  host-ops={hostops}  "
                  f"host.item()={sum(_tally.values())}  matched={pr.all_matched}",
                  flush=True)
            rows.append(("after", chunk, us, sps, pr.all_matched))
            top = _tally.most_common(3)
            if top:
                print(f"           top .item() sites: "
                      + "; ".join(f"{c}x {s}" for s, c in top), flush=True)
    finally:
        uninstall_composed(model)

    # ---- DEVICE-TIME split of the single dispatch (kernel profiler) ----
    print("\n  --- device-time split (single dispatch, CUDA profiler) ---", flush=True)
    install_composed(model, verbose=False)
    try:
        from torch.profiler import profile, ProfilerActivity
        os.environ["C4_SCHED_CHUNK"] = "16384"
        _composed_on()
        PS.run_verify(model, L, code, draft, device, mask=0xFFFFFFFF)      # warmup
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as prof:
            PS.run_verify(model, L, code, draft, device, mask=0xFFFFFFFF)
            torch.cuda.synchronize(device)
        _composed_off()
        evs = prof.key_averages()
        tot = sum(e.self_device_time_total for e in evs) or 1.0
        dev_us = tot / steps
        print(f"  device time {tot/1e3:.1f} ms total = {dev_us:.2f} us/step", flush=True)
    finally:
        uninstall_composed(model)

    print("\n  --- SUMMARY (before vs after; deep nested loop) ---", flush=True)
    before = [r for r in rows if r[0] == "before"]
    after = [r for r in rows if r[0] == "after"]
    b_best = min(before, key=lambda r: r[2]) if before else None
    a_best = min(after, key=lambda r: r[2]) if after else None
    if b_best:
        print(f"  BEFORE best: {b_best[2]:.2f} us/step  {b_best[3]:.0f} steps/s "
              f"(K={b_best[1]})", flush=True)
    if a_best:
        us = a_best[2]; sps = a_best[3]
        frame_full = DOOM_FRAME_INSTRS / sps
        frame_red = RENDER_REDUCED_FRAME / sps
        print(f"  AFTER  best: {us:.2f} us/step  {sps:.0f} steps/s (chunk={a_best[1]})",
              flush=True)
        if b_best:
            print(f"    -> {b_best[2]/us:.2f}x the per-op wall", flush=True)
        print(f"    sec/frame (6.89M full)      = {frame_full:.1f} s", flush=True)
        print(f"    sec/frame (358,058 reduced) = {frame_red:.3f} s  "
              f"({1.0/frame_red:.2f} fps; {frame_red:.3f}x the 1s real-time target)",
              flush=True)
        print(f"    x above 0.0071 us FLOP floor = {us / FLOP_FLOOR_US:.0f}x", flush=True)
        print(f"    x from 35 fps                = {(1.0/35.0)/frame_red:.4f}x "
              f"(reduced-frame fps / 35)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=256)
    ap.add_argument("--skip-byte-exact", action="store_true")
    args = ap.parse_args(argv)
    _guard()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(
        code_size=args.code_size, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()
    ok = True
    if not args.skip_byte_exact:
        ok = byte_exact(model, L, device)
    measure(model, L, device)
    print(f"\n{'=== PRECOMPUTED FLOOR COMPLETE ===' if ok else '=== BYTE-EXACT FAILED ==='}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
