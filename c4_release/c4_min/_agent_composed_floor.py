#!/usr/bin/env python3
"""_agent_composed_floor.py — THE CULMINATION: compose ALL perf levers onto the REAL
doom ``verify_blocks`` path, measure the HONEST composed full-step floor at giant K,
and BYTE-EXACT gate the whole composed stack.

Levers composed (ALL ON, all default-OFF individually):
  * C4_DEAD_BLOCK_FUSION   — 238 dead blocks -> output=x (zero attention compute)
  * C4_DIRECT_CAM_BATCHED  — global mem/stack/LEV/code heads O(1) direct-gather
  * C4_DIRECT_LOCAL_CAM    — block-0 ingest locals O(1) direct-gather
  * C4_FLASH_ATTN          — flash/banded softmax1 (no [Sq,Sk] score tensor)
  * C4_BANDED_LOCAL_ATTN   — length-30 ingest band
  * C4_FROZEN_ROW_SKIP     — [cut,N) over K query rows only
  * C4_CUT_SPAN_CHUNK      — O(K) band: block-0 S-chunked -> giant K (65k/262k)
  * C4_FUSED_MEGABLOCK     — dead-FFN [cut,N) chain as ONE on-chip fused megakernel
  * C4_FUSED_DELTA_FFN     — sparse-COO SwiGLU (only touches nnz)
  * C4_GPU_VERIFY          — vectorized whole-verify decode+compare (ONE host sync/fwd)
  * C4_EXACT_EVICT         — O(steps) last-read+1 liveness schedule (GPU-vectorized drop)

TASK 5 (byte-exact): per-step accepted/all_matched/final_ax of the FULL composed
stack == the K=1 reference, on the DIV-free battery + deep nested loops.
TASK 3 (measure): ms/step, host-syncs/forward, steps/sec, sec/frame(6.89M) at K in
{8192, 65536, 262144}.
TASK 4 (new wall): block-0 ingest attention us/step (the 97%-FLOP part kept below
the megablock cut).

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_composed_floor --device cuda:0
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
_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import (draft_pf_program, verify_blocks, set_gpu_verify,
                                   V as _V)
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested, build_loop_countdown
from c4_min.bench_composed_fast_path import _battery


DOOM_FRAME_INSTRS = 6_890_000       # a real doom title frame (~6.89M instrs)
FLOP_FLOOR_US = 0.0071              # measured sparse-nnz FLOP floor per step (us)


# --- host-sync tally (monkeypatch .item()) --------------------------------------
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


ALL_LEVERS = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FROZEN_ROW_SKIP",
              "C4_CUT_SPAN_CHUNK", "C4_FUSED_MEGABLOCK", "C4_FUSED_DELTA_FFN",
              "C4_OVERLAY_BATCHED", "C4_STREAM_EMBED", "C4_GRAPH_BLOCK0"]


def _levers_on(cut_chunk):
    os.environ["C4_DEAD_BLOCK_FUSION"] = "1"
    os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
    os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
    os.environ["C4_FLASH_ATTN"] = "1"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "1"
    os.environ["C4_FROZEN_ROW_SKIP"] = "1"
    os.environ["C4_FUSED_MEGABLOCK"] = "1"
    os.environ["C4_FUSED_DELTA_FFN"] = "1"
    os.environ["C4_CUT_SPAN_CHUNK"] = str(cut_chunk)
    # BLOCK-0 GRAPH (lever 1): CUDA-graph block-0's S-chunked ingest attn+FFN per-chunk
    # body — kills the ~640-707 us host dispatch over the chunk loop (the profiled wall).
    os.environ["C4_GRAPH_BLOCK0"] = "1"
    # LAUNCH-COLLAPSE: batch the ~465K per-row overlay scalar HtoD copies into ONE
    # index_put_ (the profiled #1 wall, 28.6% of the composed step); stream-embed
    # builds each block-0 chunk's embed+overlay on demand (kills the last O(K*30)).
    os.environ["C4_OVERLAY_BATCHED"] = "1"
    os.environ["C4_STREAM_EMBED"] = "1"
    # qrow chunk defaults to cut chunk; keep block-0 dead-kv drop on for giant K.
    os.environ["C4_BLOCK0_DROP_DEAD_KV"] = "1"
    # RUNG 2 + RUNG 3 (this task): whole-step block-0 chunk-loop graph (folds the ingest
    # gather + kills the ~640 per-chunk host syncs) + block-0 dense FFN -> fused-delta COO.
    if os.environ.get("C4_RUNGS_23", "1") not in ("0", "", "false", "False"):
        os.environ["C4_WHOLE_STEP_GRAPH"] = "1"
        os.environ["C4_BLOCK0_FUSED_FFN"] = "1"


def _levers_off():
    for f in ALL_LEVERS + ["C4_BLOCK0_DROP_DEAD_KV", "C4_QROW_CHUNK",
                           "C4_WHOLE_STEP_GRAPH", "C4_BLOCK0_FUSED_FFN"]:
        os.environ.pop(f, None)


def _verify(model, L, code, draft, K, device, *, gpu, mega, cut_chunk):
    if mega:
        _levers_on(cut_chunk)
    set_gpu_verify(gpu)
    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    set_gpu_verify(None)
    return vr, stats


def byte_exact(model, L, device):
    """TASK 5: full composed stack (mega + gpu-verify + O(K) band + exact-evict) is
    byte-exact — its per-step AX/PC/SP/BP decode matches the CERTIFIED DRAFT (the exact
    reference-ISA interpreter, the K=1 ground truth) at EVERY step — on the DIV-free
    battery + deep nested loops.

    The draft IS the K=1 per-step reference: ``draft_pf_program`` runs ``ref_interpret``
    (the bit-exact C4 ISA) one step at a time.  ``all_matched and accepted_steps ==
    step_count`` means the model's decoded (AX,PC,SP,BP) equalled the draft's at every
    single step — that is the per-step byte-exact trace vs the K=1 reference.  We ALSO
    cross-check the scalar (per-step decode) verify against the GPU-vectorized verify at
    each K so both decode paths agree."""
    print("\n=== TASK 5: BYTE-EXACT (full composed stack per-step == certified draft) ===",
          flush=True)
    progs = []
    for name, prog, seed in _battery():
        if seed:
            continue                 # composed doom path is DIV-free / seed-mem-free
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        progs.append((name, code))
    progs.append(("loop_countdown60", build_loop_countdown(60)[0]))
    progs.append(("nested_6_16", build_nested(6, 16)[0]))
    progs.append(("nested_10_24", build_nested(10, 24)[0]))    # deep nested loop
    Ks = [512, 2048, 8192]
    all_ok = True
    install_composed(model, verbose=False)
    print(f"{'prog':>16} {'K':>7} {'steps':>7} {'scalar(acc,fin,m)':>22} "
          f"{'gpu(acc,fin,m)':>22} {'ok':>4}", flush=True)
    try:
        for name, code in progs:
            draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
            if not draft.halted:
                print(f"{name:>16}  draft did not halt, skip", flush=True)
                continue
            for K in Ks:
                # SCALAR (per-step decode) vs GPU-vectorized decode, both full-stack.
                vs, _ = _verify(model, L, code, draft, K, device,
                                gpu=False, mega=True, cut_chunk=256)
                _levers_off()
                vg, _ = _verify(model, L, code, draft, K, device,
                                gpu=True, mega=True, cut_chunk=256)
                _levers_off()
                sc = (vs.accepted_steps, vs.decoded_final_ax, vs.all_matched)
                gp = (vg.accepted_steps, vg.decoded_final_ax, vg.all_matched)
                # per-step byte-exact vs the certified draft: matched at EVERY step.
                draft_exact = (vs.all_matched and vg.all_matched
                               and vs.accepted_steps == draft.step_count
                               and vg.accepted_steps == draft.step_count
                               and vs.decoded_final_ax == draft.final_ax_masked
                               and vg.decoded_final_ax == draft.final_ax_masked)
                # scalar decode path agrees with the GPU-vectorized decode path.
                paths_agree = (sc == gp)
                ok = draft_exact and paths_agree
                all_ok = all_ok and ok
                print(f"{name:>16} {K:>7} {draft.step_count:>7} {str(sc):>22} "
                      f"{str(gp):>22} {'OK' if ok else 'FAIL':>4}", flush=True)
                if not ok:
                    print(f"    scalar first_mismatch={vs.first_mismatch}", flush=True)
                    print(f"    gpu    first_mismatch={vg.first_mismatch}", flush=True)
            _guard()
    finally:
        uninstall_composed(model)
    print(f"\n  -> {'ALL BYTE-EXACT' if all_ok else 'DIVERGENCE FOUND'}", flush=True)
    return all_ok


def measure(model, L, device):
    """TASK 3+4: composed full-step at giant K + host-sync count + block-0 wall."""
    print("\n=== TASK 3: COMPOSED FULL-STEP FLOOR at giant K ===", flush=True)
    # the deep nested loop is the doom-representative long DIV-free stream.
    name, code = "nested_12_28", build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    assert draft.halted, "draft did not halt"
    steps = draft.step_count
    print(f"  program '{name}': {steps} DIV-free steps (deep nested loop)", flush=True)

    install_composed(model, verbose=False)
    Ks = [8192, 65536, 262144]
    rows = []
    try:
        for K in Ks:
            _guard()
            # WARMUP (build the megablock CUDA graphs + prime allocator).
            _verify(model, L, code, draft, K, device, gpu=True, mega=True,
                    cut_chunk=256)
            _levers_off()
            # TIMED + host-sync tally.
            _levers_on(256)
            set_gpu_verify(True)
            torch.cuda.synchronize(device)
            _tally.clear(); _tracing[0] = True
            stats = {}
            t0 = time.perf_counter()
            vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                               evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                               evict_interval_steps=8, exact_evict=True)
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - t0
            _tracing[0] = False
            set_gpu_verify(None)
            _levers_off()
            fwds = stats.get("forwards", vr.forwards)
            syncs = sum(_tally.values())
            ms_step = wall * 1e3 / steps
            sps = steps / wall
            sec_frame = DOOM_FRAME_INSTRS / sps
            rows.append((K, ms_step, syncs, syncs / max(fwds, 1), sps, sec_frame,
                         fwds, vr.all_matched))
            top = _tally.most_common(4)
            print(f"  K={K:>7}: {ms_step*1e3:8.2f} us/step  {sps:12.0f} steps/s  "
                  f"frame={sec_frame:7.2f}s  fwds={fwds:4d}  "
                  f"syncs={syncs} ({syncs/max(fwds,1):.2f}/fwd)  "
                  f"matched={vr.all_matched}", flush=True)
            print(f"           top .item() sites: "
                  + "; ".join(f"{c}x {s}" for s, c in top), flush=True)
    finally:
        uninstall_composed(model)

    print("\n  --- floor comparison (composed vs baselines) ---", flush=True)
    best = min(rows, key=lambda r: r[1])
    print(f"  composed best: {best[1]*1e3:.2f} us/step at K={best[0]}", flush=True)
    print(f"    vs 1.11 ms/step decode-only baseline: "
          f"{1.11 / best[1]:.1f}x faster", flush=True)
    print(f"    vs 0.034 ms/step lean-forward:        "
          f"{0.034 / best[1]:.2f}x", flush=True)
    print(f"    vs {FLOP_FLOOR_US} us FLOP floor:            "
          f"{best[1]*1e3 / FLOP_FLOOR_US:.1f}x above compute-bound", flush=True)
    return rows, draft


def measure_block0(model, L, device, draft):
    """TASK 4: the NEW dominant wall after the composition — the block-0 ingest dense
    SwiGLU/attention GEMM (the 97%-FLOP part a3efe5e kept ABOVE the megablock cut).
    Quantified by a CUDA-profiler kernel split of ONE composed forward: the
    ``ampere_sgemm`` (block-0 + 3 live CAM blocks' dense GEMM) vs the megablock kernels
    (``_down_delta_inplace`` / ``_fused_upgate_silu``) vs the residual overlay HtoD."""
    print("\n=== TASK 4: NEW DOMINANT WALL (CUDA kernel split) ===", flush=True)
    from torch.profiler import profile, ProfilerActivity
    code = build_nested(12, 28)[0]
    steps = draft.step_count
    install_composed(model, verbose=False)
    K = 8192
    try:
        _levers_on(256); set_gpu_verify(True)
        _verify(model, L, code, draft, K, device, gpu=True, mega=True, cut_chunk=256)
        _levers_off(); _levers_on(256); set_gpu_verify(True)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as prof:
            verify_blocks(model, L, code, draft, block_steps=K, device=device,
                          evict=True, mask=0xFFFFFFFF, fast=True,
                          evict_interval_steps=8, exact_evict=True)
            torch.cuda.synchronize(device)
        set_gpu_verify(None); _levers_off()
    finally:
        uninstall_composed(model)

    evs = prof.key_averages()
    tot_us = sum(e.self_device_time_total for e in evs) or 1.0
    def cat(pred):
        return sum(e.self_device_time_total for e in evs if pred(e.key))
    sgemm = cat(lambda k: "sgemm" in k or "gemm" in k.lower())
    mega = cat(lambda k: "delta_inplace" in k or "upgate_silu" in k)
    htod = cat(lambda k: "HtoD" in k)
    print(f"  ONE composed forward, {steps} steps, total device time "
          f"{tot_us/1e3:.1f} ms ({tot_us/steps:.2f} us/step):", flush=True)
    print(f"    block-0/live DENSE GEMM (ampere_sgemm) : {sgemm/1e3:7.1f} ms "
          f"({sgemm/tot_us*100:5.1f}%)  {sgemm/steps:7.2f} us/step  <== NEW WALL",
          flush=True)
    print(f"    megablock kernels (delta+upgate_silu)  : {mega/1e3:7.1f} ms "
          f"({mega/tot_us*100:5.1f}%)  {mega/steps:7.2f} us/step", flush=True)
    print(f"    overlay HtoD (batched)                 : {htod/1e3:7.1f} ms "
          f"({htod/tot_us*100:5.1f}%)  {htod/steps:7.2f} us/step", flush=True)
    print(f"  -> the NEW dominant wall is the block-0 (+3 live CAM) DENSE SwiGLU/attn "
          f"GEMM = {sgemm/steps:.1f} us/step = {sgemm/steps/FLOP_FLOOR_US:.0f}x the FLOP "
          f"floor.  The megablock ATE the dead-FFN [cut,N) region (now {mega/tot_us*100:.0f}%); "
          f"block-0's dense ingest GEMM (the 97%-FLOP part kept above the cut) is the "
          f"next lever (fused-delta / megablock the block-0 FFN).", flush=True)


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
    rows, draft = measure(model, L, device)
    measure_block0(model, L, device, draft)
    print(f"\n{'=== COMPOSED FLOOR COMPLETE ===' if ok else '=== BYTE-EXACT FAILED ==='}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
