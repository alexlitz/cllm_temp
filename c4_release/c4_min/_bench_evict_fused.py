#!/usr/bin/env python3
"""Lean fused-eviction validator for the DEEP-loop fast path (#667 follow-up).

Runs ONLY the block-wise ``verify_blocks`` (perfect-draft speculation + the FUSED
on-GPU KV eviction) over a deep program, and reports:

  * the eviction/forward WALL SPLIT (``stats['t_evict']`` / ``stats['t_forward']``)
    — the #667 bottleneck instrumentation.  With the per-block host-synced prune
    loop, eviction dominated (GPU idle 41%); fused it is a small fraction;
  * BYTE-IDENTITY vs the perfect draft (the model accepted every step-query row)
    AND vs a short prefix of the token-by-token cached driver (model==model);
  * the cache/eviction stats (bounded cache, #row-evictions) + peak VRAM.

Unlike ``bench_fast_path`` it does NOT run the full naive per-step baseline (which
is minutes and, on a shared box, contention-dominated), so the eviction/forward
SPLIT is measured cleanly on the SAME run and is contention-robust (both phases
share the box, so the RATIO is fair).

    OMP_NUM_THREADS=4 python -m c4_min._bench_evict_fused nested --outer 6 \
        --inner 100 --device cuda:1 --block-steps 64 --block-moe
"""
from __future__ import annotations

import argparse
import time
import sys
from typing import List, Optional


def _build(kind: str, args):
    from c4_min.bench_fast_path import (
        build_nested, build_loop_countdown, build_matmul, build_malloc)
    if kind == "nested":
        return build_nested(args.outer, args.inner)
    if kind == "loop":
        return build_loop_countdown(args.n)
    if kind == "matmul":
        return build_matmul(args.n)
    if kind == "malloc":
        return build_malloc(args.n)
    raise ValueError(kind)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["nested", "loop", "matmul", "malloc"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--inner", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--block-steps", type=int, default=64)
    ap.add_argument("--block-moe", action="store_true")
    ap.add_argument("--prune-interval", type=int, default=60)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--spotcheck-steps", type=int, default=25,
                    help="run the token-by-token cached driver this many steps and "
                         "assert the fast output matches it (model==model).")
    ap.add_argument("--sp-init", type=lambda s: int(s, 0), default=0xFC)
    args = ap.parse_args(argv)

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = args.sp_init

    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.pf_speculative import draft_pf_program, verify_blocks
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _cuda = device.startswith("cuda")

    code, expected, data, label = _build(args.kind, args)
    print(f"=== {label} ===", flush=True)

    draft = draft_pf_program(code, max_steps=5_000_000, mask=0xFFFFFFFF)
    ndivmod = sum(1 for f in draft.frames if f["op"] in ("DIV", "MOD"))
    print(f"  draft_steps={draft.step_count}  halted={draft.halted}  "
          f"divmod={ndivmod}  final_ax={draft.final_ax_masked}", flush=True)
    if not draft.halted:
        print("  DRAFT DID NOT HALT"); return 2

    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        sparse = sparse.to(device)
    print(f"  blocks={len(sparse.blocks)}  moved to {device}", flush=True)

    # -- FAST verify (fused eviction) with the timing split -------------------
    fast_out: List[int] = []
    fast_stats: dict = {}
    if _cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    t = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=args.block_steps,
                       device=device, evict=(not args.no_evict),
                       prune_interval=args.prune_interval, mask=0xFFFFFFFF,
                       stats=fast_stats, fast=True, collect_out=fast_out,
                       block_moe=args.block_moe)
    if _cuda:
        torch.cuda.synchronize()
    t_fast = time.time() - t

    t_ev = fast_stats.get("t_evict", 0.0)
    n_pr = fast_stats.get("n_prunes", 0)
    t_rest = max(t_fast - t_ev, 0.0)
    peak_gb = (torch.cuda.max_memory_allocated(torch.device(device)) / 1e9
               if _cuda else 0.0)
    print(f"  FAST: {vr.forwards} forwards in {t_fast:.1f}s  "
          f"all_matched={vr.all_matched} accepted={vr.accepted_steps}/"
          f"{draft.step_count}", flush=True)
    print(f"  WALL SPLIT: evict={t_ev:.2f}s ({100*t_ev/max(t_fast,1e-9):.2f}%, "
          f"{n_pr} prunes, {1000*t_ev/max(n_pr,1):.2f} ms/prune)  "
          f"forward+overlay={t_rest:.1f}s ({100*t_rest/max(t_fast,1e-9):.1f}%)",
          flush=True)
    print(f"  CACHE: max_cache={fast_stats.get('max_cache_size')}  "
          f"row_evictions={fast_stats.get('total_evicted')}  "
          f"peak_vram={peak_gb:.2f} GB  "
          f"block_moe_speedup={fast_stats.get('block_moe_speedup',1.0):.2f}x",
          flush=True)
    if not vr.all_matched:
        print(f"  FAST FAIL: {vr.first_mismatch}"); return 1

    # -- BYTE-IDENTITY --------------------------------------------------------
    out_match_draft = (fast_out == (draft.out or []))
    final_ax_match = (vr.decoded_final_ax == draft.final_ax_masked)
    # short cached-driver prefix (model==model), evicting on the same interval.
    n_spot = min(args.spotcheck_steps, draft.step_count)
    drv_out: List[int] = []
    drv_tr = run_pure_forward_cached(
        sparse, L, code, max_steps=n_spot, mask=0xFFFFFFFF,
        evict=(not args.no_evict), prune_interval=args.prune_interval,
        out=drv_out, data_seg=data)
    draft_prefix_ax = [f["ax"] & 0xFFFFFFFF for f in draft.frames[:len(drv_tr)]]
    drv_match = (drv_tr == draft_prefix_ax)
    print(f"  BYTE-IDENTITY: fast_out==draft={out_match_draft}  "
          f"final_ax_match={final_ax_match} (fast={vr.decoded_final_ax} "
          f"draft={draft.final_ax_masked})  "
          f"cached_driver_prefix=={drv_match} ({len(drv_tr)} steps)", flush=True)
    ok = vr.all_matched and final_ax_match and drv_match
    if expected is not None:
        exp_ok = (vr.decoded_final_ax == (expected & 0xFFFFFFFF))
        print(f"  final_ax == expected(32-bit ref): {exp_ok} "
              f"(expected={expected & 0xFFFFFFFF})", flush=True)
    print(f"  RESULT: {'OK' if ok else 'MISMATCH'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
