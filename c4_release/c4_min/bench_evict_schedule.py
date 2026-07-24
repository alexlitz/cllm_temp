#!/usr/bin/env python3
"""A/B bench for the DRAFT-DRIVEN eviction schedule (C4_EVICT_SCHEDULE).

Builds ONE sparse model, drafts the program once, and runs the block-wise verify
BOTH ways — the content-comparison fused eviction (``evict_all_blocks_fused``, the
O(S^2) pairwise near-dup) and the schedule-driven eviction (a single O(stores)
liveness precompute + an O(dropped)/round position drop) — reporting for each:

  * the EVICTION WALL SPLIT (``stats['t_evict']`` / share of the fast wall) and
    ms/step — the cost the schedule targets;
  * BYTE-IDENTITY: same accept, same decoded final AX, same visible output, AND the
    same KV-cache contents (retained absolute positions per block).

The intended config is DROP-KV local attention + content-bound global heads (the
memory-heavy fast path), where the ONLY content comparison is the §Memory store
heads' address supersession — exactly what the liveness schedule reproduces off the
draft.  ``--classic`` runs the full-H cache (no split) to expose the raw O(S^2)
content-comparison cost (there the schedule governs only the store rows, so it is
NOT a drop-in — reported honestly).

    OMP_NUM_THREADS=4 python -m c4_min.bench_evict_schedule malloc --n 48 \
        --device cuda:0 --block-steps 32 --evict-interval-steps 8
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


def _cache_positions(caches):
    out = []
    for c in caches:
        if getattr(c, "pos", None) is None:
            out.append([])
        else:
            out.append(sorted(int(p) for p in c.pos.tolist()))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["nested", "loop", "matmul", "malloc"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--inner", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--block-steps", type=int, default=32)
    ap.add_argument("--evict-interval-steps", type=int, default=8)
    ap.add_argument("--block-moe", action="store_true")
    ap.add_argument("--local-window", type=int, default=64)
    ap.add_argument("--classic", action="store_true",
                    help="full-H cache (no local/content-bound split) — exposes the "
                         "raw O(S^2) content compare; schedule NOT a drop-in there.")
    ap.add_argument("--sp-init", type=lambda s: int(s, 0), default=0xFC)
    ap.add_argument("--reps", type=int, default=2, help="timed repeats (min taken).")
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
    import c4_min.pf_speculative as PS

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _cuda = device.startswith("cuda")

    code, expected, data, label = _build(args.kind, args)
    print(f"=== {label}  (device={device}, K={args.block_steps}, "
          f"evict_interval={args.evict_interval_steps}, "
          f"{'CLASSIC full-H' if args.classic else 'DROP-KV+content-bound'}) ===",
          flush=True)

    draft = draft_pf_program(code, max_steps=5_000_000, mask=0xFFFFFFFF)
    print(f"  draft_steps={draft.step_count}  halted={draft.halted}  "
          f"stores={len(draft.store_log)}  loads={len(draft.load_log or {})}  "
          f"final_ax={draft.final_ax_masked}", flush=True)
    if not draft.halted:
        print("  DRAFT DID NOT HALT"); return 2

    t = time.time()
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    print(f"  build {time.time()-t:.1f}s  blocks={len(sparse.blocks)}", flush=True)
    if device != "cpu":
        sparse = sparse.to(device)

    if not args.classic:
        from c4_min.local_attention import install_local_attention
        install_local_attention(sparse, window=args.local_window,
                                drop_local_kv=True, content_bound_global=True,
                                verbose=False)
        print(f"  DROP-KV + content-bound global installed (window={args.local_window})",
              flush=True)

    def run(evict_schedule: bool):
        # capture the caches (both eviction fns take caches as first arg).
        captured = {"caches": None}
        of, os_ = PS.evict_all_blocks_fused, PS.evict_all_blocks_scheduled
        def cf(caches, *a, **k):
            captured["caches"] = caches; return of(caches, *a, **k)
        def cs(caches, *a, **k):
            captured["caches"] = caches; return os_(caches, *a, **k)
        PS.evict_all_blocks_fused = cf
        PS.evict_all_blocks_scheduled = cs
        best = None
        try:
            for _ in range(max(1, args.reps)):
                st = {}; out = []
                if _cuda:
                    torch.cuda.synchronize()
                t0 = time.time()
                vr = verify_blocks(sparse, L, code, draft,
                                   block_steps=args.block_steps, device=device,
                                   evict=True, mask=0xFFFFFFFF, stats=st, fast=True,
                                   collect_out=out, block_moe=args.block_moe,
                                   evict_interval_steps=args.evict_interval_steps,
                                   evict_schedule=evict_schedule)
                if _cuda:
                    torch.cuda.synchronize()
                wall = time.time() - t0
                if best is None or wall < best[0]:
                    best = (wall, vr, st, out, _cache_positions(captured["caches"]))
        finally:
            PS.evict_all_blocks_fused = of
            PS.evict_all_blocks_scheduled = os_
        return best

    print("  running CONTENT-comparison eviction ...", flush=True)
    wc, vrc, stc, oc, posc = run(False)
    print("  running SCHEDULE-driven eviction ...", flush=True)
    ws, vrs, sts, os_, poss = run(True)

    def line(tag, wall, vr, st):
        tev = st.get("t_evict", 0.0)
        n = draft.step_count
        print(f"  [{tag}] wall={wall:.2f}s  t_evict={tev:.3f}s "
              f"({100*tev/max(wall,1e-9):.1f}%)  rounds={st.get('evict_rounds')}  "
              f"ms/step={1000*wall/max(n,1):.2f}  forwards={vr.forwards}  "
              f"matched={vr.all_matched}  final={vr.decoded_final_ax}  "
              f"maxcache={vr.max_cache_size}  evicted={vr.total_evicted}", flush=True)
    line("content ", wc, vrc, stc)
    line("schedule", ws, vrs, sts)

    tev_c, tev_s = stc.get("t_evict", 0.0), sts.get("t_evict", 0.0)
    print(f"  --- EVICTION COST: content t_evict={tev_c:.3f}s -> schedule "
          f"t_evict={tev_s:.3f}s  ({tev_c/max(tev_s,1e-9):.1f}x less eviction wall)",
          flush=True)
    print(f"  --- ms/step: content {1000*wc/max(draft.step_count,1):.2f} -> "
          f"schedule {1000*ws/max(draft.step_count,1):.2f}", flush=True)

    # byte-identity
    ax_ok = vrc.all_matched and vrs.all_matched
    final_ok = (vrc.decoded_final_ax == vrs.decoded_final_ax == draft.final_ax_masked)
    accept_ok = (vrc.accepted_steps == vrs.accepted_steps == draft.step_count)
    out_ok = (oc == os_)
    kv_ok = (posc == poss)
    print(f"  --- BYTE-IDENTITY: accept={accept_ok}  final_ax={final_ok} "
          f"(c={vrc.decoded_final_ax} s={vrs.decoded_final_ax} draft={draft.final_ax_masked})"
          f"  out={out_ok}  KV-contents-match={kv_ok}", flush=True)
    if not kv_ok:
        for b, (a, bb) in enumerate(zip(posc, poss)):
            if a != bb:
                print(f"      block {b} DIFFERS: content={a}\n"
                      f"                     schedule={bb}", flush=True)
                break
    ok = ax_ok and final_ok and accept_ok and out_ok and (kv_ok or args.classic)
    print(f"  RESULT: {'OK' if ok else 'MISMATCH'}"
          + ("  (classic: KV-contents differ by design — schedule governs only "
             "store rows)" if args.classic and not kv_ok else ""), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
