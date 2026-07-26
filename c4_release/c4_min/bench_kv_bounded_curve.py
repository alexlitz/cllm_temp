#!/usr/bin/env python3
"""KV-size-over-time curve: confirm the EXACT O(steps) eviction (C4_EXACT_EVICT) keeps
the KV BOUNDED without the O(S^2) content prune, on a long-running program.

Runs verify_blocks THREE ways and captures the GLOBAL-cache size at EVERY eviction
round (by wrapping ``evict_all_blocks_scheduled`` / ``evict_all_blocks_fused``, which
both receive the live ``caches`` list):

  * NO-EVICTION       — evict=False: the cache just GROWS (the unbounded baseline).
  * CONTENT (O(S^2))  — the fused cdist/cosine prune.
  * EXACT  (O(steps)) — the unified min(supersession, last-read+1) schedule, NO content
                        prune at all.

The curve is the max-over-blocks global cache size sampled at each eviction round.  For
a loop / nested / malloc program the EXACT curve must stay FLAT (bounded by the local
register window + the live working set), while NO-EVICTION grows ~linearly in steps.

    OMP_NUM_THREADS=4 python -m c4_min.bench_kv_bounded_curve nested --outer 6 --inner 40 \
        --device cuda:0 --block-steps 32 --evict-interval-steps 8
"""
from __future__ import annotations

import argparse
from typing import List, Optional


def _build(kind: str, args):
    from c4_min.bench_fast_path import (
        build_nested, build_loop_countdown, build_malloc, build_malloc_free)
    if kind == "nested":
        return build_nested(args.outer, args.inner)
    if kind == "loop":
        return build_loop_countdown(args.n)
    if kind == "malloc":
        return build_malloc(args.n)
    if kind == "malloc_free":
        return build_malloc_free(args.n)
    raise ValueError(kind)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["nested", "loop", "malloc", "malloc_free"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--inner", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--block-steps", type=int, default=32)
    ap.add_argument("--evict-interval-steps", type=int, default=8)
    ap.add_argument("--local-window", type=int, default=64)
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
    import c4_min.pf_speculative as PS

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    code, expected, data, label = _build(args.kind, args)
    draft = draft_pf_program(code, max_steps=5_000_000, mask=0xFFFFFFFF)
    print(f"=== {label}  (device={device})  steps={draft.step_count} "
          f"stores={len(draft.store_log)} ===", flush=True)
    if not draft.halted:
        print("  DRAFT DID NOT HALT"); return 2

    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        sparse = sparse.to(device)
    from c4_min.local_attention import install_local_attention
    install_local_attention(sparse, window=args.local_window, drop_local_kv=True,
                            content_bound_global=True, verbose=False)

    def run(evict: bool, evict_schedule: bool, exact_evict: bool):
        # sample max-over-blocks global cache size at each eviction round (POST-drop).
        sizes: List[int] = []
        of, os_ = PS.evict_all_blocks_fused, PS.evict_all_blocks_scheduled

        def _snap(caches):
            live = [c for c in caches if getattr(c, "pos", None) is not None]
            sizes.append(max((int(c.pos.shape[0]) for c in live), default=0))

        def cf(caches, *a, **k):
            r = of(caches, *a, **k); _snap(caches); return r

        def cs(caches, *a, **k):
            r = os_(caches, *a, **k); _snap(caches); return r
        PS.evict_all_blocks_fused = cf
        PS.evict_all_blocks_scheduled = cs
        st = {}; out = []
        try:
            vr = verify_blocks(sparse, L, code, draft, block_steps=args.block_steps,
                               device=device, evict=evict, mask=0xFFFFFFFF, stats=st,
                               fast=True, collect_out=out,
                               evict_interval_steps=args.evict_interval_steps,
                               evict_schedule=evict_schedule, exact_evict=exact_evict)
        finally:
            PS.evict_all_blocks_fused = of
            PS.evict_all_blocks_scheduled = os_
        return vr, st, sizes

    print("  NO-EVICTION (cache grows) ...", flush=True)
    vr_n, st_n, _ = run(evict=False, evict_schedule=False, exact_evict=False)
    print("  CONTENT O(S^2) ...", flush=True)
    vr_c, st_c, sizes_c = run(evict=True, evict_schedule=False, exact_evict=False)
    print("  EXACT O(steps) (no content prune) ...", flush=True)
    vr_x, st_x, sizes_x = run(evict=True, evict_schedule=True, exact_evict=True)

    print(f"\n  no-evict  : matched={vr_n.all_matched} final={vr_n.decoded_final_ax} "
          f"MAX cache={vr_n.max_cache_size}  (== grows to ~seq)", flush=True)
    print(f"  content   : matched={vr_c.all_matched} final={vr_c.decoded_final_ax} "
          f"MAX cache={vr_c.max_cache_size}", flush=True)
    print(f"  EXACT     : matched={vr_x.all_matched} final={vr_x.decoded_final_ax} "
          f"MAX cache={vr_x.max_cache_size}  read_after_free="
          f"{st_x.get('sched_read_after_free')}", flush=True)
    byte_ok = (vr_n.decoded_final_ax == vr_c.decoded_final_ax
               == vr_x.decoded_final_ax == draft.final_ax_masked
               and vr_n.all_matched and vr_c.all_matched and vr_x.all_matched)
    print(f"  BYTE-EXACT (all 3 == draft {draft.final_ax_masked}): {byte_ok}", flush=True)

    def curve(name, sizes):
        if not sizes:
            print(f"  {name} curve: (no eviction rounds)"); return
        show = sizes if len(sizes) <= 24 else (sizes[:12] + ["..."] + sizes[-8:])
        print(f"  {name} per-round KV-size (max over blocks): {show}")
        print(f"        rounds={len(sizes)}  min={min(sizes)} max={max(sizes)} "
              f"final={sizes[-1]}  ceiling={max(sizes)}")
    curve("CONTENT", sizes_c)
    curve("EXACT  ", sizes_x)
    # the headline: EXACT ceiling vs the growing no-evict / seq.
    exact_ceiling = max((x for x in sizes_x if isinstance(x, int)), default=0)
    print(f"\n  --- BOUNDED-KV: EXACT ceiling={exact_ceiling} rows "
          f"(vs no-evict grows to {vr_n.max_cache_size}, seq={1 + draft.step_count*30}); "
          f"EXACT max<=content max: {vr_x.max_cache_size <= vr_c.max_cache_size}", flush=True)
    ok = byte_ok and st_x.get("sched_read_after_free", -1) == 0 and \
        vr_x.max_cache_size <= vr_c.max_cache_size
    print(f"  RESULT: {'OK' if ok else 'MISMATCH'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
