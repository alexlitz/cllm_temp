#!/usr/bin/env python3
"""_agent_attn_audit.py — AUDIT the residual ATTENTION ARITHMETIC in the fully
composed doom forward (direct-CAM + direct-local + dead-block-fusion + banded).

Pins WHERE score matmuls / softmax1 still run after all the lookup levers are on,
by monkeypatching torch.einsum / softmax1 with call counters.  LEAN STREAMING
(C4_PF_CFM=1); no dense load.

Run:
    cd c4_release
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_attn_audit --device cuda:0
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from collections import defaultdict

import torch


def _mem_avail_gb() -> float:
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--block-moe", action="store_true",
                    help="enable batched-block-skip (skip dead divmod span per non-divmod span)")
    ap.add_argument("--dead-fusion", action="store_true",
                    help="install live-head-attn + dead-block-fusion into the composed forward")
    a = ap.parse_args(argv)

    _mem_guard("startup")
    dev = a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc

    os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
    os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "1"
    os.environ["C4_DIRECT_CAM_LIVE_LOCAL"] = "1"
    os.environ["C4_FROZEN_ROW_SKIP"] = "1"

    print(f"[build] lean streaming dev={dev} MemAvail={_mem_avail_gb():.1f}GB", flush=True)
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    if dev != "cpu":
        model = model.to(dev)
    _mem_guard("post-build")

    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)

    if a.block_moe:
        os.environ["C4_BATCHED_BLOCK_SKIP"] = "1"

    if a.dead_fusion:
        from .live_head_attention import (install_live_head_attention,
                                          install_dead_block_fusion)
        st1 = install_live_head_attention(model, verbose=False)
        st2 = install_dead_block_fusion(model, verbose=False)
        print(f"[dead-fusion] live-head-attn scored={st1['live_head_slots']}/"
              f"{st1['total_head_slots']} slots; dead-block-fusion bypassed "
              f"{st2['fused_blocks']}/{st2['n_blocks']} blocks", flush=True)

    names = list(getattr(L, "_block_names", []))
    print(f"[build] {len(model.blocks)} blocks dim={model.embed.shape[1]}", flush=True)

    from .local_attention import live_value_heads, _ingest_slope
    from .direct_cam_batched import cam_head_map
    ings = _ingest_slope()
    chm = cam_head_map(model, L)
    print("\n=== LIVE-ATTENTION BLOCK MAP (post-compose) ===", flush=True)
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        live = live_value_heads(at)
        if not live:
            continue
        nm = names[bi] if bi < len(names) else "?"
        ing = [h for h in live if abs(float(at.alibi_slopes[h]) - ings) <= 1e-3]
        glob = [h for h in live if h not in ing]
        is_cam = bi in chm
        print(f"  blk {bi:3d} {nm:20s} live={len(live):2d} ingest-local={len(ing):2d} "
              f"other/global={len(glob):2d} CAM={is_cam} "
              f"cam_heads={[h for h,_ in chm.get(bi,[])]}", flush=True)

    import c4_min.banded_local_attn as BLA
    import c4_min.blogspec_model as BM
    _orig_einsum = torch.einsum
    _orig_softmax1 = BM.softmax1
    counters = {"banded_einsum": 0, "banded_flop": 0.0,
                "softmax1": 0, "softmax1_elems": 0.0}
    per_blk_softmax = defaultdict(int)      # block_name -> softmax1 call count
    per_blk_flop = defaultdict(float)

    # tag the active block by wrapping each block's attn.forward
    active = {"blk": "?"}

    def _wrap_attn(bi, nm):
        at = model.blocks[bi].attn
        orig = at.forward
        def wrapped(x, *args, **kw):
            prev = active["blk"]
            active["blk"] = f"{bi}:{nm}"
            try:
                return orig(x, *args, **kw)
            finally:
                active["blk"] = prev
        at.forward = wrapped
    for bi in range(len(model.blocks)):
        _wrap_attn(bi, names[bi] if bi < len(names) else "?")

    def _einsum(eq, *ops):
        if eq in ("bhqd,bhqwd->bhqw", "bhqw,bhqwd->bhqd"):
            counters["banded_einsum"] += 1
            f = 2.0 * float(ops[0].numel()) * float(ops[1].shape[-1])
            counters["banded_flop"] += f
            per_blk_flop[active["blk"]] += f
        return _orig_einsum(eq, *ops)

    def _softmax1(x, dim=-1):
        counters["softmax1"] += 1
        counters["softmax1_elems"] += float(x.numel())
        per_blk_softmax[active["blk"]] += 1
        return _orig_softmax1(x, dim=dim)

    torch.einsum = _einsum
    BM.softmax1 = _softmax1
    BLA.softmax1 = _softmax1
    import c4_min.direct_cam_batched as DCB
    import c4_min.local_attention as LAT
    if hasattr(DCB, "softmax1"):
        DCB.softmax1 = _softmax1
    if hasattr(LAT, "softmax1"):
        LAT.softmax1 = _softmax1

    code = build_malloc(16)[0]
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    assert draft.halted
    print(f"\n[run] malloc draft steps={draft.step_count} @ K={a.K}", flush=True)

    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=a.K, device=dev,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True)
    print(f"\n[run] all_matched={vr.all_matched} accepted={vr.accepted_steps}/"
          f"{draft.step_count} final_ax={vr.decoded_final_ax}", flush=True)

    torch.einsum = _orig_einsum
    BM.softmax1 = _orig_softmax1

    print("\n=== RESIDUAL ATTENTION ARITHMETIC (composed forward) ===", flush=True)
    print(f"  banded-local einsum calls : {counters['banded_einsum']:6d}  "
          f"flop={counters['banded_flop']/1e9:.3f} G", flush=True)
    print(f"  softmax1 calls            : {counters['softmax1']:6d}  "
          f"elems={counters['softmax1_elems']/1e6:.3f} M", flush=True)
    print("\n  -> softmax1 calls > 0 means residual attention softmax still runs.",
          flush=True)
    print("  -> banded einsum > 0 means the local-ingest score+ctx matmul still runs.",
          flush=True)

    print("\n=== PER-BLOCK ATTRIBUTION (where the residual attn compute lives) ===",
          flush=True)
    for blk in sorted(per_blk_softmax, key=lambda k: -per_blk_softmax[k]):
        print(f"  {blk:24s} softmax1={per_blk_softmax[blk]:5d} "
              f"banded_flop={per_blk_flop.get(blk,0.0)/1e9:.3f} G", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
