#!/usr/bin/env python3
"""Byte-exact check: dead-block-fusion on the PER-STEP re-embed forward.

Directly compares the model's per-block-stack output on a battery of embedded
frame streams BEFORE vs AFTER installing live-head-attn + dead-block-fusion,
asserting L-inf == 0 (the fused dead blocks are the identity on the residual, and
live blocks score only their live heads — both byte-exact).  Fast (no full
autoregressive reference).

Run: C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_zeroattn_perstep_verify --device cuda:0
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    dev = a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    from . import isa
    from .compact_alloc import build_compact_sparse_streaming
    from .nibble_pure_forward_complete import make_overlay_complete
    from .live_head_attention import (install_live_head_attention,
                                      install_dead_block_fusion,
                                      uninstall_live_head_attention,
                                      uninstall_dead_block_fusion)

    model, L, _ = build_compact_sparse_streaming(code_size=64, compute_mode="dense_kernel")
    if dev != "cpu":
        model = model.to(dev)

    # build a battery of embedded frame streams (a few hundred rows each).
    from ._flop_gauge import _make_stream
    streams = []
    for S in (91, 200, 350):
        toks, store_log = _make_stream(L, S)
        code = isa.assemble([("IMM", 0), ("HALT", 0)])
        overlay = make_overlay_complete(code, L, store_log=store_log)
        x = model.embed[torch.tensor([toks], device=dev)].clone()
        overlay(x)
        streams.append((S, x))

    def _run_full(x):
        with torch.no_grad():
            h = x
            for blk in model.blocks:
                h = blk(h)
            return h

    # baseline (no fusion)
    base = [_run_full(x) for _, x in streams]

    st1 = install_live_head_attention(model, verbose=False)
    st2 = install_dead_block_fusion(model, verbose=False)
    print(f"[per-step] live-head slots={st1['live_head_slots']}/{st1['total_head_slots']}; "
          f"fused {st2['fused_blocks']}/{st2['n_blocks']} dead blocks", flush=True)

    fused = [_run_full(x) for _, x in streams]

    ok = True
    for (S, _), b, f in zip(streams, base, fused):
        linf = float((b - f).abs().max())
        row_ok = linf == 0.0
        ok = ok and row_ok
        print(f"  S={S:4d}  L-inf(base vs fused) = {linf:.3e}  "
              f"{'BYTE-EXACT' if row_ok else 'DIFFERS'}", flush=True)

    uninstall_dead_block_fusion(model)
    uninstall_live_head_attention(model)
    print(f"\n[per-step] {'ALL BYTE-EXACT (L-inf=0) with dead-block-fusion' if ok else 'DIVERGENCE'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
