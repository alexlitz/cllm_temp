#!/usr/bin/env python3
"""Inspect the BUILT L15 attention head count + LEV head activity.
spec_k=0, hook-free.  Usage: python tools/_probe_l15_heads.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
import torch  # noqa


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    nb = len(model.blocks)
    print(f"n_blocks (physical) = {nb}")
    # Find the L15 memory_lookup block: it's the one with the most heads
    # in the late region. Print num_heads for each block.
    for bi, blk in enumerate(model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        nh = getattr(attn, "num_heads", None)
        wq = attn.W_q.shape if hasattr(attn, "W_q") else None
        # Identify by nonzero alibi pin of 0.05 (L15 load heads pinned to 0.05)
        slopes = getattr(attn, "alibi_slopes", None)
        slope_str = ""
        if slopes is not None:
            sv = [round(float(s), 4) for s in slopes.tolist()]
            slope_str = f" alibi={sv}"
        print(f"  block {bi:2d}: num_heads={nh} W_q={tuple(wq) if wq else None}{slope_str}")


if __name__ == "__main__":
    main()
