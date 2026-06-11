#!/usr/bin/env python3
"""Dump the token-level CLEAN_EMBED_LO/HI for a sweep of byte values.

This isolates whether the high-nibble mis-encoding lives in the *token
embedding* (model.embed of a bare value token) or is introduced later by
the L5 fetch / STACK0 plumbing.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_token_clean_embed.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from tools.probe_groundtruth import build_groundtruth_probe


def band(row, dp, name, width=16, thr=0.3):
    base = dp.get(name)
    if base is None:
        return None
    cells = [round(float(row[base + i].item()), 2) for i in range(width)]
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device

    # The embedding table: what token id encodes a literal byte value V?
    # Values 0..255 are usually direct vocab entries (vocab=276). Probe the
    # raw embed of token id = V for V in a sweep.
    print("Vocab:", model.embed.embed.weight.shape if hasattr(model.embed, 'embed') else '?')
    vals = list(range(0, 256, 1))
    bad = []
    for v in vals:
        tok = torch.tensor([[v]], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(tok)[0, 0]
        lo = band(emb, dp, "CLEAN_EMBED_LO")
        hi = band(emb, dp, "CLEAN_EMBED_HI")
        exp_lo = v & 0xF
        exp_hi = (v >> 4) & 0xF
        lo_cell = lo[0][0] if len(lo) == 1 else None
        hi_cell = hi[0][0] if len(hi) == 1 else None
        ok = (lo_cell == exp_lo and hi_cell == exp_hi)
        if not ok:
            bad.append((v, exp_lo, exp_hi, lo, hi))
    print(f"Token-embed CLEAN_EMBED mismatches ({len(bad)}/{len(vals)}):")
    for v, el, eh, lo, hi in bad[:40]:
        print(f"  val={hex(v):>5} exp(LO={el},HI={eh})  got LO={lo} HI={hi}")


if __name__ == "__main__":
    main()
