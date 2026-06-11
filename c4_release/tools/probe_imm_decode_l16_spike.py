#!/usr/bin/env python3
"""Diagnose the IMM-decode mis-decode (lo-nibble-8 + hi-nibble-{E,F})
and pin WHICH L16 lev_routing rule spikes OUTPUT cell 8 on a plain
``IMM v; EXIT`` row.

spec_k=0, hook-free.  Uses the GroundTruthProbe residual_at path.

Sections:
  [1] Confirm the bug: IMM v -> emitted code for the affected family.
  [2] Block-by-block OUTPUT_LO/HI trace at the AX decode row -> find the
      physical block where cell 8 spikes.
  [3] At the spike block input (residual after the PRIOR block), evaluate
      every candidate l16 rule's condition-AND score vs threshold on the
      IMM-EXIT row, to pin which rule(s) actually fire.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/probe_imm_decode_l16_spike.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            bc.append(op[0] | (op[1] << 8))
        else:
            bc.append(op)
    return bc


def band(row, dp, name, thr=0.3):
    b = dp.get(name)
    if b is None:
        return None
    return [(i, round(float(row[b + i].item()), 2)) for i in range(16)
            if abs(row[b + i].item()) > thr]


def find_ax_row(model, dp, dev, ctx):
    """Return the AX decode row index (last MARK_AX row before final SE)."""
    S = len(ctx)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(toks)[0]
    axb, seb = dp["MARK_AX"], dp["MARK_SE_ONLY"]
    axr = [r for r in range(S) if emb[r, axb].abs().item() > 0.5]
    ser = [r for r in range(S) if emb[r, seb].abs().item() > 0.5]
    se = ser[-1] if ser else None
    if not axr:
        return S - 1
    ax = max((r for r in axr if (se is None or r < se)), default=axr[-1])
    return ax


def section1(probe):
    print("=== [1] IMM v; EXIT decode (affected families) ===")
    fam = [0x00, 0x0F, 0x2A, 0x70, 0x80, 0xAB,        # OK control
           0x08, 0x18, 0x28, 0x38, 0x48, 0x88, 0xD8,  # lo-nibble-8
           0xE0, 0xE5, 0xEF, 0xF0, 0xFF]              # hi-nibble E/F
    bad = []
    for v in fam:
        _, code = probe.emitted_result(_mk([(Opcode.IMM, v), Opcode.EXIT]),
                                       max_steps=10)
        ok = code == v
        if not ok:
            bad.append(v)
        print(f"    IMM {hex(v):>5} -> {hex(code):>8} {'OK' if ok else 'CORRUPT'}")
    print(f"    -> {len(bad)} corrupt in sample: {[hex(x) for x in bad]}\n")


def section2(probe, model, dp, dev):
    print("=== [2] OUTPUT cell trace at AX decode row (IMM 0xFF; EXIT) ===")
    bc = _mk([(Opcode.IMM, 0xFF), Opcode.EXIT])
    ctx = probe._final_context(bc, max_steps=10)
    ax = find_ax_row(model, dp, dev, ctx)
    print(f"    ctx len={len(ctx)} ax_row={ax}")
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    nblocks = len(model.blocks)
    prev8lo = prev8hi = 0.0
    for blk in range(nblocks):
        with torch.no_grad():
            r = model.forward(toks, stop_after_block=blk)[0]
        row = r[ax]
        lo = band(row, dp, "OUTPUT_LO")
        hi = band(row, dp, "OUTPUT_HI_THIS_STEP")
        c8lo = round(float(row[dp["OUTPUT_LO"] + 8].item()), 1)
        c8hi = round(float(row[dp["OUTPUT_HI_THIS_STEP"] + 8].item()), 1)
        c14hi = round(float(row[dp["OUTPUT_HI_THIS_STEP"] + 14].item()), 1)
        spike = "  <== cell8 jump" if abs(c8lo - prev8lo) > 5 or abs(c8hi - prev8hi) > 5 else ""
        prev8lo, prev8hi = c8lo, c8hi
        lg = getattr(model.blocks[blk], "_logical_layer", blk)
        print(f"    blk{blk:>2} L{lg:>2}: OLO8={c8lo:>9} OHI8={c8hi:>9} "
              f"OHI14={c14hi:>9}{spike}")
        if blk >= 8:
            print(f"           OLO={lo}")
            print(f"           OHI={hi}")
    print()


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    section1(probe)
    section2(probe, model, dp, dev)


if __name__ == "__main__":
    main()
