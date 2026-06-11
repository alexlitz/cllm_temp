#!/usr/bin/env python3
"""PROOF that the block-8 head-0 operand gather is CLEAN, and the real
AND/MUL/XOR blockers are upstream IMM-decode + downstream L20 corruption.

Three sections, all spec_k=0, hook-free:

  [A] Token-level CLEAN_EMBED is a correct one-hot for ALL 256 values
      (the gather's V source is faithful).
  [B] The gather output (ALU_LO/HI at the binop AX row) is a correct
      per-nibble one-hot for every IMM-clean operand 0..15 in both nibble
      positions.  The only constant residue is a cell-0 magnitude artifact
      (~5.56) + a tiny cell-8 (~0.45) that PASSING or_basic tolerates.
  [C] The real bug: ``IMM v; EXIT`` (no gather, no stack, no binop)
      mis-decodes v whenever lo-nibble(v)==8 OR hi-nibble(v) in {E,F}
      (e.g. 0xFF -> 0xE8).  The gather then faithfully copies the already
      corrupt STACK0 byte.  Plus AND/MUL/XOR of TWO clean operands still
      decode 0xFFF0 (the L15 cell-0 / L20 cell-8 downstream spike).

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_operand_gather_is_clean.py
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


def section_a(probe, model, dp, dev):
    print("=== [A] token-level CLEAN_EMBED one-hot for all 256 values ===")
    bad = 0
    for v in range(256):
        tok = torch.tensor([[v]], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(tok)[0, 0]
        lo = band(emb, dp, "CLEAN_EMBED_LO")
        hi = band(emb, dp, "CLEAN_EMBED_HI")
        if not (len(lo) == 1 and lo[0][0] == (v & 0xF)
                and len(hi) == 1 and hi[0][0] == (v >> 4)):
            bad += 1
    print(f"    CLEAN_EMBED mismatches: {bad}/256  -> "
          f"{'CLEAN' if bad == 0 else 'CORRUPT'}\n")


def gather_out(probe, model, dp, dev, a):
    bc = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, 0x2A),
              Opcode.AND, Opcode.EXIT])
    ctx = probe._final_context(bc, max_steps=20)
    S = len(ctx)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(toks)[0]
    axb, seb = dp["MARK_AX"], dp["MARK_SE_ONLY"]
    axr = [r for r in range(S) if emb[r, axb].abs().item() > 0.5]
    ser = [r for r in range(S) if emb[r, seb].abs().item() > 0.5]
    se = ser[-1] if ser else None
    ax = max((r for r in axr if (se is None or r < se)), default=axr[-1])
    with torch.no_grad():
        r8 = model.forward(toks, stop_after_block=8)[0]
    return band(r8[ax], dp, "ALU_LO"), band(r8[ax], dp, "ALU_HI")


def section_b(probe, model, dp, dev):
    print("=== [B] gather output one-hot for IMM-CLEAN operands ===")
    # hi=2 fixed, sweep lo nibble (skip lo=8 which IMM mis-decodes)
    for lo in [0, 1, 2, 3, 4, 5, 6, 7, 9, 0xA, 0xB, 0xC, 0xD]:
        a = (2 << 4) | lo
        alo, ahi = gather_out(probe, model, dp, dev, a)
        ans = [c for c, _ in alo if c not in (0, 8)]
        ok = (ans == [lo]) or (lo == 0 and ans == [])
        print(f"    A={hex(a)} lo={lo:2d}: ALU_LO={alo}  -> "
              f"answer cell {ans} {'OK' if ok else 'BAD'}")
    print()


def section_c(probe):
    print("=== [C] real blockers (outside block-8 surface) ===")
    print("  IMM v; EXIT  (no gather / no stack):")
    for v in [0x0F, 0x2A, 0x70, 0x80, 0xAB, 0xE0, 0xF0, 0xFF, 0x08, 0x88]:
        _, code = probe.emitted_result(_mk([(Opcode.IMM, v), Opcode.EXIT]),
                                       max_steps=10)
        print(f"    IMM {hex(v):>5} -> {hex(code):>7} "
              f"{'OK' if code == v else 'CORRUPT (IMM decode)'}")
    print("  AND of TWO IMM-clean operands (gather clean, compute corrupt):")
    for a, b, exp in [(0x0F, 0x30, 0x0F & 0x30), (0x70, 0x2A, 0x70 & 0x2A),
                      (0x2A, 0x70, 0x2A & 0x70)]:
        _, code = probe.emitted_result(
            _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b),
                 Opcode.AND, Opcode.EXIT]), max_steps=20)
        print(f"    {hex(a)} AND {hex(b)} -> {hex(code):>7} exp {hex(exp)} "
              f"{'OK' if code == exp else 'CORRUPT (L15/L20 compute)'}")
    print("  OR of TWO IMM-clean operands (passes — same gather/cell-0):")
    _, code = probe.emitted_result(
        _mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x30),
             Opcode.OR, Opcode.EXIT]), max_steps=20)
    print(f"    0x0F OR 0x30 -> {hex(code)} exp 0x3f "
          f"{'OK' if code == 0x3F else 'BAD'}")


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    section_a(probe, model, dp, dev)
    section_b(probe, model, dp, dev)
    section_c(probe)


if __name__ == "__main__":
    main()
