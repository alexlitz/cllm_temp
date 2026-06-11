#!/usr/bin/env python3
"""Verify the IMM-decode fix: sweep all 256 immediates through
``IMM v; EXIT`` and report mis-decodes, then check the binop cluster.

spec_k=0, hook-free.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/probe_imm_sweep_verify.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

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


def main():
    probe = build_groundtruth_probe()
    print("=== IMM v; EXIT  full 0..255 sweep ===")
    bad = []
    for v in range(256):
        _, code = probe.emitted_result(_mk([(Opcode.IMM, v), Opcode.EXIT]),
                                        max_steps=10)
        if code != v:
            bad.append((v, code))
    print(f"    mis-decodes: {len(bad)}/256")
    # group by family
    lo8 = [(v, c) for v, c in bad if (v & 0xF) == 8]
    hiEF = [(v, c) for v, c in bad if (v >> 4) in (0xE, 0xF)]
    other = [(v, c) for v, c in bad
             if (v & 0xF) != 8 and (v >> 4) not in (0xE, 0xF)]
    print(f"    lo-nibble-8 family bad: {len(lo8)}  "
          f"{[(hex(v), hex(c)) for v, c in lo8[:8]]}")
    print(f"    hi-nibble-E/F family bad: {len(hiEF)}  "
          f"{[(hex(v), hex(c)) for v, c in hiEF[:8]]}")
    print(f"    OTHER (new regressions?): {len(other)}  "
          f"{[(hex(v), hex(c)) for v, c in other]}")
    print()

    print("=== spot-check the formerly-corrupt values ===")
    for v in [0x08, 0x18, 0x28, 0x38, 0x48, 0x88, 0xD8,
              0xE0, 0xE5, 0xEF, 0xF0, 0xFF]:
        _, code = probe.emitted_result(_mk([(Opcode.IMM, v), Opcode.EXIT]),
                                        max_steps=10)
        print(f"    IMM {hex(v):>5} -> {hex(code):>8} "
              f"{'OK' if code == v else 'STILL CORRUPT'}")
    print()

    print("=== binop cluster (post-IMM-fix) ===")
    cases = [
        ("AND", Opcode.AND, 0x0F, 0x30, 0x0F & 0x30),
        ("AND", Opcode.AND, 0x70, 0x2A, 0x70 & 0x2A),
        ("AND", Opcode.AND, 0xFF, 0x2A, 0xFF & 0x2A),
        ("OR", Opcode.OR, 0x0F, 0x30, 0x0F | 0x30),
        ("XOR", Opcode.XOR, 0x0F, 0x30, 0x0F ^ 0x30),
        ("MUL", Opcode.MUL, 6, 7, 42),
    ]
    for nm, op, a, b, exp in cases:
        _, code = probe.emitted_result(
            _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), op, Opcode.EXIT]),
            max_steps=20)
        print(f"    {hex(a)} {nm} {hex(b)} -> {hex(code):>8} exp {hex(exp)} "
              f"{'OK' if code == exp else 'CORRUPT'}")


if __name__ == "__main__":
    main()
