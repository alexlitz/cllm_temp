#!/usr/bin/env python3
"""Fast check of the 3 residual IMM values + a regression spot-check.
spec_k=0, hook-free."""
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
        bc.append(op[0] | (op[1] << 8) if isinstance(op, tuple) else op)
    return bc


def main():
    probe = build_groundtruth_probe()
    print("=== 3 residual values + neighbors + LEA-relevant controls ===")
    for v in [0xD8, 0xE0, 0xE8, 0xE1, 0xD0, 0xF8, 0x08, 0xFF, 0x70, 0xAB]:
        _, code = probe.emitted_result(_mk([(Opcode.IMM, v), Opcode.EXIT]),
                                        max_steps=10)
        print(f"    IMM {hex(v):>5} -> {hex(code):>8} "
              f"{'OK' if code == v else 'CORRUPT'}")


if __name__ == "__main__":
    main()
