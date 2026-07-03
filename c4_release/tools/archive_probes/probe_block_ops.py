#!/usr/bin/env python3
"""Print per-physical-block FFN op identity + which OUTPUT cells each block
writes on the IMM 0xFF; EXIT AX row (block-by-block OUTPUT delta).

Identifies the op at block 30 (lev_routing?) and block 36 (the final 0xE8
authoritative writer).  spec_k=0, hook-free.
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


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device

    # Per-block op metadata
    print("=== block FFN identity ===")
    for phys, blk in enumerate(model.blocks):
        ffn = getattr(blk, "ffn", None)
        op_names = getattr(ffn, "_op_names", None) or getattr(blk, "_op_name", None)
        lg = getattr(blk, "_logical_layer", phys)
        hd = getattr(ffn, "hidden_dim", "?")
        # try to find any op-name tag
        tags = []
        for attr in ("_op_name", "_op_names", "_source_ops", "_ops", "name"):
            v = getattr(blk, attr, None) or getattr(ffn, attr, None)
            if v:
                tags.append(f"{attr}={v}")
        print(f"  blk{phys:>2} L{lg:>2} ffn={type(ffn).__name__} hidden={hd} "
              f"{' '.join(tags)}")


if __name__ == "__main__":
    main()
