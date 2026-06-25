#!/usr/bin/env python3
"""Trace OUTPUT band evolution at id325 step-14 AX byte1/2/3 dump rows per-block.

Find WHERE (which block) the sign-extension byte (want 0xFF) gets overwritten to
0x00, and identify a stable discriminator at the byte-1/2/3 predictor rows that
says "this AX dump's byte-0 is a negative LEA-local frame address".
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 50; x = x + 7; return x; }"
STEP = 14


def nib(row, base, dp):
    """Return (lo_argmax, hi_argmax, lo_max_val, hi_max_val) of a byte band."""
    bl = dp.get(base + "_LO") if isinstance(base, str) else None
    return None


def band_argmax(row, base, dp, width=16):
    b = dp.get(base)
    if b is None:
        return None
    vals = [float(row[b + i].item()) for i in range(width)]
    am = max(range(width), key=lambda i: vals[i])
    return am, vals[am], vals


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEPT = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)

    bc, _ = compile_c(SRC)
    ctx = p._final_context(bc, max_steps=30)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    base = prefix_len + STEP * STEPT
    seg = ctx[base:base + STEPT]
    ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
    ax_marker_row = base + ax_i
    rows = {bi: ax_marker_row + bi for bi in range(4)}

    nblocks = len(model.blocks)
    print(f"nblocks={nblocks} ax_marker_row={ax_marker_row}")
    # Per-block: report OUTPUT_LO/HI argmax at each byte row.
    for b in range(nblocks):
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        line = [f"blk{b:2d}"]
        for bi in range(4):
            row = resid[rows[bi]]
            lo = band_argmax(row, "OUTPUT_LO", dp)
            hi = band_argmax(row, "OUTPUT_HI", dp)
            if lo and hi:
                line.append(f"b{bi}:LO{lo[0]:2d}({lo[1]:6.0f})HI{hi[0]:2d}({hi[1]:6.0f})")
        print("  ".join(line))


if __name__ == "__main__":
    main()
