#!/usr/bin/env python3
"""Identify which op owns blocks 35 and 45 (the byte-1 0xFF writer + the nuke),
and dump candidate sign-extension discriminator dims at the byte-1/2/3 rows.
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


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEPT = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)
    nblocks = len(model.blocks)
    print(f"nblocks={nblocks}")
    # Try to get block->name map
    for attr in ("block_op_names", "_block_op_names", "block_names", "op_names"):
        v = getattr(model, attr, None)
        if v is not None:
            print(f"model.{attr} = {v}")
    # Per-block try to read a 'name' attribute
    for b, blk in enumerate(model.blocks):
        nm = getattr(blk, "name", None) or getattr(blk, "_name", None)
        post = getattr(blk, "post_ops", None)
        npost = len(post) if post is not None else 0
        if b in range(33, 48):
            print(f"  block {b}: name={nm} post_ops={npost}")

    bc, _ = compile_c(SRC)
    ctx = p._final_context(bc, max_steps=30)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    base = prefix_len + STEP * STEPT
    seg = ctx[base:base + STEPT]
    ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
    ax_marker_row = base + ax_i

    # Candidate discriminator dims at byte-1/2/3 rows at block 44 (just before nuke).
    cands = ["OP_LEA", "HAS_SE", "IS_BYTE", "MARK_AX", "MARK_SE",
             "SE_NEG", "SIGN", "NEG", "OUTPUT_HI", "OUTPUT_LO",
             "AX_CARRY_LO", "AX_CARRY_HI", "ALU_HI", "ALU_LO",
             "OUTPUT_HI_THIS_STEP", "FETCH_HI", "ADDR_B0_HI"]
    for blk in (35, 41, 44, 45):
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=blk)[0]
        print(f"\n=== block {blk} byte-rows ===")
        for bi in range(4):
            row = resid[ax_marker_row + bi]
            parts = []
            for nm in cands:
                b = dp.get(nm)
                if b is None:
                    continue
                # show scalar if single, else argmax of 16-wide band
                if nm.startswith(("OUTPUT", "ALU", "AX_CARRY", "FETCH", "ADDR")):
                    vals = [float(row[b + i].item()) for i in range(16)]
                    am = max(range(16), key=lambda i: abs(vals[i]))
                    parts.append(f"{nm}[am{am}={vals[am]:.1f}]")
                else:
                    parts.append(f"{nm}={float(row[b].item()):.2f}")
            print(f"  byte{bi}: " + " ".join(parts))


if __name__ == "__main__":
    main()
