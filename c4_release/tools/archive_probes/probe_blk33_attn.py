#!/usr/bin/env python3
"""Decompose block-33 OUTPUT_LO/HI change into attention vs FFN, and if
attention, find which head + which K row it copies from.

Usage: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       python tools/probe_blk33_attn.py 816
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token

BLK = int(os.environ.get("BLK", "33"))


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def _ax_rows(ctx):
    return [i for i, t in enumerate(ctx) if t == Token.REG_AX]


def main(idx, mul_step):
    p = build_groundtruth_probe()
    dp = p.model.dim_positions
    OL = int(dp["OUTPUT_LO"]); OH = int(dp["OUTPUT_HI"])
    bc, exp, desc = _corpus(idx)
    ctx = p._final_context(bc, max_steps=12)
    pos = _ax_rows(ctx)[mul_step]
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    print(f"=== id={idx} {desc} blk{BLK} MUL row pos={pos} ===")

    block = p.model.blocks[BLK]
    # input to block
    xin = p.model.forward(padded, stop_after_block=BLK - 1)  # [1,S,D]
    def cell(t, base):
        r = t[0, pos, base:base+16]
        r = (r.to_dense() if r.is_sparse else r).float().cpu()
        v = r.tolist(); m = max(range(16), key=lambda i: v[i])
        return m, round(v[m], 2)
    with torch.no_grad():
        post_attn = block.attn(xin)        # returns new x (residual added)
        post_ffn = block.ffn(post_attn)
    print("  block INPUT  OUTPUT_LO", cell(xin, OL), "OUTPUT_HI", cell(xin, OH))
    print("  post-ATTN    OUTPUT_LO", cell(post_attn, OL), "OUTPUT_HI", cell(post_attn, OH))
    print("  post-FFN     OUTPUT_LO", cell(post_ffn, OL), "OUTPUT_HI", cell(post_ffn, OH))
    full = p.model.forward(padded, stop_after_block=BLK)
    print("  post-BLOCK   OUTPUT_LO", cell(full, OL), "OUTPUT_HI", cell(full, OH))


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 816
    # add_mul MUL=step5; standalone mul (100-149) MUL=step3
    default = 3 if 100 <= idx < 150 else 5
    mul_step = {816: 5, 800: 5, 850: 3, 825: 6}.get(idx, default)
    main(idx, mul_step)
