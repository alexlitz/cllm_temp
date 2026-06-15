#!/usr/bin/env python3
"""Decode the byte-1 emit: logits at the generating position + OUTPUT band
trace across blocks at THAT position (the position whose next-token is b1).

In the AX register section the tokens are: REG_AX, byte0, byte1, byte2, byte3.
Token byte1 (= ctx[ax_pos+2]) is produced by the argmax of logits at
position ax_pos+1 (the byte0 token) when the context length is ax_pos+2.
So we read residual/logits at row (ax_pos+1) over the TRUNCATED context
ctx[:ax_pos+2].

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/probe_mul_byte1_logits.py A B
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
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def prog(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL, Opcode.EXIT])


def argmax_nib(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    cells = [float(row[base + i].item()) for i in range(width)]
    mx = max(range(width), key=lambda i: cells[i])
    return mx, round(cells[mx], 2)


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)

    bc = prog(a, b)
    ctx_full = probe._final_context(bc, max_steps=20)
    exp = a * b
    ax_pos = None
    for i in range(len(ctx_full) - 1, -1, -1):
        if ctx_full[i] == int(Token.REG_AX):
            ax_pos = i
            break
    # byte1 is generated at position (ax_pos+1) with context ctx[:ax_pos+2]
    gen_pos = ax_pos + 1
    trunc = ctx_full[:ax_pos + 2]
    print(f"A={a} B={b} exp=0x{exp:04x} b1=0x{(exp>>8)&0xff:02x} ax_pos={ax_pos} "
          f"gen_pos(byte0_token)={gen_pos} trunc_len={len(trunc)} emitted_b1=0x{ctx_full[ax_pos+2]&0xff:02x}")

    toks = torch.tensor([trunc], dtype=torch.long, device=dev)
    # full logits at gen_pos (model.forward returns [1,S,V] logits)
    with torch.no_grad():
        logits = model.forward(toks)[0]
    top = torch.topk(logits[gen_pos], 6)
    print(f"  top-6 logits @gen_pos: " +
          ", ".join(f"tok{int(i)}={float(v):.2f}" for v, i in zip(top.values, top.indices)))
    print(f"  argmax token = {int(logits[gen_pos].argmax())}")

    prev = None
    for blk in range(nblocks):
        with torch.no_grad():
            r = model.forward(toks, stop_after_block=blk)[0]
        lg = getattr(model.blocks[blk], "_logical_layer", blk)
        ol = argmax_nib(r[gen_pos], dp, 'OUTPUT_LO')
        oh = argmax_nib(r[gen_pos], dp, 'OUTPUT_HI')
        afl = argmax_nib(r[gen_pos], dp, 'AX_FULL_LO')
        afh = argmax_nib(r[gen_pos], dp, 'AX_FULL_HI')
        acl = argmax_nib(r[gen_pos], dp, 'AX_CARRY_LO')
        ach = argmax_nib(r[gen_pos], dp, 'AX_CARRY_HI')
        byte = (oh[0] << 4) | ol[0]
        cur = (ol, oh)
        marker = " <==OUT CHANGED" if cur != prev else ""
        print(f"  blk{blk:2d}(L{lg:2d}) OUT_LO={str(ol):12s} OUT_HI={str(oh):12s} "
              f"AXF_LO={str(afl):11s} AXF_HI={str(afh):11s} AXC_LO={str(acl):11s} AXC_HI={str(ach):11s} "
              f"byte0x{byte:02x}{marker}")
        prev = cur


if __name__ == "__main__":
    main()
