#!/usr/bin/env python3
"""Compare the STACK0_BYTE0 carrier token (the operand-A value head-0
gathers) between the PASSING or_basic (operand 0x0F) and the FAILING
and_basic/xor_basic (operand 0xFF), to localize whether the value
0xFF -> 0xE8 corruption is in PSH (upstream) or the gather.

Reconstructs head-0's argmax-attended row at the binop AX row and prints
that row's token, so we see EXACTLY which value the gather copies.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_psh_stack0_carrier.py
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
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


PROGRAMS = {
    "or_basic(0x0F)":  _mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x30), Opcode.OR,  Opcode.EXIT]),
    "and_basic(0xFF)": _mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]),
    "or_hi(0xF0)":     _mk([(Opcode.IMM, 0xF0), Opcode.PSH, (Opcode.IMM, 0x0F), Opcode.OR,  Opcode.EXIT]),
    "and_lo(0x0F)":    _mk([(Opcode.IMM, 0x0F), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]),
    "and_70(0x70)":    _mk([(Opcode.IMM, 0x70), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]),
    "and_E0(0xE0)":    _mk([(Opcode.IMM, 0xE0), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]),
}


def head0_attended(model, dp, toks, ax_row, S, dev):
    with torch.no_grad():
        r7 = model.forward(toks, stop_after_block=7)[0]
    block = model.blocks[8]
    attn = block.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    normed = r7
    for attr in ("norm1", "ln1", "attn_norm", "ln_attn"):
        if hasattr(block, attr):
            normed = getattr(block, attr)(r7)
            break
    with torch.no_grad():
        q_all = normed @ attn.W_q.t()
        k_all = normed @ attn.W_k.t()
    qh = q_all[:, 0:HD]
    kh = k_all[:, 0:HD]
    scores = qh[ax_row] @ kh.t()
    slope = float(attn.alibi_slopes[0].item()) if getattr(attn, "alibi_slopes", None) is not None else 0.0
    pos = torch.arange(S, device=dev, dtype=scores.dtype)
    dist = (ax_row - pos).clamp(min=0)
    s = scores - slope * dist
    s[pos > ax_row] = float("-inf")
    w = torch.softmax(s, dim=-1)
    top = torch.topk(w, 3)
    return [(int(i), round(float(v), 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())], slope


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]

    for pname, bc in PROGRAMS.items():
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        se_row = se_rows[-1] if se_rows else None
        ax_row = max((r for r in ax_rows if (se_row is None or r < se_row)),
                     default=(ax_rows[-1] if ax_rows else None))
        top, slope = head0_attended(model, dp, toks, ax_row, S, dev)
        print(f"=== {pname} S={S} ax_row={ax_row} slope={slope:.2f} ===")
        for ri, w in top:
            print(f"    attended row{ri} tok={ctx[ri]}(={hex(ctx[ri])}) w={w}")
        # block8 ALU output
        with torch.no_grad():
            r8 = model.forward(toks, stop_after_block=8)[0]
        def band(name):
            base = dp.get(name)
            return [(i, round(float(r8[ax_row, base+i].item()),2)) for i in range(16) if abs(r8[ax_row, base+i].item())>0.3]
        print(f"    blk8 ALU_LO={band('ALU_LO')} ALU_HI={band('ALU_HI')}")
        print()


if __name__ == "__main__":
    main()
