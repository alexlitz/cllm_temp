#!/usr/bin/env python3
"""Locate the TRUE operand-A row and the head-0 attention pattern.

The previous probe found the head-0 gather attends a row whose token is a
stale scratch value (0xE8), not the pushed operand (0xFF). This probe:
  1. lists every row token + its STACK0/AX/SE/PSH markers near the binop
  2. reconstructs head-0's attention weights at the AX row (hook-free:
     recompute QK^T + alibi from the post-block-7 residual and the baked
     W_q/W_k of block 8) to see which row it actually selects and why.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_operand_locate.py [VAL]
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


def main():
    a = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0xFF
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]

    bc = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT])
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
    print(f"A={hex(a)} S={S} ax_row={ax_row} ax_rows={ax_rows} se_rows={se_rows}")

    # markers of interest at residual-before-block-8
    with torch.no_grad():
        r7 = model.forward(toks, stop_after_block=7)[0]
    marker_names = ["MARK_AX", "MARK_STACK0", "STACK0_BYTE0", "MARK_SP",
                    "MARK_BP", "PSH_AT_SP", "OP_PSH", "OP_AND", "CONST"]
    bases = {n: dp.get(n) for n in marker_names}

    # Which rows carry token == a?
    print(f"\nRows with token=={hex(a)}: {[r for r in range(S) if ctx[r]==a]}")
    print(f"\nContext window around ax_row ({ax_row}):")
    lo = max(0, ax_row - 50)
    for r in range(lo, min(S, ax_row + 3)):
        flags = []
        for n in marker_names:
            b = bases[n]
            if b is None:
                continue
            v = r7[r, b].item()
            if abs(v) > 0.3:
                flags.append(f"{n}={round(v,2)}")
        if flags or ctx[r] == a:
            mark = " <== TOK==A" if ctx[r] == a else ""
            print(f"  row{r:4d} tok={ctx[r]:>4}({hex(ctx[r])}) {' '.join(flags)}{mark}")

    # ---- Reconstruct head-0 attention at the AX row in block 8 ----
    block = model.blocks[8]
    attn = block.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    # apply the block's pre-attn layernorm if any
    x = r7  # [S, d]
    # the model forward applies norm before attn; emulate by reading W_q/W_k
    # directly on the residual (alibi handled separately). This is an
    # approximation but argmax is robust.
    with torch.no_grad():
        # try to find the norm
        normed = x
        for attr in ("norm1", "ln1", "attn_norm", "ln_attn"):
            if hasattr(block, attr):
                normed = getattr(block, attr)(x)
                break
        q_all = normed @ attn.W_q.t() if attn.W_q.shape[1] == normed.shape[1] else None
        k_all = normed @ attn.W_k.t() if attn.W_k.shape[1] == normed.shape[1] else None
    if q_all is None:
        print("\n(could not reconstruct QK: W_q shape", attn.W_q.shape, ")")
        return
    h = 0
    qh = q_all[:, h*HD:(h+1)*HD]
    kh = k_all[:, h*HD:(h+1)*HD]
    scores = qh[ax_row] @ kh.t()  # [S]
    # alibi
    slope = float(attn.alibi_slopes[h].item()) if getattr(attn, "alibi_slopes", None) is not None else 0.0
    pos = torch.arange(S, device=dev, dtype=scores.dtype)
    dist = (ax_row - pos).clamp(min=0)
    scores_alibi = scores - slope * dist
    # causal mask
    scores_alibi[pos > ax_row] = float("-inf")
    w = torch.softmax(scores_alibi, dim=-1)
    top = torch.topk(w, 8)
    print(f"\nhead-0 slope={slope}; top attended rows from AX row {ax_row}:")
    for wi, ri in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   row{ri:4d} tok={ctx[ri]:>4}({hex(ctx[ri])}) w={round(wi,3)} "
              f"rawscore={round(float(scores[ri]),2)} dist={ax_row-ri}")


if __name__ == "__main__":
    main()
