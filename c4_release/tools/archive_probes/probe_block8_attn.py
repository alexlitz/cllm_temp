#!/usr/bin/env python3
"""Inspect block-8 operand-gather head-0 attention + V source, spec_k=0.

Reconstructs head-0's attention weights over the binop-step AX query row by
reading the PRE-block-8 residual (stop_after_block=7) for all rows, applying
the block's LayerNorm + head-0 W_q/W_k + alibi, softmax, and reporting the
top attended source rows and their CLEAN_EMBED_LO/HI + SP_OLD residual cells.
This shows WHERE the @0 magnitude artifact in ALU_LO/HI comes from.

Hook-free: uses probe_groundtruth.residual_at-style truncated forward to get
the block-input residual, then does the head math in numpy/torch directly.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


PROG = _mk([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42),
            Opcode.EQ, Opcode.EXIT])


def main(block=8):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]

    ctx = probe._final_context(PROG, max_steps=20)
    S = len(ctx)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(toks)[0]
    ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
    q_row = ax_rows[-1]

    # Block-input residual = output of block (block-1), i.e. stop_after_block=block-1.
    with torch.no_grad():
        resid = model.forward(toks, stop_after_block=block - 1)[0]  # [S, D]

    import torch.nn.functional as F
    blk = model.blocks[block]
    attn = blk.attn
    # AutoregressiveAttention.forward: no LayerNorm, alibi bias, softmax1.
    x = resid.unsqueeze(0)  # [1, S, D]
    H = attn.num_heads
    HD = attn.head_dim
    head = 0
    with torch.no_grad():
        Q = F.linear(x, attn.W_q).view(1, S, H, HD).transpose(1, 2)  # [1,H,S,HD]
        K = F.linear(x, attn.W_k).view(1, S, H, HD).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale  # [1,H,S,S]
        slopes = getattr(attn, "alibi_slopes", None)
        pos = torch.arange(S, device=dev).float()
        if slopes is not None:
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # [S,S]
            scores = scores - slopes.view(1, H, 1, 1) * dist.view(1, 1, S, S)
        causal = torch.triu(torch.full((S, S), float("-inf"), device=dev),
                            diagonal=1)
        scores = scores + causal.view(1, 1, S, S)
        # softmax1: append a zero-logit sink column.
        if getattr(attn, "use_softmax1", False):
            sink = torch.zeros(1, H, S, 1, device=dev)
            ext = torch.cat([scores, sink], dim=-1)  # [1,H,S,S+1]
            w_ext = F.softmax(ext, dim=-1)
            w_all = w_ext[..., :S]  # drop sink weight
        else:
            w_all = F.softmax(scores, dim=-1)
    w = w_all[0, head, q_row]  # [S] attention from q_row (may sum<1 w/ softmax1)
    raw = scores[0, head, q_row]  # raw logit (alibi+causal incl.)
    # raw QK only (no alibi/causal) for component view
    with torch.no_grad():
        qk = torch.matmul(Q, K.transpose(-2, -1))[0, head, q_row] * attn.scale
    top = torch.topk(w, k=min(10, S))
    print(f"q_row={q_row} S={S} head={head} slope="
          f"{float(slopes[head]) if slopes is not None else None} "
          f"sum_w={float(w.sum()):.3f}")
    ce_lo = dp["CLEAN_EMBED_LO"]
    ce_hi = dp["CLEAN_EMBED_HI"]
    for w_i, r in zip(top.values.tolist(), top.indices.tolist()):
        lo = [round(float(resid[r, ce_lo + k]), 2) for k in range(16)]
        hi = [round(float(resid[r, ce_hi + k]), 2) for k in range(16)]
        lo_nz = [f"{v}@{k}" for k, v in enumerate(lo) if abs(v) > 0.3]
        hi_nz = [f"{v}@{k}" for k, v in enumerate(hi) if abs(v) > 0.3]
        tok = ctx[r]
        print(f"  w={w_i:.3f} qk={float(qk[r]):.2f} raw={float(raw[r]):.2f} "
              f"row={r} tok={tok} CE_LO={lo_nz} CE_HI={hi_nz}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
