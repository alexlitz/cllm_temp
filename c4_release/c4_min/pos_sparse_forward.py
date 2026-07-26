"""STEP 2 — POSITION-LEVEL sparsity on top of the op-level block-skip.

STEP-1 measured (``_pos_sparse_measure``) that EVERY op-level-live block affects
the decoded registers at exactly ONE stream position — the QUERY ROW (the last
token, which the driver decodes) — and is a nil passthrough at all the other ~S-1
positions.  The redundancy factor is therefore ~S (the block is computed at all S
positions but only the query row matters), and for the 186-block DIV/MOD span that
means the heavy long-division gadget is computed ~S times more than needed.

This module runs each block computing the DECODE-RELEVANT work at ONLY the query
row.  A transformer block is ``x + Attn(x)`` then ``a + FFN(a)``:

  * FFN is POSITION-INDEPENDENT: FFN(a)[q] depends only on a[q].  So we compute the
    FFN at the query row alone; every other position carries ``a`` unchanged (its
    exact passthrough), byte-exact for those rows.
  * ATTENTION mixes positions: the query row's output reads K/V from ALL positions.
    So K and V (the cheap projections) run at every position, but Q, the score /
    softmax / context, and the W_o output projection run for the SINGLE query row.
    Prior rows carry ``x`` unchanged (their attention output does not feed the
    decode — proven by STEP-1's per-position ablation).

Because prior rows are frozen at their INPUT residual, a block's FFN/attn-out no
longer updates the K/V those prior rows would contribute to the NEXT block's query
attention.  STEP-1's ablation ALREADY tested exactly this (mask a position's FFN,
run the WHOLE remaining stack, check the decode) and found every non-query position
is a nil passthrough — so this is byte-exact for the DECODE.  We nonetheless VERIFY
it end-to-end through the real driver (``verify_byte_exact``), never assume.

Gate: ``C4_POS_SPARSE`` (install-time opt-in, default OFF).  OFF -> the driver runs
the unmodified full-position forward (golden byte-identical, fingerprint unchanged).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from . import isa
from .step_block_skip import build_live_index


def pos_sparse_enabled() -> bool:
    return os.environ.get("C4_POS_SPARSE", "0") == "1"


def _attn_query_only(attn, x: torch.Tensor, q_idx: int) -> torch.Tensor:
    """Compute the SparseAttn block output but only the query row ``q_idx`` gets the
    real attention output; all other rows carry ``x`` unchanged (passthrough).

    Byte-exact to ``attn.forward(x)[:, q_idx]`` for the query row.  K/V run over all
    positions (needed); Q/scores/softmax/context/W_o run for the single query row.
    """
    B, S, D = x.shape
    H, HD = attn.n_heads, attn.head_dim
    # K, V over ALL positions (the query attends to every key).
    K = attn.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)          # [B,H,S,HD]
    V = attn.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)
    # Q for the SINGLE query row only.
    xq = x[:, q_idx:q_idx + 1]                                        # [B,1,D]
    Q = attn.W_q.linear(xq).view(B, 1, H, HD).transpose(1, 2)         # [B,H,1,HD]
    from .blogspec_model import softmax1
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale       # [B,H,1,S]
    # ALiBi + causal for query position q_idx (matches the un-cached path where
    # q_pos = arange(S): dist = |q_idx - k|, causal keeps k <= q_idx).
    pos = torch.arange(S, device=x.device)
    dist = (pos - q_idx).abs().float()                               # [S]
    scores = scores - attn.alibi_slopes.view(1, H, 1, 1) * dist.view(1, 1, 1, S)
    causal = torch.where(pos > q_idx,
                         torch.full_like(dist, float("-inf")),
                         torch.zeros_like(dist))
    scores = scores + causal.view(1, 1, 1, S)
    a = softmax1(scores, dim=-1)                                      # [B,H,1,S]
    ctx = torch.matmul(a, V).transpose(1, 2).contiguous().view(B, 1, D)  # [B,1,D]
    out = x.clone()
    out[:, q_idx:q_idx + 1] = xq + attn.W_o.linear(ctx)
    return out


def _ffn_query_only(ffn, x: torch.Tensor, q_idx: int, routed: bool) -> torch.Tensor:
    """Compute the FFN at ONLY the query row; other rows carry ``x`` unchanged.

    Byte-exact: the SwiGLU FFN is position-independent, and its residual add makes
    the passthrough at a skipped row exactly the input ``x[row]``."""
    xq = x[:, q_idx:q_idx + 1]
    fq = ffn.forward(xq) if not routed else ffn(xq)
    out = x.clone()
    out[:, q_idx:q_idx + 1] = fq
    return out


def _block_query_only(blk, x: torch.Tensor, q_idx: int) -> torch.Tensor:
    a = _attn_query_only(blk.attn, x, q_idx)
    return _ffn_query_only(blk.ffn, a, q_idx, blk._routed)


class PositionSparseRunner:
    """Op-level block-skip PLUS position-level (query-row-only) heavy compute.

    ``forward(x, op)`` applies the op's live blocks; each live block runs its
    attention K/V over all positions but its Q/context/W_o and its whole FFN at the
    single query row (the last position).  Byte-exact for the decoded query row iff
    every non-query position is a nil passthrough for the decode (STEP-1 verified;
    ``verify_byte_exact`` re-checks end-to-end)."""

    def __init__(self, model, L):
        self.model = model
        self.blocks = model.blocks
        self.L = L
        self.live_index = build_live_index(model, L)
        self.n_blocks = len(model.blocks)

    def live_count(self, op) -> int:
        return len(self.live_index.get(op, self.live_index[None]))

    def forward(self, x: torch.Tensor, op) -> torch.Tensor:
        live = self.live_index.get(op)
        q = x.shape[1] - 1                       # the query row = last position
        with torch.no_grad():
            if live is None:
                for blk in self.blocks:
                    x = _block_query_only(blk, x, q)
                return x
            live_set = set(live)
            for bi, blk in enumerate(self.blocks):
                if bi in live_set:
                    x = _block_query_only(blk, x, q)
        return x


# ---------------------------------------------------------------------------
# MAC accounting: dense-over-positions vs query-row-only.
# ---------------------------------------------------------------------------
def block_macs(blk, S: int, pos_sparse: bool):
    """Effective NONZERO-weight MACs for one block at seq-len S.

    We count NONZERO weights only (the model is ~99.9% sparse; a MAC is a
    nonzero-weight multiply-add).  Attn: Q/K/V/O nnz; FFN: up/gate/down nnz.  With
    ``pos_sparse``: FFN + Q + O run at 1 row, K + V at S rows; else everything at S.
    (The S*S score/context term is tiny here — head_dim*nnz-free — and identical in
    both, so we report the WEIGHT-MAC part, the dominant cost.)"""
    def nnz(sw):
        if getattr(sw, "dense", None) is not None:
            return int((sw.dense != 0).sum().item())
        if getattr(sw, "dense_resident", None) is not None:
            return int((sw.dense_resident != 0).sum().item())
        if getattr(sw, "csr", None) is not None:
            return int(sw.nnz)
        return int((sw != 0).sum().item())
    A = blk.attn
    q_nnz, k_nnz, v_nnz, o_nnz = (nnz(A.W_q), nnz(A.W_k), nnz(A.W_v), nnz(A.W_o))
    if blk._routed:
        f = blk.ffn
        up = int((f.W_up != 0).sum().item()); ga = int((f.W_gate != 0).sum().item())
        dn = int((f.W_down != 0).sum().item())
    else:
        f = blk.ffn
        up, ga, dn = nnz(f.W_up), nnz(f.W_gate), nnz(f.W_down)
    ffn_nnz = up + ga + dn
    if pos_sparse:
        # K,V over S rows; Q,O,FFN over 1 row.
        return (k_nnz + v_nnz) * S + (q_nnz + o_nnz + ffn_nnz) * 1
    return (q_nnz + k_nnz + v_nnz + o_nnz + ffn_nnz) * S


__all__ = ["pos_sparse_enabled", "PositionSparseRunner", "block_macs",
           "_block_query_only"]
