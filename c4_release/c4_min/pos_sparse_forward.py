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


def _ffn_forward_fp64(ffn, xq: torch.Tensor) -> torch.Tensor:
    """SwiGLU FFN forward for a SINGLE query row in float64 (fp-STABLE decode).

    The 1-row query-only GEMM the pos-sparse path uses selects a cuBLAS GEMV
    kernel whose fp32 accumulation ORDER differs from the reference S-row GEMM.
    For fp-FRAGILE FFN gadgets — notably ``lea-addr-nib``, whose frame-byte decode
    sums ~1050 near-cancelling saturating-silu steps — that order flip crosses the
    integer decode margin (measured: single-row fp32 gives AXB_LO=-21, the S-row
    fp32 GEMM gives 12; fp64 gives 12 in BOTH — 12 is the true integer LEA_Q&0xFF).
    So we run the SINGLE query row in float64: it is one D-vector (microseconds) and
    computes the EXACT integer the decode intends, byte-exact to the S-row reference
    for EVERY op (fp64 >= fp32 accuracy, and the downstream nibble decode integer-
    snaps).  Only the resident dense weights are used (materialize_dense'd)."""
    W_up = ffn.W_up.dense_resident if ffn.W_up.dense_resident is not None else ffn.W_up.dense
    W_gate = ffn.W_gate.dense_resident if ffn.W_gate.dense_resident is not None else ffn.W_gate.dense
    W_down = ffn.W_down.dense_resident if ffn.W_down.dense_resident is not None else ffn.W_down.dense
    xq64 = xq.double()
    up = F.linear(xq64, W_up.double()) + ffn.b_up.double()
    gate = F.linear(xq64, W_gate.double()) + ffn.b_gate.double()
    hidden = F.silu(up) * gate
    out = xq64 + F.linear(hidden, W_down.double()) + ffn.b_down.double()
    return out.to(xq.dtype)


def _ffn_query_only(ffn, x: torch.Tensor, q_idx: int, routed: bool,
                    fp64: bool = True) -> torch.Tensor:
    """Compute the FFN at ONLY the query row; other rows carry ``x`` unchanged.

    Byte-exact: the SwiGLU FFN is position-independent, and its residual add makes
    the passthrough at a skipped row exactly the input ``x[row]``.  ``fp64`` (default
    ON) runs the single query row's SwiGLU in float64 so the 1-row GEMV accumulation
    matches the integer-exact reference (fixes the ``lea-addr-nib`` decode tie the
    fp32 1-row kernel flips — see ``_ffn_forward_fp64``).  Routed (Top-1 MoE) FFNs
    keep their own forward (their decode is not fp-fragile; the router picks <=K
    dense rows)."""
    xq = x[:, q_idx:q_idx + 1]
    if routed:
        fq = ffn(xq)
    elif fp64 and getattr(ffn, "W_up", None) is not None \
            and hasattr(ffn.W_up, "dense_resident"):
        fq = _ffn_forward_fp64(ffn, xq)
    else:
        fq = ffn.forward(xq)
    out = x.clone()
    out[:, q_idx:q_idx + 1] = fq
    return out


def _block_query_only(blk, x: torch.Tensor, q_idx: int,
                      fp64_ffn: bool = True) -> torch.Tensor:
    a = _attn_query_only(blk.attn, x, q_idx)
    return _ffn_query_only(blk.ffn, a, q_idx, blk._routed, fp64=fp64_ffn)


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
