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


# ---------------------------------------------------------------------------
# INTEGER LEA-address snap (replaces the ONLY fp64 block, ``lea-addr-nib``).
#
# WHY fp64 was there:  the ``lea-addr-nib`` FFN decodes the frame byte
# ``AXB_LO/AXB_HI = split_nibbles(LEA_Q & 0xFF)`` by summing up to ~768 saturating
# silu UNIT steps (edges from -256..512), each routing ``±val`` (val 0..15) into
# ``AXB_LO``.  For a large positive integer ``LEA_Q=k`` roughly ``k+256`` of those
# ``+val``/``-val`` terms are ON, so the reduction is a long near-cancelling chain
# whose partial sums reach O(few-thousand).  fp32 has only ~2^24 integer exactness
# and the 1-row cuBLAS GEMV REASSOCIATES the reduction order, so the accumulated
# rounding error GROWS with ``LEA_Q`` — measured up to ``77`` on ``AXB_LO`` (q=224:
# fp32 decodes -20, the true nibble is 0).  #741 ran that one block in fp64 to
# restore exactness (fp64's 2^53 mantissa absorbs the chain).  It is NOT a
# half-integer tie at the ``_step_ge`` staircase — the staircase itself is exact;
# it is the fp32 *accumulation order/precision* of the O(few-hundred-to-thousand)-
# term W_down GEMV that diverges.
#
# The block computes an INTEGER address (``LEA_Q`` in [-256,511] is already snapped
# to an exact integer by ``lea-q-snap`` upstream, and its ONLY output is the two
# 4-bit nibbles of ``LEA_Q & 0xFF``).  So it is computed EXACTLY, and much faster,
# in integer/deterministic arithmetic — no float64, no ~768-step silu sum at all:
#
#     gate = silu(S*(g-0.5)) / SILU_HALF        # the block's own LEA gate ramp (g=OP_IS[LEA])
#     byte = round(LEA_Q) mod 256               # exact integer, |LEA_Q|<=512 << 2^24
#     AXB_LO := AXB_LO + gate*((byte & 0xF) - AXB_LO)      # SET (clear+re-add) fused
#     AXB_HI := AXB_HI + gate*((byte>>4)     - AXB_HI)
#
# On a LEA (g exactly 1.0) the delta reproduces the block's SET; off-LEA (g=0) the
# delta is 0 -> byte-identical passthrough.  Verified 0/768 snapped-nibble mismatch
# vs the fp64 block over the whole [-256,511] LEA_Q range (and vs the integer-exact
# reference).  Gate: ``C4_LEA_INT_SNAP`` (default OFF -> golden byte-identical; the
# runner still installs the fp64/fp32 path when OFF).
# ---------------------------------------------------------------------------
LEA_ADDR_NIB_BLOCK_NAME = "lea-addr-nib"


def lea_int_snap_enabled() -> bool:
    """The integer LEA-address snap that eliminates the ONLY fp64 block.

    DEFAULT OFF.  When ON, the ``lea-addr-nib`` block's query-row FFN is computed
    with the exact-integer nibble split of ``LEA_Q & 0xFF`` (int32-exact) instead
    of the fp64 SwiGLU — no float64 anywhere in the forward, and the ~768-step
    silu decode is replaced by a handful of elementwise ops.  ``C4_LEA_INT_SNAP=1``
    to enable.  Escape hatch: unset / ``=0`` -> the runner keeps the fp64 (or
    fp32) path and the build stays byte-identical to golden ``069cc32f``."""
    return os.environ.get("C4_LEA_INT_SNAP", "0") == "1"


def resolve_lea_snap_dims(L):
    """Resolve the residual-dim indices the integer LEA snap reads/writes:
    ``(gate=OP_IS[LEA], lea_q, axb_lo, axb_hi)``.  Returns ``None`` if the layout
    does not expose them (non-exact-steps model) so the caller falls back."""
    from . import isa
    try:
        return (int(L.OP_IS) + int(isa.LEA), int(L.LEA_Q),
                int(L.AXB_LO), int(L.AXB_HI))
    except Exception:
        return None


def apply_int_lea_addr_nib(aout: torch.Tensor, dims) -> torch.Tensor:
    """Byte-exact INTEGER replacement for the ``lea-addr-nib`` FFN at the query row.

    ``aout`` is the block's attention output ([...,D]); ``dims`` is the tuple
    ``(gate_dim, lea_q_dim, axb_lo_dim, axb_hi_dim)`` from ``resolve_lea_snap_dims``.
    Returns the FFN output (residual-add form): identical to ``aout`` everywhere
    except ``AXB_LO``/``AXB_HI``, which get the SET result the fp64 block computes,
    gated by the block's own LEA gate ramp.  No float64; no cuBLAS GEMV; exact."""
    from .nibble_pure_forward_complete import S as _S, SILU_HALF as _SH
    g_dim, q_dim, lo_dim, hi_dim = dims
    g = aout[..., g_dim]
    # the block's LEA gate ramp: silu(S*(g-0.5))/SILU_HALF (==1 for g=1, ==0 for g=0).
    gate = F.silu(_S * (g - 0.5)) / _SH
    # LEA_Q is a snapped exact integer in [-256,511]; byte = LEA_Q & 0xFF (int-exact).
    byte = torch.remainder(torch.round(aout[..., q_dim]), 256.0)
    new_lo = torch.remainder(byte, 16.0)
    new_hi = torch.floor(byte / 16.0)
    out = aout.clone()
    out[..., lo_dim] = aout[..., lo_dim] + gate * (new_lo - aout[..., lo_dim])
    out[..., hi_dim] = aout[..., hi_dim] + gate * (new_hi - aout[..., hi_dim])
    return out


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
           "_block_query_only", "lea_int_snap_enabled", "resolve_lea_snap_dims",
           "apply_int_lea_addr_nib", "LEA_ADDR_NIB_BLOCK_NAME"]
