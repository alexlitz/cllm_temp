"""STATIC block-sparse FFN for the SparseTransformer (materialize-dense path).

STEP 1 verdict (``_bs_step1*.py``): the c4_min FFN weights are ~99.93% sparse and
SCATTERED — each hidden unit reads a MEDIAN of 1 residual dim (mean 1.2), and
different units read different dims, so the nonzeros do NOT cluster into dense
tiles (natural 32x32 tiles are 98% zeros; even a greedy row+col re-block reaches
only ~2.5% useful). Dense-sub-block matmuls would therefore be ~97-99% padding.

The representation that MATCHES a "~1 nonzero per row" matrix is NOT a dense
block — it is a GATHER · SCALE · SCATTER (the COO form): each output element is one
input dim times one scalar, accumulated. This module runs the FFN's three GEMMs
that way with STATIC (weight-fixed) index tensors, so the whole thing is a fixed
sequence of ``index_select`` / ``scatter_add`` — capturable in ONE CUDA graph.

Two static forms are provided and both are BYTE-EXACT vs the dense F.linear GEMM
only up to fp-reduction ORDER (a nonzero-order sum vs the dense contiguous sum).
For a row that reads 1 dim the order is identical -> L-inf=0; a row reading k>1
dims can differ by fp-accum residue far below the integer decode margin. We also
provide a ``dense_active`` form (compact active-row x active-col dense sub-block)
which the STEP 1 numbers show is 99% padding — kept only to MEASURE that dense
sub-blocks lose, honestly.

Gated by ``C4_BLOCK_SPARSE_FFN`` (install-time opt-in); OFF -> the model is the
unmodified dense/materialize-dense forward (golden byte-identical).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


def _coo_of(w: torch.Tensor):
    """Return (rows, cols, vals) of the nonzeros of a dense [out,in] weight."""
    idx = (w != 0).nonzero(as_tuple=False)      # [nnz, 2] (row, col)
    rows = idx[:, 0].contiguous()
    cols = idx[:, 1].contiguous()
    vals = w[rows, cols].contiguous()
    return rows, cols, vals


@dataclass
class CooLinear:
    """A [out,in] linear stored as its nonzero (row,col,val) triplets.

    ``linear(x)`` for x [N,in] computes y[N,out] = x @ Wᵀ as:
        contrib[n, e] = x[n, cols[e]] * vals[e]        (gather + scale)
        y[n, rows[e]] += contrib[n, e]                 (scatter-add over out)
    a fixed-shape (nnz-length) index program -> CUDA-graph-capturable. y is
    materialized as a fresh zeros tensor each call (graph-safe: static shape)."""

    out_dim: int
    in_dim: int
    rows: torch.Tensor       # [nnz] long, output index
    cols: torch.Tensor       # [nnz] long, input index
    vals: torch.Tensor       # [nnz] float
    nnz: int

    @classmethod
    def from_dense(cls, w: torch.Tensor) -> "CooLinear":
        r, c, v = _coo_of(w)
        return cls(w.shape[0], w.shape[1], r, c, v, r.numel())

    def to(self, device):
        self.rows = self.rows.to(device)
        self.cols = self.cols.to(device)
        self.vals = self.vals.to(device)
        return self

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2d = x.reshape(-1, self.in_dim)                        # [N, in]
        N = x2d.shape[0]
        # gather the read column for every nonzero, scale by the weight value.
        contrib = x2d.index_select(1, self.cols) * self.vals   # [N, nnz]
        y = torch.zeros(N, self.out_dim, dtype=x2d.dtype, device=x2d.device)
        # scatter-add each nonzero's contribution into its output row.
        y.index_add_(1, self.rows, contrib)                    # [N, out]
        return y.reshape(*orig[:-1], self.out_dim)


@dataclass
class DenseActiveLinear:
    """Compact active-row x active-col dense sub-block (STEP 1 shows ~1% useful;
    kept to MEASURE that dense sub-blocks lose). y[:, active_rows] = xc @ Wcᵀ,
    xc = x[:, active_cols]. Fixed shapes -> graph-capturable."""

    out_dim: int
    in_dim: int
    active_rows: torch.Tensor        # [ar] output rows that are nonempty
    active_cols: torch.Tensor        # [ac] input cols that are nonempty
    wsub: torch.Tensor               # [ar, ac] dense compacted weight

    @classmethod
    def from_dense(cls, w: torch.Tensor) -> "DenseActiveLinear":
        m = (w != 0)
        arow = m.any(dim=1).nonzero(as_tuple=False).flatten().contiguous()
        acol = m.any(dim=0).nonzero(as_tuple=False).flatten().contiguous()
        wsub = w[arow][:, acol].contiguous()
        return cls(w.shape[0], w.shape[1], arow, acol, wsub)

    def to(self, device):
        self.active_rows = self.active_rows.to(device)
        self.active_cols = self.active_cols.to(device)
        self.wsub = self.wsub.to(device)
        return self

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2d = x.reshape(-1, self.in_dim)
        N = x2d.shape[0]
        xc = x2d.index_select(1, self.active_cols)             # [N, ac]
        ysub = F.linear(xc, self.wsub)                         # [N, ar]
        y = torch.zeros(N, self.out_dim, dtype=x2d.dtype, device=x2d.device)
        y.index_copy_(1, self.active_rows, ysub)
        return y.reshape(*orig[:-1], self.out_dim)


class BlockSparseFFN:
    """SwiGLU FFN whose W_up/W_gate/W_down run through a STATIC scatter linear
    (``mode='coo'`` gather-scale-scatter, or ``mode='dense_active'`` compact dense
    sub-block).  Same arithmetic as ``SparseFFN.forward``; byte-exact up to
    fp-reduction order (L-inf=0 for the 1-nnz-per-row majority)."""

    def __init__(self, ffn, mode: str = "coo"):
        self.mode = mode
        L = CooLinear if mode == "coo" else DenseActiveLinear

        def dense_of(sw):
            if not sw.is_sparse:
                return sw.dense
            if getattr(sw, "dense_resident", None) is not None:
                return sw.dense_resident
            return sw.csr.to_dense()

        self.W_up = L.from_dense(dense_of(ffn.W_up))
        self.W_gate = L.from_dense(dense_of(ffn.W_gate))
        self.W_down = L.from_dense(dense_of(ffn.W_down))
        self.b_up = ffn.b_up.detach().clone()
        self.b_gate = ffn.b_gate.detach().clone()
        self.b_down = ffn.b_down.detach().clone()

    def to(self, device):
        for w in (self.W_up, self.W_gate, self.W_down):
            w.to(device)
        self.b_up = self.b_up.to(device)
        self.b_gate = self.b_gate.to(device)
        self.b_down = self.b_down.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.W_up.linear(x) + self.b_up
        gate = self.W_gate.linear(x) + self.b_gate
        hidden = F.silu(up) * gate
        return x + self.W_down.linear(hidden) + self.b_down


def block_sparse_ffn_enabled() -> bool:
    return os.environ.get("C4_BLOCK_SPARSE_FFN", "0") == "1"


def install_block_sparse_ffn(model, mode: str = "coo", verbose: bool = False):
    """Swap every non-routed SparseBlock's FFN for a BlockSparseFFN (static scatter).

    Byte-exact up to fp-reduction order (the 1-nnz-per-row majority is L-inf=0).
    Returns a small stats dict. Idempotent-ish: re-installs from the current
    (dense-materialized or CSR) weights.
    """
    n_swapped = 0
    n_routed = 0
    # Map each DISTINCT original SparseFFN object -> its BlockSparseFFN. Keyed by the
    # live object (not id(), which the GC can recycle to a later distinct object) so
    # a recurrent build's shared FFN is converted once and reused. Blocks that share
    # an ffn are handled by the identity map; already-converted ffns are left as-is.
    converted = {}
    for b in model.blocks:
        if getattr(b, "_routed", False):
            n_routed += 1
            continue
        cur = b.ffn
        if isinstance(cur, BlockSparseFFN):
            continue                              # already swapped (shared object)
        existing = converted.get(id(cur))
        if existing is not None and existing[0] is cur:
            b.ffn = existing[1]
            continue
        bsf = BlockSparseFFN(cur, mode=mode)
        converted[id(cur)] = (cur, bsf)           # keep cur alive -> id stays valid
        b.ffn = bsf
        n_swapped += 1
    if verbose:
        print(f"[block-sparse-ffn] mode={mode} swapped {n_swapped} distinct FFNs "
              f"({n_routed} routed kept dense)", flush=True)
    return {"swapped": n_swapped, "routed": n_routed, "mode": mode}
