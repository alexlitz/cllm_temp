"""#804 follow-up — a COO/CSR-scatter sparse-GEMM that runs ONLY the useful
nonzero FLOPs of the c4_min per-block FFN, replacing the densely-padded
``F.linear`` GEMM that multiplies ~99.97% zeros.

Why COO-scatter and NOT a fused-concat / band-grouped dense GEMM
================================================================
The measured nnz structure (``_agent_ffn_nnz_probe``) is:

  * every block shares D=1679 input; Dff in 1..4320 (56 distinct hidden widths).
  * W_up / W_gate rows read ~1.45 nonzeros each (68% read exactly ONE input dim);
    W_down rows read ~35 each.
  * the per-block (active_rows x active_cols) rectangle is only ~1.24% full — the
    nonzeros are SCATTERED, they do NOT cluster into dense bands (confirmed by
    _agent_band_structure: even the tightest active sub-block is >98% padding).

So ANY dense tiling — band-group, active-row/col gather, fused-concat GEMM — pads
the useful nnz back up to a dense rectangle that is >98% zeros.  A real
COO/CSR-scatter that touches ONLY the (row, col, val) triples is the only form
whose work is ∝ nnz.  This module builds one CSR per projection per block and a
row-segmented Triton kernel that, for each output row and a tile of K token
columns, accumulates ``sum_over_nnz(val * X[col, :])`` in registers — NO atomics
(each row is one program), NO densification.

Byte-exactness
==============
``Y[u,k] = sum_{(u,c,v) in row u} v * X[c,k]`` is the SAME arithmetic the dense
``F.linear`` computes on the nonzeros (the zero entries contribute exactly 0).
The only fp difference is accumulation ORDER within a row; with <=~35 terms and
fp32 this is ~1e-6, far below the integer-nibble decode margin.  The kernel sums
nonzeros in stored (column-sorted) order to keep the residue minimal and stable.

The composed forward stays graph-capturable (the kernel is a plain Triton launch
with static shapes per block/K), so it drops into #804's single big CUDA graph.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Row-segmented CSR SpMM:  Y[M, K] = W[M, D-sparse] @ X[D, K]      (+ optional bias)
#   W stored CSR: crow[M+1], col[nnz], val[nnz].  One program per (row, K-tile).
#   Each program walks its row's nonzeros and accumulates val * X[col, k-tile].
# ---------------------------------------------------------------------------
@triton.jit
def _csr_spmm_kernel(
    crow_ptr, col_ptr, val_ptr,          # CSR of W [M, D]
    x_ptr,                               # X [D, K]  (row-major, stride_xd, stride_xk)
    bias_ptr,                            # [M] or None
    y_ptr,                               # Y [M, K]  (row-major)
    M, K, D,
    stride_xd, stride_xk,
    stride_ym, stride_yk,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kblk = tl.program_id(1)
    k_off = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    start = tl.load(crow_ptr + row)
    end = tl.load(crow_ptr + row + 1)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + row)

    # walk the row's nonzeros (few — typically 1..35); accumulate val * X[col, :].
    for p in range(start, end):
        c = tl.load(col_ptr + p)
        v = tl.load(val_ptr + p)
        xrow = tl.load(x_ptr + c * stride_xd + k_off * stride_xk,
                       mask=k_mask, other=0.0)
        acc += v * xrow

    tl.store(y_ptr + row * stride_ym + k_off * stride_yk, acc, mask=k_mask)


class CsrSpmm:
    """A single [M, D] sparse weight in CSR, applied as ``W @ X`` for X [D, K].

    ``forward(x_dk, bias)`` returns Y [M, K].  ``x_dk`` is X laid out [D, K]
    (feature-major), i.e. the transpose of the usual [K, D] activation batch.
    """

    def __init__(self, w_dense: torch.Tensor, block_k: int = 64):
        # w_dense: [M, D].  Build CSR index arrays (int32 for cheap loads).
        M, D = w_dense.shape
        nz = torch.nonzero(w_dense, as_tuple=False)          # [nnz, 2] (row, col)
        # sort by (row, col) so each row's nonzeros are contiguous + col-ordered
        # (stable accumulation order -> minimal, deterministic fp residue).
        order = torch.argsort(nz[:, 0] * D + nz[:, 1])
        nz = nz[order]
        rows = nz[:, 0]
        cols = nz[:, 1].to(torch.int32).contiguous()
        vals = w_dense[rows, nz[:, 1]].contiguous()
        crow = torch.zeros(M + 1, dtype=torch.int32, device=w_dense.device)
        crow[1:] = torch.bincount(rows, minlength=M).cumsum(0).to(torch.int32)
        self.M, self.D = M, D
        self.crow = crow
        self.col = cols
        self.val = vals.to(torch.float32)
        self.nnz = int(vals.numel())
        self.block_k = block_k

    def to(self, device):
        self.crow = self.crow.to(device)
        self.col = self.col.to(device)
        self.val = self.val.to(device)
        return self

    def forward(self, x_dk: torch.Tensor, bias: Optional[torch.Tensor] = None,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
        D, K = x_dk.shape
        assert D == self.D, (D, self.D)
        if out is None:
            out = torch.empty(self.M, K, device=x_dk.device, dtype=torch.float32)
        grid = (self.M, triton.cdiv(K, self.block_k))
        _csr_spmm_kernel[grid](
            self.crow, self.col, self.val,
            x_dk, bias if bias is not None else x_dk, out,
            self.M, K, self.D,
            x_dk.stride(0), x_dk.stride(1),
            out.stride(0), out.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_K=self.block_k,
        )
        return out


# ---------------------------------------------------------------------------
# The composed sparse-COO FFN: three CsrSpmm (up/gate/down) + SwiGLU, byte-exact
# to ``SparseFFN.forward`` on the nonzeros.
# ---------------------------------------------------------------------------
def _dense_of(w):
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


class CooSpmmFFN:
    """Drop-in for ``SparseFFN`` whose W_up/W_gate/W_down GEMMs are COO-scatter
    SpMMs running only the nonzero FLOPs.  Forward is [B, S, D] -> [B, S, D],
    byte-identical to ``SparseFFN.forward`` at the nibble-decode margin.
    """

    def __init__(self, ffn, device, block_k: int = 64):
        self.dim = _dense_of(ffn.W_up).shape[1]
        self.Dff = _dense_of(ffn.W_up).shape[0]
        self.up = CsrSpmm(_dense_of(ffn.W_up).to(device), block_k)
        self.gate = CsrSpmm(_dense_of(ffn.W_gate).to(device), block_k)
        self.down = CsrSpmm(_dense_of(ffn.W_down).to(device), block_k)
        self.b_up = ffn.b_up.to(device)
        self.b_gate = ffn.b_gate.to(device)
        self.b_down = ffn.b_down.to(device)
        self._down_bias_zero = bool((self.b_down != 0).sum() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # X as [D, K] feature-major (K = B*S).  contiguous for coalesced col loads.
        xk = x.reshape(B * S, D).transpose(0, 1).contiguous()      # [D, K]
        up = self.up.forward(xk, self.b_up)                        # [Dff, K]
        gate = self.gate.forward(xk, self.b_gate)                  # [Dff, K]
        hidden = torch.nn.functional.silu(up) * gate               # [Dff, K]
        down = self.down.forward(hidden,
                                 None if self._down_bias_zero else self.b_down)  # [D, K]
        out = down.transpose(0, 1).reshape(B, S, D)                # [K, D] -> [B,S,D]
        if not self._down_bias_zero:
            pass  # bias already folded in the kernel
        return x + out
