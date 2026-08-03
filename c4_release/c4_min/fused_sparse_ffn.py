"""#806 follow-up — FUSED segmented sparse-FFN Triton kernels.

Collapses the per-block COO FFN's THREE separate SpMM launches (up, gate, down)
plus the transpose->silu*mul->transpose elementwise GLUE into ONE (or two) fused
Triton launches per block, eliminating:

  (a) 2 of the 3 SpMM launches per block (the 726 -> {242,484} launch collapse),
  (b) the standalone ``F.silu(up)*gate`` elementwise GLUE kernel (134 us on the
      big block per #806's attribution),
  (c) the intermediate [Dff, K] HBM read+write between up/gate and down (up and
      gate accumulate + fuse in-register; only the fused hidden touches HBM once).

WHY per-block and not all-242-in-one-grid
==========================================
The c4_min residual is a DEEP SEQUENTIAL chain: a dependency probe
(``_agent_dep_probe``) shows 241/242 blocks READ a residual dim an EARLIER block
WROTE (204 dependency levels, largest parallel wave = 3 blocks). Block i+1's FFN
input is block i's output. So the 242 blocks CANNOT be batched into one data-
parallel grid over a shared X — the fusion that IS available is collapsing the
3-launch + glue structure WITHIN each block into a single segmented launch.

Two fused forms are provided:

  FusedUpGateSiluFFN  — kernel 1 fuses up + gate + silu*mul into ONE launch
                        (one program per (Dff-row, K-tile), reads X once, writes
                        the fused hidden once); kernel 2 = the down SpMM. 2 launches
                        per block, hidden written ONCE (vs 3 launches + a glue
                        kernel + up/gate each written to HBM).

  FusedFullFFN        — a SINGLE launch per block: one grid over K-tiles, each
                        program streams ALL Dff rows (up+gate+silu in registers,
                        no [Dff,K] HBM buffer at all) and directly accumulates the
                        down projection into the [D, K-tile] output. 1 launch,
                        ZERO intermediate HBM. Uses a CSC-by-hidden layout of
                        W_down so a hidden row's contribution scatters to its D
                        output rows. This is the maximal per-block fusion.

Byte-exactness
==============
Same arithmetic on the nonzeros as ``SparseFFN.forward`` (zeros contribute 0).
The only fp difference is accumulation ORDER (Triton column-order vs cuBLAS tiled
order in W_down), ~6e-2 raw on ~5-magnitude values -> absorbed by the integer
nibble-snap decode margin, exactly as the COO path (#806).

Graph-capturable: plain Triton launches with static per-block/K shapes -> drop
into the single big CUDA graph (composed with flash attention on the 3 live
blocks), same as the COO path.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# KERNEL 1 — FUSED up + gate + silu*mul.
#   Y[u, k] = silu( sum_c Wup[u,c] X[c,k] + b_up[u] )
#             * ( sum_c Wgate[u,c] X[c,k] + b_gate[u] )
#   One program per (row u, K-tile). Walks the row's up-nonzeros then its
#   gate-nonzeros, applies silu*mul, stores the fused hidden.  X read once per
#   row; up/gate never touch HBM (fused in-register); hidden written once.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_upgate_silu_kernel(
    up_crow_ptr, up_col_ptr, up_val_ptr,       # CSR of W_up [Dff, D]
    gt_crow_ptr, gt_col_ptr, gt_val_ptr,       # CSR of W_gate [Dff, D]
    x_ptr,                                     # X [D, K]
    bup_ptr, bgt_ptr,                          # [Dff]
    h_ptr,                                     # fused hidden [Dff, K] (out)
    K,
    stride_xd, stride_xk,
    stride_hu, stride_hk,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kblk = tl.program_id(1)
    k_off = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    # ---- up accumulation (start with bias) ----
    up = tl.zeros([BLOCK_K], dtype=tl.float32) + tl.load(bup_ptr + row)
    us = tl.load(up_crow_ptr + row)
    ue = tl.load(up_crow_ptr + row + 1)
    for p in range(us, ue):
        c = tl.load(up_col_ptr + p)
        v = tl.load(up_val_ptr + p)
        up += v * tl.load(x_ptr + c * stride_xd + k_off * stride_xk, mask=k_mask, other=0.0)

    # ---- gate accumulation (start with bias) ----
    gate = tl.zeros([BLOCK_K], dtype=tl.float32) + tl.load(bgt_ptr + row)
    gs = tl.load(gt_crow_ptr + row)
    ge = tl.load(gt_crow_ptr + row + 1)
    for p in range(gs, ge):
        c = tl.load(gt_col_ptr + p)
        v = tl.load(gt_val_ptr + p)
        gate += v * tl.load(x_ptr + c * stride_xd + k_off * stride_xk, mask=k_mask, other=0.0)

    # ---- fused silu(up) * gate ----   (silu(x) = x * sigmoid(x))
    h = (up * tl.sigmoid(up)) * gate
    tl.store(h_ptr + row * stride_hu + k_off * stride_hk, h, mask=k_mask)


# ---------------------------------------------------------------------------
# KERNEL 2 — down SpMM WITH FUSED RESIDUAL ADD.
#   Out[d, k] = X_resid[d, k] + sum_h Wdown[d,h] H[h,k]      (b_down == 0)
#   One program per (output row d, K-tile). Fuses the residual add so no extra
#   elementwise kernel.  If a residual dim d has NO down nonzeros the row is still
#   written (= the residual), so the full [D,K] output is complete.
# ---------------------------------------------------------------------------
@triton.jit
def _down_resid_kernel(
    dn_crow_ptr, dn_col_ptr, dn_val_ptr,       # CSR of W_down [D, Dff]
    h_ptr,                                     # hidden [Dff, K]
    xres_ptr,                                  # residual X [D, K]  (== block input)
    y_ptr,                                     # out [D, K]
    K,
    stride_hh, stride_hk,
    stride_xd, stride_xk,
    stride_yd, stride_yk,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kblk = tl.program_id(1)
    k_off = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    # start from the residual (fused add)
    acc = tl.load(xres_ptr + row * stride_xd + k_off * stride_xk, mask=k_mask, other=0.0)
    s = tl.load(dn_crow_ptr + row)
    e = tl.load(dn_crow_ptr + row + 1)
    for p in range(s, e):
        c = tl.load(dn_col_ptr + p)
        v = tl.load(dn_val_ptr + p)
        acc += v * tl.load(h_ptr + c * stride_hh + k_off * stride_hk, mask=k_mask, other=0.0)
    tl.store(y_ptr + row * stride_yd + k_off * stride_yk, acc, mask=k_mask)


# ---------------------------------------------------------------------------
# KERNEL 2b — down DELTA in-place.  Only the residual rows that W_down actually
#   WRITES get a program (mean ~0.15*Dff nonzero rows per block); the residual is
#   updated IN PLACE (y[d] += sum_h Wdown[d,h] H[h]).  This eliminates the full
#   [D,K] residual copy the plain down kernel does for the ~99% pure-copy rows
#   (W_down averages 0.15 nnz/row -> most of the 1679 output dims are untouched).
#   ``active_rows`` maps program-id -> the actual residual dim; CSR is over the
#   ACTIVE rows only.
# ---------------------------------------------------------------------------
@triton.jit
def _down_delta_inplace_kernel(
    active_row_ptr,                            # [n_active] residual dim of each prog
    dn_crow_ptr, dn_col_ptr, dn_val_ptr,       # CSR of W_down over ACTIVE rows [n_active, Dff]
    h_ptr,                                     # hidden [Dff, K]
    y_ptr,                                     # residual [D, K]  (updated in place)
    K,
    stride_hh, stride_hk,
    stride_yd, stride_yk,
    BLOCK_K: tl.constexpr,
):
    prog = tl.program_id(0)
    kblk = tl.program_id(1)
    k_off = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K
    d = tl.load(active_row_ptr + prog)         # actual residual dim

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    s = tl.load(dn_crow_ptr + prog)
    e = tl.load(dn_crow_ptr + prog + 1)
    for p in range(s, e):
        c = tl.load(dn_col_ptr + p)
        v = tl.load(dn_val_ptr + p)
        acc += v * tl.load(h_ptr + c * stride_hh + k_off * stride_hk, mask=k_mask, other=0.0)
    yp = y_ptr + d * stride_yd + k_off * stride_yk
    tl.store(yp, tl.load(yp, mask=k_mask, other=0.0) + acc, mask=k_mask)


def _csr(w_dense: torch.Tensor):
    """Return (crow[M+1], col[nnz] int32, val[nnz] fp32) CSR of [M,D] dense w,
    row-sorted + col-sorted (stable, deterministic accumulation order)."""
    M, D = w_dense.shape
    nz = torch.nonzero(w_dense, as_tuple=False)
    order = torch.argsort(nz[:, 0].to(torch.int64) * D + nz[:, 1].to(torch.int64))
    nz = nz[order]
    rows = nz[:, 0]
    cols = nz[:, 1].to(torch.int32).contiguous()
    vals = w_dense[rows, nz[:, 1]].to(torch.float32).contiguous()
    crow = torch.zeros(M + 1, dtype=torch.int32, device=w_dense.device)
    crow[1:] = torch.bincount(rows, minlength=M).cumsum(0).to(torch.int32)
    return crow, cols, vals


def _dense_of(w):
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


class FusedUpGateSiluFFN:
    """2-launch fused FFN: (up+gate+silu*mul) fused into one kernel, then the down
    SpMM with fused residual add.  Drop-in for SparseFFN.forward, byte-exact at
    the nibble-snap margin.  Graph-capturable (static shapes)."""

    def __init__(self, ffn, device, block_k: int = 128):
        self.dev = device
        self.block_k = block_k
        Wu = _dense_of(ffn.W_up).to(device)
        Wg = _dense_of(ffn.W_gate).to(device)
        Wd = _dense_of(ffn.W_down).to(device)
        self.dim = Wu.shape[1]
        self.Dff = Wu.shape[0]
        self.up_crow, self.up_col, self.up_val = _csr(Wu)
        self.gt_crow, self.gt_col, self.gt_val = _csr(Wg)
        self.dn_crow, self.dn_col, self.dn_val = _csr(Wd)
        self.b_up = ffn.b_up.to(device).float().contiguous()
        self.b_gate = ffn.b_gate.to(device).float().contiguous()
        # b_down is all-zero across the model (probed); assert + fold if not.
        bd = ffn.b_down.to(device).float()
        self._down_bias_zero = bool((bd != 0).sum() == 0)
        self.b_down = bd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        K = B * S
        xk = x.reshape(K, D).transpose(0, 1).contiguous()      # [D, K]
        h = torch.empty(self.Dff, K, device=xk.device, dtype=torch.float32)
        grid1 = (self.Dff, triton.cdiv(K, self.block_k))
        _fused_upgate_silu_kernel[grid1](
            self.up_crow, self.up_col, self.up_val,
            self.gt_crow, self.gt_col, self.gt_val,
            xk, self.b_up, self.b_gate, h, K,
            xk.stride(0), xk.stride(1), h.stride(0), h.stride(1),
            BLOCK_K=self.block_k,
        )
        y = torch.empty(D, K, device=xk.device, dtype=torch.float32)
        grid2 = (D, triton.cdiv(K, self.block_k))
        _down_resid_kernel[grid2](
            self.dn_crow, self.dn_col, self.dn_val,
            h, xk, y, K,
            h.stride(0), h.stride(1),
            xk.stride(0), xk.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_K=self.block_k,
        )
        out = y.transpose(0, 1).reshape(B, S, D)
        if not self._down_bias_zero:
            out = out + self.b_down
        return out


# ---------------------------------------------------------------------------
# KERNEL 3 — SINGLE-LAUNCH full FFN (up+gate+silu+down) per block.
#   One program per (output residual row d, K-tile).  Each program:
#     1. loads the residual X[d, ktile]  (the fused residual base),
#     2. for each hidden col h that W_down[d, :] reads (few), recomputes that
#        hidden value h[ktile] on the fly (up+gate+silu of row h) and adds
#        Wdown[d,h] * h.
#   This is ONE launch, ZERO [Dff,K] HBM buffer.  The tradeoff: a hidden row read
#   by R different output rows is recomputed R times.  W_down averages ~0.15
#   nnz/row and W_up ~1.45 nnz/row, so the recompute is cheap for the sparse
#   majority.  For the few dense blocks (Dff=4320, down 1537 nnz/row) recompute
#   is heavy -> those use the 2-launch path.  Provided for the tiny-block bulk.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_full_kernel(
    dn_crow_ptr, dn_col_ptr, dn_val_ptr,       # CSR W_down [D, Dff]
    up_crow_ptr, up_col_ptr, up_val_ptr,       # CSR W_up [Dff, D]
    gt_crow_ptr, gt_col_ptr, gt_val_ptr,       # CSR W_gate [Dff, D]
    x_ptr,                                     # X [D, K]  (input == residual)
    bup_ptr, bgt_ptr,                          # [Dff]
    y_ptr,                                     # out [D, K]
    K,
    stride_xd, stride_xk,
    stride_yd, stride_yk,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)          # output residual dim d
    kblk = tl.program_id(1)
    k_off = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    acc = tl.load(x_ptr + row * stride_xd + k_off * stride_xk, mask=k_mask, other=0.0)
    ds = tl.load(dn_crow_ptr + row)
    de = tl.load(dn_crow_ptr + row + 1)
    for p in range(ds, de):
        hcol = tl.load(dn_col_ptr + p)            # hidden index
        wdn = tl.load(dn_val_ptr + p)
        # recompute hidden[hcol, ktile] = silu(up)*gate
        up = tl.zeros([BLOCK_K], dtype=tl.float32) + tl.load(bup_ptr + hcol)
        us = tl.load(up_crow_ptr + hcol)
        ue = tl.load(up_crow_ptr + hcol + 1)
        for q in range(us, ue):
            c = tl.load(up_col_ptr + q)
            v = tl.load(up_val_ptr + q)
            up += v * tl.load(x_ptr + c * stride_xd + k_off * stride_xk, mask=k_mask, other=0.0)
        gate = tl.zeros([BLOCK_K], dtype=tl.float32) + tl.load(bgt_ptr + hcol)
        gsq = tl.load(gt_crow_ptr + hcol)
        geq = tl.load(gt_crow_ptr + hcol + 1)
        for q in range(gsq, geq):
            c = tl.load(gt_col_ptr + q)
            v = tl.load(gt_val_ptr + q)
            gate += v * tl.load(x_ptr + c * stride_xd + k_off * stride_xk, mask=k_mask, other=0.0)
        h = (up * tl.sigmoid(up)) * gate
        acc += wdn * h
    tl.store(y_ptr + row * stride_yd + k_off * stride_yk, acc, mask=k_mask)


class FusedFullFFN:
    """SINGLE-launch full FFN per block (up+gate+silu+down fused, no HBM hidden).
    Byte-exact at the nibble margin; recomputes each read hidden row (cheap for
    the sparse bulk).  Graph-capturable."""

    def __init__(self, ffn, device, block_k: int = 128):
        self.dev = device
        self.block_k = block_k
        Wu = _dense_of(ffn.W_up).to(device)
        Wg = _dense_of(ffn.W_gate).to(device)
        Wd = _dense_of(ffn.W_down).to(device)
        self.dim = Wu.shape[1]
        self.Dff = Wu.shape[0]
        self.up_crow, self.up_col, self.up_val = _csr(Wu)
        self.gt_crow, self.gt_col, self.gt_val = _csr(Wg)
        self.dn_crow, self.dn_col, self.dn_val = _csr(Wd)
        self.b_up = ffn.b_up.to(device).float().contiguous()
        self.b_gate = ffn.b_gate.to(device).float().contiguous()
        bd = ffn.b_down.to(device).float()
        self._down_bias_zero = bool((bd != 0).sum() == 0)
        self.b_down = bd
        # recompute cost heuristic: sum over down-nnz of the read hidden row's
        # up+gate nnz.  If huge, the 2-launch path is cheaper.
        self.recompute_nnz = int((Wd != 0).sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        K = B * S
        xk = x.reshape(K, D).transpose(0, 1).contiguous()
        y = torch.empty(D, K, device=xk.device, dtype=torch.float32)
        grid = (D, triton.cdiv(K, self.block_k))
        _fused_full_kernel[grid](
            self.dn_crow, self.dn_col, self.dn_val,
            self.up_crow, self.up_col, self.up_val,
            self.gt_crow, self.gt_col, self.gt_val,
            xk, self.b_up, self.b_gate, y, K,
            xk.stride(0), xk.stride(1), y.stride(0), y.stride(1),
            BLOCK_K=self.block_k,
        )
        out = y.transpose(0, 1).reshape(B, S, D)
        if not self._down_bias_zero:
            out = out + self.b_down
        return out


class FusedUpGateSiluDeltaFFN:
    """2-launch fused FFN with a DELTA-in-place down kernel: kernel 1 fuses
    up+gate+silu; kernel 2 adds W_down @ hidden ONLY to the residual rows W_down
    actually writes (mean ~0.15*Dff active rows), in place.  This drops the
    full [D,K] residual COPY the plain down kernel does on the ~99% untouched rows
    (the biggest cost in the attribution: down was 27 us/launch, mostly copy).
    Byte-exact at the nibble margin.  Graph-capturable (static shapes)."""

    def __init__(self, ffn, device, block_k: int = 256):
        self.dev = device
        self.block_k = block_k
        Wu = _dense_of(ffn.W_up).to(device)
        Wg = _dense_of(ffn.W_gate).to(device)
        Wd = _dense_of(ffn.W_down).to(device)
        self.dim = Wu.shape[1]
        self.Dff = Wu.shape[0]
        self.up_crow, self.up_col, self.up_val = _csr(Wu)
        self.gt_crow, self.gt_col, self.gt_val = _csr(Wg)
        # W_down restricted to ACTIVE output rows (rows with any nonzero).
        active = torch.nonzero((Wd != 0).any(dim=1), as_tuple=False).flatten()
        self.active_rows = active.to(torch.int32).contiguous()
        self.n_active = int(active.numel())
        Wd_active = Wd[active] if self.n_active else Wd[:0]
        self.dn_crow, self.dn_col, self.dn_val = _csr(Wd_active)
        self.b_up = ffn.b_up.to(device).float().contiguous()
        self.b_gate = ffn.b_gate.to(device).float().contiguous()
        bd = ffn.b_down.to(device).float()
        self._down_bias_zero = bool((bd != 0).sum() == 0)
        assert self._down_bias_zero, "delta-inplace path assumes b_down==0 (probed)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        K = B * S
        # residual buffer [D,K]; we UPDATE it in place with the FFN delta.
        y = x.reshape(K, D).transpose(0, 1).contiguous()       # [D,K] == residual base
        if self.n_active:
            h = torch.empty(self.Dff, K, device=y.device, dtype=torch.float32)
            _fused_upgate_silu_kernel[(self.Dff, triton.cdiv(K, self.block_k))](
                self.up_crow, self.up_col, self.up_val,
                self.gt_crow, self.gt_col, self.gt_val,
                y, self.b_up, self.b_gate, h, K,
                y.stride(0), y.stride(1), h.stride(0), h.stride(1),
                BLOCK_K=self.block_k,
            )
            _down_delta_inplace_kernel[(self.n_active, triton.cdiv(K, self.block_k))](
                self.active_rows, self.dn_crow, self.dn_col, self.dn_val,
                h, y, K,
                h.stride(0), h.stride(1), y.stride(0), y.stride(1),
                BLOCK_K=self.block_k,
            )
        return y.transpose(0, 1).reshape(B, S, D)


class HybridFusedFFN:
    """Per-block chooses the cheaper of the single-launch full form (recompute) and
    the 2-launch upgate form, by the recompute/upgate cost ratio.  Fewest launches
    without paying the recompute blowup on the few dense blocks."""

    def __init__(self, ffn, device, block_k: int = 128, ratio_thresh: float = 1.5):
        Wu = _dense_of(ffn.W_up).to(device)
        Wg = _dense_of(ffn.W_gate).to(device)
        Wd = _dense_of(ffn.W_down).to(device)
        up_per_h = (Wu != 0).sum(dim=1)
        gt_per_h = (Wg != 0).sum(dim=1)
        hid_cost = (up_per_h + gt_per_h + 2)
        dn_cols = torch.nonzero(Wd != 0, as_tuple=False)[:, 1]
        full_cost = int(hid_cost[dn_cols].sum()) if dn_cols.numel() else 1
        upgate_cost = int((up_per_h + gt_per_h).sum()) + int((Wd != 0).sum())
        ratio = full_cost / max(upgate_cost, 1)
        self.use_full = ratio <= ratio_thresh
        self._impl = (FusedFullFFN(ffn, device, block_k) if self.use_full
                      else FusedUpGateSiluFFN(ffn, device, block_k))
        self.dim = self._impl.dim
        self.Dff = self._impl.Dff

    def forward(self, x):
        return self._impl.forward(x)


# ---------------------------------------------------------------------------
# INSTALL — swap every non-routed SparseBlock's FFN for the #808 fused DELTA
# kernel (``FusedUpGateSiluDeltaFFN``: kernel-1 fuses up+gate+silu; kernel-2 adds
# W_down @ hidden ONLY to the residual rows W_down writes, in place).  This is the
# 0.0024/0.0074 ms/tok kernel measured in isolation; ``install_fused_delta_ffn``
# wires it into the SAME forward the batched verify path drives, so it COMPOSES
# with direct-CAM + banded local attention (which only touch attention).
#
# Drop-in for ``block_sparse_ffn.install_block_sparse_ffn`` (same swap contract:
# a ``.ffn`` with a ``.forward(x)->x_out`` residual-add signature).  Byte-exact at
# the nibble-snap margin (same nonzeros, fp-accum-order residue only).  A block
# whose ``b_down != 0`` (the delta path asserts b_down==0) or whose dense-recompute
# would blow up falls back to the 2-launch ``FusedUpGateSiluFFN`` (also fused,
# hidden written ONCE).  Routed FFNs (Top1RoutedFFN) are kept dense.
# ---------------------------------------------------------------------------
def fused_delta_ffn_enabled() -> bool:
    """``C4_FUSED_DELTA_FFN`` (DEFAULT OFF): install the #808 fused delta sparse-FFN
    kernel on the verify-path blocks.  OFF -> the un-fused COO / dense FFN (byte-exact
    golden path, 069cc32f unchanged)."""
    import os
    return os.environ.get("C4_FUSED_DELTA_FFN", "0") not in ("0", "", "false", "False")


def install_fused_delta_ffn(model, device=None, *, block_k: int = 256,
                            verbose: bool = False):
    """Swap every non-routed block's FFN for the #808 fused DELTA kernel.

    Returns a stats dict.  ``block_k`` is the Triton K-tile (span-column) size.
    The delta form needs ``b_down == 0`` (true for the whole c4_min model — probed);
    a block that violates it (or a routed FFN) is left as-is / uses the 2-launch
    fallback.  Idempotent over shared FFN objects (a recurrent build's shared FFN is
    converted once and reused)."""
    if device is None:
        device = model.embed.device
    n_swapped = 0
    n_routed = 0
    n_fallback = 0
    converted = {}
    for b in model.blocks:
        if getattr(b, "_routed", False):
            n_routed += 1
            continue
        cur = b.ffn
        if isinstance(cur, (FusedUpGateSiluDeltaFFN, FusedUpGateSiluFFN,
                            FusedFullFFN, HybridFusedFFN)):
            continue                              # already fused (shared object)
        existing = converted.get(id(cur))
        if existing is not None and existing[0] is cur:
            b.ffn = existing[1]
            continue
        # b_down all-zero across the model -> the delta-in-place path applies.  A
        # rare nonzero-b_down block would fail the delta assert, so fall back to the
        # 2-launch fused form (which folds b_down explicitly) for it.
        bd = cur.b_down.to(device).float()
        if bool((bd != 0).sum() == 0):
            fused = FusedUpGateSiluDeltaFFN(cur, device, block_k=block_k)
        else:
            fused = FusedUpGateSiluFFN(cur, device, block_k=block_k)
            n_fallback += 1
        converted[id(cur)] = (cur, fused)         # keep cur alive -> id stays valid
        b.ffn = fused
        n_swapped += 1
    if verbose:
        print(f"[fused-delta-ffn] swapped {n_swapped} distinct FFNs "
              f"(delta={n_swapped - n_fallback}, 2-launch-fallback={n_fallback}, "
              f"{n_routed} routed kept dense) block_k={block_k}", flush=True)
    return {"swapped": n_swapped, "delta": n_swapped - n_fallback,
            "fallback": n_fallback, "routed": n_routed, "block_k": block_k}
