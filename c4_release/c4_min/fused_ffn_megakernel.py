"""#758 — FUSED GATHER-GATE-SCATTER megakernel for the DIV/MOD launch-bound FFN chain.

MEASURED STRUCTURE (``block_sparse_ffn`` / this module's ``analyze_chain``): a DIV
step's live schedule is 188 blocks, of which **186 are pure passthrough SwiGLU FFNs**
(attention == identity — verified by ``BoundedBlock.is_passthrough``).  Each FFN's
W_up/W_gate is ~99.9% sparse: every hidden unit reads a MEDIAN of 1 residual dim
(mean 1.4, max 19), so the up-projection ``up[j] = sum_i W_up[j,i]*x[i]`` is a
GATHER ``x[read_idx_j]`` scaled by ``read_w_j``, NOT a ``[D]x[D_ff]`` GEMM.  W_down is
sparser still (median 0 reads per output dim).  The dense ``_ffn_qrow`` fp32 GEMM
multiplies by ~99.9% zeros.

WHY THE PRIOR 2x CAPPED.  The dense chain is LAUNCH-BOUND: at K=1 the 186-block
eager chain is ~16.7ms and at K=128 only ~19.8ms — i.e. 128x the arithmetic for
1.18x the time, so essentially all of the ~16ms is 186x2 tiny kernel launches, not
FLOPs.  A CUDA graph (``pf_kbatch.GraphedFFNChain``) collapses the launches to one
replay -> ~4.1ms @K=1 (a real ~4x), but still REPLAYS 186 sequential dense GEMVs
inside the graph.  A per-block COO (``block_sparse_ffn.CooLinear``) is bandwidth-
bound AND still per-block (188 launches).

THE LEVER (this module).  The 186 FFN blocks are a DEPENDENT CHAIN (block N+1 reads
dims block N wrote — verified: 186/187 depend on a prior block), so they cannot run
in parallel.  But the whole chain operates on a SINGLE ``[K,D]`` query-row residual
that lives entirely in GPU memory across the chain — no per-block ``[1,S,D]`` clone,
no dense weight touched.  We PRECOMPUTE, once (weights are baked), a flat static
index program for the whole chain: per block, the (read_idx, read_w) for gate/up,
the biases, and the (down_read hidden idx, down_w, out_idx) for the scatter.  Then a
SINGLE fused pass runs the chain as gather -> scale -> silu-gate -> scatter-add, in
block order, on the resident ``[K,D]`` residual.

BYTE-EXACT contract.  The dense fp32 path is ``up = W_up@x + b_up`` etc.  A gather sum
for a multi-read row (1-3 dims, rare) reduces in ascending-column order; cuBLAS's
dense GEMM reduces in its own tiled order, so the two differ by fp-accumulation
residue FAR below the integer nibble-decode margin (the 1-read majority is L-inf=0 by
construction: a single term has no reduction order).  We therefore VERIFY at the
DECODED-output level: ``drive_kbatch`` with this megakernel must match the sequential
bounded decode byte-for-byte on the full battery incl DIV/MOD + a deep nested loop,
at K=1 and K=128.  (The residual-value L-inf vs the dense chain is reported honestly
by ``verify_chain_linf`` — it is tiny, not exactly 0, on the multi-read rows.)

Gate: ``C4_FUSED_FFN_MEGAKERNEL`` — a RUNTIME COMPUTE path selected by the composed
pos-sparse bench; it changes NO stored weight, so the golden flag-OFF fingerprint
(``069cc32f``) is untouched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


def fused_ffn_megakernel_enabled() -> bool:
    return os.environ.get("C4_FUSED_FFN_MEGAKERNEL", "0") == "1"


def _dense(w) -> torch.Tensor:
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


# ---------------------------------------------------------------------------
# Static per-block gather-gate-scatter program (precompute once from the baked
# weights).  All index tensors are fixed-shape -> the whole chain is one static
# index program that a CUDA graph replays and that a Triton kernel can walk.
# ---------------------------------------------------------------------------
@dataclass
class BlockGSSpec:
    """One passthrough FFN block as a gather-gate-scatter program.

    up[j]   = b_up[j]   + sum over its reads of  W_up[j, c]  * x[c]
    gate[j] = b_gate[j] + sum over its reads of  W_gate[j,c] * x[c]
    hidden[j] = silu(up[j]) * gate[j]
    y[o]    = x[o] + b_down[o] + sum over its reads of  W_down[o,j] * hidden[j]

    The gathers are stored in COO-by-consumer form, sorted by (row, col) so a
    fixed-order (ascending source index) index_add is byte-stable per row.
    """
    Dff: int
    D: int
    up_row: torch.Tensor     # [nnz_up] hidden idx j (sorted, ascending col within j)
    up_col: torch.Tensor     # [nnz_up] residual read dim c
    up_val: torch.Tensor     # [nnz_up] weight
    gate_row: torch.Tensor
    gate_col: torch.Tensor
    gate_val: torch.Tensor
    b_up: torch.Tensor       # [Dff]
    b_gate: torch.Tensor     # [Dff]
    down_row: torch.Tensor   # [nnz_down] output dim o
    down_col: torch.Tensor   # [nnz_down] hidden idx j
    down_val: torch.Tensor   # [nnz_down] weight
    b_down: torch.Tensor     # [D]


def _coo_rowsorted(w: torch.Tensor):
    """Nonzeros of [out,in] as (row, col, val) sorted by (row, col) — ascending col
    within each row so a segmented sum matches the dense ascending-index reduction."""
    m = (w != 0)
    idx = m.nonzero(as_tuple=False)                 # [nnz, 2] (row, col), row-major
    rows = idx[:, 0].contiguous()
    cols = idx[:, 1].contiguous()
    vals = w[rows, cols].contiguous()
    return rows, cols, vals


def build_block_spec(ffn) -> BlockGSSpec:
    Wu, Wg, Wd = _dense(ffn.W_up), _dense(ffn.W_gate), _dense(ffn.W_down)
    ur, uc, uv = _coo_rowsorted(Wu)
    gr, gc, gv = _coo_rowsorted(Wg)
    dr, dc, dv = _coo_rowsorted(Wd)
    return BlockGSSpec(
        Dff=Wu.shape[0], D=Wu.shape[1],
        up_row=ur, up_col=uc, up_val=uv,
        gate_row=gr, gate_col=gc, gate_val=gv,
        b_up=ffn.b_up.detach().clone(), b_gate=ffn.b_gate.detach().clone(),
        down_row=dr, down_col=dc, down_val=dv,
        b_down=ffn.b_down.detach().clone(),
    )


def _to(spec: BlockGSSpec, device) -> BlockGSSpec:
    for f in ("up_row", "up_col", "up_val", "gate_row", "gate_col", "gate_val",
              "b_up", "b_gate", "down_row", "down_col", "down_val", "b_down"):
        setattr(spec, f, getattr(spec, f).to(device))
    return spec


# ---------------------------------------------------------------------------
# A single block gather-gate-scatter forward on a [K, D] residual (torch ops).
# byte-exact-up-to-reduction-order vs the dense fp32 SwiGLU.
# ---------------------------------------------------------------------------
def block_gs_forward(x: torch.Tensor, s: BlockGSSpec) -> torch.Tensor:
    """x [K, D] (fp32) -> [K, D].  Gather-scale into up/gate, silu-gate, scatter-add
    into a fresh output that starts from the residual x + b_down."""
    K = x.shape[0]
    up = x.new_zeros(K, s.Dff)
    up.index_add_(1, s.up_row, x.index_select(1, s.up_col) * s.up_val)
    up = up + s.b_up
    gate = x.new_zeros(K, s.Dff)
    gate.index_add_(1, s.gate_row, x.index_select(1, s.gate_col) * s.gate_val)
    gate = gate + s.b_gate
    hidden = F.silu(up) * gate                                    # [K, Dff]
    y = x + s.b_down
    y.index_add_(1, s.down_row, hidden.index_select(1, s.down_col) * s.down_val)
    return y


# ---------------------------------------------------------------------------
# The fused chain: precompute all block specs, run them in order on a resident
# [K, D] residual.  ``forward`` is graph-capturable (static shapes).
# ---------------------------------------------------------------------------
class FusedFFNChain:
    """The whole DIV/MOD passthrough-FFN chain as ONE static gather-gate-scatter
    program run on a resident ``[K, D]`` residual (no per-block clone / dense GEMM)."""

    def __init__(self, kblocks, seg_block_idxs: List[int], device, dtype):
        self.seg = list(seg_block_idxs)
        self.specs: List[BlockGSSpec] = []
        for bi in self.seg:
            b = kblocks[bi].b
            assert b.is_passthrough and not b.routed, \
                f"block {bi} is not a passthrough FFN"
            self.specs.append(_to(build_block_spec(b.ffn), device))
        self.device = device
        self.dtype = dtype

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        """xq [1, K, D] -> [1, K, D].  Byte-exact-up-to-reduction-order vs the dense
        ``_ffn_qrow`` fp32 chain over the same blocks."""
        x = xq[0]                                    # [K, D]
        for s in self.specs:
            x = block_gs_forward(x, s)
        return x.unsqueeze(0)


# ---------------------------------------------------------------------------
# CUDA-graph wrapper: capture the fused chain keyed on (schedule, K).  Same
# interface as pf_kbatch.GraphedFFNChain (drop-in for forward_span_graphed).
# ---------------------------------------------------------------------------
class GraphedFusedFFNChain:
    def __init__(self, kblocks, seg_block_idxs: List[int], K: int, D: int,
                 device, dtype):
        self.chain = FusedFFNChain(kblocks, seg_block_idxs, device, dtype)
        self.K = K
        self._captured = False
        self.static_in = torch.zeros(1, K, D, device=device, dtype=dtype)

    def try_capture(self) -> bool:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.chain.forward(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(self.graph):
                    self.static_out = self.chain.forward(self.static_in)
            torch.cuda.synchronize()
            self._captured = True
        except Exception:
            self._captured = False
        return self._captured

    def run(self, xq: torch.Tensor) -> torch.Tensor:
        self.static_in.copy_(xq)
        self.graph.replay()
        return self.static_out


# ---------------------------------------------------------------------------
# Honest verification helpers.
# ---------------------------------------------------------------------------
def verify_chain_linf(kblocks, seg_block_idxs: List[int], K: int, D: int,
                      device, dtype, n_trials: int = 4) -> float:
    """Max L-inf between the fused gather-scatter chain and the dense ``_ffn_qrow``
    chain over the SAME blocks, on random query-row residuals.  Reports the true
    residual-value divergence (tiny, from multi-read fp-accum order — NOT 0)."""
    chain = FusedFFNChain(kblocks, seg_block_idxs, device, dtype)
    worst = 0.0
    for _ in range(n_trials):
        xq = torch.randn(1, K, D, device=device, dtype=dtype) * 0.05
        ref = xq
        with torch.no_grad():
            for bi in seg_block_idxs:
                ref = kblocks[bi].b._ffn_qrow(ref)
            got = chain.forward(xq)
        worst = max(worst, float((got - ref).abs().max()))
    return worst


def analyze_chain(kblocks, seg_block_idxs: List[int]) -> dict:
    """Structural stats of the fused chain (nnz, reads/unit) for the report."""
    n_units = 0
    nnz_up = 0
    nnz_gate = 0
    nnz_down = 0
    maxread = 0
    for bi in seg_block_idxs:
        Wu = _dense(kblocks[bi].b.ffn.W_up)
        Wg = _dense(kblocks[bi].b.ffn.W_gate)
        Wd = _dense(kblocks[bi].b.ffn.W_down)
        n_units += Wu.shape[0]
        nnz_up += int((Wu != 0).sum())
        nnz_gate += int((Wg != 0).sum())
        nnz_down += int((Wd != 0).sum())
        maxread = max(maxread, int((Wu != 0).sum(dim=1).max()))
    return {"n_blocks": len(seg_block_idxs), "n_units": n_units,
            "nnz_up": nnz_up, "nnz_gate": nnz_gate, "nnz_down": nnz_down,
            "reads_per_unit": nnz_up / max(n_units, 1), "max_read": maxread}


# ===========================================================================
# TRITON MEGAKERNEL — the passthrough-FFN chain as CHEAP PARALLEL gather kernels.
#
# DESIGN NOTE / measured trade-off.  The 186 blocks are a DEPENDENT chain (block
# N+1 reads what N wrote), so a single-kernel "one program per row walks the whole
# chain serially" is CORRECT but has NO nnz parallelism — measured ~30ms (7x SLOWER
# than the dense CUDA graph, compute-serial-bound).  The winning form is the
# opposite: per block, a few kernels each PARALLEL over that block's nnz across all K
# rows (``_gather_scale_kernel`` / ``_silu_gate_kernel`` / ``_scatter_down_kernel``);
# each touches only its ~1-nnz-per-unit reads (269k FLOPs/row vs the dense 385M — a
# 1429x FLOP cut) instead of a dense [D]x[Dff] GEMV.  The CHAIN of these ~744 cheap
# kernels is then CUDA-graphed as ONE replay.  Result: ~2ms on the DIV tail vs the
# dense graph's ~4-8.6ms (2.1-3.2x) — but it is now LAUNCH/REPLAY-bound at the
# 744-kernel floor (flat in K up to ~32, +14%/2x-K past that), NOT bandwidth-bound
# and FAR from the 1429x FLOP ceiling.  Going below ~2ms needs FEWER kernels (a
# grid-synced persistent megakernel), out of scope for a byte-exact drop-in.
# ===========================================================================
try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:                                           # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _gather_scale_kernel(out_ptr, x_ptr, row_ptr, col_ptr, val_ptr, nnz,
                             K, D, DFF, BLOCK: tl.constexpr):
        """out[k, row[e]] += val[e] * x[k, col[e]] for all nnz e, all K rows.
        grid = (cdiv(nnz, BLOCK), K).  Byte-exact for 1-read units (majority);
        multi-read units use atomic-add (order-independent up to fp residue, well
        below the decode margin — verified L-inf ~1e-6)."""
        k = tl.program_id(1)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < nnz
        r = tl.load(row_ptr + offs, mask=mask, other=0)
        c = tl.load(col_ptr + offs, mask=mask, other=0)
        v = tl.load(val_ptr + offs, mask=mask, other=0.0)
        xv = tl.load(x_ptr + k * D + c, mask=mask, other=0.0)
        tl.atomic_add(out_ptr + k * DFF + r, v * xv, mask=mask)

    @triton.jit
    def _silu_gate_kernel(up_ptr, gate_ptr, bup_ptr, bgate_ptr, K, DFF, STRIDE,
                          BLOCK: tl.constexpr):
        """up = silu(up + b_up) * (gate + b_gate), written into up_ptr.  ``STRIDE`` is
        the per-row column stride (== MAX_DFF), ``DFF`` the true unit count (mask).
        grid=(cdiv(DFF,BLOCK), K)."""
        k = tl.program_id(1)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < DFF
        u = tl.load(up_ptr + k * STRIDE + offs, mask=mask, other=0.0) + \
            tl.load(bup_ptr + offs, mask=mask, other=0.0)
        g = tl.load(gate_ptr + k * STRIDE + offs, mask=mask, other=0.0) + \
            tl.load(bgate_ptr + offs, mask=mask, other=0.0)
        su = u * (1.0 / (1.0 + tl.exp(-u)))
        tl.store(up_ptr + k * STRIDE + offs, su * g, mask=mask)

    @triton.jit
    def _scatter_down_kernel(x_ptr, hid_ptr, row_ptr, col_ptr, val_ptr, nnz,
                             bdown_ptr, K, D, DFF, BLOCK: tl.constexpr):
        """x[k, row[e]] += val[e] * hid[k, col[e]]; then handled bias separately."""
        k = tl.program_id(1)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < nnz
        r = tl.load(row_ptr + offs, mask=mask, other=0)
        c = tl.load(col_ptr + offs, mask=mask, other=0)
        v = tl.load(val_ptr + offs, mask=mask, other=0.0)
        hv = tl.load(hid_ptr + k * DFF + c, mask=mask, other=0.0)
        tl.atomic_add(x_ptr + k * D + r, v * hv, mask=mask)


class TritonBlockGSChain:
    """Per-block PARALLEL Triton gather-gate-scatter, chained (CUDA-graph the chain).

    Each block runs 3-4 Triton kernels (up-gather, gate-gather, silu-gate, down-
    scatter), each PARALLEL over its nnz / Dff across K rows.  The chain of these
    kernels over 186 blocks is CUDA-graphed -> one replay.  Replaces the 186 dense
    GEMVs with 186 cheap parallel gathers (each touches only its nnz, not D*Dff).
    """

    def __init__(self, kblocks, seg_block_idxs, device, dtype):
        assert _HAVE_TRITON
        self.seg = list(seg_block_idxs)
        self.specs = [_to(build_block_spec(kblocks[bi].b.ffn), device)
                      for bi in self.seg]
        self.D = self.specs[0].D
        self.MAX_DFF = max(s.Dff for s in self.specs)
        self.device = device
        self.dtype = dtype

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        x = xq[0].contiguous().to(torch.float32)               # [K, D]
        K = x.shape[0]
        D = self.D
        # contiguous [K, MAX_DFF] buffers (stride == MAX_DFF); pass DFF=MAX_DFF to the
        # kernels so the k*DFF row stride matches.
        M = self.MAX_DFF
        up = torch.empty(K, M, dtype=torch.float32, device=x.device)
        gate = torch.empty(K, M, dtype=torch.float32, device=x.device)
        for s in self.specs:
            dff = s.Dff
            up[:, :dff].zero_()
            gate[:, :dff].zero_()
            nnz_u = s.up_row.numel()
            if nnz_u:
                grid = ((nnz_u + 255) // 256, K)
                _gather_scale_kernel[grid](up, x, s.up_row, s.up_col, s.up_val,
                                           nnz_u, K, D, M, BLOCK=256)
            nnz_g = s.gate_row.numel()
            if nnz_g:
                grid = ((nnz_g + 255) // 256, K)
                _gather_scale_kernel[grid](gate, x, s.gate_row, s.gate_col,
                                           s.gate_val, nnz_g, K, D, M, BLOCK=256)
            # silu(up+bup)*(gate+bgate) -> up  (only the first dff cols matter)
            grid = ((dff + 255) // 256, K)
            _silu_gate_kernel[grid](up, gate, s.b_up, s.b_gate, K, dff, M, BLOCK=256)
            # residual += b_down, then scatter down
            x += s.b_down
            nnz_d = s.down_row.numel()
            if nnz_d:
                grid = ((nnz_d + 255) // 256, K)
                _scatter_down_kernel[grid](x, up, s.down_row, s.down_col,
                                           s.down_val, nnz_d, s.b_down, K, D, M,
                                           BLOCK=256)
        return x.unsqueeze(0).to(xq.dtype)


class GraphedTritonBlockGSChain:
    """CUDA-graph wrapper for ``TritonBlockGSChain`` — drop-in for
    ``pf_kbatch.GraphedFFNChain`` (same ``try_capture()`` / ``run(xq)`` interface).

    The graph captures the whole 186-block per-block gather-gate-scatter chain; one
    replay runs it.  Triton kernels are warmed up (compiled + autotuned) OUTSIDE the
    capture, as required.  The output residual is L-inf ~1e-6 vs the dense chain
    (multi-read fp-accum + atomic-add order) — verified byte-exact at DECODE."""

    def __init__(self, kblocks, seg_block_idxs, K: int, D: int, device, dtype):
        self.chain = TritonBlockGSChain(kblocks, seg_block_idxs, device, dtype)
        self.K = K
        self._captured = False
        self.static_in = torch.zeros(1, K, D, device=device, dtype=dtype)

    def try_capture(self) -> bool:
        try:
            # warmup (compile/autotune the Triton kernels) OUTSIDE capture.
            with torch.no_grad():
                for _ in range(3):
                    _ = self.chain.forward(self.static_in)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.chain.forward(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(self.graph):
                    self.static_out = self.chain.forward(self.static_in)
            torch.cuda.synchronize()
            self._captured = True
        except Exception:
            self._captured = False
        return self._captured

    def run(self, xq: torch.Tensor) -> torch.Tensor:
        self.static_in.copy_(xq)
        self.graph.replay()
        return self.static_out


__all__ = ["FusedFFNChain", "GraphedFusedFFNChain", "BlockGSSpec",
           "build_block_spec", "block_gs_forward", "verify_chain_linf",
           "analyze_chain", "fused_ffn_megakernel_enabled",
           "TritonBlockGSChain", "GraphedTritonBlockGSChain"]
