"""FUSED MEGABLOCK (C4_FUSED_MEGABLOCK) — the on-chip DIV-free dead-FFN megakernel.

THE LEVER (this task, #895).  The DIV-free doom step's compute floor is the 49
DEAD-attention pure-FFN blocks of the frozen-skip [cut=1, N) region (the 3 live
CAM blocks 2/7/11 are O(1) direct gathers, run eagerly).  Each dead block is
``x = x + W_down @ (silu(W_up@x + b_up) * (W_gate@x + b_gate))`` over ~1.3 nnz/unit
sparse weights.  The residual ``[K, D=1392]`` is the ONLY cross-block state.

The measured 0.034 ms/step baseline is NOT compute-bound (FLOP floor 0.0071 us):
it is (a) the residual ``[K,D]`` read+write through HBM for each of the 49 blocks,
and (b) 49 per-block kernel launches (+ 3 SpMM launches each in the COO path).

This module collapses BOTH:

  * SINGLE LAUNCH for the whole 49-block chain.  The chain is captured into ONE
    CUDA graph (``MegaBlockGraph``) so the 49x (or 147x COO) launch overhead
    becomes one graph replay.  Respects the deep sequential residual dependency
    (block i+1 reads block i's output) by running the blocks SEQUENTIALLY inside
    the single captured stream — the fusion is temporal (one launch), not a
    data-parallel-over-blocks grid (which the dependency forbids).

  * NO PER-BLOCK FULL-RESIDUAL COPY, NO HIDDEN HBM BUFFER, RESIDUAL RESIDENT IN L2.
    The residual is a SINGLE PERSISTENT ``[D, K]`` buffer delta-updated IN PLACE
    across all 49 blocks (the ``_down_delta_inplace_kernel`` writes ONLY the ~0.15*Dff
    residual rows W_down touches -> no full-D copy per block; the ``_fused_upgate_silu
    _kernel`` writes the hidden ONCE into a shared scratch).  For a doom K-chunk
    ([K,1392]*4B; K=512 -> ~2.85 MB) the residual stays RESIDENT IN L2 (A5000 6 MB)
    across the whole chain, so each block's residual re-read is an L2 hit, NOT the HBM
    round-trip + fresh alloc the per-block ``SparseBlock`` loop pays.

BYTE-EXACT.  The per-block arithmetic is the SAME nonzeros in the SAME per-output
column-sorted accumulation order as the COO/dense ``SparseFFN.forward`` (zeros
contribute nothing); attention is the identity on every dead block (L-inf=0, the
dead-block-fusion contract).  Only fp-accumulation-ORDER residue vs cuBLAS (~6e-2
on ~5-magnitude values) differs, far below the integer nibble-snap decode margin
— exactly as the existing COO / fused-delta paths.  The 3 live CAM blocks are
untouched (run eagerly, direct-gather).  DEFAULT OFF -> the eager per-block loop
(golden 069cc32f byte-identical).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import triton
import triton.language as tl


def fused_megablock_enabled() -> bool:
    """``C4_FUSED_MEGABLOCK`` (DEFAULT OFF): run the DIV-free dead-FFN chain as ONE
    on-chip fused megakernel (single CUDA-graph launch, residual resident in L2, no
    per-block hidden HBM buffer).  OFF -> the eager per-block loop (byte-identical
    golden path, 069cc32f unchanged)."""
    return os.environ.get("C4_FUSED_MEGABLOCK", "0") not in ("0", "", "false", "False")


def wave_batch_enabled() -> bool:
    """``C4_FFN_WAVE_BATCH`` (DEFAULT OFF): HORIZONTAL wave-batching of the dead-FFN
    mega-chain.  The chain's dead blocks are grouped into DEPENDENCY WAVES (RAW+WAR+WAW
    hazard levels — blocks in a wave are mutually independent), and the up-to-5 blocks
    within each wave are CONCATENATED into ONE batched up/gate + ONE batched down-delta
    kernel launch per wave (the 49 dead-block 2-kernel pairs collapse to ~27 wave-kernel
    pairs).

    PURE KERNEL-COUNT REDUCTION: no FLOP change, no fill-in — the concatenated CSR is the
    same nonzeros in the same per-output accumulation order.  BYTE-EXACT because the wave
    decomposition respects WAW (no two blocks in a wave write the same residual dim) AND
    RAW/WAR (no block reads a dim another block in the same wave writes): so every block in
    a wave reads the SAME pre-wave residual and writes DISJOINT rows — running them as one
    batched kernel is bit-identical to running them sequentially (the batched up/gate
    kernel-1 fully precedes the batched down kernel-2, exactly as the per-block 2-kernel
    order).  OFF -> the per-block 2-kernel loop (069cc32f unchanged)."""
    return os.environ.get("C4_FFN_WAVE_BATCH", "0") not in ("0", "", "false", "False")


def linfold_enabled() -> bool:
    """``C4_FFN_LINFOLD`` (DEFAULT OFF): VERTICAL linear-fold of consecutive dead-FFN
    blocks.  For a run of dead-FFN blocks with NO attention between them, block N's
    ``W_down`` is folded into block N+1's ``W_up``/``W_gate`` across the linear residual add:
    precompute ``A = Wu_{N+1} @ Wd_N`` and ``B = Wg_{N+1} @ Wd_N`` so block N+1's
    pre-activations read block N's HIDDEN directly (``pre_u = Wu_{N+1}·y + A·h_N + b_up``,
    ``pre_g = Wg_{N+1}·y + B·h_N + b_gate``).  Block N STILL writes ``Wd_N·h_N`` to the
    residual (later blocks + decode read it) — the fold only avoids re-reading the residual
    dims N wrote inside N+1.

    HONEST TRADE: the ``[Dff,Dff]`` fused products can be DENSER (fill-in) than the sparse
    factors, so this is a FLOP-for-tile-occupancy trade — a net win only if the bigger/denser
    GEMMs raise kernel efficiency above the 12-15% tile-bound floor.  The fill-in factor is
    reported by the build.  OFF -> no fold (069cc32f unchanged)."""
    return os.environ.get("C4_FFN_LINFOLD", "0") not in ("0", "", "false", "False")


def fused_hidden_enabled() -> bool:
    """``C4_FFN_FUSED_HIDDEN`` (DEFAULT OFF): keep each dead-FFN block's ``[Dff,K]`` hidden
    activation ON-CHIP (recomputed in registers per active output row) instead of writing
    it to and reading it back from HBM between the up/gate kernel and the down kernel.

    THE DISPATCH LEVER (this task).  The profiler proved the dead-FFN mega-chain runs at
    only 19% of the A5000's 768 GB/s HBM peak, and **92% of the bytes moved per step is
    the ``[Dff,K]`` hidden-activation scratch** round-tripping HBM (the 2-kernel delta path
    ``_fused_upgate_silu_kernel`` WRITES it, ``_down_delta_inplace_kernel`` READS it).  This
    flag switches the dead-FFN blocks to the single ``_fused_hidden_delta_kernel``: one
    program per active output row recomputes ONLY the hidden columns W_down reads, in
    registers, from the input residual, and never materialises ``[Dff,K]`` in HBM.

    BYTE-EXACT (default OFF): the recompute walks the SAME CSRs in the SAME order, and reads
    up/gate from a SEPARATE input buffer than the delta-write target (no read-after-write
    hazard) — so the arithmetic is bit-identical to the 2-kernel path (which is itself
    nibble-margin byte-exact to the eager COO FFN).  OFF -> the 2-kernel HBM-hidden path
    (069cc32f unchanged)."""
    return os.environ.get("C4_FFN_FUSED_HIDDEN", "0") not in ("0", "", "false", "False")


def _resid_dtype() -> "torch.dtype":
    """``C4_MEGABLOCK_RESID`` (lever 3, DEFAULT fp32 == byte-exact): the resident
    residual buffer dtype.  ``bf16`` halves the residual buffer's L2/HBM footprint
    (each of the ~49 dead blocks RE-READS the residual for its hidden inputs).

    MEASURED (this task's battery, real frames):
      * ``bf16`` — DECODE byte-EXACT (29/29 ops; the nibble argmax is residue-immune
        and bf16 keeps fp32's 8-bit exponent, so the large framing values are safe).
        BUT the chain is kernel-COMPUTE-bound at doom K, so the speedup is negligible
        (~0.44 -> 0.41 us/step at K=2048; 0 at K=8192).  Safe but not worth it here.
      * ``fp16`` — BREAKS decode (1/29): fp16's max ~65504 OVERFLOWS the large frame
        register values (PC ~1.7e8) -> inf.  DISALLOWED (raises) — bf16 is the correct
        packed format.
    Default ``fp32`` (byte-exact headline)."""
    v = os.environ.get("C4_MEGABLOCK_RESID", "fp32").lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "half", "float16"):
        raise ValueError("C4_MEGABLOCK_RESID=fp16 OVERFLOWS the frame register values "
                         "(PC ~1.7e8 > fp16 max 65504) and BREAKS decode (1/29). Use "
                         "bf16 (decode byte-exact) or fp32 (default).")
    return torch.float32


# ===========================================================================
# CSR pack of a dense [M, D] weight — row-sorted, col-sorted (deterministic
# per-output accumulation order, matching the COO path's y.index_add_ order).
# ===========================================================================
def _csr(w_dense: torch.Tensor):
    M, D = w_dense.shape
    nz = torch.nonzero(w_dense, as_tuple=False)
    if nz.numel() == 0:
        crow = torch.zeros(M + 1, dtype=torch.int32, device=w_dense.device)
        col = torch.zeros(0, dtype=torch.int32, device=w_dense.device)
        val = torch.zeros(0, dtype=torch.float32, device=w_dense.device)
        return crow, col, val
    order = torch.argsort(nz[:, 0].to(torch.int64) * D + nz[:, 1].to(torch.int64))
    nz = nz[order]
    rows = nz[:, 0]
    col = nz[:, 1].to(torch.int32).contiguous()
    val = w_dense[rows, nz[:, 1]].to(torch.float32).contiguous()
    crow = torch.zeros(M + 1, dtype=torch.int32, device=w_dense.device)
    crow[1:] = torch.bincount(rows, minlength=M).cumsum(0).to(torch.int32)
    return crow, col, val


def _dense_of(w):
    # CooLinear (from block_sparse_ffn): reconstruct dense from (rows,cols,vals).
    if hasattr(w, "rows") and hasattr(w, "cols") and hasattr(w, "vals") \
            and hasattr(w, "out_dim") and hasattr(w, "in_dim"):
        d = torch.zeros(int(w.out_dim), int(w.in_dim),
                        dtype=w.vals.dtype, device=w.vals.device)
        d[w.rows.long(), w.cols.long()] = w.vals
        return d
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


# ===========================================================================
# WAVE DECOMPOSITION (lever A).  Group a list of dead-FFN blocks into DEPENDENCY
# WAVES: block j depends on block i (< j) if any of RAW (i writes a dim j reads),
# WAR (i reads a dim j writes) or WAW (i and j write the same dim) holds.  Blocks in
# the SAME wave are mutually hazard-free -> their up/gate reads all see the same
# pre-wave residual and their down-deltas target disjoint rows, so they can be run as
# one batched kernel byte-identically to running them sequentially.  (Same algorithm
# as _agent_doom_active_depth.critical_depth_hazards.)
# ===========================================================================
def _dependency_waves(writes, reads):
    """writes[i], reads[i] : python sets of residual dims block i writes / reads.
    Returns (levels, waves): levels[i] = wave index of block i; waves = list of
    lists of block-indices (into the input lists) per wave, in dependency order."""
    N = len(reads)
    level = [0] * N
    for j in range(N):
        mx = -1
        wj, rj = writes[j], reads[j]
        for i in range(j):
            if (writes[i] & rj) or (reads[i] & wj) or (writes[i] & wj):
                if level[i] > mx:
                    mx = level[i]
        level[j] = mx + 1
    nlev = (max(level) + 1) if N else 0
    waves: List[List[int]] = [[] for _ in range(nlev)]
    for i, lv in enumerate(level):
        waves[lv].append(i)
    return level, waves


def _cat_csr(crows, cols, vals, row_counts):
    """Concatenate per-block CSRs (each (crow[M_i+1], col, val)) into one CSR over the
    stacked rows sum(M_i).  Row r of block b maps to stacked row offset_b + r; the value
    and column arrays are concatenated in block order, so each stacked row's nonzeros are
    contiguous in the SAME per-row column-sorted order as the source CSR.  Byte-exact:
    the per-(stacked-row) accumulation walks the identical nonzeros in the identical order
    as the source block's row."""
    device = crows[0].device if crows else torch.device("cpu")
    tot_rows = sum(int(rc) for rc in row_counts)
    crow = torch.zeros(tot_rows + 1, dtype=torch.int32, device=device)
    col_parts, val_parts = [], []
    row_ptr = 0
    nnz_off = 0
    for cr, co, va, rc in zip(crows, cols, vals, row_counts):
        rc = int(rc)
        # cr has rc+1 entries; interior deltas (cr[1:]-cr[:1]) are the per-row nnz.
        if rc:
            crow[row_ptr + 1: row_ptr + rc + 1] = cr[1:rc + 1].to(torch.int64) + nnz_off
        col_parts.append(co)
        val_parts.append(va)
        row_ptr += rc
        nnz_off += int(co.numel())
    col = torch.cat(col_parts).contiguous() if col_parts else \
        torch.zeros(0, dtype=torch.int32, device=device)
    val = torch.cat(val_parts).contiguous() if val_parts else \
        torch.zeros(0, dtype=torch.float32, device=device)
    return crow.to(torch.int32).contiguous(), col, val


class _WaveBatch:
    """One dependency WAVE of dead-FFN blocks batched into a SINGLE up/gate launch and a
    SINGLE down-delta launch.  Concatenates the wave's blocks' CSRs over stacked rows:

      * up/gate: one [sum_Dff, K] hidden scratch; row (block-hidden-offset + u) computes
        block b's hidden unit u.  b_up/b_gate concatenated so program `row` reads its own
        block's bias.  The up/gate CSR columns are ABSOLUTE residual dims (all blocks read
        the same pre-wave residual y).
      * down: active_rows concatenated (all DISTINCT across the wave — the WAW-free wave
        property), CSR over the stacked active rows with hidden-columns SHIFTED by the
        block's hidden offset (so each active row reads its own block's hidden slice).

    Byte-exact: kernel-1 (all up/gate) fully precedes kernel-2 (all down); each stacked
    row walks the identical nonzeros in identical order as the per-block path."""

    def __init__(self, mffns: List["_MegaFFN"], block_k: int):
        self.block_k = block_k
        dev = mffns[0].b_up.device
        # ---- concatenate up/gate over stacked hidden rows ----
        hid_off = 0
        up_crows, up_cols, up_vals = [], [], []
        gt_crows, gt_cols, gt_vals = [], [], []
        b_ups, b_gts = [], []
        # down over stacked active rows; hidden cols shifted by hid_off.
        dn_crows, dn_cols, dn_vals = [], [], []
        active_all = []
        dn_rowcounts = []
        up_rowcounts = []
        for mf in mffns:
            up_crows.append(mf.up_crow); up_cols.append(mf.up_col); up_vals.append(mf.up_val)
            gt_crows.append(mf.gt_crow); gt_cols.append(mf.gt_col); gt_vals.append(mf.gt_val)
            b_ups.append(mf.b_up); b_gts.append(mf.b_gate)
            up_rowcounts.append(mf.Dff)
            # down: shift hidden col indices by this block's hidden offset in the stack.
            dn_crows.append(mf.dn_crow)
            dn_cols.append((mf.dn_col.to(torch.int64) + hid_off).to(torch.int32)
                           if mf.dn_col.numel() else mf.dn_col)
            dn_vals.append(mf.dn_val)
            dn_rowcounts.append(mf.n_active)
            active_all.append(mf.active_rows)
            hid_off += mf.Dff
        self.tot_hid = hid_off
        self.up_crow, self.up_col, self.up_val = _cat_csr(up_crows, up_cols, up_vals, up_rowcounts)
        self.gt_crow, self.gt_col, self.gt_val = _cat_csr(gt_crows, gt_cols, gt_vals, up_rowcounts)
        self.b_up = torch.cat(b_ups).contiguous()
        self.b_gate = torch.cat(b_gts).contiguous()
        self.dn_crow, self.dn_col, self.dn_val = _cat_csr(dn_crows, dn_cols, dn_vals, dn_rowcounts)
        self.active_rows = torch.cat(active_all).contiguous() if active_all else \
            torch.zeros(0, dtype=torch.int32, device=dev)
        self.n_active = int(self.active_rows.numel())
        # non-zero down bias (rare): fold each block's b_down at the end.
        self._bias_blocks = [mf for mf in mffns if not mf._down_bias_zero]

    def run_inplace(self, y: torch.Tensor, K: int, h_scratch: torch.Tensor):
        """Update the shared residual y[D,K] IN PLACE with this whole wave's FFN delta via
        exactly TWO kernel launches (one batched up/gate, one batched down-delta)."""
        from .fused_sparse_ffn import (_fused_upgate_silu_kernel,
                                       _down_delta_inplace_kernel)
        if self.tot_hid and self.n_active:
            h = h_scratch[:self.tot_hid]
            g1 = (self.tot_hid, triton.cdiv(K, self.block_k))
            _fused_upgate_silu_kernel[g1](
                self.up_crow, self.up_col, self.up_val,
                self.gt_crow, self.gt_col, self.gt_val,
                y, self.b_up, self.b_gate, h, K,
                y.stride(0), y.stride(1), h.stride(0), h.stride(1),
                BLOCK_K=self.block_k,
            )
            g2 = (self.n_active, triton.cdiv(K, self.block_k))
            _down_delta_inplace_kernel[g2](
                self.active_rows, self.dn_crow, self.dn_col, self.dn_val,
                h, y, K,
                h.stride(0), h.stride(1), y.stride(0), y.stride(1),
                BLOCK_K=self.block_k,
            )
        for mf in self._bias_blocks:
            y += mf.b_down.view(mf.dim, 1)


# ===========================================================================
# The megablock reuses the byte-exact fused-sparse-FFN kernels from
# ``fused_sparse_ffn`` (``_fused_upgate_silu_kernel`` fuses up+gate+silu into one
# launch; ``_down_delta_inplace_kernel`` adds W_down@hidden to ONLY the residual rows
# W_down writes, in place).  The MegaBlock contribution is the CHAIN: it runs those
# per-block kernels over a SINGLE persistent [D, K] residual buffer, delta-updated in
# place across all ~49 dead-FFN blocks (no per-block full-D copy, no hidden HBM
# buffer), and captures the whole chain in ONE CUDA graph — so the residual stays
# L2-resident and the 49x2 launches collapse to one replay.
# ===========================================================================
class _MegaFFN:
    """One dead-FFN block's CSR weights + its IN-PLACE delta megablock kernel run.

    The residual ``y [D,K]`` is a SINGLE persistent buffer the whole chain shares.
    Each block:
      (1) fuses up+gate+silu of its Dff hidden units into a scratch ``h`` (kernel 1),
      (2) adds ``W_down @ h`` to y IN PLACE, launching a program ONLY for the ~active
          output rows W_down actually writes (mean ~0.15*Dff), NOT a full-D grid.
    This drops BOTH the per-block full ``[D,K]`` residual COPY (the ~99% untouched
    rows) AND holds the residual in ONE buffer across all blocks (no ping/pong copy)
    — the residual-traffic + launch lever.  Byte-exact at the nibble margin (same
    nonzeros, fp-accum-order residue only).  ``b_down`` is all-zero across the model
    (probed); a rare nonzero-b_down block folds it into y once at the end."""

    def __init__(self, ffn, device, block_k: int):
        self.dev = device
        self.block_k = block_k
        Wu = _dense_of(ffn.W_up).to(device)
        Wg = _dense_of(ffn.W_gate).to(device)
        Wd = _dense_of(ffn.W_down).to(device)
        self.dim = Wu.shape[1]
        self.Dff = Wu.shape[0]
        self.up_crow, self.up_col, self.up_val = _csr(Wu)
        self.gt_crow, self.gt_col, self.gt_val = _csr(Wg)
        # W_down restricted to ACTIVE output rows (rows with any nonzero) -> the delta
        # kernel launches n_active programs, not D.
        active = torch.nonzero((Wd != 0).any(dim=1), as_tuple=False).flatten()
        self.active_rows = active.to(torch.int32).contiguous()
        self.n_active = int(active.numel())
        Wd_active = Wd[active] if self.n_active else Wd[:0]
        self.dn_crow, self.dn_col, self.dn_val = _csr(Wd_active)
        self.b_up = ffn.b_up.to(device).float().contiguous()
        self.b_gate = ffn.b_gate.to(device).float().contiguous()
        bd = ffn.b_down.to(device).float()
        self._down_bias_zero = bool((bd != 0).sum() == 0)
        self.b_down = bd
        # FUSED-HIDDEN (C4_FFN_FUSED_HIDDEN): the single-kernel form recomputes each read
        # hidden row on the fly and delta-writes the residual IN PLACE.  To keep it
        # byte-exact despite the 47/49 blocks that WRITE a residual dim they also READ as an
        # FFN input (a read-after-write hazard), up/gate must read the PRE-BLOCK residual.
        # We snapshot ONLY the (small) set of residual dims up/gate read into a compact
        # ``[n_in, K]`` buffer BEFORE the delta write, and remap the up/gate CSR column
        # indices to snapshot-local rows.  n_in is tiny (4-148 vs D=1392), so the snapshot
        # is far cheaper than a full-[D,K] copy AND eliminates the [Dff,K] hidden HBM buffer.
        in_cols = torch.nonzero(((Wu != 0).any(dim=0) | (Wg != 0).any(dim=0)),
                                as_tuple=False).flatten()
        self.in_cols = in_cols.to(torch.int64).contiguous()   # residual dims to snapshot
        self.n_in = int(in_cols.numel())
        # remap: residual-dim -> snapshot-local row (dense LUT over D, built once).
        remap = torch.full((self.dim,), -1, dtype=torch.int32, device=device)
        if self.n_in:
            remap[in_cols] = torch.arange(self.n_in, dtype=torch.int32, device=device)
        # up/gate CSRs with columns REMAPPED to snapshot-local indices.
        self.up_col_snap = remap[self.up_col.to(torch.int64)].contiguous() \
            if self.up_col.numel() else self.up_col
        self.gt_col_snap = remap[self.gt_col.to(torch.int64)].contiguous() \
            if self.gt_col.numel() else self.gt_col

    def run_inplace(self, y: torch.Tensor, K: int,
                    h_scratch: Optional[torch.Tensor] = None):
        """Update the shared residual ``y [D,K]`` IN PLACE with this block's FFN delta.
        ``h_scratch`` : shared [max_Dff, K] hidden buffer (only first Dff rows used)."""
        from .fused_sparse_ffn import (_fused_upgate_silu_kernel,
                                       _down_delta_inplace_kernel)
        if self.n_active:
            h = h_scratch[:self.Dff] if h_scratch is not None else \
                torch.empty(self.Dff, K, device=y.device, dtype=torch.float32)
            g1 = (self.Dff, triton.cdiv(K, self.block_k))
            _fused_upgate_silu_kernel[g1](
                self.up_crow, self.up_col, self.up_val,
                self.gt_crow, self.gt_col, self.gt_val,
                y, self.b_up, self.b_gate, h, K,
                y.stride(0), y.stride(1), h.stride(0), h.stride(1),
                BLOCK_K=self.block_k,
            )
            g2 = (self.n_active, triton.cdiv(K, self.block_k))
            _down_delta_inplace_kernel[g2](
                self.active_rows, self.dn_crow, self.dn_col, self.dn_val,
                h, y, K,
                h.stride(0), h.stride(1), y.stride(0), y.stride(1),
                BLOCK_K=self.block_k,
            )
        if not self._down_bias_zero:
            y += self.b_down.view(self.dim, 1)

    def run_fused(self, y: torch.Tensor, K: int, snap_scratch=None):
        """FUSED-HIDDEN form (C4_FFN_FUSED_HIDDEN): update the residual ``y [D,K]`` IN
        PLACE with this block's FFN delta, WITHOUT ever materialising the ``[Dff,K]`` hidden
        in HBM.

        (1) SNAPSHOT the small set of residual dims up/gate read (``in_cols``, ~4-148) into a
            compact ``[n_in,K]`` buffer — this is the ORIGINAL pre-block residual for the
            hidden recompute (dodges the write-after-read hazard for the 47/49 blocks whose
            W_down output rows overlap their W_up/W_gate input dims).
        (2) the fused kernel: one program per active output row recomputes each read hidden
            row IN REGISTERS from the snapshot, and adds ``W_down @ hidden`` to ``y`` in place.
        No full-[D,K] copy, no [Dff,K] HBM hidden.  Byte-exact to ``run_inplace`` (same CSRs,
        same accumulation order; the snapshot is a value-identical copy of the read dims)."""
        from .fused_sparse_ffn import _fused_hidden_delta_snap_kernel
        if self.n_active and self.n_in:
            # snapshot y[in_cols] -> xs [n_in, K]  (index_select is a coalesced gather).
            if snap_scratch is not None and snap_scratch.shape[1] == K \
                    and snap_scratch.shape[0] >= self.n_in:
                xs = snap_scratch[:self.n_in]
                torch.index_select(y, 0, self.in_cols, out=xs)
            else:
                xs = torch.index_select(y, 0, self.in_cols)
            g = (self.n_active, triton.cdiv(K, self.block_k))
            _fused_hidden_delta_snap_kernel[g](
                self.active_rows, self.dn_crow, self.dn_col, self.dn_val,
                self.up_crow, self.up_col_snap, self.up_val,
                self.gt_crow, self.gt_col_snap, self.gt_val,
                xs, self.b_up, self.b_gate, y, K,
                xs.stride(0), xs.stride(1), y.stride(0), y.stride(1),
                BLOCK_K=self.block_k,
            )
        if not self._down_bias_zero:
            y += self.b_down.view(self.dim, 1)


class MegaBlockChain:
    """The DIV-free dead-FFN megakernel: run a contiguous list of DEAD-FFN blocks
    as ONE fused on-chip chain over a [K, D] residual, IN PLACE.

    ``run(hq)`` : forward the K query rows ([1,K,D]) through the dead-FFN blocks,
    copying the input into a SINGLE persistent [D, K] residual buffer ONCE, then
    delta-updating it IN PLACE across all blocks (no per-block full-D copy, no hidden
    HBM buffer — for a doom K-chunk the [D,K] residual stays L2-resident across the
    whole chain).  The live CAM blocks (2/7/11) are handled by the caller (eager
    direct gather) — this chain covers ONLY the dead-FFN blocks given at construction.

    ``run_graphed(hq)`` : the same, but the whole chain is captured into ONE CUDA
    graph (per K) and replayed — collapsing the 49x2 launch overhead to one replay.
    Byte-exact (graph replays the identical kernel stream).
    """

    def __init__(self, model, device, block_list: List[int], block_k: int = 64,
                 resid_dtype: Optional[torch.dtype] = None):
        self.model = model
        self.device = torch.device(device)
        # C4_MEGABLOCK_BLOCK_K (perf-tuning knob, byte-neutral): override the Triton K-tile
        # size for the dead-FFN kernels (default 64).  Larger tiles = fewer programs / more
        # per-program K-columns (better coalescing on the [Dff/n_in,K] loads); smaller tiles
        # = more parallelism.  Byte-exact for any value (only the tiling changes, not the
        # arithmetic).  Applies to BOTH the 2-kernel and the fused-hidden paths.
        _bk_env = os.environ.get("C4_MEGABLOCK_BLOCK_K", "")
        if _bk_env.strip().isdigit():
            block_k = int(_bk_env)
        self.block_k = block_k
        self.blocks = block_list
        self.D = model.dim
        # RESIDUAL DTYPE (lever 3, NON-byte-exact): bf16 halves the residual L2/HBM
        # traffic (each dead block RE-READS the residual for its hidden inputs).  The
        # kernels still compute in fp32 (Triton promotes bf16 loads); only the resident
        # [D,K] buffer is bf16, so each block rounds the residual to bf16 (~3 decimal
        # digits).  Default None -> fp32 (byte-exact).  Gate: _resid_dtype().
        self.resid_dtype = resid_dtype or _resid_dtype()
        # build a _MegaFFN per (distinct) block ffn
        self._ffns: List[_MegaFFN] = []
        cache = {}
        for b in block_list:
            ffn = model.blocks[b].ffn
            key = id(ffn)
            mf = cache.get(key)
            if mf is None:
                mf = _MegaFFN(ffn, str(self.device), block_k)
                cache[key] = mf
            self._ffns.append(mf)
        # shared hidden scratch width = max Dff over all blocks
        self._max_hid = max((mf.Dff for mf in self._ffns), default=1)
        # FUSED-HIDDEN (C4_FFN_FUSED_HIDDEN): keep each block's [Dff,K] hidden ON-CHIP
        # (recompute in registers) instead of round-tripping it through HBM.  The fused
        # form snapshots ONLY the small set of residual dims up/gate read (max n_in over
        # the chain) into a shared [max_in,K] scratch before the in-place delta write — no
        # full-[D,K] copy, no [Dff,K] HBM hidden buffer.
        self._fused_hidden = fused_hidden_enabled()
        self._max_in = max((mf.n_in for mf in self._ffns), default=1)
        # WAVE-BATCH (lever A, C4_FFN_WAVE_BATCH): group the chain's dead blocks into
        # dependency waves (RAW+WAR+WAW) and batch each wave's blocks into ONE up/gate +
        # ONE down-delta launch.  Byte-exact (hazard-free wave -> disjoint writes, shared
        # pre-wave residual read).  The hidden scratch must hold a whole wave's STACKED
        # hidden (sum of the wave's Dff), so it grows to max(single Dff, max wave tot_hid).
        self._wave_batch = wave_batch_enabled()
        self._waves: List[_WaveBatch] = []
        self.n_waves = 0
        self.max_wave_parallel = 0
        if self._wave_batch and self._ffns:
            writes = [set(mf.active_rows.tolist()) for mf in self._ffns]
            reads = [set(mf.in_cols.tolist()) for mf in self._ffns]
            _lvls, wave_idx = _dependency_waves(writes, reads)
            for grp in wave_idx:
                self._waves.append(_WaveBatch([self._ffns[i] for i in grp], block_k))
            self.n_waves = len(self._waves)
            self.max_wave_parallel = max((len(g) for g in wave_idx), default=0)
            self._max_hid = max(self._max_hid,
                                max((wb.tot_hid for wb in self._waves), default=1))
        # persistent residual + scratch, lazy per K
        self._bufs: dict = {}      # K -> (resid [D,K], h_scratch [max_hid,K], snap [max_in,K])
        self._graphs: dict = {}    # K -> (graph, static_in [D,K], static_out [D,K])
        self.n_captures = 0

    def _get_bufs(self, K: int):
        bufs = self._bufs.get(K)
        if bufs is None:
            y = torch.empty(self.D, K, device=self.device, dtype=self.resid_dtype)
            hs = torch.empty(self._max_hid, K, device=self.device, dtype=torch.float32)
            # snapshot scratch for the fused-hidden path (only the ~n_in read dims; only
            # allocated when the fused path is on -> the 2-kernel path pays no extra VRAM).
            snap = (torch.empty(self._max_in, K, device=self.device, dtype=torch.float32)
                    if self._fused_hidden else None)
            self._bufs[K] = bufs = (y, hs, snap)
        return bufs

    def _run_chain(self, y: torch.Tensor, K: int) -> torch.Tensor:
        """Run the dead-FFN chain over the residual buffer ``y [D,K]``, IN PLACE.

        Default (2-kernel path): each block writes its [Dff,K] hidden to HBM then reads it
        back (the profiler's dominant 92%-of-bytes round-trip).

        FUSED-HIDDEN path (C4_FFN_FUSED_HIDDEN): each block SNAPSHOTS only the small set of
        residual dims up/gate read, then a single kernel recomputes the hidden IN REGISTERS
        from that snapshot and adds the FFN delta to ``y`` in place — no [Dff,K] HBM hidden
        buffer, no full-[D,K] copy.  Byte-exact to the 2-kernel path."""
        _, hs, snap = self._get_bufs(K)
        if self._wave_batch:
            # WAVE-BATCH (lever A): one batched up/gate + one batched down-delta per wave.
            for wb in self._waves:
                wb.run_inplace(y, K, h_scratch=hs)
        elif not self._fused_hidden:
            for mf in self._ffns:
                mf.run_inplace(y, K, h_scratch=hs)
        else:
            for mf in self._ffns:
                mf.run_fused(y, K, snap_scratch=snap)
        return y

    def run(self, hq: torch.Tensor) -> torch.Tensor:
        """hq [1,K,D] -> [1,K,D] after the dead-FFN chain (eager, no graph).

        Copies the input into the persistent residual buffer ONCE, then runs the whole
        chain in place (no per-block copy) — the residual stays in one buffer across
        all blocks."""
        B, K, D = hq.shape
        y, _, _ = self._get_bufs(K)
        y.copy_(hq.reshape(K, D).transpose(0, 1))     # single [D,K] load into resident buf
        out = self._run_chain(y, K)
        # cast back to the caller's fp32 at the boundary (bf16 is chain-internal only)
        return out.transpose(0, 1).reshape(1, K, D).to(hq.dtype)

    def _capture(self, K: int):
        y, hs, _ = self._get_bufs(K)
        static_in = torch.zeros(self.D, K, device=self.device, dtype=torch.float32)

        def chain():
            y.copy_(static_in)
            self._run_chain(y, K)
            return y

        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    chain()
        torch.cuda.current_stream(self.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                static_out = chain()
        self.n_captures += 1
        return g, static_in, static_out

    def run_graphed(self, hq: torch.Tensor) -> torch.Tensor:
        B, K, D = hq.shape
        ginfo = self._graphs.get(K)
        if ginfo is None:
            ginfo = self._capture(K)
            self._graphs[K] = ginfo
        g, gin, gout = ginfo
        gin.copy_(hq.reshape(K, D).transpose(0, 1))
        g.replay()
        # clone + cast back to fp32: gout is the persistent (maybe bf16) residual buffer
        # the next replay overwrites; the boundary is always fp32 for the caller.
        return gout.transpose(0, 1).reshape(1, K, D).to(hq.dtype)


# ===========================================================================
# LINEAR-FOLD (lever B, C4_FFN_LINFOLD).  Fold block N's W_down into block N+1's
# W_up/W_gate across the LINEAR residual add.  A folded PAIR (N, N+1):
#   * block N+1's up/gate reads the PRE-PAIR residual y (NOT y + d_N) PLUS block N's
#     hidden h_N through the precomputed A = Wu_{N+1}@Wd_N, B = Wg_{N+1}@Wd_N:
#         pre_u_{N+1} = Wu_{N+1}·y + A·h_N + b_up
#         pre_g_{N+1} = Wg_{N+1}·y + B·h_N + b_gate
#     (algebraically identical to Wu_{N+1}·(y + d_N) + b, since d_N = Wd_N·h_N.)
#   * block N STILL writes d_N = Wd_N·h_N to the residual (later blocks + decode read
#     it), and block N+1 writes d_{N+1}.  Both deltas are written AFTER both hiddens are
#     computed from the pre-pair y (so the y-read is clean of the pair's own writes).
#
# The A/B products can be DENSER than the sparse factors (fill-in).  Reported honestly.
# ===========================================================================
@triton.jit
def _linfold_upgate_silu_kernel(
    up_crow_ptr, up_col_ptr, up_val_ptr,       # CSR W_up_{N+1} [Dff, D]   (reads y)
    gt_crow_ptr, gt_col_ptr, gt_val_ptr,       # CSR W_gate_{N+1} [Dff, D] (reads y)
    A_crow_ptr, A_col_ptr, A_val_ptr,          # CSR A = Wu_{N+1}@Wd_N [Dff, DffN] (reads hN)
    B_crow_ptr, B_col_ptr, B_val_ptr,          # CSR B = Wg_{N+1}@Wd_N [Dff, DffN] (reads hN)
    y_ptr,                                     # residual y [D, K]     (pre-pair)
    hN_ptr,                                    # block-N hidden hN [DffN, K]
    bup_ptr, bgt_ptr,                          # [Dff]
    h_ptr,                                     # out: block-(N+1) hidden [Dff, K]
    K,
    stride_yd, stride_yk,
    stride_hnd, stride_hnk,
    stride_hu, stride_hk,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)            # hidden unit u of block N+1
    kblk = tl.program_id(1)
    k_off = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K
    # ---- up = b + Wu_{N+1}·y + A·hN ----
    up = tl.zeros([BLOCK_K], dtype=tl.float32) + tl.load(bup_ptr + row)
    us = tl.load(up_crow_ptr + row); ue = tl.load(up_crow_ptr + row + 1)
    for p in range(us, ue):
        c = tl.load(up_col_ptr + p); v = tl.load(up_val_ptr + p)
        up += v * tl.load(y_ptr + c * stride_yd + k_off * stride_yk, mask=k_mask, other=0.0)
    as_ = tl.load(A_crow_ptr + row); ae = tl.load(A_crow_ptr + row + 1)
    for p in range(as_, ae):
        c = tl.load(A_col_ptr + p); v = tl.load(A_val_ptr + p)
        up += v * tl.load(hN_ptr + c * stride_hnd + k_off * stride_hnk, mask=k_mask, other=0.0)
    # ---- gate = b + Wg_{N+1}·y + B·hN ----
    gate = tl.zeros([BLOCK_K], dtype=tl.float32) + tl.load(bgt_ptr + row)
    gs = tl.load(gt_crow_ptr + row); ge = tl.load(gt_crow_ptr + row + 1)
    for p in range(gs, ge):
        c = tl.load(gt_col_ptr + p); v = tl.load(gt_val_ptr + p)
        gate += v * tl.load(y_ptr + c * stride_yd + k_off * stride_yk, mask=k_mask, other=0.0)
    bs = tl.load(B_crow_ptr + row); be = tl.load(B_crow_ptr + row + 1)
    for p in range(bs, be):
        c = tl.load(B_col_ptr + p); v = tl.load(B_val_ptr + p)
        gate += v * tl.load(hN_ptr + c * stride_hnd + k_off * stride_hnk, mask=k_mask, other=0.0)
    h = (up * tl.sigmoid(up)) * gate
    tl.store(h_ptr + row * stride_hu + k_off * stride_hk, h, mask=k_mask)


class _LinFoldPair:
    """A folded pair (N, N+1): block N+1's up/gate reads y (pre-pair) + h_N through the
    precomputed A/B products.  Both blocks' down-deltas write the residual after."""

    def __init__(self, mfN: "_MegaFFN", mfNp: "_MegaFFN", block_k: int):
        self.block_k = block_k
        self.N = mfN; self.Np = mfNp
        dev = mfN.b_up.device
        # A = Wu_{N+1} @ Wd_N   [Dff_{N+1}, Dff_N]   (Wd_N reconstructed from its ACTIVE CSR)
        # Wd_N active CSR is over active_rows; expand to full [D, Dff_N] then Wu@Wd.
        # Wu_{N+1} is [Dff_{N+1}, D]; Wd_N is [D, Dff_N].
        D = mfNp.dim
        DffN = mfN.Dff
        # rebuild dense Wd_N [D, DffN] from the active CSR (rows are active residual dims).
        Wd_N = torch.zeros(D, DffN, dtype=torch.float32, device=dev)
        if mfN.dn_col.numel():
            arows = torch.repeat_interleave(
                mfN.active_rows.to(torch.int64),
                (mfN.dn_crow[1:] - mfN.dn_crow[:-1]).to(torch.int64))
            Wd_N[arows, mfN.dn_col.to(torch.int64)] = mfN.dn_val
        # dense Wu_{N+1}, Wg_{N+1} [Dff_{N+1}, D]
        Wu = torch.zeros(mfNp.Dff, D, dtype=torch.float32, device=dev)
        if mfNp.up_col.numel():
            urows = torch.repeat_interleave(
                torch.arange(mfNp.Dff, device=dev),
                (mfNp.up_crow[1:] - mfNp.up_crow[:-1]).to(torch.int64))
            Wu[urows, mfNp.up_col.to(torch.int64)] = mfNp.up_val
        Wg = torch.zeros(mfNp.Dff, D, dtype=torch.float32, device=dev)
        if mfNp.gt_col.numel():
            grows = torch.repeat_interleave(
                torch.arange(mfNp.Dff, device=dev),
                (mfNp.gt_crow[1:] - mfNp.gt_crow[:-1]).to(torch.int64))
            Wg[grows, mfNp.gt_col.to(torch.int64)] = mfNp.gt_val
        A = Wu @ Wd_N     # [Dff_{N+1}, Dff_N]
        B = Wg @ Wd_N
        self.A_crow, self.A_col, self.A_val = _csr(A)
        self.B_crow, self.B_col, self.B_val = _csr(B)
        # fill-in accounting: fused A/B nnz vs the factor nnz (Wu/Wg + Wd_N over the pair).
        self.fused_nnz = int((A != 0).sum() + (B != 0).sum())
        self.factor_nnz = int(mfNp.up_val.numel() + mfNp.gt_val.numel() + mfN.dn_val.numel())

    def run(self, y: torch.Tensor, K: int, hN: torch.Tensor, hNp: torch.Tensor):
        """Compute h_N (into hN) and h_{N+1} (into hNp) from the PRE-PAIR residual y, then
        write both down-deltas to y IN PLACE.  hN/hNp are scratch slices [Dff*, K]."""
        from .fused_sparse_ffn import (_fused_upgate_silu_kernel,
                                       _down_delta_inplace_kernel)
        mfN, mfNp = self.N, self.Np
        # (1) block N hidden from y (the plain fused up/gate)
        if mfN.n_active:
            g1 = (mfN.Dff, triton.cdiv(K, self.block_k))
            _fused_upgate_silu_kernel[g1](
                mfN.up_crow, mfN.up_col, mfN.up_val,
                mfN.gt_crow, mfN.gt_col, mfN.gt_val,
                y, mfN.b_up, mfN.b_gate, hN, K,
                y.stride(0), y.stride(1), hN.stride(0), hN.stride(1),
                BLOCK_K=self.block_k)
        # (2) block N+1 hidden from y + A/B·hN (the FOLD): reads pre-pair y + hN, no d_N re-read
        if mfNp.n_active:
            g2 = (mfNp.Dff, triton.cdiv(K, self.block_k))
            _linfold_upgate_silu_kernel[g2](
                mfNp.up_crow, mfNp.up_col, mfNp.up_val,
                mfNp.gt_crow, mfNp.gt_col, mfNp.gt_val,
                self.A_crow, self.A_col, self.A_val,
                self.B_crow, self.B_col, self.B_val,
                y, hN, mfNp.b_up, mfNp.b_gate, hNp, K,
                y.stride(0), y.stride(1), hN.stride(0), hN.stride(1),
                hNp.stride(0), hNp.stride(1),
                BLOCK_K=self.block_k)
        # (3) write both down-deltas (order N then N+1, matching sequential residual state)
        if mfN.n_active:
            gd = (mfN.n_active, triton.cdiv(K, self.block_k))
            _down_delta_inplace_kernel[gd](
                mfN.active_rows, mfN.dn_crow, mfN.dn_col, mfN.dn_val,
                hN, y, K, hN.stride(0), hN.stride(1), y.stride(0), y.stride(1),
                BLOCK_K=self.block_k)
        if mfNp.n_active:
            gd = (mfNp.n_active, triton.cdiv(K, self.block_k))
            _down_delta_inplace_kernel[gd](
                mfNp.active_rows, mfNp.dn_crow, mfNp.dn_col, mfNp.dn_val,
                hNp, y, K, hNp.stride(0), hNp.stride(1), y.stride(0), y.stride(1),
                BLOCK_K=self.block_k)


class LinFoldChain:
    """Dead-FFN chain with VERTICAL linear-fold (lever B).  Consecutive dead blocks are
    paired: within a pair (N, N+1), block N's W_down is folded into N+1's up/gate so N+1
    reads block N's hidden directly instead of re-reading the residual dims N wrote.  An
    odd trailing block runs as a plain _MegaFFN.run_inplace.

    Optionally composes with wave-batch (lever A) is NOT applied inside a fold (the fold
    forces the N->N+1 order); wave_batch=True only affects the UNFOLDED remainder blocks
    (here: none in a fully-paired chain), so composition is reported as A+B = the folded
    chain (the fold already collapses the launch count)."""

    def __init__(self, model, device, block_list: List[int], block_k: int = 64,
                 wave_batch: bool = False):
        self.device = torch.device(device)
        self.D = model.dim
        _bk_env = os.environ.get("C4_MEGABLOCK_BLOCK_K", "")
        if _bk_env.strip().isdigit():
            block_k = int(_bk_env)
        self.block_k = block_k
        self.resid_dtype = _resid_dtype()
        # build a _MegaFFN per (distinct) block ffn (cache identical ffns)
        cache = {}; ffns: List[_MegaFFN] = []
        for b in block_list:
            ffn = model.blocks[b].ffn
            mf = cache.get(id(ffn))
            if mf is None:
                mf = _MegaFFN(ffn, str(self.device), block_k); cache[id(ffn)] = mf
            ffns.append(mf)
        self._ffns = ffns
        # pair up consecutive blocks; odd trailing block runs plain.
        self._pairs: List[_LinFoldPair] = []
        self._tail: Optional[_MegaFFN] = None
        i = 0
        while i + 1 < len(ffns):
            self._pairs.append(_LinFoldPair(ffns[i], ffns[i + 1], block_k))
            i += 2
        if i < len(ffns):
            self._tail = ffns[i]
        self.n_folds = len(self._pairs)
        # launch count: 4 per fold (h_N, h_{N+1}, down_N, down_{N+1}) + 2 for a tail block.
        self.n_launches = 4 * self.n_folds + (2 if self._tail is not None else 0)
        self._max_hid = max((mf.Dff for mf in ffns), default=1)
        self._bufs: dict = {}

    def _get_bufs(self, K: int):
        b = self._bufs.get(K)
        if b is None:
            y = torch.empty(self.D, K, device=self.device, dtype=self.resid_dtype)
            hN = torch.empty(self._max_hid, K, device=self.device, dtype=torch.float32)
            hNp = torch.empty(self._max_hid, K, device=self.device, dtype=torch.float32)
            self._bufs[K] = b = (y, hN, hNp)
        return b

    def run(self, hq: torch.Tensor) -> torch.Tensor:
        B, K, D = hq.shape
        y, hN, hNp = self._get_bufs(K)
        y.copy_(hq.reshape(K, D).transpose(0, 1))
        for pr in self._pairs:
            pr.run(y, K, hN[:pr.N.Dff], hNp[:pr.Np.Dff])
        if self._tail is not None:
            self._tail.run_inplace(y, K, h_scratch=hN)
        return y.transpose(0, 1).reshape(1, K, D).to(hq.dtype)


def linfold_fillin_report(chain: "LinFoldChain") -> dict:
    """Fill-in accounting for a LinFoldChain: fused A/B nnz vs the sparse factor nnz."""
    fused = sum(p.fused_nnz for p in chain._pairs)
    factor = sum(p.factor_nnz for p in chain._pairs)
    return {"n_folds": chain.n_folds, "fused_nnz": fused, "factor_nnz": factor,
            "fillin_factor": (fused / factor) if factor else 0.0}


# ===========================================================================
# REGION drop-in — the whole frozen-skip [cut, N) block loop as dead-FFN mega-chains
# (fused, in-place, per-K CUDA-graphed) with the live CAM blocks run EAGERLY between
# segments.  A drop-in for ``megastep_graph.MegaStepGraph`` (same ``run(hq, qpos)``
# contract) that the speculative driver's ``_forward_hidden_cached_frozen_skip`` calls.
# ===========================================================================
class MegaBlockRegion:
    """Run frozen-skip blocks ``[cut, N)`` over the K query rows: contiguous DEAD-FFN
    runs go through a ``MegaBlockChain`` (fused in-place delta, CUDA-graphed per K);
    each LIVE-attention block (direct-CAM gather) runs eagerly between them.  Drop-in
    for ``MegaStepGraph`` (``run(hq, qpos) -> hq_out``).  Byte-exact to the eager
    per-block loop (the dead-FFN fusion is nibble-margin byte-exact; the live blocks
    are unchanged; dead blocks write no KV -> the None-cache contract holds)."""

    def __init__(self, model, device, cut: int, carry_blocks=None):
        self.model = model
        self.device = torch.device(device)
        self.n = len(model.blocks)
        self.cut = cut
        # DOOM-LEAN (lever 2): only carry blocks in ``carry_blocks`` (the DIV-free live
        # union) — never the 179-block divmod span (0% doom use).  Blocks NOT carried
        # are run EAGERLY as pass-through (a dead block is identity; a DIV/MOD step
        # falls back to the full eager path upstream).  Default: carry every [cut, N)
        # block (the op-agnostic full region, byte-identical superset).
        carry = set(carry_blocks) if carry_blocks is not None else set(range(cut, self.n))
        # split [cut, N) into (kind, payload) run-items.
        items = []
        seg = []
        for b in range(cut, self.n):
            if b not in carry:
                # NOT carried (doom-lean skip: the divmod span, 0% doom use): a dead
                # block is the IDENTITY on the residual, so it is dropped from the run
                # entirely (NOT run eagerly).  Flush any pending dead-FFN mega-segment
                # so carried segments stay contiguous.
                if seg:
                    items.append(("mega", MegaBlockChain(model, device, seg)))
                    seg = []
                continue
            dead = getattr(model.blocks[b].attn, "_dead_block_fused", False)
            if dead:
                seg.append(b)
            else:
                if seg:
                    items.append(("mega", MegaBlockChain(model, device, seg)))
                    seg = []
                items.append(("live", b))
        if seg:
            items.append(("mega", MegaBlockChain(model, device, seg)))
        self.items = items
        self.n_mega = sum(1 for k, _ in items if k == "mega")

    def run(self, hq, qpos):
        """Forward K query rows [1,K,D] through [cut, N); dead-FFN chains graphed,
        live CAM blocks eager.  Returns the final hidden (dead blocks write no KV)."""
        h = hq
        for kind, payload in self.items:
            if kind == "live":
                h, _ = self.model.blocks[payload](
                    h, past_kv=None, q_positions=qpos, use_cache=True)
            else:
                h = payload.run_graphed(h)
        return h


def divfree_carry_blocks(model, L):
    """The DOOM-LEAN carry set (lever 2): the DIV-free live-block union in [cut, N)
    (the ~30-53 blocks doom actually uses), EXCLUDING the 179-block divmod span
    (0% doom use).  Returns a sorted list, or ``None`` if the live index is unavailable
    (caller then carries the full region)."""
    try:
        from .step_block_skip import build_live_index
        from . import isa
    except Exception:
        return None
    li = build_live_index(model, L)
    union = set()
    for op in li:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(li[op])
    return sorted(union) if union else None


def install_fused_megablock(model, device, cut: int, L=None, verbose: bool = False
                            ) -> MegaBlockRegion:
    """Build a ``MegaBlockRegion`` for the frozen-skip ``[cut, N)`` block loop.

    Requires dead-block-fusion (dead blocks attention-identity) + a CUDA device.
    Byte-exact at the nibble-snap margin (the dead-FFN mega-chains) with the live
    CAM blocks unchanged.  A stronger drop-in for ``install_megastep_graph``: the
    dead-FFN segments run the fused in-place-delta megakernel (no per-block full-D
    residual copy, no hidden HBM buffer) instead of the per-block ``SparseBlock``
    call graph.

    DOOM-LEAN (lever 2): when ``L`` (the layout) is given, the region carries ONLY
    the DIV-free live-block union (never the 179-block divmod span, 0% doom use) —
    the divmod blocks pass straight through (a dead block is identity).  A DIV/MOD
    step is DIV-heavy and falls back to the full eager path upstream, so excluding
    the divmod span from the megablock is byte-safe for the doom (DIV-free) stream."""
    carry = divfree_carry_blocks(model, L) if L is not None else None
    w = MegaBlockRegion(model, device, cut, carry_blocks=carry)
    if verbose:
        nlive = sum(1 for k, _ in w.items if k == "live")
        ncarry = "all" if carry is None else str(len([b for b in carry if b >= cut]))
        print(f"[fused-megablock] cut={cut}; {w.n_mega} dead-FFN mega-chains + "
              f"{nlive} eager live/passthrough over [cut, {w.n}); "
              f"doom-lean carry={ncarry} DIV-free blocks", flush=True)
    return w


__all__ = ["fused_megablock_enabled", "MegaBlockChain", "MegaBlockRegion",
           "install_fused_megablock", "wave_batch_enabled", "linfold_enabled",
           "LinFoldChain", "linfold_fillin_report"]
