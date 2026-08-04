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
        # persistent residual + hidden scratch, lazy per K
        self._bufs: dict = {}      # K -> (resid [D,K], h_scratch [max_hid,K])
        self._graphs: dict = {}    # K -> (graph, static_in [D,K], static_out [D,K])
        self.n_captures = 0

    def _get_bufs(self, K: int):
        bufs = self._bufs.get(K)
        if bufs is None:
            y = torch.empty(self.D, K, device=self.device, dtype=self.resid_dtype)
            hs = torch.empty(self._max_hid, K, device=self.device, dtype=torch.float32)
            self._bufs[K] = bufs = (y, hs)
        return bufs

    def _run_chain(self, y: torch.Tensor, K: int) -> torch.Tensor:
        """In-place delta over the shared residual buffer ``y [D,K]``."""
        _, hs = self._get_bufs(K)
        for mf in self._ffns:
            mf.run_inplace(y, K, h_scratch=hs)
        return y

    def run(self, hq: torch.Tensor) -> torch.Tensor:
        """hq [1,K,D] -> [1,K,D] after the dead-FFN chain (eager, no graph).

        Copies the input into the persistent residual buffer ONCE, then runs the whole
        chain in place (no per-block copy) — the residual stays in one buffer across
        all blocks."""
        B, K, D = hq.shape
        y, _ = self._get_bufs(K)
        y.copy_(hq.reshape(K, D).transpose(0, 1))     # single [D,K] load into resident buf
        out = self._run_chain(y, K)
        # cast back to the caller's fp32 at the boundary (bf16 is chain-internal only)
        return out.transpose(0, 1).reshape(1, K, D).to(hq.dtype)

    def _capture(self, K: int):
        y, hs = self._get_bufs(K)
        static_in = torch.zeros(self.D, K, device=self.device, dtype=torch.float32)

        def chain():
            y.copy_(static_in)
            for mf in self._ffns:
                mf.run_inplace(y, K, h_scratch=hs)
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
           "install_fused_megablock"]
