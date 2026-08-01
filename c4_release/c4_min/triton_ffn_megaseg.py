"""TRITON per-block gather-gate-scatter MEGAKERNEL wired into the doom
verify_blocks block-stack (drop-in for ``forward_hidden_cached``).

After ``install_dead_block_fusion`` the doom stack has a few LIVE-attention blocks
(0/2/7/11) and long DEAD segments of pure-FFN passthrough blocks; the big one is
``(12, 242)`` = 230 pure SwiGLU FFNs.  ``GraphedFusedForward`` CUDA-graphs those
segments but replays 230 DENSE ``[S,D]x[D,Dff]`` GEMVs inside the graph — FLOP-bound
at doom's large S.  This module instead runs each dead FFN block as the
``fused_ffn_megakernel`` PER-BLOCK gather-gate-scatter (each hidden unit reads ~1
residual dim, so up/gate is a GATHER not a GEMM), and CUDA-graphs the whole chain of
those cheap Triton kernels as ONE replay (``GraphedTritonBlockGSChain``).

Byte-exactness: the 1-read majority is L-inf=0; multi-read units differ by fp-accum /
atomic-add residue FAR below the integer nibble-decode margin (verified at DECODE:
the verifier's accepted==total is the byte-exact gate).  The live blocks stay eager
(their attention builds a dynamic ALiBi/causal mask over the growing KV — not
graph-static).

Gate: composed at install time by the agent (``C4_FFN_MEGAKERNEL=triton``).  Changes
NO stored weight -> golden ``069cc32f`` unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .graphed_fused_forward import _dead_segments
from .fused_ffn_megakernel import (GraphedTritonBlockGSChain, GraphedFusedFFNChain,
                                   _HAVE_TRITON)


class _SegAdapter:
    """Adapts a list of SparseBlocks to the ``kblocks[bi].b.ffn`` interface the
    fused_ffn_megakernel chains expect (they read ``kblocks[bi].b.ffn``)."""

    class _B:
        def __init__(self, ffn):
            self.ffn = ffn
            self.is_passthrough = True   # dead-segment blocks are pure-FFN passthrough
            self.routed = False

    class _K:
        def __init__(self, b):
            self.b = b

    def __init__(self, blocks):
        self._items = [self._K(self._B(blk.ffn)) for blk in blocks]

    def __getitem__(self, i):
        return self._items[i]


class TritonMegaSegForward:
    """Graph the big dead segment(s) with a per-block Triton gather-gate-scatter
    chain; other segments dense-graphed; live blocks eager.  Drop-in for
    ``model.forward_hidden_cached`` (same signature/return).

    ``min_seg_len`` — only segments at least this long use the Triton chain (short
    segments have too little to amortize; dense-graph them instead).
    ``use_triton`` — if Triton is unavailable, fall back to the dense fused chain
    (``GraphedFusedFFNChain``) for the big segment too."""

    def __init__(self, model, device, min_seg_len: int = 16, use_triton: bool = True):
        self.model = model
        self.device = torch.device(device)
        self.n = len(model.blocks)
        self.live, self.segs = _dead_segments(model)
        self._live_set = set(self.live)
        self._orig_fhc = model.forward_hidden_cached
        self.min_seg_len = min_seg_len
        self.use_triton = use_triton and _HAVE_TRITON
        # COMPOSITION NOTE: this megakernel IS the graphed COO gather-gate-scatter,
        # so it is mutually EXCLUSIVE with an already-installed BlockSparseFFN
        # (C4_BLOCK_SPARSE_FFN) — that swaps the FFN to a NON-graphed CooLinear form
        # with no materialize_dense / dense weight tensor for us to re-index.  The
        # graphed COO here strictly dominates the un-graphed one, so the intended
        # composition is triton-megakernel WITHOUT C4_BLOCK_SPARSE_FFN.  We assert a
        # clear message rather than crash cryptically if both are on.
        from .block_sparse_ffn import BlockSparseFFN
        if any(isinstance(blk.ffn, BlockSparseFFN) for blk in model.blocks):
            raise RuntimeError(
                "triton_ffn_megaseg does not compose with an installed "
                "C4_BLOCK_SPARSE_FFN (both are COO gather-gate-scatter; the graphed "
                "megakernel here is strictly faster).  Run the megakernel WITHOUT "
                "C4_BLOCK_SPARSE_FFN.")
        model.materialize_dense(device=str(self.device))
        self.D = model.dim
        # per-S graph cache: S -> {seg -> chain-graph}
        self._graphs: Dict[int, Dict[Tuple[int, int], object]] = {}
        # adapter over ALL blocks (indexable by absolute block idx)
        self._adapter = _SegAdapter(model.blocks)

    def _build_seg_graph(self, seg, S):
        s0, s1 = seg
        seg_idxs = list(range(s0, s1))
        seglen = s1 - s0
        if seglen >= self.min_seg_len and self.use_triton:
            g = GraphedTritonBlockGSChain(self._adapter, seg_idxs, S, self.D,
                                          self.device, torch.float32)
        else:
            g = GraphedFusedFFNChain(self._adapter, seg_idxs, S, self.D,
                                     self.device, torch.float32)
        ok = g.try_capture()
        return (g if ok else None, seg_idxs)

    def _graphs_for_S(self, S):
        gs = self._graphs.get(S)
        if gs is None:
            gs = {seg: self._build_seg_graph(seg, S) for seg in self.segs}
            self._graphs[S] = gs
        return gs

    def _eager_seg(self, h, seg_idxs, q_positions):
        for i in seg_idxs:
            h, _ = self.model.blocks[i](h, past_kv=None, q_positions=q_positions,
                                        use_cache=True)
        return h

    def forward_hidden_cached(self, x, past_key_values=None, q_positions=None,
                              use_cache: bool = False):
        B, S, D = x.shape
        if B != 1 or q_positions is None:
            return self._orig_fhc(x, past_key_values=past_key_values,
                                  q_positions=q_positions, use_cache=use_cache)
        gs = self._graphs_for_S(S)
        if past_key_values is None:
            past_key_values = [None] * self.n
        new_caches: List[Optional[object]] = [None] * self.n
        h = x
        bi = 0
        seg_ptr = 0
        while bi < self.n:
            if bi in self._live_set:
                h, kv = self.model.blocks[bi](
                    h, past_kv=past_key_values[bi], q_positions=q_positions,
                    use_cache=True)
                new_caches[bi] = kv
                bi += 1
            else:
                seg = self.segs[seg_ptr]
                seg_ptr += 1
                g, seg_idxs = gs[seg]
                if g is not None:
                    # chain graph expects [1, S, D]; run + clone (own the output).
                    h = g.run(h).clone()
                else:
                    h = self._eager_seg(h, seg_idxs, q_positions)
                bi = seg[1]
        return h, new_caches

    def install(self):
        self.model.forward_hidden_cached = self.forward_hidden_cached
        return self

    def uninstall(self):
        self.model.forward_hidden_cached = self._orig_fhc


def install_triton_ffn_megaseg(model, device, min_seg_len: int = 16,
                               use_triton: bool = True, verbose: bool = False):
    w = TritonMegaSegForward(model, device, min_seg_len=min_seg_len,
                             use_triton=use_triton).install()
    if verbose:
        seg_lens = [e - s for s, e in w.segs]
        which = "TRITON" if w.use_triton else "DENSE-fused (no triton)"
        print(f"[triton-megaseg] live={w.live} segs={w.segs} (lens {seg_lens}); "
              f"big-seg via {which} (min_seg_len={min_seg_len}); graphs per-S on first use")
    return w
