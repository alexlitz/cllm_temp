"""MEGASTEP GRAPH (C4_GRAPH_MEGAKERNEL) — collapse the post-cut query-row block
loop of the composed dead-fusion + frozen-skip step into CUDA graphs.

CONTEXT.  With ``C4_DEAD_BLOCK_FUSION`` the 238 dead-attention blocks are pure
identity on the residual (attention output = x, no Q/K/V/W_o), so a dead block is
just ``x = FFN(x)``.  With ``C4_FROZEN_ROW_SKIP`` the blocks past the cut (==1 for
doom: only block-0 ingest reads frozen-row KV) run over ONLY the K query rows.
So ``_forward_hidden_cached_frozen_skip`` runs blocks ``[cut, N)`` (241 blocks for
doom) as a PER-BLOCK PYTHON LOOP over a tiny ``[1, K, D]`` tensor — 241 launches of
a ~5,650-FLOP FFN each, PURE launch/dispatch overhead (#865: the step is
overhead-bound, ~0.00002% util).

Because attention is now EXACTLY ZERO on the dead blocks (no dynamic ALiBi /
causal mask, no growing-KV read — the exact dynamic shapes that INVALIDATED
#851's whole-step capture with ``cudaErrorStreamCaptureInvalidated``), the
contiguous DEAD-block FFN segments are FIXED-SHAPE, past-independent,
position-independent computations that CAPTURE cleanly into a CUDA graph.  A dead
segment's graph replays as ONE launch instead of ``len(seg)`` FFN launches.

The ~4 LIVE-attention blocks (0=ingest, 2=code-select, 7=mem-cam, 11=stack-pop-cam)
stay eager: block 0 is < cut (not in this region), and blocks 2/7/11 do a
draft-resolved DIRECT GATHER (``direct_cam_batched``) whose per-row host lookup is
not graph-safe.  They are cheap O(1)-per-query-row gathers.

BYTE-EXACT.  A dead segment's graphed output is the SAME dense ``F.linear`` GEMM
chain (post ``materialize_dense``) in the SAME accumulation order as the eager
``SparseBlock.__call__`` loop, with attention the identity throughout (L-inf=0).
The live blocks are unchanged.  Dead blocks write NO KV in the frozen-skip region
(the post-cut caches are None), so the cache contract is unchanged.

Graphs are keyed by K (query-row count).  The verifier's spans are mostly a
constant K, so one capture amortizes over the whole verify; a new K captures a
fresh graph set on first use.  Default OFF -> the eager per-block loop (golden
069cc32f unchanged).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch


def megastep_graph_enabled() -> bool:
    """``C4_GRAPH_MEGAKERNEL`` (DEFAULT OFF): CUDA-graph the post-cut dead-FFN
    segments of the frozen-skip query-row block loop.  Requires dead-block-fusion
    (the dead blocks must be attention-identity) + a CUDA device.  OFF -> the eager
    per-block Python loop (byte-identical golden path)."""
    return os.environ.get("C4_GRAPH_MEGAKERNEL", "0") not in ("0", "", "false", "False")


def _dead_segments_in_range(model, lo: int, hi: int
                            ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Split blocks ``[lo, hi)`` into LIVE blocks (attention not dead-fused) and
    contiguous DEAD ``[start, end)`` segments (dead-fused, pure-FFN, graphable)."""
    live = [i for i in range(lo, hi)
            if not getattr(model.blocks[i].attn, "_dead_block_fused", False)]
    segs: List[Tuple[int, int]] = []
    start = lo
    for lv in live + [hi]:
        if lv > start:
            segs.append((start, lv))
        start = lv + 1
    return live, segs


class MegaStepGraph:
    """CUDA-graphs the DEAD-FFN segments of the post-cut block loop ``[cut, N)``.

    Install with ``install(model, cut)``: it materialises the model dense (byte-exact,
    skips per-call ``csr.to_dense``), splits ``[cut, N)`` into live blocks + dead
    segments, and exposes ``run(hq, qpos)`` that replays each dead segment's graph
    and runs each live block eagerly — a drop-in for the ``[cut, N)`` half of
    ``_forward_hidden_cached_frozen_skip``.
    """

    def __init__(self, model, device, cut: int):
        self.model = model
        self.device = torch.device(device)
        self.n = len(model.blocks)
        self.cut = cut
        self.live, self.segs = _dead_segments_in_range(model, cut, self.n)
        self._live_set = set(self.live)
        # graph cache: K (query rows) -> {seg -> (graph, static_in, static_out)}
        self._graphs: Dict[int, Dict[Tuple[int, int], object]] = {}
        self.n_captures = 0
        model.materialize_dense(device=str(self.device))

    # -- capture one dead segment's FFN chain at a fixed K -------------------
    def _capture_segment(self, seg: Tuple[int, int], K: int, qpos):
        s0, s1 = seg
        D = self.model.dim
        static_in = torch.zeros(1, K, D, device=self.device)

        def run(x):
            h = x
            for i in range(s0, s1):
                h, _ = self.model.blocks[i](
                    h, past_kv=None, q_positions=qpos, use_cache=True)
            return h

        st = torch.cuda.Stream(device=self.device)
        st.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(st):
            for _ in range(3):
                with torch.no_grad():
                    run(static_in)
        torch.cuda.current_stream(self.device).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                static_out = run(static_in)
        self.n_captures += 1
        return g, static_in, static_out

    def _graphs_for_K(self, K: int, qpos):
        gs = self._graphs.get(K)
        if gs is None:
            gs = {seg: self._capture_segment(seg, K, qpos) for seg in self.segs}
            self._graphs[K] = gs
        return gs

    def run(self, hq, qpos):
        """Forward the K query rows through blocks ``[cut, N)``: replay each dead
        segment's graph, run each live block eagerly.  Returns the final hidden
        (dead blocks write no KV; the caller's post-cut caches are None)."""
        K = hq.shape[1]
        gs = self._graphs_for_K(K, qpos)
        h = hq
        bi = self.cut
        seg_ptr = 0
        while bi < self.n:
            if bi in self._live_set:
                h, _ = self.model.blocks[bi](
                    h, past_kv=None, q_positions=qpos, use_cache=True)
                bi += 1
            else:
                seg = self.segs[seg_ptr]
                seg_ptr += 1
                g, gin, gout = gs[seg]
                gin.copy_(h)
                g.replay()
                h = gout.clone()
                bi = seg[1]
        return h


def install_megastep_graph(model, device, cut: int, verbose: bool = False
                           ) -> MegaStepGraph:
    """Build + return a ``MegaStepGraph`` for the post-cut block loop.

    Requires dead-block-fusion installed (dead blocks are attention-identity) and a
    CUDA device.  Byte-exact: the dead segments' graphed FFN output equals the eager
    fused chain (L-inf=0); the live blocks are unchanged.
    """
    w = MegaStepGraph(model, device, cut)
    if verbose:
        seg_lens = [e - s for s, e in w.segs]
        print(f"[megastep-graph] cut={cut} live post-cut blocks {w.live}; "
              f"dead segments {w.segs} (lens {seg_lens}, total {sum(seg_lens)}); "
              f"graphs captured per-K on first use", flush=True)
    return w
