"""CUDA-GRAPHED fused forward — collapse the 235 dead-block FFN launches.

After ``install_dead_block_fusion`` the ~235 dead-attention blocks are pure
``x = FFN_i(x)`` (attention is the identity: output ``x``, no KV).  They form
CONTIGUOUS depth segments between the ~3 live-attention blocks.  Each dead
segment is a fixed-shape, past-independent, position-independent computation
(the FFN reads only its input residual), so — once the sparse CSR weights are
DENSIFIED once (``materialize_dense``, so ``.linear`` stops re-running
``csr.to_dense()`` per call) — the whole segment captures cleanly into a CUDA
graph.  Replaying the graph collapses the segment's 3·len·(tiny-GEMM) kernel
LAUNCHES to a single graph launch, killing the per-block Python-dispatch +
launch overhead the ms/step sweep pinned as the wall.

The ~3 LIVE-attention blocks stay EAGER (their attention builds dynamic ALiBi /
causal masks + reads the growing per-block KV cache, which does not capture into
a fixed-shape graph).  They are cheap (~2.4 ms of a ~12 ms forward).

BYTE-EXACT.  A dead segment's graphed output is the SAME arithmetic as the eager
``SparseBlock.__call__`` chain (same densified GEMM, same accum order, no
attention) — verified L-inf=0.  The live blocks are unchanged (eager).  Dead
blocks return ``None`` KV in the fused path, so the graphed forward returns the
same per-block cache list (``None`` for every dead block, real KV for live).

Graphs are keyed by S (span row count): the verifier's block-forward runs a
fixed span structure, so S is usually constant across a run; a new S captures a
fresh graph set on first use (amortized over the whole verify).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def _dead_segments(model) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (live_block_indices, dead_segments) where a dead segment is a
    contiguous ``[start, end)`` run of dead-attention (fused) blocks."""
    n = len(model.blocks)
    live = [i for i in range(n)
            if not getattr(model.blocks[i].attn, "_dead_block_fused", False)]
    segs: List[Tuple[int, int]] = []
    start = 0
    for lv in live + [n]:
        if lv > start:
            segs.append((start, lv))
        start = lv + 1
    return live, segs


class GraphedFusedForward:
    """Wraps a dead-block-fused SparseTransformer with a CUDA-graphed dead-segment
    forward.  Install with ``install(model)``; it monkeypatches
    ``model.forward_hidden_cached`` to the graphed version and returns the wrapper
    (call ``uninstall()`` to revert).

    Requires the model to already have ``install_dead_block_fusion`` applied and to
    be on a CUDA device.  Calls ``materialize_dense`` (idempotent, byte-exact) so
    the graphed GEMMs skip the per-call CSR densify.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)
        self.n = len(model.blocks)
        self.live, self.segs = _dead_segments(model)
        self._live_set = set(self.live)
        # graph cache: S -> {seg -> (graph, static_in, static_out)}
        self._graphs: Dict[int, Dict[Tuple[int, int], object]] = {}
        self._orig_fhc = model.forward_hidden_cached
        # ensure dense-resident weights (byte-exact; skips per-call csr.to_dense).
        model.materialize_dense(device=str(self.device))

    # -- capture one dead segment's graph at a fixed S ----------------------
    def _capture_segment(self, seg: Tuple[int, int], S: int, qpos):
        s0, s1 = seg
        D = self.model.dim
        static_in = torch.zeros(1, S, D, device=self.device)

        def run(x):
            h = x
            for i in range(s0, s1):
                h, _ = self.model.blocks[i](
                    h, past_kv=None, q_positions=qpos, use_cache=True)
            return h

        # warm up on a side stream (allocator + any lazy init) then capture.
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
        return g, static_in, static_out

    def _graphs_for_S(self, S: int, qpos):
        gs = self._graphs.get(S)
        if gs is None:
            gs = {seg: self._capture_segment(seg, S, qpos) for seg in self.segs}
            self._graphs[S] = gs
        return gs

    # -- the graphed forward (drop-in for forward_hidden_cached) -------------
    def forward_hidden_cached(self, x, past_key_values=None, q_positions=None,
                              use_cache: bool = False):
        B, S, D = x.shape
        # fall back to eager for anything that doesn't fit the captured contract
        # (batch != 1, or a live block needs a graph — we only graph the dead segs).
        if B != 1 or q_positions is None:
            return self._orig_fhc(x, past_key_values=past_key_values,
                                  q_positions=q_positions, use_cache=use_cache)
        qpos = q_positions
        gs = self._graphs_for_S(S, qpos)
        if past_key_values is None:
            past_key_values = [None] * self.n
        new_caches: List[Optional[object]] = [None] * self.n
        h = x
        bi = 0
        seg_ptr = 0
        while bi < self.n:
            if bi in self._live_set:
                h, kv = self.model.blocks[bi](
                    h, past_kv=past_key_values[bi], q_positions=qpos,
                    use_cache=True)
                new_caches[bi] = kv
                bi += 1
            else:
                seg = self.segs[seg_ptr]
                seg_ptr += 1
                g, gin, gout = gs[seg]
                gin.copy_(h)
                g.replay()
                # clone so the next segment's capture-input copy (or the caller's
                # downstream use) does not alias this graph's static output buffer.
                h = gout.clone()
                # dead blocks write NO KV (fused): leave new_caches[seg] == None.
                bi = seg[1]
        return h, new_caches

    def install(self):
        self.model.forward_hidden_cached = self.forward_hidden_cached
        return self

    def uninstall(self):
        self.model.forward_hidden_cached = self._orig_fhc


def install_graphed_fused_forward(model, device, verbose: bool = False):
    """Install the CUDA-graphed dead-segment forward on a dead-block-fused model.

    Returns the ``GraphedFusedForward`` wrapper (already installed).  Byte-exact:
    the dead segments' graphed output equals the eager fused chain (L-inf=0), the
    live blocks are unchanged.
    """
    w = GraphedFusedForward(model, device).install()
    if verbose:
        seg_lens = [e - s for s, e in w.segs]
        print(f"[graphed-fused] live blocks {w.live}; dead segments "
              f"{w.segs} (lens {seg_lens}); materialize_dense done; "
              f"graphs captured per-S on first use")
    return w
