"""WHOLE-STEP GRAPH (C4_WHOLE_STEP_GRAPH) — RUNG 2: collapse the composed doom
step's remaining PER-BLOCK / PER-CHUNK host dispatch into as few CUDA-graph replays
as possible, per-row-compacted to the DIV-free live set, residual on-chip.

CONTEXT (this task, RUNG 2).  Measured on the REAL doom ``verify_blocks`` path with
the WHOLE composed stack on (dead-fusion + direct-CAM + frozen-skip + O(K)
cut-span-chunk + fused-megablock + fused-delta + block-0 graph + stream-embed +
gpu-verify + exact-evict):

  * wall  ~943 us/step at K=8192, but device only ~267-363 us/step
    -> a ~580 us/step HOST-DISPATCH gap.
  * that gap is dominated (probed: 81%) by the BLOCK-0 S-CHUNK LOOP: block 0 is the
    SOLE block over all ``S = K*30`` span rows, and the cut-span-chunk path walks it in
    ~640 host iterations (chunk=256), each launching: a stream-embed build, an ingest
    ``gather_ingest_out`` (``new_zeros`` + index scatter), the block-0 graph replay
    (its own static ``.copy_``s), a ``bool(m.any())`` HOST SYNC, and an ``hq0`` scatter.
    ~640 iterations x ~10 launches + 1 sync each = the launch/dispatch wall.

THE LEVER.  Two collapses, both byte-exact and per-row-compacted (block 0 is a per-row
independent map under direct-local-CAM, so the chunk axis is bit-identical):

  1. KILL THE 640 PER-CHUNK HOST SYNCS.  The ``bool(m.any())`` that guards the ``hq0``
     scatter forces a device->host sync EVERY chunk.  The per-chunk query membership is
     KNOWN ON THE HOST ahead of time (``qpos_to_qi`` is built from the draft's
     ``win_starts``), so we precompute, per chunk, the (chunk-local row -> hq0 row) index
     pairs on the host — zero device syncs in the loop.

  2. FOLD THE INGEST GATHER + hq0 SCATTER INTO THE BLOCK-0 GRAPH.  The gather's index
     pattern per chunk is FIXED (the chunk's absolute positions are ``arange(lo0,hi0)``),
     so only the gathered nibble VALUES vary.  We capture ONE graph whose static inputs
     are the overlaid embed ``xc [1,C,D]`` and the chunk's resolved ingest nibbles
     ``nib [C, N_ROLES]`` (precomputed, gathered from the draft once), and whose body is:
     scatter ``nib`` into the ingest-head ``out`` slots -> ``W_o`` residual -> FFN.  The
     graph writes its output into a persistent ``[1,C,D]`` static buffer the host then
     scatters into ``hq0`` via the precomputed index pairs.  One replay + two static
     copies per chunk instead of ~10 launches + a sync.

BYTE-EXACT.  The gathered nibbles are the SAME ``rf.nib_lo/nib_hi`` rows the eager
``gather_ingest_out`` scatters (identical ``pos_map`` index math, precomputed once from
the SAME ``ResolvedFrames``); the ``W_o`` + FFN body is the SAME GEMM chain as
``block0_graph.Block0ChunkGraph`` (which is byte-exact to the eager
``direct_local_forward``); the ``hq0`` scatter writes the SAME query rows to the SAME
slots.  L-inf=0 vs the block-0-graph path (VALIDATED: per-step AX/PC/SP/BP identical to
the K=1 reference on the DIV-free battery + the 5455-step nested_12_28 loop at K up to
262144).  DEFAULT OFF -> the per-chunk eager loop (golden 069cc32f unchanged).

Requires: direct-local-CAM (block 0 per-row independent), cut-span-chunk (fixed chunk
size), stream-embed OFF is fine (works on both the pre-built and streamed embed), a CUDA
device.  Keyed by the fixed chunk size (constant across the verify) -> one capture
amortises over every chunk of every span.

HONEST MEASURED OUTCOME (this task, cuda:1 A5000).  RUNG 2 collapses the block-0
chunk-loop slice it targets (region wall ~790 -> ~672 us/step probed) but the FULL
composed-step wall moves only ~2% at K=8192 (886 -> 866 us/step) and ~0-3% at giant K.
The ~558 us/step host-dispatch gap (wall ~847 us/step vs device ~289 us/step) is NOT
concentrated in one graphable fixed-shape region: it is per-op Python dispatch + hidden
``.cpu()``/index-build host work spread across the live-CAM direct-gather head loop
(``direct_cam_batched._head_out_vec``, ~3600 host calls/forward), the overlay index
construction (``apply_overlay_window_batched``, ~640/forward), the eviction schedule
walk and the decode — none a fixed-shape GEMM chain a CUDA graph can capture.  A larger
cut-span-chunk (16x fewer block-0 iterations, 256->4096) also only cuts the wall ~8%,
confirming the gap is per-op-bound, NOT iteration/launch-count-bound.  So the whole-step
graph does NOT beat the per-block-graphed path by a decisive margin — the measurement is
the deliverable (cf. a1ee254's whole-step megakernel loss).  Byte-exact + gated + golden-
safe regardless; the residual wall is the CPython interpreter dispatch, not device FLOPs.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch


def whole_step_graph_enabled() -> bool:
    """``C4_WHOLE_STEP_GRAPH`` (DEFAULT OFF): collapse the block-0 S-chunk loop's
    per-chunk host dispatch (gather + graph + sync + scatter) into ONE CUDA-graph
    replay + precomputed sync-free scatter per chunk.  OFF -> the per-chunk eager loop
    (byte-identical golden path)."""
    return os.environ.get("C4_WHOLE_STEP_GRAPH", "0") not in ("0", "", "false", "False")


class Block0LoopGraph:
    """CUDA-graphs block-0's per-chunk body — ingest-gather scatter + W_o residual +
    FFN — at a FIXED chunk size, taking the overlaid embed ``xc`` and the chunk's
    resolved ingest nibbles as static inputs (no per-chunk ``gather_ingest_out``
    ``new_zeros``/scatter launch, no ``bool(.any())`` host sync).

    ``run(xc, nib_lo, nib_hi, row_mask)`` replays the graph for one fixed-size chunk and
    returns the block-0 output ``[1, chunk, D]``.  ``nib_lo/nib_hi`` are ``[chunk,
    N_ROLES]`` (0 on non-query rows); ``row_mask`` selects the ingest heads' rows (all
    rows; non-query rows gather 0, harmless).  The per-chunk scatter into ``hq0`` is done
    by the caller with a PRECOMPUTED (host-side) index (no sync).
    """

    def __init__(self, model, device, chunk: int, ingest_heads: List[int],
                 n_roles: int, cut: int = 1):
        self.model = model
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.cut = int(cut)
        self.D = model.dim
        blk = model.blocks[0]
        self.attn = blk.attn
        self.ffn = blk.ffn
        self.H = self.attn.n_heads
        self.HD = self.attn.head_dim
        self.n_roles = int(n_roles)
        # ingest heads (role index == head index; bake_frame_ingest: head h -> role h).
        self.ing_heads = torch.tensor(sorted(ingest_heads), dtype=torch.long,
                                      device=self.device)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_xc: Optional[torch.Tensor] = None
        self._static_lo: Optional[torch.Tensor] = None
        self._static_hi: Optional[torch.Tensor] = None
        self._static_res: Optional[torch.Tensor] = None
        model.materialize_dense(device=str(self.device))

    def _body(self):
        """Fixed-shape per-chunk body over the static buffers.  IDENTICAL algebra to the
        eager gather + ``direct_local_forward`` residual + ``SparseBlock`` FFN."""
        C, D, H, HD = self.chunk, self.D, self.H, self.HD
        xc = self._static_xc                                   # [1, C, D]
        # ingest-head gather scatter: out[0, head, row, 0/1] = nib_lo/hi[row, role==head].
        out = xc.new_zeros(1, H, C, HD)
        heads = self.ing_heads                                 # [n_ing] head==role idx
        rows = torch.arange(C, device=self.device)             # [C]
        # broadcast to [C, n_ing]
        h_idx = heads.unsqueeze(0).expand(C, -1)               # [C, n_ing]
        r_idx = rows.unsqueeze(1).expand(-1, heads.numel())    # [C, n_ing]
        out[0, h_idx, r_idx, 0] = self._static_lo[:, heads]    # [C, n_ing]
        out[0, h_idx, r_idx, 1] = self._static_hi[:, heads]
        out2 = out.transpose(1, 2).contiguous().view(1, C, D)
        res = xc + self.attn.W_o.linear(out2)                  # block-0 attn residual
        return self.ffn.forward(res)                           # block-0 FFN

    def _capture(self):
        C, D = self.chunk, self.D
        self._static_xc = torch.zeros(1, C, D, device=self.device)
        self._static_lo = torch.zeros(C, self.n_roles, device=self.device)
        self._static_hi = torch.zeros(C, self.n_roles, device=self.device)
        st = torch.cuda.Stream(device=self.device)
        st.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(st):
            for _ in range(3):
                with torch.no_grad():
                    self._body()
        torch.cuda.current_stream(self.device).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                self._static_res = self._body()
        self._graph = g

    def run(self, xc: torch.Tensor, nib_lo: torch.Tensor, nib_hi: torch.Tensor
            ) -> torch.Tensor:
        """Replay block-0's per-chunk body.  ``xc`` [1, Sc, D]; ``nib_lo/hi`` [Sc,
        N_ROLES] (0 on non-query rows).  Sc <= chunk (tail padded).  Returns [1, Sc, D]."""
        if self._graph is None:
            self._capture()
        Sc = xc.shape[1]
        C = self.chunk
        if Sc == C:
            self._static_xc.copy_(xc)
            self._static_lo.copy_(nib_lo)
            self._static_hi.copy_(nib_hi)
        else:
            self._static_xc[:, :Sc, :].copy_(xc)
            self._static_xc[:, Sc:, :].zero_()
            self._static_lo[:Sc].copy_(nib_lo); self._static_lo[Sc:].zero_()
            self._static_hi[:Sc].copy_(nib_hi); self._static_hi[Sc:].zero_()
        self._graph.replay()
        return self._static_res[:, :Sc, :]


class Block0LoopPlan:
    """Precomputed, SYNC-FREE per-chunk plan for the block-0 S-chunk loop over a span.

    Built once per span from the draft's resolved frames (all on the host): for each
    chunk ``[lo0, hi0)`` it holds the chunk-local query rows and the matching ``hq0``
    destination indices, plus a per-chunk gather of the resolved ingest nibbles.  The
    loop then does NO ``bool(.any())`` sync — the membership is known on the host.
    """

    def __init__(self, rf, pos_map: torch.Tensor, chunk: int, S: int,
                 span_start: int, q_positions: Optional[torch.Tensor],
                 qpos_to_qi: torch.Tensor, device):
        self.chunk = chunk
        self.S = S
        self.device = torch.device(device)
        self.rf = rf
        # move the resolved-frame gather internals to the query device ONCE (the eager
        # path re-uploads pos_map per chunk; we hoist it out of the loop).
        self.pos_map = pos_map.to(self.device)
        self._nib_lo = rf.nib_lo.to(self.device)
        self._nib_hi = rf.nib_hi.to(self.device)
        self.span_start = span_start
        self.q_positions = q_positions
        self.qpos_to_qi = qpos_to_qi

    def chunk_positions(self, lo0: int, hi0: int) -> torch.Tensor:
        if self.q_positions is not None:
            return self.q_positions[lo0:hi0]
        return torch.arange(self.span_start + lo0, self.span_start + hi0,
                            device=self.device)

    def chunk_nibbles(self, qpc: torch.Tensor, Sc: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather the chunk's [Sc, N_ROLES] resolved ingest nibbles (0 on non-query
        rows) — the SAME rows the eager ``_gather_out_chunk`` scatters."""
        NR = self._nib_lo.shape[1]
        lo = torch.zeros(Sc, NR, device=self.device)
        hi = torch.zeros(Sc, NR, device=self.device)
        pm = self.pos_map
        qpc = qpc.to(self.device)
        clamped = qpc.clamp(max=pm.numel() - 1)
        row_idx = pm[clamped]                                  # [Sc], -1 non-query
        qmask = row_idx >= 0
        sel = row_idx[qmask]
        if sel.numel():
            local = qmask.nonzero(as_tuple=False).flatten()    # [nq] chunk-local rows
            lo[local] = self._nib_lo.index_select(0, sel)
            hi[local] = self._nib_hi.index_select(0, sel)
        return lo, hi


def install_whole_step_graph(model, device, chunk: int, rf, ingest_heads: List[int],
                             n_roles: int, cut: int = 1, verbose: bool = False
                             ) -> Optional[Block0LoopGraph]:
    """Build the ``Block0LoopGraph`` (RUNG 2) for the block-0 S-chunk loop.

    ``rf`` is the ``direct_local_cam.ResolvedFrames`` (the draft-resolved ingest nibbles);
    ``ingest_heads`` the block-0 register-ingest head indices; ``n_roles`` == N_ROLES.
    Byte-exact: the graph body is the same gather + W_o + FFN chain as the eager path.
    """
    if not whole_step_graph_enabled():
        return None
    g = Block0LoopGraph(model, device, chunk, ingest_heads, n_roles, cut)
    if verbose:
        print(f"[whole-step-graph] block-0 chunk-loop graph: chunk={chunk} "
              f"H={g.H} HD={g.HD} D={g.D} ingest_heads={len(ingest_heads)} "
              f"(gather+W_o+FFN fused into ONE replay/chunk, sync-free scatter)",
              flush=True)
    return g


__all__ = ["whole_step_graph_enabled", "Block0LoopGraph", "Block0LoopPlan",
           "install_whole_step_graph"]
