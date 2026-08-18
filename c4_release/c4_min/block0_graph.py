"""BLOCK-0 GRAPH (C4_GRAPH_BLOCK0) — CUDA-graph block-0's S-chunked ingest attention
+ FFN loop into ONE fixed-shape replay per chunk.

CONTEXT.  Once the composed stack (dead-block fusion + fused megablock + O(K) band +
direct-CAM) lands, the composed doom step is HOST-DISPATCH bound: the pure GPU device
time is ~131 us/step but the wall is ~770-838 us/step.  The ~640-707 us gap is host
dispatch over block-0's S-chunked ingest loop (``C4_CUT_SPAN_CHUNK`` /
``C4_BLOCK0_SQ_CHUNK``): block 0 is the SOLE block that runs over all ``S = K*30`` span
rows (every other block runs on the K query rows only, under ``C4_FROZEN_ROW_SKIP``), and
the cut-span-chunk path processes those S rows in fixed-size chunks, launching block-0's
whole forward (direct-CAM gather + ``W_o`` GEMM + dense SwiGLU FFN) PER CHUNK on the host.
At K=8192 that is ~960 chunks x ~13 kernels each = pure launch/dispatch overhead.

Because block-0 is a PER-ROW INDEPENDENT map under ``C4_DIRECT_LOCAL_CAM`` (the ingest
heads resolve their value by a direct GATHER from the draft-resolved frame, NOT by scoring
the span — there is no cross-row attention), the per-chunk block-0 compute is a
FIXED-SHAPE, past-independent, position-independent GEMM chain: given a fixed
``[1, chunk, D]`` overlaid embed and a fixed ``[1, H, chunk, HD]`` ingest-gather result, it
computes ``res = xc + W_o.linear(out2)`` then ``ffn.forward(res)``.  That captures cleanly
into a CUDA graph (the SAME technique ``MegaStepGraph`` / ``fused_megablock`` use for the
``[cut, N)`` FFN chain).  A single graph replays as ONE launch instead of ~13 per chunk.

BYTE-EXACT.  The graphed per-chunk output is the SAME dense ``F.linear`` GEMM chain (post
``materialize_dense``) in the SAME accumulation order as the eager
``direct_local_forward`` + ``SparseBlock.__call__`` per-chunk body (L-inf=0):
  * The ingest-gather ``static_out`` is filled by the IDENTICAL ``_gather_out_chunk`` index
    scatter the eager path uses (computed OUTSIDE the graph, into the static buffer — a
    cheap index op, no GEMM — so the graph body is fixed-shape and sync-free).
  * ``res = xc + W_o.linear(out2)`` is exactly the direct-local per-chunk residual
    (``direct_local_forward`` chunked path, ``res[:, lo0:hi0, :] = xc + W_o.linear(out2c)``).
  * ``ffn.forward(res)`` is exactly ``SparseBlock.__call__``'s ``out = self.ffn.forward(a)``.
  * The TAIL chunk (S not divisible by ``chunk``) is PADDED up to ``chunk`` rows; the pad
    rows are non-query frozen rows whose block-0 output is dropped downstream, so padding is
    output-irrelevant (the per-row map means a pad row NEVER perturbs a real row).
Block-0's KV is dead under direct-local (``C4_BLOCK0_DROP_DEAD_KV``): the graph returns
only the residual, no KV — same contract as the eager chunked drop-dead-kv path.

Graphs are keyed by the fixed chunk size (``C4_CUT_SPAN_CHUNK``), which is CONSTANT across
the whole verify, so ONE capture amortises over every chunk of every span.  Default OFF ->
the eager per-chunk block-0 loop (golden unchanged).
"""
from __future__ import annotations

import os
from typing import Optional

import torch


def block0_graph_enabled() -> bool:
    """``C4_GRAPH_BLOCK0`` (DEFAULT OFF): CUDA-graph block-0's S-chunked ingest
    attention + FFN per-chunk body.  Requires direct-local-CAM (block 0 per-row
    independent), the cut-span-chunk path (fixed-size chunks), drop-dead-kv, and a CUDA
    device.  OFF -> the eager per-chunk block-0 loop (byte-identical golden path)."""
    return os.environ.get("C4_GRAPH_BLOCK0", "0") not in ("0", "", "false", "False")


class Block0ChunkGraph:
    """CUDA-graphs block-0's per-chunk ingest-attention + FFN body at a fixed chunk size.

    ``run(xc, out)`` replays the graph for one fixed-size ``[1, chunk, D]`` chunk given the
    overlaid embed ``xc`` and the ingest-gather ``out`` ``[1, H, chunk, HD]`` (the caller
    fills ``out`` with the direct-CAM scatter, exactly as the eager path does), returning
    the ``[1, chunk, D]`` block-0 output.  Drop-in for the eager
    ``model.blocks[0](xc, ...)`` inside the cut-span-chunk loop, chunk shapes fixed.
    """

    def __init__(self, model, device, chunk: int, cut: int = 1):
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
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_xc: Optional[torch.Tensor] = None
        self._static_out: Optional[torch.Tensor] = None
        self._static_res: Optional[torch.Tensor] = None
        model.materialize_dense(device=str(self.device))

    # -- the fixed-shape per-chunk block-0 body (attn residual + FFN) -----------
    def _body(self, xc, out):
        """xc: [1, chunk, D] overlaid embed; out: [1, H, chunk, HD] ingest gather.
        Returns [1, chunk, D] block-0 output.  IDENTICAL algebra to the eager
        ``direct_local_forward`` chunked residual + ``SparseBlock.__call__`` FFN."""
        B = xc.shape[0]
        out2 = out.transpose(1, 2).contiguous().view(B, self.chunk, self.D)
        res = xc + self.attn.W_o.linear(out2)          # block-0 attention residual
        return self.ffn.forward(res)                    # block-0 dense SwiGLU FFN

    def _capture(self):
        C, D = self.chunk, self.D
        self._static_xc = torch.zeros(1, C, D, device=self.device)
        self._static_out = torch.zeros(1, self.H, C, self.HD, device=self.device)
        st = torch.cuda.Stream(device=self.device)
        st.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(st):
            for _ in range(3):
                with torch.no_grad():
                    self._body(self._static_xc, self._static_out)
        torch.cuda.current_stream(self.device).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                self._static_res = self._body(self._static_xc, self._static_out)
        self._graph = g

    def run(self, xc, out):
        """Replay block-0's per-chunk body.  ``xc`` [1, Sc, D], ``out`` [1, H, Sc, HD]
        with ``Sc <= chunk`` (the tail chunk is padded to ``chunk``).  Returns the
        block-0 output at the first ``Sc`` rows ([1, Sc, D])."""
        if self._graph is None:
            self._capture()
        Sc = xc.shape[1]
        C = self.chunk
        if Sc == C:
            self._static_xc.copy_(xc)
            self._static_out.copy_(out)
        else:
            # TAIL chunk: pad to the fixed capture size.  Pad rows are non-query frozen
            # rows (their block-0 output is dropped downstream) — the per-row map means a
            # pad row NEVER perturbs a real row, so padding is output-irrelevant.
            self._static_xc[:, :Sc, :].copy_(xc)
            self._static_xc[:, Sc:, :].zero_()
            self._static_out[:, :, :Sc, :].copy_(out)
            self._static_out[:, :, Sc:, :].zero_()
        self._graph.replay()
        return self._static_res[:, :Sc, :]


def install_block0_graph(model, device, chunk: int, cut: int = 1, verbose: bool = False
                         ) -> Block0ChunkGraph:
    """Build + return a ``Block0ChunkGraph`` for block-0's fixed-size S-chunk body.

    Requires direct-local-CAM (block 0 per-row independent) + cut-span-chunking + CUDA.
    Byte-exact: the graphed per-chunk body equals the eager ``direct_local_forward`` +
    FFN chain (L-inf=0)."""
    w = Block0ChunkGraph(model, device, chunk, cut)
    if verbose:
        print(f"[block0-graph] chunk={chunk} H={w.H} HD={w.HD} D={w.D} "
              f"(one graph replay per chunk, captured on first use)", flush=True)
    return w
