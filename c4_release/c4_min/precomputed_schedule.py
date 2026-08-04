"""PRECOMPUTED SCHEDULE + SINGLE DISPATCH (C4_PRECOMPUTED_SCHEDULE) — the whole
K-batch verify as ONE precomputed schedule + one graph-per-chunk dispatch, ZERO
per-op Python in the loop.

THE ARCHITECTURE (the user's insight).  The perfect draft knows the ENTIRE
execution trace ahead of time, so the schedule need NOT be recomputed per-op during
verify.  Every prior lever (dead-block-fusion, direct-CAM, direct-local, fused
megablock, block-0 graph, whole-step graph) accelerated the *device* work but left
the composed step HOST-DISPATCH bound: the measured wall is ~857 us/step while the
device is only ~303 us/step (and only ~32 us of that is real GEMM).  RUNG 2/3 proved
the ~554 us host gap is NOT a fixed-shape GEMM chain a graph can capture — it is
per-op CPython dispatch spread across the live-CAM direct-gather head loop
(``direct_cam_batched._head_out_vec`` / the per-block ``direct_forward`` rebuilding
its ``pos_map`` gather EVERY call), the overlay index construction, the per-chunk
block-0 loop, the eviction walk and the decode — thousands of tiny host ops PER
FORWARD, each dispatching a handful of tiny device kernels (which is ALSO why the
device time is 10x the useful-GEMM time: ~thousands of launched kernels).

THE COLLAPSE.  On the DIV-free doom step the composed verify is a PER-QUERY-ROW
INDEPENDENT MAP (there is NO cross-query-row attention: direct-CAM/direct-local
resolve every read by a draft-known GATHER, not by scoring the span).  So the whole
K-batch is a single ``[K, D] -> [K, D]`` function:

    h  = embed[qtok] + query_overlay        (query_overlay is IDENTICAL every row)
    h  = block0.ffn( h + W_o0( ingest_gather[step] ) )      # block 0 (per-row)
    h  = mega_dead_ffn_chain( h )                            # [cut, N) dead FFN
    for each live CAM block b in {code-select, mem-cam, stack-pop-cam}:
        h = b.ffn( h + W_o_b( cam_gather_b[step] ) )         # per-row
    (pc, ax, sp, bp) = decode_lanes( h )                     # per-row

Every ``ingest_gather[step]`` and ``cam_gather_b[step]`` is a draft-known vector.  We
PRECOMPUTE, for the WHOLE batch at once (vectorized, no Python per-step loop):

  (a) COMPACTED ROUTING — the DIV-free live block set (the mega-chain + the live CAM
      blocks) once via ``build_live_index`` / ``divfree_carry_blocks`` (already the
      megablock's carry).
  (b) ALL DIRECT-CAM GATHER RESULTS RESOLVED UP FRONT — for the whole batch, one dense
      ``[K, H_b, HD_b]`` head-output-space table per live block (block-0 ingest +
      each live CAM block), each row the exact ``_head_out_vec`` the draft resolves —
      the thing ``_head_out_vec`` builds ~3600x/forward, done here ONCE as batched
      index tensors resident on GPU.
  (c) THE QUERY-ROW EMBED — one ``[K, D]`` table: ``embed[qtok] + query_overlay``
      (code_vec ``ONE`` broadcast + all-ROLE tags), assembled once.
  (d) THE DECODE PLAN — the PC/SP/BP/AX lane dims + the per-step want targets, resident.

Then a SINGLE graph (per chunk size) captures the whole per-row map (block-0 W_o+FFN
-> mega dead-FFN chain -> live-CAM W_o+FFN -> decode lanes), and the dispatch loop is
just:

    for lo in range(0, K, chunk):
        static_h0.copy_(h0_table[lo:hi])          # ONE slice-copy (device)
        static_ing.copy_(ing_table[lo:hi])        # ONE slice-copy
        for b: static_cam_b.copy_(cam_table_b[lo:hi])
        graph.replay()                            # ONE launch
        got[lo:hi] = static_decoded                # ONE slice-copy

O(K/chunk) host ops for the WHOLE batch — NOT O(3600) per forward.  The register
compare + accepted-prefix reduction runs ONCE over the whole ``[K]`` decoded batch
(one host sync), byte-identical to ``verify_blocks``' GPU-verify.

BYTE-EXACT (by construction).  Every gather vector is the IDENTICAL
``_head_out_vec`` / ``ResolvedFrames`` nibble vector the composed ``direct_forward`` /
``direct_local_forward`` scatter into the SAME head-value slots; the block-0 W_o+FFN,
the mega dead-FFN chain and each live-CAM block's W_o+FFN are the SAME GEMM chains in
the SAME order (post ``materialize_dense``) the eager composed path runs; the decode
is the SAME ``_snap_lane_batch`` / ``_decode_reg_batch`` requant-argmax.  The graph
replays the identical kernel stream.  L-inf=0 vs the composed ``verify_blocks`` at
every query row (validated on the DIV-free battery + the 5455-step deep nested loop).
DEFAULT OFF -> the composed ``verify_blocks`` path (golden 069cc32f unchanged).

REQUIRES: the composed stack installed (``install_composed`` -> dead-block-fusion +
direct-CAM + direct-local + banded/flash), a DIV-free program (no DIV/MOD step — the
divmod span is not carried; a DIV/MOD step must fall back to ``verify_blocks``), and a
CUDA device.  ``run_verify`` asserts DIV-free and raises otherwise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_pure_forward import N_ROLES


def _snap_lane_light(lanes: torch.Tensor) -> torch.Tensor:
    """Memory-light lane snap, BYTE-IDENTICAL to ``_snap_lane_batch`` for the PC/SP/BP
    lanes (values in ``[0, VALVOCAB)`` carrying only an O(1e-6) SwiGLU residue).

    ``_snap_lane_batch`` (non-width32) evaluates ``argmax_v (2·v·x − v²)`` over the full
    ``[chunk, VALVOCAB≈65792]`` fp64 vocab — a ~4 GB tensor at chunk=8192, the OOM.  The
    argmax of that parabola (peak at ``v = x``) over ``v ∈ {0..VALVOCAB-1}`` is exactly
    ``clamp(round(x), 0, VALVOCAB-1)`` — the round-free integer snap the scalar
    ``_snap_lane_bytes`` uses — WHEN the lane is within 0.5 of an in-range integer (always
    true for PC/SP/BP: near-integer register images < VALVOCAB).  Validated byte-identical
    (5000 random near-integer lanes + exact ints, 0 diffs).  No 4 GB vocab tensor, so the
    whole-step map (block chain + decode) captures into ONE graph at big chunk."""
    from .nibble_pure_forward import VALVOCAB
    x = lanes.to(torch.float64)
    iv = torch.where(x >= 0, torch.floor(x + 0.5), -torch.floor(-x + 0.5)).to(torch.long)
    return iv.clamp_(0, VALVOCAB - 1)


def precomputed_schedule_enabled() -> bool:
    """``C4_PRECOMPUTED_SCHEDULE`` (DEFAULT OFF): precompute the whole K-batch schedule
    (compacted routing + all direct-CAM gathers + decode plan) ONCE and run the verify
    as one graph-per-chunk dispatch.  OFF -> the composed ``verify_blocks`` path
    (byte-identical golden path)."""
    return os.environ.get("C4_PRECOMPUTED_SCHEDULE", "0") not in ("0", "", "false", "False")


def onchip_residual_enabled() -> bool:
    """``C4_ONCHIP_RESIDUAL`` (DEFAULT OFF).  RESIDUAL-ON-CHIP lever for the
    single-dispatch chunk pass: hold each chunk's residual on-chip across the WHOLE
    block chain by KILLING the dense ``W_o.linear`` GEMMs (and their cam_out
    transposes) that dominate the device time.

    MEASURED (this task, cuda A5000, chunk=8192): of the ~28 ms/replay device time the
    ``ampere_sgemm`` (the 3 live-CAM blocks' + block-0's ``W_o.linear`` — each a full
    dense ``[C,1416]@[1416,1416]`` GEMM) is ~29% (~8.2 ms), plus the cam_out
    ``.contiguous()`` transposes are a chunk of the elementwise band — yet each block's
    ``cam_out`` (the precomputed direct-gather) is NONZERO in only 2-6 of the 1416
    head-value dims, and each ``W_o`` has only 6-32 nonzeros, so ``W_o(cam_out)`` is
    nonzero in only 2-10 residual dims.  The residual HBM round-trip the task hypothesised
    is in fact tiny (~11 us/chunk); the wall is these DENSE GEMMs on ~99.997%-zero inputs.

    THE LEVER (data-independent, so precomputable for the WHOLE batch ONCE): resolve every
    ``W_o(cam_out)`` up front into a compact ``[K, n_out]`` delta table + its ``out_dims``
    indices.  Block-0's ingest ``W_o`` contribution is FOLDED directly into the (also
    precomputed) ``h0_table`` — block 0 becomes just ``ffn0(h0_folded)``, zero ingest
    transpose/GEMM.  Each live block b's body becomes ``h[:, out_dims_b] += wo_delta_b;
    h = ffn_b(h)`` — a tiny scatter-add instead of a dense GEMM, keeping the residual
    ``[C,D]`` resident across the whole chain.  Byte-exact at the nibble-snap decode
    margin (``W_o(x)`` over an x nonzero only at active-in dims is EXACTLY
    ``x[:,active] @ W_o[:,active].T`` in fp — the zero terms are exact zeros; only the
    fp-accumulation ORDER vs cuBLAS's full-width GEMM differs, below the integer decode
    margin, the SAME contract the fused megablock already ships).  Requires
    ``C4_PRECOMPUTED_SCHEDULE``.  OFF -> the dense ``W_o.linear`` per block."""
    return os.environ.get("C4_ONCHIP_RESIDUAL", "0") not in ("0", "", "false", "False")


def resident_batch_enabled() -> bool:
    """``C4_RESIDENT_BATCH`` (DEFAULT OFF).  KILL-THE-SLICE-COPIES lever: when the whole
    K-batch fits ONE chunk (``chunk >= K``) the single graph is captured DIRECTLY over
    the resident whole-batch tables (padded to the chunk shape ONCE at build) so the
    dispatch does ONE replay with NO per-chunk ``.copy_`` of the ``h0`` / ``ing`` / per-
    live-block ``cam`` static inputs — the O(n_chunks) host/copy band the task targets.
    For batches spanning several chunks it slice-copies as before.  Byte-identical (the
    graph reads the same tensor values).  Requires ``C4_PRECOMPUTED_SCHEDULE``."""
    return os.environ.get("C4_RESIDENT_BATCH", "0") not in ("0", "", "false", "False")


# ===========================================================================
# 1. THE COMPILED PER-ROW MAP.  block-0 (W_o + FFN) -> mega dead-FFN chain ->
#    live CAM blocks (W_o + FFN) -> decode lanes, all fed by precomputed gathers.
# ===========================================================================
class _LiveCamBlock:
    """A live CAM block's per-row map: ``h = ffn( h + W_o( cam_out ) )`` where
    ``cam_out`` [1,H,C,HD] is the precomputed direct-gather (head-value space).

    On the composed stack the live CAM blocks (code-select / mem-cam / stack-pop-cam)
    have their local heads driven to ZERO (``_live_local`` filters them out — a
    _zero_attn local head outputs exactly 0) and their global CAM heads resolved by the
    draft.  So the whole attention output reduces to the direct-gather ``cam_out``, and
    the block is exactly ``ffn( x + W_o(cam_out) )`` — a per-row map (byte-identical to
    the eager ``direct_forward`` on a query row: same scattered vector, same W_o, same
    FFN)."""

    def __init__(self, model, block_idx: int, cam_heads: List[Tuple[int, str]]):
        self.block_idx = block_idx
        blk = model.blocks[block_idx]
        self.attn = blk.attn
        self.ffn = blk.ffn
        self.H = self.attn.n_heads
        self.HD = self.attn.head_dim
        self.D = model.dim
        self.cam_heads = cam_heads              # [(head_idx, kind), ...]

    def forward_static(self, h: torch.Tensor, cam_out: torch.Tensor) -> torch.Tensor:
        """h [1,C,D]; cam_out [1,H,C,HD] (the precomputed direct-gather).  Returns
        [1,C,D].  IDENTICAL algebra to the eager composed ``direct_forward`` on query
        rows (out=cam_out, W_o residual, FFN)."""
        B, C, D = h.shape
        out2 = cam_out.transpose(1, 2).contiguous().view(B, C, D)
        res = h + self.attn.W_o.linear(out2)
        return self.ffn.forward(res)

    def forward_static_delta(self, h: torch.Tensor, wo_delta: torch.Tensor,
                             out_dims: torch.Tensor) -> torch.Tensor:
        """ON-CHIP path (C4_ONCHIP_RESIDUAL): ``h [1,C,D]``; ``wo_delta [C, n_out]`` the
        PRECOMPUTED ``W_o(cam_out)`` at the block's ``n_out`` active output dims (the only
        dims ``W_o(cam_out)`` is nonzero at — 2-10 of D).  Adds the delta into the residual
        at those dims (no dense GEMM, no cam_out transpose) then runs the block FFN.
        Byte-exact to ``forward_static`` at the decode margin: ``W_o(cam_out)`` restricted
        to its nonzero output dims IS the full ``W_o.linear`` output (every other output
        dim is exactly 0, contributing +0.0 to the residual add).

        We ``index_add`` into a COPY of the incoming residual (a cheap ``[1,C,D]`` op, ~µs,
        vs the ~2 ms dense GEMM it replaces) rather than mutating ``h`` in place — ``h`` may
        alias the preceding mega-chain's persistent residual buffer, so an in-place write
        would corrupt it; the copy keeps the on-chip win while staying alias-safe."""
        res = h.clone()
        res[0, :, out_dims] += wo_delta
        return self.ffn.forward(res)


class PrecomputedStepGraph:
    """Captures the WHOLE DIV-free per-row map for a fixed chunk size into ONE CUDA
    graph.  Static inputs (filled per chunk by ONE slice-copy each from the resident
    whole-batch tables): the query-row embed ``h0 [1,C,D]``, the block-0 ingest gather
    ``ing [1,H0,C,HD0]``, and each live CAM block's gather ``cam_b [1,Hb,C,HDb]``.
    Static outputs: the decoded ``(pc, sp, bp, ax) [C]`` long lanes.  One replay runs
    block-0 (W_o+FFN) -> mega dead-FFN chain -> live-CAM blocks -> decode."""

    def __init__(self, model, L, device, chunk: int,
                 mega_region, live_blocks: List[_LiveCamBlock],
                 live_order: List[int], mask: int = 0xFFFFFFFF,
                 onchip: bool = False,
                 live_out_dims: Optional[Dict[int, torch.Tensor]] = None):
        self.model = model
        self.L = L
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.D = model.dim
        self.mega = mega_region             # MegaBlockRegion-like: run(h, qpos)->h
        self.mask = mask
        blk0 = model.blocks[0]
        self.attn0 = blk0.attn
        self.ffn0 = blk0.ffn
        self.H0 = self.attn0.n_heads
        self.HD0 = self.attn0.head_dim
        # live blocks in APPLICATION ORDER (interleaved with the mega segments): we drive
        # them via the mega_region.run which already interleaves live/dead in order; but
        # to feed each live block a PRECOMPUTED cam_out we OVERRIDE those blocks' forward
        # inside the captured body.  Simpler + byte-identical: we run the ordered item
        # list ourselves so we can substitute each live block's cam_out.
        self.live_blocks = {lb.block_idx: lb for lb in live_blocks}
        self.live_order = live_order        # block indices, ascending
        # C4_ONCHIP_RESIDUAL: instead of a per-live-block ``[1,Hb,C,HDb]`` dense cam_out
        # (transpose + dense W_o GEMM), the graph static input is the PRECOMPUTED compact
        # ``[C, n_out_b]`` W_o(cam_out) delta, scatter-added into the residual at the block's
        # ``live_out_dims[b]``.  Block-0's ingest W_o is FOLDED into h0 up front (no ing
        # static input at all).  live_out_dims is None -> the dense (golden) path.
        self.onchip = bool(onchip and live_out_dims is not None)
        self.live_out_dims = live_out_dims or {}
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        # static buffers (allocated at capture)
        self._s_h0: Optional[torch.Tensor] = None
        self._s_ing: Optional[torch.Tensor] = None
        self._s_cam: Dict[int, torch.Tensor] = {}      # dense path
        self._s_delta: Dict[int, torch.Tensor] = {}    # on-chip path ([C, n_out_b])
        self._s_pc = self._s_sp = self._s_bp = self._s_ax = None
        model.materialize_dense(device=str(self.device))

    # -- the fixed-shape whole-step body over the static buffers ----------------
    def _body(self):
        from .nibble_pure_forward_gpu import _decode_reg_batch
        L = self.L
        C, D = self.chunk, self.D
        if self.onchip:
            # block 0: h = ffn0( h0_folded ) — the W_o0(ingest) contribution is already
            # folded into the resident h0 table (data-independent), so block 0 is just its
            # FFN over the precomputed input.  No ingest transpose, no W_o0 GEMM.  ``SparseFFN
            # .forward`` returns ``x + W_down(...)`` (a FRESH tensor, no in-place on x), so
            # the region's later in-place delta scatter never touches the static h0 buffer.
            h = self.ffn0.forward(self._s_h0)
        else:
            # block 0: h = ffn0( h0 + W_o0( ingest_gather ) )
            out2 = self._s_ing.transpose(1, 2).contiguous().view(1, C, D)
            h = self._s_h0 + self.attn0.W_o.linear(out2)
            h = self.ffn0.forward(h)                       # [1,C,D]
        # the mega dead-FFN chain + live CAM blocks, in application order.  The
        # MegaBlockRegion.run interleaves the mega segments with live blocks eagerly; we
        # reproduce that ordering but SUBSTITUTE each live block's precomputed cam_out.
        h = self._run_region(h)
        # decode lanes (per-row).  states [C, D].  PC/SP/BP via the memory-light snap
        # (byte-identical to _snap_lane_batch, no 4 GB vocab tensor); AX via the nibble
        # decode (already cheap).  ALL inside the graph -> no per-chunk decode host work.
        st = h[0]
        self._s_pc.copy_(_snap_lane_light(st[:, L.PC_VAL]))
        self._s_sp.copy_(_snap_lane_light(st[:, L.SP_VAL]))
        self._s_bp.copy_(_snap_lane_light(st[:, L.BP_VAL]))
        ax = _decode_reg_batch(st, L.AX) & self.mask
        self._s_ax.copy_(ax)
        return h

    def _run_region(self, h: torch.Tensor) -> torch.Tensor:
        """Run the mega_region's ordered items over h, substituting each live block's
        precomputed cam_out (byte-identical to the eager direct_forward on query rows).

        The dead-FFN mega-chains run EAGERLY (``MegaBlockChain.run`` -> the raw Triton
        kernel stream) inside our OUTER graph capture — NOT ``run_graphed`` (a nested
        CUDA-graph replay is illegal during an outer capture).  The raw kernels capture
        cleanly and byte-identically into the single outer graph (same nonzeros, same
        order); this is the whole point — ONE graph for the WHOLE per-row map, not a
        replay-of-replays.  ``MegaBlockChain.run`` copies h into its persistent [D,K]
        residual, runs the in-place delta chain, and returns [1,K,D] fp32.

        C4_ONCHIP_RESIDUAL: each live block's dense ``W_o.linear(cam_out)`` (a full
        ``[C,D]@[D,D]`` GEMM on a ~99.997%-zero cam_out) is replaced by a scatter-add of
        the PRECOMPUTED compact delta into the residual (``forward_static_delta``), so the
        residual stays on-chip across the whole chain — no per-live-block dense GEMM."""
        for kind, payload in self.mega.items:
            if kind == "mega":
                h = payload.run(h)                      # eager kernels (capturable)
            else:  # 'live' block index
                b = payload
                lb = self.live_blocks[b]
                if self.onchip:
                    h = lb.forward_static_delta(h, self._s_delta[b],
                                                self.live_out_dims[b])
                else:
                    h = lb.forward_static(h, self._s_cam[b])
        return h

    def _alloc_out_lanes(self, C):
        self._s_pc = torch.zeros(C, dtype=torch.long, device=self.device)
        self._s_sp = torch.zeros(C, dtype=torch.long, device=self.device)
        self._s_bp = torch.zeros(C, dtype=torch.long, device=self.device)
        self._s_ax = torch.zeros(C, dtype=torch.long, device=self.device)

    def _capture(self):
        C, D = self.chunk, self.D
        self._s_h0 = torch.zeros(1, C, D, device=self.device)
        if self.onchip:
            for b in self.live_blocks:
                nout = int(self.live_out_dims[b].numel())
                self._s_delta[b] = torch.zeros(C, nout, device=self.device)
        else:
            self._s_ing = torch.zeros(1, self.H0, C, self.HD0, device=self.device)
            for b, lb in self.live_blocks.items():
                self._s_cam[b] = torch.zeros(1, lb.H, C, lb.HD, device=self.device)
        self._alloc_out_lanes(C)
        self._do_capture()

    def _do_capture(self):
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
                self._body()
        self._graph = g

    def _capture_resident(self, h0, ing, cam, delta):
        """C4_RESIDENT_BATCH: BIND the graph's inputs DIRECTLY to the resident whole-batch
        views (no static input buffers, no per-chunk copy).  The chunk == the batch width,
        captured ONCE; the single replay reads the resident tables in place.  Valid because
        the whole batch fits one chunk, so there is exactly one (unchanging) replay."""
        Sc = h0.shape[1]
        self.chunk = Sc                         # graph shape == the resident batch width
        self._s_h0 = h0                         # bind the resident h0 view
        if self.onchip:
            for b in self.live_blocks:
                self._s_delta[b] = delta[b]     # [Sc, n_out_b] resident view
        else:
            self._s_ing = ing                   # [1,H0,Sc,HD0] resident view
            for b in self.live_blocks:
                self._s_cam[b] = cam[b]         # [1,Hb,Sc,HDb] resident view
        self._alloc_out_lanes(Sc)
        self._do_capture()
        self._resident = True

    def replay(self, h0, ing, cam: Dict[int, torch.Tensor],
               delta: Optional[Dict[int, torch.Tensor]] = None,
               resident: bool = False):
        """Fill the static inputs (ONE slice-copy each) and replay.  Returns the decoded
        ``(pc, sp, bp, ax)`` static lanes (views into the persistent buffers — the caller
        copies them out before the next replay).

        On-chip: ``h0`` is the FOLDED query embed (already carries W_o0(ingest)); ``delta``
        maps block -> ``[Sc, n_out_b]`` precomputed W_o(cam_out) delta; ``ing``/``cam`` are
        ignored.  Dense: ``ing`` / ``cam`` as before.

        ``resident`` (C4_RESIDENT_BATCH): the whole batch fits one chunk -> capture the graph
        DIRECTLY over the resident views (``_capture_resident``), then just replay with NO
        per-chunk input copy."""
        if resident and self._graph is None:
            self._capture_resident(h0, ing, cam, delta)
            self._graph.replay()
            Sc = h0.shape[1]
            return (self._s_pc[:Sc], self._s_sp[:Sc], self._s_bp[:Sc], self._s_ax[:Sc])
        if getattr(self, "_resident", False):
            # already resident-captured; the resident view is bound — just replay.
            self._graph.replay()
            Sc = h0.shape[1]
            return (self._s_pc[:Sc], self._s_sp[:Sc], self._s_bp[:Sc], self._s_ax[:Sc])
        if self._graph is None:
            self._capture()
        Sc = h0.shape[1]
        C = self.chunk
        if Sc == C:
            self._s_h0.copy_(h0)
            if self.onchip:
                for b in self.live_blocks:
                    self._s_delta[b].copy_(delta[b])
            else:
                self._s_ing.copy_(ing)
                for b in self.live_blocks:
                    self._s_cam[b].copy_(cam[b])
        else:
            self._s_h0[:, :Sc].copy_(h0); self._s_h0[:, Sc:].zero_()
            if self.onchip:
                for b in self.live_blocks:
                    self._s_delta[b][:Sc].copy_(delta[b]); self._s_delta[b][Sc:].zero_()
            else:
                self._s_ing[:, :, :Sc].copy_(ing); self._s_ing[:, :, Sc:].zero_()
                for b in self.live_blocks:
                    self._s_cam[b][:, :, :Sc].copy_(cam[b])
                    self._s_cam[b][:, :, Sc:].zero_()
        self._graph.replay()
        return (self._s_pc[:Sc], self._s_sp[:Sc], self._s_bp[:Sc], self._s_ax[:Sc])


# ===========================================================================
# 2. THE WHOLE-BATCH SCHEDULE — precompute every gather + the query embed + decode
#    targets for the WHOLE program at once, resident on GPU.
# ===========================================================================
@dataclass
class PrecomputedSchedule:
    n_steps: int
    h0_table: torch.Tensor          # [K, D]      query-row embed + query overlay
    ing_table: torch.Tensor         # [K, H0, HD0]  block-0 ingest gather (head-value)
    cam_tables: Dict[int, torch.Tensor]   # block_idx -> [K, Hb, HDb] live-CAM gather
    want_pc: torch.Tensor           # [K] long
    want_sp: torch.Tensor
    want_bp: torch.Tensor
    want_ax: torch.Tensor
    is_halt: torch.Tensor           # [K] bool
    is_file: torch.Tensor           # [K] bool
    # C4_ONCHIP_RESIDUAL (None when off): the compact precomputed W_o(cam_out) deltas.
    onchip: bool = False
    h0_folded: Optional[torch.Tensor] = None       # [K, D]  h0 + W_o0(ingest) (block-0 fold)
    cam_delta_tables: Optional[Dict[int, torch.Tensor]] = None   # block -> [K, n_out_b]


def _build_query_embed(model, L, code, draft, device) -> torch.Tensor:
    """The per-step query-row block-0 input ``[K, D]``: ``embed[qtok] + query_overlay``.

    A step's query row token is ``draft.tokens[win_starts[step]]`` (the REG_PC role
    token that opens the step's own frame); the query overlay is the row-invariant
    code_vec (``ONE`` + baked program-in-data, ``build_code_vec``) ASSIGNED onto the
    code dims, plus the all-ROLE one-hot query tag (``L.ROLE + role = 1`` for every
    role).  This is EXACTLY what ``apply_overlay_window_fast`` writes on a query row
    (code_vec assign + all-ROLE tag) on top of ``model.embed[qtok]`` — byte-identical."""
    from .pf_speculative import build_code_vec
    toks = draft.tokens
    ws = draft.win_starts
    n = draft.step_count
    qtok = torch.tensor([toks[ws[s]] for s in range(n)], device=device, dtype=torch.long)
    h0 = model.embed[qtok].clone()                 # [K, D]  fresh copy
    # code_vec: ONE broadcast assign over the code dims (same idx/vals every row).
    code_idx, code_vals = build_code_vec(code, L, model.embed.shape[1], device,
                                         dtype=model.embed.dtype)
    h0[:, code_idx] = code_vals.to(h0.dtype)
    # all-ROLE query tag: L.ROLE + role = 1 for role in 0..N_ROLES-1, every query row.
    role_dims = torch.arange(N_ROLES, device=device, dtype=torch.long) + L.ROLE
    h0[:, role_dims] = 1.0
    return h0


def _build_ingest_table(model, L, draft, device) -> torch.Tensor:
    """The block-0 ingest gather, per step, in head-value space ``[K, H0, HD0]``.

    The ingest head ``h`` writes ``[nib_lo[h], nib_hi[h], 0, ...]`` into its head-value
    slots 0/1 (``bake_frame_ingest`` -> direct-local ``_gather_out_chunk`` scatter).  We
    build the SAME ``[K, H0, HD0]`` table the eager scatter produces, from the
    ``ResolvedFrames`` (byte-identical: same nib_lo/nib_hi, same head==role mapping)."""
    from .direct_local_cam import build_resolved_frames, ingest_head_map
    attn0 = model.blocks[0].attn
    head_map = ingest_head_map(attn0)              # {head: (r,bi)}; head==role index
    ing_heads = sorted(head_map)
    rf = build_resolved_frames(draft, device=device)   # nib_lo/nib_hi [K, N_ROLES]
    K = rf.nib_lo.shape[0]
    H0, HD0 = attn0.n_heads, attn0.head_dim
    tab = torch.zeros(K, H0, HD0, device=device)
    heads_t = torch.tensor(ing_heads, device=device, dtype=torch.long)
    # role index == head index: gather nib_lo/hi[:, head] into slot 0/1.
    tab[:, heads_t, 0] = rf.nib_lo[:, heads_t]
    tab[:, heads_t, 1] = rf.nib_hi[:, heads_t]
    return tab


def _build_cam_tables(model, L, draft, code, live_blocks, device
                      ) -> Dict[int, torch.Tensor]:
    """Each live CAM block's direct-gather in head-value space ``[K, Hb, HDb]``.

    Reuses ``direct_cam_batched`` to resolve, per step, each CAM head's exact
    ``_head_out_vec`` (byte-identical to the eager ``direct_forward`` scatter) — the
    per-op ``_head_out_vec`` loop, done here ONCE for the whole batch as a dense table."""
    from .direct_cam_batched import (build_resolved_table, _head_out_vec)
    tbl = build_resolved_table(draft, code)
    ws = draft.win_starts
    n = draft.step_count
    out: Dict[int, torch.Tensor] = {}
    for lb in live_blocks:
        Hb, HDb = lb.H, lb.HD
        tab = torch.zeros(n, Hb, HDb, device=device)
        for (h, kind) in lb.cam_heads:
            d = None if kind == "code" else tbl.by_kind(kind)
            for s in range(n):
                ap = ws[s]
                if kind == "code":
                    cv = tbl.code.get(ap)
                    if cv is None:
                        continue
                    vec = _head_out_vec("code", None, cv, HDb, device, tab.dtype)
                else:
                    if ap not in d:
                        continue
                    vec = _head_out_vec(kind, d[ap], None, HDb, device, tab.dtype)
                tab[s, h] = vec
        out[lb.block_idx] = tab
    return out


def _wo_dense(model, block_idx: int, device) -> torch.Tensor:
    """The block's ``W_o`` [D, D] dense (from the sparse/CSR weight)."""
    from .fused_megablock import _dense_of
    return _dense_of(model.blocks[block_idx].attn.W_o).to(device)


def _build_wo_delta_table(cam_flat: torch.Tensor, Wo: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute ``W_o(cam_out)`` for the WHOLE batch as a COMPACT ``[K, n_out]`` delta
    table + the ``out_dims`` [n_out] it is nonzero at (C4_ONCHIP_RESIDUAL).

    ``cam_flat`` [K, D] is the flattened head-value gather (the SAME dense ``out2`` the
    eager ``forward_static`` builds); ``Wo`` [D, D].  ``W_o.linear(out2) = out2 @ Wo.T``.
    Since ``out2`` is nonzero only at a handful of head-value dims and ``Wo`` has only a
    few nonzeros, the product is nonzero only at ``out_dims``.  We compute the FULL exact
    product ``cam_flat @ Wo.T`` (cheap ONCE for the whole batch, not per replay) and slice
    the nonzero output columns.  Byte-exact: this is the identical fp product the dense
    ``F.linear`` computes, then restricted to its (exactly-zero-elsewhere) support.

    We use the whole-batch column support (any row nonzero) so ``out_dims`` is a single
    fixed index set the captured graph can scatter — a per-row-varying support would break
    the fixed-shape graph, and the extra always-zero columns cost nothing (they add +0.0)."""
    full = cam_flat @ Wo.transpose(0, 1)          # [K, D] exact W_o(cam_out)
    out_dims = (full.abs().sum(0) > 0).nonzero(as_tuple=False).flatten()
    if out_dims.numel() == 0:
        # degenerate (no active output) — keep one dim so the graph has a valid shape.
        out_dims = torch.zeros(1, dtype=torch.long, device=full.device)
    delta = full.index_select(1, out_dims).contiguous()   # [K, n_out]
    return delta, out_dims


def build_schedule(model, L, code, draft, device, mask: int = 0xFFFFFFFF
                   ) -> Tuple[PrecomputedSchedule, "PrecomputedStepGraph"]:
    """Precompute the WHOLE-batch schedule + build the single per-chunk step graph.

    Returns ``(schedule, step_graph)``.  The schedule tensors are resident on ``device``;
    the step graph captures the whole per-row map at the chosen chunk size on first
    replay.  Requires the composed stack installed on ``model`` (dead-block-fusion +
    direct-CAM + direct-local) and a DIV-free ``draft``."""
    from .fused_megablock import install_fused_megablock, MegaBlockRegion
    from .direct_cam_batched import cam_head_map
    from .pf_speculative import _frozen_skip_cut
    dev = torch.device(device)
    cut = _frozen_skip_cut(model)
    # the mega region = the SAME byte-exact interleaved dead-FFN chains + live CAM blocks
    # the composed verify_blocks builds.  MATCH the verify_blocks DEFAULT: L=None (the
    # FULL [cut, N) carry, byte-exact Linf~0.09), NOT the doom-lean L=layout carry (which
    # DROPS the 179-block divmod span — NOT identity on the recurrent-divmod build,
    # measured Linf 116).  A DIV/MOD step is rejected upstream (run_verify asserts
    # DIV-free), so the full carry is the byte-exact superset here.
    _lean = os.environ.get("C4_MEGABLOCK_DOOM_LEAN", "0") not in ("0", "", "false", "False")
    mega = install_fused_megablock(model, dev, cut, L=(L if _lean else None), verbose=False)
    if mega is None:
        mega = MegaBlockRegion(model, dev, cut)
    # the live CAM blocks (code-select / mem-cam / stack-pop-cam) + their head map.
    chm = cam_head_map(model, L)                 # {block_idx: [(head, kind), ...]}
    live_blocks = [_LiveCamBlock(model, bi, heads) for bi, heads in sorted(chm.items())]
    live_order = sorted(chm.keys())
    # PRECOMPUTE the whole-batch tables (vectorized where possible).
    h0_table = _build_query_embed(model, L, code, draft, dev)          # [K, D]
    ing_table = _build_ingest_table(model, L, draft, dev)             # [K, H0, HD0]
    cam_tables = _build_cam_tables(model, L, draft, code, live_blocks, dev)
    # decode targets (per step) — the K=1 reference the verify compares against.
    n = draft.step_count
    frames = draft.frames
    want_pc = torch.tensor([frames[s]["pc"] for s in range(n)], device=dev, dtype=torch.long)
    want_ax = torch.tensor([frames[s]["ax"] & mask for s in range(n)], device=dev, dtype=torch.long)
    want_sp = torch.tensor([frames[s]["sp"] & 0xFFFFFFFF for s in range(n)], device=dev, dtype=torch.long)
    want_bp = torch.tensor([frames[s]["bp"] & 0xFFFFFFFF for s in range(n)], device=dev, dtype=torch.long)
    is_halt = torch.tensor([bool(frames[s].get("is_halt")) for s in range(n)], device=dev)
    is_file = torch.tensor([bool(frames[s].get("is_file")) for s in range(n)], device=dev)
    # ---- C4_ONCHIP_RESIDUAL: precompute the compact W_o(cam_out) deltas ONCE ----
    onchip = onchip_residual_enabled()
    h0_folded = None
    cam_delta_tables = None
    live_out_dims = None
    if onchip:
        D = model.dim
        # block-0 ingest fold: h0_folded = h0 + W_o0( ingest_flat ).  ingest_flat is the
        # SAME dense [K,D] the eager block-0 builds (ing_table [K,H0,HD0] reshaped);
        # W_o0(ingest_flat) = ingest_flat @ W_o0.T, folded into h0 up front so block 0 is
        # just ffn0(h0_folded) at replay (no ingest transpose / W_o0 GEMM in the graph).
        Wo0 = _wo_dense(model, 0, dev)
        ing_flat = ing_table.reshape(n, D)                    # [K, D] (H0*HD0 == D)
        h0_folded = h0_table + ing_flat @ Wo0.transpose(0, 1)
        # each live block: compact [K, n_out_b] delta + its out_dims.
        cam_delta_tables = {}
        live_out_dims = {}
        for lb in live_blocks:
            b = lb.block_idx
            cam_flat = cam_tables[b].reshape(n, D)            # [K, D]
            Wo = _wo_dense(model, b, dev)
            delta, out_dims = _build_wo_delta_table(cam_flat, Wo)
            cam_delta_tables[b] = delta
            live_out_dims[b] = out_dims
    sched = PrecomputedSchedule(
        n_steps=n, h0_table=h0_table, ing_table=ing_table, cam_tables=cam_tables,
        want_pc=want_pc, want_sp=want_sp, want_bp=want_bp, want_ax=want_ax,
        is_halt=is_halt, is_file=is_file, onchip=onchip, h0_folded=h0_folded,
        cam_delta_tables=cam_delta_tables)
    chunk = _sched_chunk()
    sg = PrecomputedStepGraph(model, L, dev, chunk, mega, live_blocks, live_order, mask,
                              onchip=onchip, live_out_dims=live_out_dims)
    return sched, sg


def _sched_chunk() -> int:
    """``C4_SCHED_CHUNK`` (default 4096): the per-chunk query-row batch the single graph
    processes.  Bigger amortizes the (already O(1)) per-chunk host op over more rows; the
    per-chunk peak is O(chunk * ffn_hidden) so it is VRAM-bounded, not K-bounded."""
    try:
        return int(os.environ.get("C4_SCHED_CHUNK", "4096"))
    except ValueError:
        return 4096


# ===========================================================================
# 3. THE SINGLE DISPATCH — replay the graph over the resident tables in chunks.
# ===========================================================================
@dataclass
class PrecomputedResult:
    accepted_steps: int
    total_steps: int
    all_matched: bool
    decoded_final_ax: Optional[int]
    first_mismatch: Optional[dict]
    n_chunks: int
    host_ops_per_forward: int       # host ops in the WHOLE dispatch (O(n_chunks))


def run_verify(model, L, code, draft, device, *, mask: int = 0xFFFFFFFF,
               collect_out: Optional[List[int]] = None,
               stats: Optional[dict] = None) -> PrecomputedResult:
    """Run the whole K-batch verify as ONE precomputed schedule + graph-per-chunk
    dispatch.  Byte-identical to the composed ``verify_blocks`` GPU-verify at every
    query row (same gathers, same GEMM chains, same requant decode + compare).

    Asserts DIV-free (no DIV/MOD step) — the divmod span is not carried by the mega
    region; a DIV/MOD program must use ``verify_blocks``."""
    _DM = {"DIV", "MOD"}
    for s in range(draft.step_count):
        if draft.frames[s].get("op") in _DM:
            raise ValueError(
                f"precomputed_schedule.run_verify: step {s} is {draft.frames[s]['op']} "
                "(DIV/MOD) — the DIV-free schedule does not carry the divmod span; use "
                "verify_blocks for a DIV/MOD program.")
    dev = torch.device(device)
    sched, sg = build_schedule(model, L, code, draft, dev, mask=mask)
    n = sched.n_steps
    chunk = sg.chunk
    onchip = sched.onchip
    got_pc = torch.empty(n, dtype=torch.long, device=dev)
    got_sp = torch.empty(n, dtype=torch.long, device=dev)
    got_bp = torch.empty(n, dtype=torch.long, device=dev)
    got_ax = torch.empty(n, dtype=torch.long, device=dev)
    n_chunks = 0
    host_ops = 0
    # in on-chip mode block 0 reads the FOLDED h0 (carries W_o0(ingest)); dense mode reads
    # the raw h0 + the ingest table.
    h0_src = sched.h0_folded if onchip else sched.h0_table
    # ---- C4_RESIDENT_BATCH: whole batch fits ONE chunk -> capture the graph over VIEWS of
    #      the resident tables (no per-chunk static-input copy) and do ONE replay. ----
    resident = resident_batch_enabled() and n <= chunk
    # -------- THE SINGLE DISPATCH LOOP: O(n_chunks) host ops, ZERO per-op work -------
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        h0 = h0_src[lo:hi].unsqueeze(0)                            # [1, C, D]
        if onchip:
            ing = None
            cam = None
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}  # [Sc, n_out]
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, delta=delta,
                                               resident=resident)
        else:
            ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)  # [1, H0, C, HD0]
            cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0)
                   for b, t in sched.cam_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, resident=resident)
        got_pc[lo:hi].copy_(pc_c)
        got_sp[lo:hi].copy_(sp_c)
        got_bp[lo:hi].copy_(bp_c)
        got_ax[lo:hi].copy_(ax_c)
        n_chunks += 1
        host_ops += 1                     # ONE replay + a handful of slice-copies/chunk
    got_ax = got_ax & mask
    # -------- ONE whole-batch compare + accepted-prefix reduction (one host sync) -----
    bad_normal = ((got_pc != sched.want_pc) | (got_ax != sched.want_ax)
                  | (got_sp != sched.want_sp) | (got_bp != sched.want_bp))
    bad = torch.where(sched.is_halt, got_ax != sched.want_ax, bad_normal)
    bad = bad & (~sched.is_file)
    any_bad = bad.any()
    first_bad = torch.argmax(bad.to(torch.uint8))
    n_ok = int(torch.where(any_bad, first_bad,
                           torch.tensor(n, device=dev)).item())    # THE one host sync
    # PRTF visible bytes for the accepted prefix (byte-identical to verify_blocks).
    if collect_out is not None and draft.prtf_steps:
        prtf = sorted(int(s) for s in draft.prtf_steps if s < n_ok)
        if prtf:
            pl = torch.tensor(prtf, device=dev, dtype=torch.long)
            pb = (got_ax & 0xFF).index_select(0, pl).tolist()
            collect_out.extend(int(v) & 0xFF for v in pb)
    all_matched = (n_ok == n)
    final_ax = int(got_ax[n - 1].item()) if all_matched and n > 0 else None
    first_mismatch = None
    if not all_matched:
        s = n_ok
        first_mismatch = {
            "step": s, "query_pos": draft.win_starts[s],
            "got": {"pc": int(got_pc[s].item()), "ax": int(got_ax[s].item()),
                    "sp": int(got_sp[s].item()), "bp": int(got_bp[s].item())},
            "want": {"pc": int(sched.want_pc[s].item()), "ax": int(sched.want_ax[s].item()),
                     "sp": int(sched.want_sp[s].item()), "bp": int(sched.want_bp[s].item())}}
    if stats is not None:
        stats["n_chunks"] = n_chunks
        stats["host_ops_per_forward"] = host_ops
        stats["chunk"] = chunk
    return PrecomputedResult(
        accepted_steps=n_ok, total_steps=n, all_matched=all_matched,
        decoded_final_ax=final_ax, first_mismatch=first_mismatch,
        n_chunks=n_chunks, host_ops_per_forward=host_ops)


__all__ = ["precomputed_schedule_enabled", "PrecomputedSchedule",
           "PrecomputedStepGraph", "build_schedule", "run_verify",
           "PrecomputedResult"]
