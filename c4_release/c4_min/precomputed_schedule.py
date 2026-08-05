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
    whole-step map (block chain + decode) captures into ONE graph at big chunk.

    WIDE-PC / WIDTH32 (``C4_PC_WIDE`` / ``C4_VM_WIDTH32`` / ``C4_SP_WIDE``, e.g. doom's
    547k-instruction program whose PC / return-PC / below-init SP exceed VALVOCAB≈65792):
    mirror ``_snap_lane_batch``'s wide branch EXACTLY — the round-free ``floor(x+½)`` snap
    reduced to the unsigned 32-bit word ``& 0xFFFFFFFF`` (NO VALVOCAB clamp, which would
    truncate a wide PC to 65791).  Bit-identical to ``_snap_lane_batch`` per element in
    BOTH regimes; the flags are default-OFF so the golden (narrow-PC) path is unchanged."""
    from .nibble_pure_forward import VALVOCAB
    from .nibble_vm import vm_width32, _pc_wide_enabled
    x = lanes.to(torch.float64)
    iv = torch.where(x >= 0, torch.floor(x + 0.5), -torch.floor(-x + 0.5)).to(torch.long)
    if vm_width32() or _pc_wide_enabled():
        # wide lane: unsigned 32-bit word, no VALVOCAB clamp (matches _snap_lane_batch).
        return iv & 0xFFFFFFFF
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
        # FAITHFUL SINGLE-DISPATCH (C4_FAITHFUL_SINGLE_DISPATCH): also decode the
        # model's OWN query address at each live CAM block IN the same graph (one
        # extra W_q(x) + sign-decode per live block, near-zero marginal — the block's
        # W_q(x) would otherwise be discarded).  The verify then rides on the one fast
        # dispatch pass (no second megachain forward).  ``_faithful_heads`` : list of
        # (block_idx, head_idx, kind, n_bits) to decode; ``_s_qaddr`` : per-key output
        # lane [C] long.  OFF -> the plain fast graph (byte-identical).
        from .faithful_single_dispatch import faithful_single_dispatch_enabled
        self._faithful = bool(faithful_single_dispatch_enabled())
        self._faithful_heads = []
        self._s_qaddr = {}
        if self._faithful:
            from .direct_cam_batched import _n_addr_bits
            for b, lb in self.live_blocks.items():
                for (hh, knd) in lb.cam_heads:
                    self._faithful_heads.append((b, int(hh), knd, _n_addr_bits(knd)))

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
                # FAITHFUL: decode the model's OWN query address at THIS block's genuine
                # input residual (before the draft cam_out inject overwrites it), IN-graph.
                if self._faithful:
                    self._decode_qaddr_at(b, lb, h)
                if self.onchip:
                    h = lb.forward_static_delta(h, self._s_delta[b],
                                                self.live_out_dims[b])
                else:
                    h = lb.forward_static(h, self._s_cam[b])
        return h

    def _decode_qaddr_at(self, b, lb, h):
        """In-graph sign-decode of the model's queried address at live block ``b`` from its
        genuine input residual ``h`` (``sign(W_q(x))`` per address bit == the model's
        binary-address key).  Writes each CAM head's decoded address into ``_s_qaddr``."""
        Bx, Cx, Dx = h.shape
        Q = lb.attn.W_q.linear(h).view(Bx, Cx, lb.H, lb.HD)
        for (hh, knd) in lb.cam_heads:
            n_bits = self._n_bits_of.get((b, hh), None)
            if n_bits is None:
                continue
            Qh = Q[0, :, hh, :n_bits]                       # [C, n_bits]
            bits = (Qh > 0).to(torch.long)
            w = (1 << torch.arange(n_bits, device=h.device, dtype=torch.long))
            self._s_qaddr[(b, hh, knd)].copy_((bits * w.unsqueeze(0)).sum(dim=1))

    def decode_model_query_addrs(self, h0, ing, cam, delta=None):
        """FAITHFUL VERIFY (C4_FAITHFUL_SINGLE_DISPATCH): run the SAME fused per-row map
        EAGERLY (not graphed) and, at each live CAM block's INPUT residual, compute the
        model's OWN query ``Q = W_q.linear(h)`` and sign-decode its queried address per
        CAM head.  Returns ``{(block_idx, head_idx): LongTensor[C]}`` -- the model's
        decoded query address at every query row for that head (0 when the query band is
        not asserted in this residual view).

        This is the model's GENUINE computation feeding the independent routing/address/
        value verify: the same block-0 + mega dead-FFN chain + prior-live-block residual
        the fast graph runs, byte-identically (same kernels, same order), but stopping to
        read ``W_q(x)`` before each live CAM block overwrites the head-value slot.  The
        sign decode is exactly ``direct_cam_batched._decode_model_query_addr`` (the query
        for bit b is ``Q[.., head, b] = smag*(2*bit-1)``)."""
        from .direct_cam_batched import _decode_model_query_addr, _n_addr_bits
        # block 0 (identical to _body).
        if self.onchip:
            h = self.ffn0.forward(h0)
        else:
            B, C, D = h0.shape
            out2 = ing.transpose(1, 2).contiguous().view(1, C, D)
            h = h0 + self.attn0.W_o.linear(out2)
            h = self.ffn0.forward(h)
        out = {}
        for kind, payload in self.mega.items:
            if kind == "mega":
                h = payload.run(h)                       # eager kernels (same as graph)
            else:  # 'live' block index
                b = payload
                lb = self.live_blocks[b]
                attn = lb.attn
                # the model's OWN query at THIS block's genuine input residual.
                Bx, Cx, Dx = h.shape
                Q = attn.W_q.linear(h).view(Bx, Cx, lb.H, lb.HD)   # [1,C,H,HD]
                for (hh, knd) in lb.cam_heads:
                    n_bits = _n_addr_bits(knd)
                    Qh = Q[0, :, hh, :]                            # [C, HD]
                    out[(b, hh, knd)] = _decode_model_query_addr(Qh, n_bits)
                # advance the residual exactly as the graph does (inject the draft gather
                # -- byte-identical to _run_region; the verify already read Q pre-inject).
                if self.onchip:
                    h = lb.forward_static_delta(h, delta[b], self.live_out_dims[b])
                else:
                    h = lb.forward_static(h, cam[b])
        return out

    def _alloc_out_lanes(self, C):
        self._s_pc = torch.zeros(C, dtype=torch.long, device=self.device)
        self._s_sp = torch.zeros(C, dtype=torch.long, device=self.device)
        self._s_bp = torch.zeros(C, dtype=torch.long, device=self.device)
        self._s_ax = torch.zeros(C, dtype=torch.long, device=self.device)
        if self._faithful:
            self._n_bits_of = {(b, hh): nb for (b, hh, knd, nb) in self._faithful_heads}
            self._s_qaddr = {(b, hh, knd): torch.zeros(C, dtype=torch.long,
                                                       device=self.device)
                             for (b, hh, knd, nb) in self._faithful_heads}

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

    def last_qaddr(self, Sc):
        """FAITHFUL: the model's decoded query addresses from the LAST replay's in-graph
        sign-decode.  ``{(block, head, kind): LongTensor[Sc]}`` (empty when not faithful).
        Valid only until the next replay overwrites the lanes (the caller copies out)."""
        if not self._faithful:
            return {}
        return {k: v[:Sc] for k, v in self._s_qaddr.items()}


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


def _draft_tokens_np(draft):
    """``np.asarray(draft.tokens)`` — the ONE full ~30*n-token stream conversion (a real
    O(tokens) one-time cost the whole-frame giant-K exposes).  Cached on the draft so the
    query-embed + ingest-table + frame-decode builds SHARE it (three consumers, one
    conversion) under C4_SCHED_GPU_BUILD."""
    import numpy as _np
    tk = getattr(draft, "_tokens_np_cache", None)
    if tk is None:
        tk = _np.asarray(draft.tokens, dtype=_np.int64)
        try:
            draft._tokens_np_cache = tk
        except Exception:
            pass
    return tk


def _build_query_embed(model, L, code, draft, device, toks_np=None) -> torch.Tensor:
    """The per-step query-row block-0 input ``[K, D]``: ``embed[qtok] + query_overlay``.

    A step's query row token is ``draft.tokens[win_starts[step]]`` (the REG_PC role
    token that opens the step's own frame); the query overlay is the row-invariant
    code_vec (``ONE`` + baked program-in-data, ``build_code_vec``) ASSIGNED onto the
    code dims, plus the all-ROLE one-hot query tag (``L.ROLE + role = 1`` for every
    role).  This is EXACTLY what ``apply_overlay_window_fast`` writes on a query row
    (code_vec assign + all-ROLE tag) on top of ``model.embed[qtok]`` — byte-identical."""
    from .pf_speculative import build_code_vec
    import numpy as _np
    toks = draft.tokens
    ws = draft.win_starts
    n = draft.step_count
    if toks_np is not None:
        # C4_SCHED_GPU_BUILD: gather qtok from the shared token array via a vectorized
        # index (win_starts) — no per-step Python generator.  Byte-identical.
        ws_np = _np.asarray(ws[:n], dtype=_np.int64)
        qtok = torch.from_numpy(toks_np[ws_np]).to(device)
    else:
        # the query-row token per step (toks[win_starts[s]]) via ``np.fromiter`` (n reads,
        # no full 30*n-token list conversion), then ONE host->device transfer — no slow
        # torch.tensor(python-list) at K.
        qtok = torch.from_numpy(
            _np.fromiter((toks[ws[s]] for s in range(n)), dtype=_np.int64, count=n)).to(device)
    h0 = model.embed[qtok].clone()                 # [K, D]  fresh copy
    # code_vec: ONE broadcast assign over the code dims (same idx/vals every row).
    code_idx, code_vals = build_code_vec(code, L, model.embed.shape[1], device,
                                         dtype=model.embed.dtype)
    h0[:, code_idx] = code_vals.to(h0.dtype)
    # all-ROLE query tag: L.ROLE + role = 1 for role in 0..N_ROLES-1, every query row.
    role_dims = torch.arange(N_ROLES, device=device, dtype=torch.long) + L.ROLE
    h0[:, role_dims] = 1.0
    return h0


def _resolved_frame_nibs_vec(draft, toks_np, device):
    """The vectorized frame-decode inner of ``build_resolved_frames`` WITHOUT the
    ``ResolvedFrames`` object's O(n) ``pos_to_local`` Python dict (unused by the schedule
    build).  Returns ``(nib_lo, nib_hi)`` [K, N_ROLES] on ``device``.  Shares the token
    array ``toks_np`` (C4_SCHED_GPU_BUILD).  Byte-identical to ``build_resolved_frames``'s
    nibbles (same fstart / REG_PC guard / 0..255 clamp)."""
    from .direct_local_cam import _ROLE_TO_LOCAL
    import numpy as _np
    FL = V.FRAME_LEN
    NR = len(_ROLE_TO_LOCAL)
    n = draft.step_count
    local_of_role = _np.asarray([_ROLE_TO_LOCAL[r] for r in range(NR)], dtype=_np.int64)
    T = toks_np.shape[0]
    if n == 0:
        z = torch.zeros(0, NR, dtype=torch.float32, device=device)
        return z, z.clone()
    pos = _np.asarray(draft.win_starts[:n], dtype=_np.int64)          # [n]
    fstart = pos - (FL - 1)
    valid = fstart >= 0
    fs_c = _np.clip(fstart, 0, None)
    head_tok = toks_np[_np.clip(fs_c, 0, T - 1)]
    valid = valid & (head_tok == V.REG_PC)
    gather_idx = fs_c[:, None] + local_of_role[None, :]              # [n, NR]
    gather_idx = _np.clip(gather_idx, 0, T - 1)
    B = toks_np[gather_idx]                                          # [n, NR]
    B = _np.where((B >= 0) & (B <= 255), B, 0)
    B = _np.where(valid[:, None], B, 0)
    nib_lo = (B & 0xF).astype(_np.float32)
    nib_hi = ((B >> 4) & 0xF).astype(_np.float32)
    return (torch.from_numpy(nib_lo).to(device), torch.from_numpy(nib_hi).to(device))


def _build_ingest_table(model, L, draft, device, toks_np=None) -> torch.Tensor:
    """The block-0 ingest gather, per step, in head-value space ``[K, H0, HD0]``.

    The ingest head ``h`` writes ``[nib_lo[h], nib_hi[h], 0, ...]`` into its head-value
    slots 0/1 (``bake_frame_ingest`` -> direct-local ``_gather_out_chunk`` scatter).  We
    build the SAME ``[K, H0, HD0]`` table the eager scatter produces, from the
    ``ResolvedFrames`` (byte-identical: same nib_lo/nib_hi, same head==role mapping)."""
    from .direct_local_cam import build_resolved_frames, ingest_head_map
    attn0 = model.blocks[0].attn
    head_map = ingest_head_map(attn0)              # {head: (r,bi)}; head==role index
    ing_heads = sorted(head_map)
    if toks_np is not None:
        # C4_SCHED_GPU_BUILD: skip the ResolvedFrames pos_to_local O(n) dict (unused here)
        # + share the token array — byte-identical nibbles.
        nib_lo, nib_hi = _resolved_frame_nibs_vec(draft, toks_np, device)
    else:
        rf = build_resolved_frames(draft, device=device)   # nib_lo/nib_hi [K, N_ROLES]
        nib_lo, nib_hi = rf.nib_lo, rf.nib_hi
    K = nib_lo.shape[0]
    H0, HD0 = attn0.n_heads, attn0.head_dim
    tab = torch.zeros(K, H0, HD0, device=device)
    heads_t = torch.tensor(ing_heads, device=device, dtype=torch.long)
    # role index == head index: gather nib_lo/hi[:, head] into slot 0/1.
    tab[:, heads_t, 0] = nib_lo[:, heads_t]
    tab[:, heads_t, 1] = nib_hi[:, heads_t]
    return tab


def _sched_fast_build() -> bool:
    """``C4_SCHED_FAST_BUILD`` (DEFAULT ON): build the per-step CAM gather tables with a
    VECTORIZED numpy nibble-scatter (one dense buffer per (block, head), scattered at the
    resolved step rows) instead of the per-step Python ``for s in range(n): _head_out_vec``
    loop.  At 358 K steps the old loop is ~1.4 M ``torch.zeros(HD)`` + per-nibble Python
    calls — the O(n_steps) one-time build cost the whole-frame giant-K exposes.  The
    scatter is byte-identical (the SAME nibbles at the SAME value slots ``_head_out_vec``
    writes).  OFF -> the original per-step loop (kept as the byte-exact cross-check)."""
    return os.environ.get("C4_SCHED_FAST_BUILD", "1") not in ("0", "", "false", "False")


def _sched_cache_resolved() -> bool:
    """``C4_SCHED_CACHE_RESOLVED`` (DEFAULT OFF).  BUILD-REDUCTION for the CONTINUOUS render:
    the query-embed gather + ingest-nibble decode + CAM-sparse latest-write-wins resolution
    are pure functions of the IMMUTABLE draft, so cache the RESOLVED numpy/torch arrays ONCE
    (on the draft, keyed by device/mask) and COPY them per frame instead of re-resolving on
    every ``build_schedule_tables_only``.  The profiler pins that resolution (cam_sparse 56% +
    ingest 16% at doom n=120K) as the dominant per-frame BUILD cost, so caching turns the
    build into a copy + the GPU W_o-delta GEMM (the dispatch-bound residual).  Byte-identical
    (same arrays); DEFAULT OFF -> the per-frame re-resolve.  This is the composed stack's
    build-bound lever (build 2.9 > dispatch 1.9 us/step on real doom)."""
    return os.environ.get("C4_SCHED_CACHE_RESOLVED", "0") not in ("0", "", "false", "False")


def _sched_gpu_build() -> bool:
    """``C4_SCHED_GPU_BUILD`` (DEFAULT OFF).  The DEEP build vectorization: collapse the
    three-stage Python-dict CAM pipeline (``resolve_load_rows`` -> ``build_resolved_table``
    -> ``_build_cam_sparse``) AND the per-step ``decode_targets`` frame loop into ONE
    vectorized numpy pass over flat arrays, plus a torch device gather for the value-nibble
    scatter.  Profiled (n=364 K): that three-stage chain is ~70% of the schedule build
    (cam_sparse 32% + resolve_load_rows 20% + build_resolved_table 18%) — each stage
    builds/iterates a Python dict keyed by absolute position, all O(reads+stores+steps)
    CPython.  The vectorized form does the SAME latest-write-wins resolution as a stable
    argsort + per-address ``searchsorted`` (no per-read ``ResolvedRead`` dataclass, no
    position-keyed dict), maps read frame -> step -> query row by a single vectorized
    ``searchsorted``, and scatters the resolved value nibbles into the compact
    ``vals[K, n_active]`` arrays with vectorized bit-shifts — byte-identical to the
    per-read scatter ``_build_cam_sparse`` does (same active columns, same nibbles, same
    step rows).  OFF -> the Python-dict path (``_build_cam_sparse`` / the decode loop),
    the byte-exact cross-check.  Requires ``C4_SCHED_FAST_BUILD`` + ``C4_ONCHIP_RESIDUAL``
    (the compact-sparse -> W_o-delta on-chip path)."""
    return os.environ.get("C4_SCHED_GPU_BUILD", "0") not in ("0", "", "false", "False")


def _draft_read_store_arrays(draft):
    """Flatten ``draft.read_log`` / ``store_log`` into numpy arrays ONCE (the only
    O(reads+stores) Python touch), returning
      reads:  (rframe[R], rhead_code[R], raddr[R])   head_code: mem=0 pop=1 lev=2 uni=3
      stores: (sframe[S], saddr[S], sval[S])         (ascending sframe)
    ``rhead_code`` lets the resolver process all heads together; the caller filters per
    live-block head kind.  Mirrors ``resolve_load_rows``' inputs exactly."""
    import numpy as _np
    _HC = {"mem": 0, "pop": 1, "lev": 2, "uni": 3}
    read_log = draft.read_log or {}
    store_log = draft.store_log or {}
    # reads (order within a frame preserved; frames in ascending order for determinism).
    rf_l, rh_l, ra_l = [], [], []
    for f in sorted(read_log):
        for (head, addr) in read_log[f]:
            hc = _HC.get(head)
            if hc is None:
                continue
            rf_l.append(f); rh_l.append(hc); ra_l.append(addr & 0xFFFFFFFF)
    rframe = _np.asarray(rf_l, dtype=_np.int64)
    rhead = _np.asarray(rh_l, dtype=_np.int64)
    raddr = _np.asarray(ra_l, dtype=_np.int64)
    sf_l = sorted(store_log)
    sframe = _np.asarray(sf_l, dtype=_np.int64)
    saddr = _np.fromiter((store_log[f][0] & 0xFFFFFFFF for f in sf_l),
                         dtype=_np.int64, count=len(sf_l))
    sval = _np.fromiter((store_log[f][1] & 0xFFFFFFFF for f in sf_l),
                        dtype=_np.int64, count=len(sf_l))
    return (rframe, rhead, raddr), (sframe, saddr, sval)


def _resolve_reads_vec(reads, stores):
    """VECTORIZED latest-write-wins: for each read ``(rframe, raddr)`` return the value of
    the store to the SAME address with the LARGEST store frame ``< rframe`` (0 if none —
    the softmax1 +1-sink ZFOD).  Byte-identical to ``resolve_load_rows``' per-read
    ``latest.get(addr)`` walk (a store at frame S is visible to a read at frame > S; the
    VM never has a store and a read at the SAME frame resolving against each other — SI
    reads then stores at DISTINCT frames, and the store is committed after the read).

    Method: group stores by address via a stable argsort on ``(saddr, sframe)``; within
    each address group the store frames are ascending, so a read's winner is the store at
    ``searchsorted(group_frames, rframe, 'left') - 1`` (the last store frame strictly
    ``< rframe``).  A single vectorized pass, no per-read Python."""
    import numpy as _np
    rframe, rhead, raddr = reads
    sframe, saddr, sval = stores
    R = rframe.shape[0]
    val = _np.zeros(R, dtype=_np.int64)
    if R == 0 or sframe.shape[0] == 0:
        return val
    # stable sort stores by (addr, frame); frames within an addr group are then ascending.
    order = _np.lexsort((sframe, saddr))          # primary saddr, secondary sframe
    s_addr_s = saddr[order]
    s_frame_s = sframe[order]
    s_val_s = sval[order]
    # group boundaries per distinct address in the sorted store array.
    uniq_addr, grp_start = _np.unique(s_addr_s, return_index=True)
    grp_end = _np.empty_like(grp_start)
    grp_end[:-1] = grp_start[1:]
    grp_end[-1] = s_addr_s.shape[0]
    # map each read address to its store group (or -1 if the address was never stored).
    gi = _np.searchsorted(uniq_addr, raddr)
    in_range = gi < uniq_addr.shape[0]
    matched = _np.zeros(R, dtype=_np.bool_)
    matched[in_range] = uniq_addr[gi[in_range]] == raddr[in_range]
    ridx = _np.nonzero(matched)[0]
    if ridx.shape[0] == 0:
        return val
    g = gi[ridx]                                  # group index per matched read
    gs = grp_start[g]; ge = grp_end[g]
    # within [gs, ge) the frames s_frame_s are ascending; find the last frame < rframe.
    # searchsorted over the WHOLE sorted-frame array restricted to the group window:
    # local position = searchsorted(s_frame_s[gs:ge], rframe, 'left') - 1.
    rfr = rframe[ridx]
    # vectorized per-group searchsorted via a global searchsorted on a group-shifted key:
    # build a monotone global key = group*BIG + frame so a single searchsorted respects
    # both the group and the ascending frame order.
    BIG = int(s_frame_s.max()) + int(rfr.max()) + 2 if s_frame_s.size else 1
    # per-store group id (== group index of each sorted store), then a globally-ascending
    # key = group*BIG + frame so a single searchsorted respects both group and frame order.
    store_group = _np.searchsorted(uniq_addr, s_addr_s)
    store_key = store_group * BIG + s_frame_s
    read_key = g * BIG + rfr
    loc = _np.searchsorted(store_key, read_key, side="left") - 1
    # a valid winner must fall inside the read's own group window [gs, ge).
    ok = loc >= gs
    good = ridx[ok]
    val[good] = s_val_s[loc[ok]]
    return val


def _build_cam_sparse_gpu(draft, code, live_blocks, device):
    """C4_SCHED_GPU_BUILD: the vectorized CAM-sparse build.  Produces the SAME
    ``{block: (vals[K,nc] float32, active_cols[nc] int64, HDb)}`` compact form
    ``_build_cam_sparse`` does, but resolves the reads with ``_resolve_reads_vec`` and
    scatters value nibbles with vectorized bit-shifts (no per-position Python dict, no
    ``np.fromiter`` over dict generators).  ``vals`` are returned on the CPU (numpy) —
    the W_o-delta step reads them to device; the heavy per-step resolution is what moved
    to array ops.  Byte-identical to ``_build_cam_sparse``."""
    import numpy as _np
    from .direct_cam_batched import CODE_ADDR_BITS, ADDR_BITS, _n_seed
    from .nibble_pure_forward_complete import IMM_NIBS, _pf_cfm_enabled
    from .blogspec_layout import NIB_PER_REG

    _HC = {"mem": 0, "pop": 1, "lev": 2, "uni": 3}
    n = draft.step_count
    ws = _np.asarray(draft.win_starts[:n], dtype=_np.int64)          # [n] query-row abs pos

    # ---- read/store arrays + vectorized latest-write-wins ----
    reads, stores = _draft_read_store_arrays(draft)
    rframe, rhead, raddr = reads
    rval = _resolve_reads_vec(reads, stores)                         # [R] resolved value

    # ---- map each read frame -> its STEP (the query row it decodes) ----
    # replay the frame counter EXACTLY as _frame_to_step: frame_idx starts at n_seed,
    # advances +1 per step (primary frame) then + n_byte_stores for a file step.  The
    # per-step primary frame index is thus a cumulative sum; build it vectorized.
    n_seed = _n_seed(draft)
    frames = draft.frames
    # n_byte_stores per step (0 unless a file step); tiny Python touch (one attr per step).
    nbs = _np.zeros(n, dtype=_np.int64)
    # only file steps carry n_byte_stores; scan is unavoidable but a single cheap pass.
    for s in range(n):
        f = frames[s]
        if f.get("is_file"):
            nbs[s] = int(f.get("n_byte_stores", 0) or 0)
    # primary frame index of step s = n_seed + 1 + s + sum(nbs[:s])
    prefix = _np.zeros(n, dtype=_np.int64)
    if n > 1:
        prefix[1:] = _np.cumsum(nbs[:-1])
    primary_frame = n_seed + 1 + _np.arange(n, dtype=_np.int64) + prefix   # [n]
    # invert: read frame -> step via searchsorted (primary_frame is strictly increasing).
    # a read frame that is NOT a primary frame (a store frame) maps to no step -> drop.
    ridx = _np.searchsorted(primary_frame, rframe)
    okr = (ridx < n)
    okr[okr] &= (primary_frame[ridx[okr]] == rframe[okr])
    step_of_read = _np.where(okr, ridx.clip(0, n - 1), -1)           # step index or -1

    # ---- CODE FETCH@PC: per-step (op, imm) resolved from the pre-step PC ----
    code_op = None; code_imm = None
    if _pf_cfm_enabled():
        # pre-step PC per step: pc0=0, pc[s] = frames[s-1]["pc"].  One cheap Python pass to
        # pull the post-step pc array (frames are dicts), then shift.
        post_pc = _np.fromiter((int(frames[s]["pc"]) for s in range(n)),
                               dtype=_np.int64, count=n)
        pre_pc = _np.empty(n, dtype=_np.int64)
        pre_pc[0] = 0
        if n > 1:
            pre_pc[1:] = post_pc[:-1]
        in_code = (pre_pc >= 0) & (pre_pc < len(code))
        # code ops/imms as arrays (len(code) is tiny vs n).
        op_arr = _np.asarray([int(ins.op) for ins in code], dtype=_np.int64)
        imm_arr = _np.asarray([int(ins.imm) & 0xFFFFFFFF for ins in code], dtype=_np.int64)
        code_op = _np.where(in_code, op_arr[pre_pc.clip(0, len(code) - 1)], -1)
        code_imm = _np.where(in_code, imm_arr[pre_pc.clip(0, len(code) - 1)], 0)

    # ---- scatter into the compact per-block vals[K, n_active] ----
    out = {}
    for lb in live_blocks:
        Hb, HDb = lb.H, lb.HD
        col_of = {}; col_list = []
        for (h, kind) in lb.cam_heads:
            if kind == "code":
                v0 = CODE_ADDR_BITS + 4
                slots = [v0] + [v0 + 1 + j for j in range(IMM_NIBS)]
            else:
                b0 = ADDR_BITS + 3
                slots = [b0 + j for j in range(NIB_PER_REG)]
            for sl in slots:
                fc = h * HDb + sl
                if fc not in col_of:
                    col_of[fc] = len(col_list); col_list.append(fc)
        nc = len(col_list)
        vals = _np.zeros((n, nc), dtype=_np.float32)
        for (h, kind) in lb.cam_heads:
            if kind == "code":
                if code_op is None:
                    continue
                have = code_op >= 0                     # steps with an in-range fetch
                rows = _np.nonzero(have)[0]
                if rows.size == 0:
                    continue
                ops = code_op[rows]; imms = code_imm[rows]
                v0 = CODE_ADDR_BITS + 4
                vals[rows, col_of[h * HDb + v0]] = ops.astype(_np.float32)
                for j in range(IMM_NIBS):
                    vals[rows, col_of[h * HDb + v0 + 1 + j]] = \
                        ((imms >> (4 * j)) & 0xF).astype(_np.float32)
            else:
                hc = _HC[kind]
                # this head's reads: matching head code AND a valid step.
                sel = (rhead == hc) & okr
                if not sel.any():
                    continue
                rows = step_of_read[sel]
                vv = rval[sel]
                b0 = ADDR_BITS + 3
                for j in range(NIB_PER_REG):
                    vals[rows, col_of[h * HDb + b0 + j]] = \
                        ((vv >> (4 * j)) & 0xF).astype(_np.float32)
        out[lb.block_idx] = (vals, _np.asarray(col_list, dtype=_np.int64), HDb)
    return out


def _decode_targets_gpu(draft, mask, device):
    """C4_SCHED_GPU_BUILD: vectorized decode-target extraction.  The frames are Python
    dicts, so a SINGLE pass over ``frames`` pulls all six fields (pc/ax/sp/bp/is_halt/
    is_file) into preallocated numpy arrays at once — iterating the Python dict list ONCE
    (vs six separate ``np.fromiter`` generators, 6x the per-dict Python overhead) — then ONE
    host->device transfer per lane.  Byte-identical to the per-step loop in
    ``build_schedule`` (same fields, same masks)."""
    import numpy as _np
    n = draft.step_count
    frames = draft.frames
    _pc = _np.empty(n, dtype=_np.int64); _ax = _np.empty(n, dtype=_np.int64)
    _sp = _np.empty(n, dtype=_np.int64); _bp = _np.empty(n, dtype=_np.int64)
    _hl = _np.empty(n, dtype=_np.bool_); _fl = _np.empty(n, dtype=_np.bool_)
    for s in range(n):
        f = frames[s]
        _pc[s] = f["pc"]; _ax[s] = f["ax"] & mask
        _sp[s] = f["sp"] & 0xFFFFFFFF; _bp[s] = f["bp"] & 0xFFFFFFFF
        _hl[s] = f.get("is_halt") or False; _fl[s] = f.get("is_file") or False
    dev = torch.device(device)
    return (torch.from_numpy(_pc).to(dev), torch.from_numpy(_ax).to(dev),
            torch.from_numpy(_sp).to(dev), torch.from_numpy(_bp).to(dev),
            torch.from_numpy(_hl).to(dev), torch.from_numpy(_fl).to(dev))


def _build_cam_tables_loop(model, L, draft, code, live_blocks, device
                           ) -> Dict[int, torch.Tensor]:
    """The ORIGINAL per-step Python loop (kept as the byte-exact reference for the
    vectorized builder).  Resolves, per step, each CAM head's exact ``_head_out_vec``."""
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


def _build_cam_sparse(draft, code, live_blocks):
    """Build the live-CAM gather as a COMPACT SPARSE representation (no dense
    ``[K, Hb, HDb]`` tensor): for each block a numpy ``vals[K, n_active_cols]`` of the
    active head-value NIBBLES + the ``active_cols`` flat-slot indices (flat = head*HDb +
    slot) they map to.  The dense ``[K, Hb, HDb]`` cam_out is exactly ``scatter(vals into
    active_cols)`` — 99.7% zeros — so the compact form carries the SAME information at ~50x
    less memory (the ~16 nibble slots per active head, not 24*59 dims).  Vectorized numpy
    scatter (``C4_SCHED_FAST_BUILD``).  Returns ``{block: (vals[K,nc] np.float32, active_cols
    [nc] np.int64, HDb)}``."""
    import numpy as _np
    from .direct_cam_batched import build_resolved_table, CODE_ADDR_BITS, ADDR_BITS
    from .nibble_pure_forward_complete import IMM_NIBS
    from .blogspec_layout import NIB_PER_REG
    tbl = build_resolved_table(draft, code)
    n = draft.step_count
    ws = _np.asarray(draft.win_starts, dtype=_np.int64)

    def _rows_for(positions):
        if not positions:
            return None
        p = _np.fromiter(positions, dtype=_np.int64, count=len(positions))
        idx = _np.searchsorted(ws, p)
        ok = (idx < n) & (ws[idx.clip(0, n - 1)] == p)
        return idx[ok], ok

    out = {}
    for lb in live_blocks:
        Hb, HDb = lb.H, lb.HD
        # active flat columns (head*HDb + slot) + a per-column filler into a compact buffer.
        col_of = {}                             # flat_col -> compact index
        col_list = []
        # first pass: enumerate the active columns this block writes.
        for (h, kind) in lb.cam_heads:
            if kind == "code":
                v0 = CODE_ADDR_BITS + 4
                slots = [v0] + [v0 + 1 + j for j in range(IMM_NIBS)]
            else:
                b0 = ADDR_BITS + 3
                slots = [b0 + j for j in range(NIB_PER_REG)]
            for sl in slots:
                fc = h * HDb + sl
                if fc not in col_of:
                    col_of[fc] = len(col_list); col_list.append(fc)
        nc = len(col_list)
        vals = _np.zeros((n, nc), dtype=_np.float32)
        for (h, kind) in lb.cam_heads:
            if kind == "code":
                items = list(tbl.code.items())
                if not items:
                    continue
                r = _rows_for([p for p, _cv in items])
                if r is None:
                    continue
                rows, ok = r
                cv = [items[i][1] for i in range(len(items)) if ok[i]]
                ops = _np.fromiter((int(o) for (o, _im) in cv), dtype=_np.int64, count=len(cv))
                imms = _np.fromiter((int(im) & 0xFFFFFFFF for (_o, im) in cv),
                                    dtype=_np.int64, count=len(cv))
                v0 = CODE_ADDR_BITS + 4
                vals[rows, col_of[h * HDb + v0]] = ops.astype(_np.float32)
                for j in range(IMM_NIBS):
                    vals[rows, col_of[h * HDb + v0 + 1 + j]] = \
                        ((imms >> (4 * j)) & 0xF).astype(_np.float32)
            else:
                items = list(tbl.by_kind(kind).items())
                if not items:
                    continue
                r = _rows_for([p for p, _v in items])
                if r is None:
                    continue
                rows, ok = r
                vv = _np.fromiter((int(items[i][1]) & 0xFFFFFFFF
                                   for i in range(len(items)) if ok[i]),
                                  dtype=_np.int64, count=int(ok.sum()))
                b0 = ADDR_BITS + 3
                for j in range(NIB_PER_REG):
                    vals[rows, col_of[h * HDb + b0 + j]] = ((vv >> (4 * j)) & 0xF).astype(_np.float32)
        out[lb.block_idx] = (vals, _np.asarray(col_list, dtype=_np.int64), HDb)
    return out


def _build_cam_tables(model, L, draft, code, live_blocks, device
                      ) -> Dict[int, torch.Tensor]:
    """Each live CAM block's direct-gather in head-value space ``[K, Hb, HDb]``.

    VECTORIZED (``C4_SCHED_FAST_BUILD``, default ON): resolve every read of the draft ONCE
    (``build_resolved_table``), then for each (block, head) scatter the resolved value's
    nibbles into a dense numpy ``[K, HDb]`` buffer at the reading STEP rows — ONE build,
    no per-step Python loop / per-step ``torch.zeros(HD)`` alloc.  Byte-identical to the
    per-step ``_head_out_vec`` scatter (same slots, same nibbles).  This is the O(n_steps)
    build the whole-frame giant-K exposes; the numpy scatter makes it a draft-fast one-time
    cost.  OFF -> ``_build_cam_tables_loop`` (the original loop, the byte-exact reference)."""
    if not _sched_fast_build():
        return _build_cam_tables_loop(model, L, draft, code, live_blocks, device)
    import numpy as _np
    from .direct_cam_batched import build_resolved_table, CODE_ADDR_BITS, ADDR_BITS
    from .nibble_pure_forward_complete import IMM_NIBS
    from .blogspec_layout import NIB_PER_REG
    tbl = build_resolved_table(draft, code)
    n = draft.step_count
    # win_starts is strictly increasing (each step has a unique query row); build the
    # abs-position -> step-index inverse via searchsorted so a resolver keyed by abs pos
    # maps directly to step rows (vectorized, no per-step dict lookup).
    ws = _np.asarray(draft.win_starts, dtype=_np.int64)      # [n], strictly increasing

    def _rows_for(positions):
        """Map a list of absolute query-row positions to their step-row indices (the ones
        that are actually a step's own query row — always true for resolver keys)."""
        if not positions:
            return None
        p = _np.fromiter(positions, dtype=_np.int64, count=len(positions))
        idx = _np.searchsorted(ws, p)
        ok = (idx < n) & (ws[idx.clip(0, n - 1)] == p)
        return idx[ok], p[ok], ok

    out: Dict[int, torch.Tensor] = {}
    for lb in live_blocks:
        Hb, HDb = lb.H, lb.HD
        buf = _np.zeros((n, Hb, HDb), dtype=_np.float32)
        for (h, kind) in lb.cam_heads:
            if kind == "code":
                items = list(tbl.code.items())
                if not items:
                    continue
                r = _rows_for([p for p, _cv in items])
                if r is None:
                    continue
                rows, _pk, ok = r
                cv = [items[i][1] for i in range(len(items)) if ok[i]]
                ops = _np.fromiter((int(o) for (o, _im) in cv), dtype=_np.int64, count=len(cv))
                imms = _np.fromiter((int(im) & 0xFFFFFFFF for (_o, im) in cv),
                                    dtype=_np.int64, count=len(cv))
                v0 = CODE_ADDR_BITS + 4
                buf[rows, h, v0] = ops.astype(_np.float32)
                for j in range(IMM_NIBS):
                    buf[rows, h, v0 + 1 + j] = ((imms >> (4 * j)) & 0xF).astype(_np.float32)
            else:
                d = tbl.by_kind(kind)
                items = list(d.items())
                if not items:
                    continue
                r = _rows_for([p for p, _v in items])
                if r is None:
                    continue
                rows, _pk, ok = r
                vals = _np.fromiter((int(items[i][1]) & 0xFFFFFFFF
                                     for i in range(len(items)) if ok[i]),
                                    dtype=_np.int64, count=int(ok.sum()))
                b0 = ADDR_BITS + 3
                for j in range(NIB_PER_REG):
                    buf[rows, h, b0 + j] = ((vals >> (4 * j)) & 0xF).astype(_np.float32)
        out[lb.block_idx] = torch.from_numpy(buf).to(device)
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
    onchip = onchip_residual_enabled()
    sched, live_out_dims = _build_schedule_tables(model, L, code, draft, dev, live_blocks,
                                                  onchip, mask)
    chunk = _sched_chunk()
    sg = PrecomputedStepGraph(model, L, dev, chunk, mega, live_blocks, live_order, mask,
                              onchip=onchip, live_out_dims=live_out_dims)
    return sched, sg


def _build_schedule_tables(model, L, code, draft, dev, live_blocks, onchip, mask):
    """Build the O(n) whole-batch schedule TABLES (query embed + ingest + CAM gather + decode
    targets + on-chip W_o deltas).  Shared by ``build_schedule`` (first frame, also builds the
    graph) and ``build_schedule_tables_only`` (continuous frames, reuses the captured graph).
    Returns ``(PrecomputedSchedule, live_out_dims)``."""
    fast = _sched_fast_build()
    gpu_build = _sched_gpu_build()               # C4_SCHED_GPU_BUILD (deep vectorization)
    # ON-CHIP FAST BUILD: the dense ``[K, Hb, HDb]`` cam_tables (~623 MB/block, 99.7% zeros)
    # are ONLY used to compute the compact ``W_o(cam_out)`` delta, then discarded (replay
    # uses the delta).  So skip materializing them on GPU entirely — build the delta DIRECTLY
    # from the compact sparse nibble form (a ``[K, n_active_cols] @ [n_active_cols, n_out]``
    # tiny GEMM instead of a ``[K, D] @ [D, D]`` dense one on a 99.7%-zero input).  This is the
    # dominant build cost (~62% of the schedule build) at whole-frame K.  OFF (or non-onchip):
    # the dense cam_tables path (unchanged).
    # C4_SCHED_GPU_BUILD: convert the ~30*n-token stream to numpy ONCE, shared by the
    # query-embed + ingest-table + frame-decode builds (three consumers, one conversion).
    # ---- C4_SCHED_CACHE_RESOLVED: the RESOLVED-GATHER cache (build-reduction lever) ----
    # The query-embed ``qtok`` gather, the ingest-nibble frame decode, and the CAM-sparse
    # latest-write-wins resolution (``_resolve_reads_vec`` lexsort/searchsorted + the value-
    # nibble scatter) are ALL pure functions of the IMMUTABLE draft — the profiler pins them
    # as ~73% of the doom per-frame build (cam_sparse 56% + ingest 16% at n=120K).  In a
    # CONTINUOUS render the SAME draft is rebuilt every frame, so cache the resolved arrays
    # ONCE (keyed by device+mask) and COPY them per frame instead of re-resolving.  The GPU
    # W_o-delta GEMM (which reads cam_sparse) + the block-0 fold still recur (they are the
    # dispatch-bound residual); the CPU resolution becomes a one-time cost.  Byte-identical
    # (same arrays); DEFAULT OFF -> the per-frame re-resolve (composed number unchanged).
    _cache_resolved = _sched_cache_resolved()
    _rc = getattr(draft, "_resolved_cache", None) if _cache_resolved else None
    _rc_key = (str(dev), int(mask), bool(onchip), bool(gpu_build))
    if _rc is not None and _rc.get("_key") != _rc_key:
        _rc = None
    toks_np = _draft_tokens_np(draft) if (gpu_build or _cache_resolved) else None
    if _rc is not None:
        h0_table = _rc["h0_table"].clone()               # fresh copy (fold mutates it)
        ing_table = _rc["ing_table"]                     # read-only after build
    else:
        h0_table = _build_query_embed(model, L, code, draft, dev, toks_np=toks_np)   # [K, D]
        ing_table = _build_ingest_table(model, L, draft, dev, toks_np=toks_np)      # [K, H0, HD0]
    if onchip and fast:
        # C4_SCHED_GPU_BUILD: collapse the resolve_load_rows -> build_resolved_table ->
        # _build_cam_sparse Python-dict chain into one vectorized numpy pass (the dominant
        # ~70% of the schedule build at whole-frame K).  OFF -> the Python-dict _build_cam_sparse.
        if _rc is not None:
            cam_sparse = _rc["cam_sparse"]              # cached resolved (vals/cols/HDb)
        else:
            cam_sparse = (_build_cam_sparse_gpu(draft, code, live_blocks, dev) if gpu_build
                          else _build_cam_sparse(draft, code, live_blocks))  # {b:(vals,cols,HDb)}
        cam_tables = {}                                               # dense not materialized
    else:
        cam_sparse = None
        cam_tables = _build_cam_tables(model, L, draft, code, live_blocks, dev)
    # populate the resolved-gather cache (one-time; keyed by device/mask/onchip/gpu_build).
    if _cache_resolved and _rc is None:
        try:
            draft._resolved_cache = {"_key": _rc_key, "h0_table": h0_table.clone(),
                                     "ing_table": ing_table, "cam_sparse": cam_sparse}
        except Exception:
            pass
    # decode targets (per step) — the K=1 reference the verify compares against.  ONE pass
    # over frames (not 6 separate range(n) comprehensions) into numpy, one host->device xfer
    # each — draft-fast at 358 K steps.
    n = draft.step_count
    frames = draft.frames
    import numpy as _np
    if gpu_build:
        want_pc, want_ax, want_sp, want_bp, is_halt, is_file = _decode_targets_gpu(
            draft, mask, dev)
    else:
        _pc = _np.empty(n, dtype=_np.int64); _ax = _np.empty(n, dtype=_np.int64)
        _sp = _np.empty(n, dtype=_np.int64); _bp = _np.empty(n, dtype=_np.int64)
        _hl = _np.empty(n, dtype=_np.bool_); _fl = _np.empty(n, dtype=_np.bool_)
        for s in range(n):
            f = frames[s]
            _pc[s] = f["pc"]; _ax[s] = f["ax"] & mask
            _sp[s] = f["sp"] & 0xFFFFFFFF; _bp[s] = f["bp"] & 0xFFFFFFFF
            _hl[s] = bool(f.get("is_halt")); _fl[s] = bool(f.get("is_file"))
        want_pc = torch.from_numpy(_pc).to(dev)
        want_ax = torch.from_numpy(_ax).to(dev)
        want_sp = torch.from_numpy(_sp).to(dev)
        want_bp = torch.from_numpy(_bp).to(dev)
        is_halt = torch.from_numpy(_hl).to(dev)
        is_file = torch.from_numpy(_fl).to(dev)
    # ---- C4_ONCHIP_RESIDUAL: precompute the compact W_o(cam_out) deltas ONCE ----
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
        # IN-PLACE fold into h0_table (it is dead after this — the onchip replay reads only
        # h0_folded), so no separate 2.6 GB h0_folded allocation.  addmm_ avoids the [K,D]
        # matmul temporary too.
        h0_table.addmm_(ing_flat, Wo0.transpose(0, 1))
        h0_folded = h0_table
        del ing_flat
        # each live block: compact [K, n_out_b] delta + its out_dims.
        cam_delta_tables = {}
        live_out_dims = {}
        for lb in live_blocks:
            b = lb.block_idx
            Wo = _wo_dense(model, b, dev)
            if cam_sparse is not None:
                # DIRECT sparse->delta: full[:, c] = sum_a vals[:,a] * Wo[c, active_cols[a]]
                # = vals @ Wo[:, active_cols].T (all D output cols; inactive cam cols are
                # exactly 0 so dropping them is exact).  Then slice the nonzero output dims.
                # A [K, n_active] @ [n_active, D] tiny GEMM — no dense [K,D] cam_flat, no
                # [K,D]@[D,D] GEMM on a 99.7%-zero input.
                vals_np, cols_np, _HDb = cam_sparse[b]
                vals = torch.from_numpy(vals_np).to(dev)                 # [K, n_active]
                cols = torch.from_numpy(cols_np).to(dev)                 # [n_active]
                Wo_sub = Wo.index_select(1, cols)                        # [D, n_active]
                # SUPPORT-FIRST (VRAM-frugal): out_dims = the output dims W_o(cam) CAN be
                # nonzero at = the rows of Wo with any nonzero over the active cam columns
                # (data-independent — a [D, n_active] reduction, NOT a [K, D] matmul).  Then
                # build the compact delta ``vals @ Wo_sub[out_dims].T`` = [K, n_out] DIRECTLY
                # (n_out is a handful), never materializing the 2.44 GB [K, D] product.  Byte-
                # exact: every column NOT in out_dims is exactly 0 for every row (same set the
                # full-product .abs().sum(0)>0 selects — a row is nonzero only where Wo_sub is).
                out_dims = (Wo_sub.abs().sum(1) > 0).nonzero(as_tuple=False).flatten()
                if out_dims.numel() == 0:
                    out_dims = torch.zeros(1, dtype=torch.long, device=vals.device)
                delta = (vals @ Wo_sub.index_select(0, out_dims).transpose(0, 1)).contiguous()
                del vals, Wo_sub
            else:
                cam_flat = cam_tables[b].reshape(n, D)            # [K, D]
                delta, out_dims = _build_wo_delta_table(cam_flat, Wo)
            cam_delta_tables[b] = delta
            live_out_dims[b] = out_dims
    # ON-CHIP replay reads ONLY h0_folded + cam_delta_tables — ing_table / cam_tables are
    # dead after the fold/delta build, so drop them (a big VRAM saving: ing_table is [K,D],
    # freeing ~2.6 GB — critical for the continuous double-buffer where two schedules coexist).
    if onchip:
        ing_table = None
        cam_tables = {}
    sched = PrecomputedSchedule(
        n_steps=n, h0_table=h0_table, ing_table=ing_table, cam_tables=cam_tables,
        want_pc=want_pc, want_sp=want_sp, want_bp=want_bp, want_ax=want_ax,
        is_halt=is_halt, is_file=is_file, onchip=onchip, h0_folded=h0_folded,
        cam_delta_tables=cam_delta_tables)
    return sched, live_out_dims


def build_schedule_tables_only(model, L, code, draft, device, sg, mask: int = 0xFFFFFFFF
                               ) -> "PrecomputedSchedule":
    """Rebuild the O(n) schedule TABLES for a frame, REUSING an already-built (and captured)
    ``PrecomputedStepGraph`` ``sg`` — the honest per-frame BUILD cost in a CONTINUOUS stream
    (the megablock install + live-block map + graph capture are one-time; only the tables
    recur per frame).  Returns a fresh ``PrecomputedSchedule`` whose tables the SAME captured
    ``sg`` replays over.  Byte-identical tables to ``build_schedule``."""
    dev = torch.device(device)
    live_blocks = list(sg.live_blocks.values())
    sched, _lod = _build_schedule_tables(model, L, code, draft, dev, live_blocks,
                                         sg.onchip, mask)
    return sched


# ===========================================================================
# 4. THE TRUE OVERLAP: build(N+1) on a background THREAD + dedicated CUDA streams,
#    concurrent with the graph replay of frame N (C4_SCHED_PIPELINE).
# ===========================================================================
def sched_pipeline_enabled() -> bool:
    """``C4_SCHED_PIPELINE`` (DEFAULT OFF).  TRUE double-buffer overlap: run the frame
    N+1 schedule BUILD (the ~90%-CPU numpy resolve + the H2D uploads + the small W_o-delta
    GEMMs) on a BACKGROUND THREAD bound to a DEDICATED CUDA build stream, CONCURRENT with
    the frame-N graph REPLAY on the default compute stream.

    WHY the naive ``with torch.cuda.stream(side): build()`` in the harness did NOT overlap
    (measured 2% gain, not the ~max(build,dispatch) the double-buffer promises): the build
    is ~90% CPU numpy (``_resolve_reads_vec`` / the frame-decode loop / the token
    ``np.asarray``) + H2D copies — a ``torch.cuda.stream`` context only redirects GPU
    KERNEL LAUNCHES, it does NOT make CPU work run off the calling thread.  So the whole
    CPU build ran to completion on the main thread BEFORE the dispatch replay was ever
    issued — they serialized on the CPython thread, not on the GPU.

    THE FIX (this builder):
      (a) THREAD: the build runs in a real ``threading.Thread``.  Every heavy numpy op
          (``lexsort`` / ``searchsorted`` / bit-shift scatter / ``np.asarray`` of the
          ~30·n-token stream) releases the GIL, so the main thread proceeds to issue the
          frame-N replay launches concurrently.
      (b) STREAM: inside the thread the current stream is set to a dedicated ``build``
          stream, so the build's GPU ops (embed gather, the H2D ``.to(device)`` copies,
          the W_o-delta GEMMs) are enqueued on a SEPARATE stream from the default compute
          stream the replay uses — they overlap on the GPU (subject to SM availability),
          and the H2D copies do not serialize behind the replay's compute.
      (c) EVENT SYNC AT THE SWAP ONLY: ``wait()`` joins the thread, then records a build
          event and makes the default stream ``wait_event`` it, so the next frame's replay
          (which reads the freshly-built buffer) is ordered strictly AFTER the build
          completes — no reading a half-built buffer.  Correctness == serial (the swap is
          the only synchronization point).

    Byte-identical to the serial build (same numpy, same H2D values); DEFAULT OFF."""
    return os.environ.get("C4_SCHED_PIPELINE", "0") not in ("0", "", "false", "False")


def _faithful_precompute_thread_enabled() -> bool:
    """``C4_FAITHFUL_PRECOMPUTE_THREAD`` (DEFAULT OFF).  Run the faithful value-verify
    PRECOMPUTE (``build_faithful_precompute``, the ~1.1 us/step genuine latest-write-wins
    re-resolution) on a SECOND inner thread inside ``PipelinedScheduleBuilder._run``,
    CONCURRENT with the schedule-table build (which is itself ~90% GIL-releasing numpy + GPU
    ops on the dedicated build stream) — instead of serializing the two on the one build
    thread.  So the build thread's steady-state cost becomes ``max(precompute, schedule-build)``
    rather than their sum, keeping the faithful build below the dispatch when the precompute is
    NOT cached (a genuinely re-drafting continuous render).  Both are GIL-releasing so they
    overlap on separate CPython threads; the precompute is joined before the swap.  Byte-
    identical (same numpy, same result); DEFAULT OFF -> serial precompute-then-build."""
    return os.environ.get("C4_FAITHFUL_PRECOMPUTE_THREAD", "0") not in ("0", "", "false", "False")


class PipelinedScheduleBuilder:
    """Double-buffered background-thread schedule builder for the CONTINUOUS render loop.

    Usage (the harness / render loop drives it)::

        pb = PipelinedScheduleBuilder(model, L, code, draft, device, sg, mask)
        cur = pb.build_blocking()          # frame 0 (no overlap possible yet)
        pb.start()                         # kick off frame 1's build on the thread
        while rendering:
            dispatch(cur)                  # frame-N replay on the default stream (overlaps)
            nxt = pb.wait()                # join the build thread + event-sync the swap
            del cur; cur = nxt
            pb.start()                     # kick off frame N+2's build

    The build thread owns a dedicated CUDA build stream; ``wait`` event-syncs the default
    stream to the build stream so the returned schedule is fully materialized before the
    caller replays over it.  ``draft`` is a fixed trace in this steady-state harness (the
    continuous loop re-drafts per real frame; here the build cost — the target — is the
    same tensor work regardless), so the thread re-runs ``build_schedule_tables_only``."""

    def __init__(self, model, L, code, draft, device, sg, mask: int = 0xFFFFFFFF,
                 faithful: bool = False):
        import threading
        self._threading = threading
        self.model = model
        self.L = L
        self.code = code
        self.draft = draft
        self.device = torch.device(device)
        self.sg = sg
        self.mask = mask
        # dedicated build stream: the build's GPU ops (embed gather / H2D / W_o GEMMs) run
        # on ``_build_stream`` so they do NOT serialize behind the default compute stream the
        # graph replay uses (priority 0 == least-priority default, so the replay is never
        # preempted by build kernels — the build kernels fill replay idle cycles).
        self._build_stream = torch.cuda.Stream(device=self.device)
        self._thread: Optional["threading.Thread"] = None
        self._result: Optional[PrecomputedSchedule] = None
        self._precomp = None                       # the faithful precompute for this frame
        self._exc: Optional[BaseException] = None
        self._done_event: Optional[torch.cuda.Event] = None
        # FAITHFUL SINGLE-DISPATCH: also PIPELINE the value-verify's genuine value
        # re-resolution.  It is a PURE function of the store-log — knowable BEFORE the
        # dispatch — so it runs on THIS build thread (GIL-releasing numpy: lexsort /
        # searchsorted latest-write-wins), concurrent with the previous frame's replay,
        # exactly like the schedule build.  The critical path then only does the cheap
        # vectorized model-vs-precomputed compare (``verify_faithful_fast``).
        self.faithful = bool(faithful)
        # C4_FAITHFUL_PRECOMPUTE_THREAD: overlap the value-precompute with the schedule build
        # on a second inner thread (the lever for the uncached / genuinely-redrafting case).
        self._precompute_on_thread = bool(faithful) and _faithful_precompute_thread_enabled()
        self._plan = None
        self._ws = None
        if self.faithful:
            import numpy as _np
            from .faithful_single_dispatch import build_faithful_plan
            # the plan (committed store arrays + draft addr/val/code dicts) is fixed per
            # draft — build it ONCE; the per-frame precompute re-resolves the genuine values.
            self._plan = build_faithful_plan(model, L, code, draft)
            self._ws = _np.asarray(draft.win_starts[:draft.step_count], dtype=_np.int64)

    def _build_precompute(self):
        """The faithful value-verify precompute (GIL-releasing numpy) — the piece we moved
        off the critical path onto this build thread."""
        from .faithful_single_dispatch import build_faithful_precompute
        return build_faithful_precompute(self.draft, self._plan, self._ws,
                                         self.draft.step_count, mask=self.mask)

    def build_blocking(self) -> PrecomputedSchedule:
        """Frame-0 build (no concurrent dispatch yet); runs on the caller thread/stream.
        Also builds the frame-0 faithful precompute (stored in ``self._precomp``)."""
        if self.faithful:
            self._precomp = self._build_precompute()
        return build_schedule_tables_only(self.model, self.L, self.code, self.draft,
                                          self.device, self.sg, mask=self.mask)

    def last_precompute(self):
        """The faithful precompute matching the schedule the last ``wait``/``build_blocking``
        returned (None when not faithful).  The critical path passes this to
        ``verify_faithful_fast`` with the dispatch's decoded model addresses."""
        return self._precomp

    def _run(self):
        try:
            # the faithful value-verify re-resolution — pure numpy, releases the GIL.  The
            # schedule build below is also ~90% GIL-releasing numpy + GPU ops on the build
            # stream.  By DEFAULT the two run SERIALLY on this thread (precompute first, then
            # the schedule build); with C4_FAITHFUL_PRECOMPUTE_THREAD the precompute runs on a
            # SECOND inner thread CONCURRENT with the schedule build so they OVERLAP EACH OTHER
            # (the build thread's steady-state cost becomes max(precompute, schedule-build)
            # instead of their sum).  With C4_FAITHFUL_PRECOMPUTE_CACHE the precompute is a
            # cheap cache-hit copy after frame 0, so serialization is already free — the second
            # thread is the lever for the UNCACHED (genuinely re-drafting) case.
            precomp_thr = None
            precomp_box = {}
            if self.faithful and self._precompute_on_thread:
                def _pc():
                    try:
                        precomp_box["v"] = self._build_precompute()
                    except BaseException as e:       # surface at the join below
                        precomp_box["e"] = e
                precomp_thr = self._threading.Thread(target=_pc, name="faithful-precompute",
                                                     daemon=True)
                precomp_thr.start()
            precomp = None
            if self.faithful and precomp_thr is None:
                precomp = self._build_precompute()
            with torch.cuda.stream(self._build_stream):
                with torch.no_grad():
                    sched = build_schedule_tables_only(
                        self.model, self.L, self.code, self.draft, self.device,
                        self.sg, mask=self.mask)
                # record completion on the BUILD stream; the caller's default stream will
                # wait on this event at the swap so the replay reads a fully-built buffer.
                ev = torch.cuda.Event()
                ev.record(self._build_stream)
                self._done_event = ev
            if precomp_thr is not None:
                precomp_thr.join()                   # the two overlapped; join before swap
                if "e" in precomp_box:
                    raise precomp_box["e"]
                precomp = precomp_box.get("v")
            self._result = sched
            self._precomp_next = precomp
        except BaseException as e:                 # surface build errors at wait()
            self._exc = e

    def start(self):
        """Kick off the NEXT frame's build on the background thread + build stream.  Returns
        immediately; the caller then issues the current frame's replay (which overlaps)."""
        assert self._thread is None, "a build is already in flight; call wait() first"
        self._result = None
        self._precomp_next = None
        self._exc = None
        self._done_event = None
        self._thread = self._threading.Thread(target=self._run, name="sched-build",
                                              daemon=True)
        self._thread.start()

    def wait(self) -> PrecomputedSchedule:
        """Join the build thread and EVENT-SYNC the default stream to the build stream so the
        freshly-built schedule is fully materialized before the caller replays over it.  Also
        swaps in the freshly-built faithful precompute (``last_precompute``)."""
        assert self._thread is not None, "no build in flight; call start() first"
        self._thread.join()
        self._thread = None
        if self._exc is not None:
            raise self._exc
        # order the default (replay) stream strictly AFTER the build completed on the build
        # stream — the ONLY synchronization point (no reading a half-built buffer).
        if self._done_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._done_event)
        self._precomp = self._precomp_next
        return self._result


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


def run_faithful_verify(model, L, code, draft, device, *, mask=0xFFFFFFFF,
                        collect_out=None, stats=None):
    """FAITHFUL SINGLE-DISPATCH (C4_FAITHFUL_SINGLE_DISPATCH): the FAST fused single
    dispatch made GENUINE — ONE dispatch pass produces BOTH the byte-exact register
    decode AND the model's own decoded query addresses (in-graph ``sign(W_q(x))`` at each
    live CAM block, near-zero marginal since that block's ``W_q(x)`` would be discarded).
    We then independently verify every draft-derived quantity (routing op / read address /
    read value) against that genuine computation with a first-divergence terminal FAIL.

    Returns a ``PrecomputedResult`` identical to ``run_verify`` on correct execution, but
    ``accepted_steps`` is clipped to the first step where EITHER the register compare OR
    the genuine routing/address/value check diverges (so a self-consistent wrong draft is
    REJECTED, unlike the plain fast path).  The value re-resolution is latest-write-wins
    over the committed (evicted / bounded) stores at the MODEL's address — the
    ``C4_FAITHFUL_ATTN_EVICT`` mechanism, wired into the single dispatch (not the slow
    verify_blocks fallback)."""
    import numpy as _np
    import time as _time
    from .faithful_single_dispatch import (build_faithful_plan,
                                           build_faithful_precompute, verify_faithful_fast)
    dev = torch.device(device)
    _DM = {"DIV", "MOD"}
    for s in range(draft.step_count):
        if draft.frames[s].get("op") in _DM:
            raise ValueError("run_faithful_verify: DIV/MOD step — use verify_blocks")
    t0 = _time.perf_counter()
    sched, sg = build_schedule(model, L, code, draft, dev, mask=mask)
    n = sched.n_steps
    chunk = sg.chunk
    onchip = sched.onchip
    got_pc = torch.empty(n, dtype=torch.long, device=dev)
    got_sp = torch.empty(n, dtype=torch.long, device=dev)
    got_bp = torch.empty(n, dtype=torch.long, device=dev)
    got_ax = torch.empty(n, dtype=torch.long, device=dev)
    h0_src = sched.h0_folded if onchip else sched.h0_table
    resident = resident_batch_enabled() and n <= chunk
    per_head = {}          # (block, head, kind) -> list of (lo, hi, np arr)
    n_chunks = 0
    host_ops = 0
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        h0 = h0_src[lo:hi].unsqueeze(0)
        if onchip:
            ing = None; cam = None
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, delta=delta, resident=resident)
        else:
            ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
            cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0)
                   for b, t in sched.cam_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, resident=resident)
        got_pc[lo:hi].copy_(pc_c); got_sp[lo:hi].copy_(sp_c)
        got_bp[lo:hi].copy_(bp_c); got_ax[lo:hi].copy_(ax_c)
        # the in-graph decoded model query addresses for this chunk.
        for key, t in sg.last_qaddr(hi - lo).items():
            per_head.setdefault(key, []).append(
                (lo, hi, t.detach().to("cpu").numpy().astype("int64")))
        n_chunks += 1; host_ops += 1
    got_ax = got_ax & mask
    # ---- REGISTER compare (byte-exact, exactly run_verify) ----
    bad_normal = ((got_pc != sched.want_pc) | (got_ax != sched.want_ax)
                  | (got_sp != sched.want_sp) | (got_bp != sched.want_bp))
    bad = torch.where(sched.is_halt, got_ax != sched.want_ax, bad_normal)
    bad = bad & (~sched.is_file)
    any_bad = bad.any()
    first_bad = torch.argmax(bad.to(torch.uint8))
    n_ok_reg = int(torch.where(any_bad, first_bad,
                               torch.tensor(n, device=dev)).item())
    t_fast = _time.perf_counter() - t0
    # ---- GENUINE verify (routing / address / value from the model's query) ----
    # PIPELINED SPLIT: the genuine value re-resolution (latest-write-wins over the committed
    # store-log) is a PURE function of the store-log — knowable BEFORE the dispatch — so it is
    # done as a DRAFT-ONLY precompute (``build_faithful_precompute``, GIL-releasing numpy) that
    # in the continuous render loop rides on the background build thread (concurrent with the
    # prior frame's replay, via ``PipelinedScheduleBuilder``).  The critical path here only
    # does the cheap fully-vectorized model-vs-precomputed compare (``verify_faithful_fast``).
    ws = _np.asarray(draft.win_starts[:n], dtype=_np.int64)
    t0 = _time.perf_counter()
    plan = build_faithful_plan(model, L, code, draft)
    pre = build_faithful_precompute(draft, plan, ws, n, mask=mask)
    t_precomp = _time.perf_counter() - t0
    model_addrs = {}
    for (b, hh, knd), chunks in per_head.items():
        full = _np.zeros(n, dtype=_np.int64)
        for (lo, hi, arr) in chunks:
            full[lo:hi] = arr[:hi - lo]
        model_addrs[(b, knd)] = full
    t0 = _time.perf_counter()
    verdict = verify_faithful_fast(pre, model_addrs)
    t_verify = _time.perf_counter() - t0
    n_ok_gen = verdict.first_bad_step if not verdict.ok else n
    n_ok = min(n_ok_reg, n_ok_gen)
    # PRTF visible bytes for the accepted prefix (byte-identical to verify_blocks).
    if collect_out is not None and draft.prtf_steps:
        prtf = sorted(int(s) for s in draft.prtf_steps if s < n_ok)
        if prtf:
            pl = torch.tensor(prtf, device=dev, dtype=torch.long)
            pb = (got_ax & 0xFF).index_select(0, pl).tolist()
            collect_out.extend(int(v) & 0xFF for v in pb)
    all_matched = (n_ok == n)
    final_ax = int(got_ax[n - 1].item()) if all_matched and n > 0 else None
    if stats is not None:
        stats["n_chunks"] = n_chunks
        stats["host_ops_per_forward"] = host_ops
        stats["chunk"] = chunk
        stats["fast_dispatch_secs"] = t_fast
        stats["faithful_precompute_secs"] = t_precomp    # pipelined onto the build thread
        stats["faithful_verify_secs"] = t_verify         # the residual critical-path compare
        stats["faithful_addr_checked"] = verdict.n_addr_checked
        stats["faithful_value_checked"] = verdict.n_value_checked
        stats["faithful_routing_checked"] = verdict.n_routing_checked
        stats["faithful_ok"] = (n_ok == n)
    first_mismatch = None
    if not all_matched:
        s = n_ok
        if n_ok_gen <= n_ok_reg and not verdict.ok:
            first_mismatch = {"step": s, "query_pos": int(ws[s]) if s < n else None,
                              "kind": verdict.kind, "detail": verdict.detail}
            if stats is not None:
                stats["faithful_divergence"] = dict(first_mismatch)
        else:
            first_mismatch = {
                "step": s, "query_pos": draft.win_starts[s],
                "got": {"pc": int(got_pc[s].item()), "ax": int(got_ax[s].item()),
                        "sp": int(got_sp[s].item()), "bp": int(got_bp[s].item())},
                "want": {"pc": int(sched.want_pc[s].item()),
                         "ax": int(sched.want_ax[s].item()),
                         "sp": int(sched.want_sp[s].item()),
                         "bp": int(sched.want_bp[s].item())}}
    return PrecomputedResult(
        accepted_steps=n_ok, total_steps=n, all_matched=all_matched,
        decoded_final_ax=final_ax, first_mismatch=first_mismatch,
        n_chunks=n_chunks, host_ops_per_forward=host_ops)


__all__ = ["precomputed_schedule_enabled", "PrecomputedSchedule",
           "PrecomputedStepGraph", "build_schedule", "build_schedule_tables_only",
           "run_verify", "run_faithful_verify", "PrecomputedResult",
           "sched_pipeline_enabled", "PipelinedScheduleBuilder"]
