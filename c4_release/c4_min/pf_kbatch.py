"""#747 — WITHIN-PROGRAM BIG-K batching on the BOUNDED-KV pos-sparse path.

#746 (``pos_sparse_bounded.BoundedPosSparseRunner``) collapsed the K/V-over-S floor:
each VM step's attention is now over a BOUNDED row set (local ingest window ~W rows +
store-only global CAM ~n_store rows), so ms/step is FLAT in S.  But it still runs ONE
``forward`` per VM step (a single query row ``q = S-1``).  At ~4.2ms/step and 23.2M
self-emu steps that is the 18hr K=1 wall.

This module packs K VM steps into ONE forward on that bounded path.  The perfect
draft (``pf_speculative.draft_pf_program``) materialises the K steps' 30-token frames
+ store_log DETERMINISTICALLY (CPU, microseconds/step).  We teacher-force the whole
K-frame stream and VERIFY all K steps in ONE forward: each of the K query rows (the
last token of each frame, absolute pos ``30*(s+1)``) decodes with its OWN bounded
local window + store-row subset, causal to that row.  Because softmax1 is causal and
every dropped row's true weight is exactly 0 (the #746 bounding), the K-batched decode
of step ``s`` is BYTE-IDENTICAL to the sequential single-step decode of step ``s`` —
the model does the computing, K-batching only verifies K at a time.

  forwards = steps / K.

The K ceiling: the score matrix is per-query-row bounded (each row attends only to its
~W + n_store keys, NOT to all K*30 span rows), so the attention cost is O(K * (W +
n_store)) — LINEAR in K, not O(K^2).  The FFN is over all K query rows (K-row batch).
So this path does NOT suffer the O(K^2) within-batch blowup the full-stack KV-cached
verifier (``pf_speculative.verify_blocks``) hits — its ceiling is K-row FFN + the K
per-row bounded attentions, both linear in K.

#748 — SUB-HOUR self-forward.  #747 landed ~1.14hr @ K=128; #747 pinned the residual
as the byte-exact fp64 FFN (a39ae2c ran EVERY live block's query-row SwiGLU in fp64 to
fix the ``lea-addr-nib`` decode tie).  Three byte-exact levers cut it below 1hr:
  (a) SELECTIVE fp64 — the analyzer (``analyze_fp64_blocks``) proved a SINGLE block,
      ``lea-addr-nib``, is the only fp-fragile one; fp32-ing the other ~11-185 live
      blocks (A5000 fp64 ~1/32 fp32) gave ~1.55x @ K=128 (1.13 -> 0.73hr).
  (b) IN-PLACE forward_span — clone the teacher-forced stream ONCE (not per block;
      the per-block [1,S,D] clone was ~29MB x #live-blocks) -> 0.73 -> 0.66hr.
  (c) VECTORIZED bounded attention — the per-query Python loop (128 aranges + 128
      indexed writes) was the K=128 launch bottleneck; ``_local_layout`` /
      ``_store_layout`` build the ``[Q, Rmax]`` key layout in a couple of broadcasts
      -> 0.66 -> ~0.30hr (the big win).  Plus (d) the MEGAKERNEL: the tail
      passthrough-FFN segment is S-INDEPENDENT (block 0 is the only attention block),
      so ``GraphedFFNChain`` CUDA-graphs it keyed on (schedule, K) — amortizes across
      the self-emu, 2x on DIV/MOD's 185-block FFN chain, ~1.06x on the weighted mix.
RESULT: byte-exact @ K=1..128; weighted op-mix ~0.046 ms/step_eff @ K=128 ->
self-forward ~0.30hr (eager) / ~0.29hr (graph), 52x vs the 18hr K=1 baseline.  The
residual is now the block-0 attention einsum (bounded but GPU-compute-bound, the
[1,K,W,H,HD] score/ctx tensors), NOT fp64 and NOT launch.

Gate: ``C4_POS_SPARSE`` (the composed path's flag).  Golden flag-OFF is untouched.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .blogspec_model import softmax1
from .pos_sparse_bounded import BoundedBlock
from .step_block_skip import build_live_index


# ---------------------------------------------------------------------------
# SELECTIVE fp64 (#748): the query-row SwiGLU FFN runs fp64 ONLY on the fp-fragile
# blocks whose nibble decode a39ae2c pinned; every other block runs fp32.  The
# analyzer (``analyze_fp64_blocks``) proved that a SINGLE block — ``lea-addr-nib``,
# the LEA address-nibble gadget — is the only block whose fp32 1-row GEMV flips the
# integer decode margin (LEA_Q&0xFF: fp32 -> -21, fp64 -> 12).  Fixing it fp64
# makes the whole battery + a deep nested loop byte-exact at K=1..32.  We resolve
# it by NAME (robust to block-count / layout shifts), overrideable by env
# ``C4_FP64_BLOCK_NAMES`` (comma-sep) / ``C4_FP64_ALL=1`` (a39ae2c all-fp64).
# ---------------------------------------------------------------------------
FP_FRAGILE_BLOCK_NAMES = ("lea-addr-nib",)


def default_fp64_block_idxs(L, n_blocks: int):
    """Resolve the fp-fragile block NAMES to indices in the built model.

    Returns ``None`` (all-fp64) if ``C4_FP64_ALL=1``; else the indices of the
    fp-fragile blocks (env-overridable via ``C4_FP64_BLOCK_NAMES``)."""
    import os
    if os.environ.get("C4_FP64_ALL", "0") == "1":
        return None
    names_env = os.environ.get("C4_FP64_BLOCK_NAMES")
    frag = tuple(n.strip() for n in names_env.split(",") if n.strip()) \
        if names_env else FP_FRAGILE_BLOCK_NAMES
    block_names = list(getattr(L, "_block_names", []))
    idxs = [i for i, nm in enumerate(block_names) if nm in frag and i < n_blocks]
    return idxs


# ---------------------------------------------------------------------------
# MULTI-QUERY bounded block: run K query rows in one forward.  Each query row s
# has its own local window (last-W rows <= its pos) and store-row subset (IS_STORE
# rows <= its pos).  Byte-identical to K sequential single-query BoundedBlock calls.
# ---------------------------------------------------------------------------
class KBatchBoundedBlock:
    """A ``BoundedBlock`` that processes a LIST of query rows in one forward.

    For each query row we gather its bounded keys (local window ``[q-W+1, q]`` for
    local heads; store rows ``<= q`` + self for global heads), run softmax1+ALiBi,
    and write the query-row attention output + FFN in place.  The query rows are
    processed with a per-row ragged key set — but because each row's key count is
    bounded (``W`` or ``n_store``), the total work is ``O(K*(W+n_store))`` (linear
    in K), and the FFN is a single K-row batch.
    """

    def __init__(self, blk, window: int, store_dim: int):
        self.b = BoundedBlock(blk, window, store_dim)
        # SELF-EMU DIRECT-CAM (C4_SELFEMU_DIRECT_CAM): when armed, the global CAM
        # heads DIRECT-GATHER the draft-resolved value (O(1)) instead of softmax1
        # over the store rows (O(n_store)).  Populated by the runner via
        # ``arm_direct_cam``; None -> the vanilla store-row softmax path.
        self._dcam_kinds = None                # {head_idx: "mem"/"pop"/"lev"}
        self._dcam_tbl = None                  # DirectCamTable

    # ------------------------------------------------------------------
    def _local_layout(self, q_t, W):
        """VECTORIZED local-window layout for ALL queries at once (no Python loop).

        Local heads attend the last-``W`` rows ``[q-W+1, q]`` for each query ``q``.
        Every query has the SAME window WIDTH ``W`` (queries near the stream start
        clamp to 0 -> the below-0 slots are masked off).  Builds ``[Q, W]``
        ``key_idx`` / ``valid`` / ``dist`` in a couple of broadcasts — byte-identical
        to the per-query aranges the ragged path built."""
        Q = q_t.numel()
        offs = torch.arange(W, device=q_t.device)                        # 0..W-1
        # key = q - (W-1) + off  == q - (W-1-off) ; rows below 0 are invalid.
        key_idx = q_t.view(Q, 1) - (W - 1) + offs.view(1, W)             # [Q,W]
        valid = key_idx >= 0
        key_idx = key_idx.clamp_min(0)                                   # pad -> row 0
        dist = (key_idx.float() - q_t.view(Q, 1).float()).abs()
        return key_idx, valid, dist

    def _store_layout(self, q_t, all_store):
        """VECTORIZED store-row layout for ALL queries at once (no Python loop).

        Global CAM heads attend the IS_STORE rows ``<= q`` + the query row itself.
        For query grid ``q_t`` [Q] and the shared sorted ``all_store`` [P], build a
        ``[Q, P+1]`` layout: column p holds store row p if ``all_store[p] <= q``
        (else masked), and the last column always holds the query row ``q`` (the
        causal self row).  If ``q`` is already a store row it appears twice — the
        DUPLICATE query-self column is masked off there so softmax1 counts it once
        (byte-identical to ``_store_rows_for``'s dedup, which drops the appended
        self when ``rows[-1] == q``)."""
        Q = q_t.numel()
        P = all_store.numel()
        # store columns: valid iff store_row <= q.
        st = all_store.view(1, P)                                        # [1,P]
        le = st <= q_t.view(Q, 1)                                        # [Q,P] valid
        # self column (always the query row).
        self_col = q_t.view(Q, 1)                                       # [Q,1]
        # a store row that EQUALS q is the causal self already -> mask the appended
        # self column there (matches _store_rows_for: it appends self only when the
        # last <=q store row != q).
        q_is_store = (st == q_t.view(Q, 1)).any(dim=1, keepdim=True)     # [Q,1]
        key_idx = torch.cat([all_store.view(1, P).expand(Q, P), self_col], dim=1)
        valid = torch.cat([le, ~q_is_store], dim=1)                     # [Q,P+1]
        key_idx = torch.where(valid, key_idx, torch.zeros_like(key_idx))
        dist = (key_idx.float() - q_t.view(Q, 1).float()).abs()
        return key_idx, valid, dist

    # ------------------------------------------------------------------
    def _attend_dense(self, x, q_t, head_idx, key_idx, valid, dist):
        """softmax1+ALiBi context for ``head_idx`` given a DENSE ``[Q, Rmax]`` key
        layout (built vectorized by ``_local_layout`` / ``_store_layout``).  Pad
        keys (``~valid``) score -inf so their softmax1 weight is exactly 0.  Returns
        ``[B, Q, nh, HD]``.  Byte-identical to the per-query ragged path."""
        b = self.b
        B = x.shape[0]
        HD, H = b.HD, b.H
        Q, Rmax = key_idx.shape
        hset = head_idx
        nh = int(hset.numel())
        # project K/V over the UNION of referenced rows once (dedup keeps the GEMM
        # bounded), then index per (q, key) — byte-identical to per-row projection.
        uniq, inv = torch.unique(key_idx[valid], return_inverse=True)     # [U]
        xr = x[:, uniq]                                                   # [B,U,D]
        Kk = b.attn.W_k.linear(xr).view(B, -1, H, HD)[:, :, hset]         # [B,U,nh,HD]
        Vv = b.attn.W_v.linear(xr).view(B, -1, H, HD)[:, :, hset]
        full_inv = torch.zeros(Q, Rmax, dtype=torch.long, device=x.device)
        full_inv[valid] = inv                                            # [Q,Rmax] -> U
        xq = x[:, q_t]                                                    # [B,Q,D]
        Qh = b.attn.W_q.linear(xq).view(B, Q, H, HD)[:, :, hset]          # [B,Q,nh,HD]
        Kg = Kk[:, full_inv]                                             # [B,Q,Rmax,nh,HD]
        Vg = Vv[:, full_inv]
        sc = torch.einsum("bqhd,bqrhd->bqrh", Qh, Kg) * b.scale          # [B,Q,Rmax,nh]
        sc = sc - b.alibi[hset].view(1, 1, 1, nh) * dist.view(1, Q, Rmax, 1)
        sc = sc.masked_fill(~valid.view(1, Q, Rmax, 1), float("-inf"))
        a = softmax1(sc, dim=2)                                          # [B,Q,Rmax,nh]
        ctx = torch.einsum("bqrh,bqrhd->bqhd", a, Vg)                    # [B,Q,nh,HD]
        return ctx

    def forward(self, x: torch.Tensor, q_idxs: List[int],
                all_store: Optional[torch.Tensor] = None,
                q_t: Optional[torch.Tensor] = None,
                inplace: bool = False) -> torch.Tensor:
        """Process the K query rows of ``x`` in one forward.

        ``inplace=True`` mutates ``x``'s query rows IN PLACE and returns ``x`` —
        skipping the per-block ``x.clone()`` (a full [1,S,D] copy that at K=128
        S~4000 is ~29MB, and ONLY the Q<=128 query rows ever change).  Byte-
        identical to the clone path: each block reads the FULL stream (frozen
        non-query rows + the prior blocks' query-row updates) and writes ONLY the
        query rows.  The caller (``forward_span``) clones the input ONCE and runs
        the block chain in place.  ``q_t`` may be a precomputed LongTensor of the
        query indices (shared across blocks, one fewer host->device sync)."""
        b = self.b
        B, S, D = x.shape
        HD, H = b.HD, b.H
        Q = len(q_idxs)
        if q_t is None:
            q_t = torch.tensor(q_idxs, device=x.device, dtype=torch.long)
        xq = x[:, q_t]                                                   # [B,Q,D]
        if b.is_passthrough:
            aout = xq
        else:
            ctx_full = x.new_zeros(B, Q, H, HD)
            W = b.window
            if b.local_idx.numel() > 0:
                # VECTORIZED local-window layout (was a 128-iter Python loop + 128
                # aranges — the K=128 launch bottleneck).
                ki, vld, dst = self._local_layout(q_t, W)
                ctx_full[:, :, b.local_idx] = self._attend_dense(
                    x, q_t, b.local_idx, ki, vld, dst)
            if b.global_idx.numel() > 0:
                if self._dcam_kinds is not None:
                    # SELF-EMU DIRECT-CAM: O(1) gather of the resolved value's V
                    # vector per query row -> NO O(n_store) softmax over stores.
                    from .selfemu_direct_cam import gather_global_ctx
                    gctx = gather_global_ctx(
                        self._dcam_kinds, self._dcam_tbl, q_idxs, b.global_idx,
                        HD, x.device, ctx_full.dtype)          # [Q, nG, HD]
                    ctx_full[:, :, b.global_idx] = gctx.unsqueeze(0)
                else:
                    if all_store is None:
                        col = x[0, :, b.store_dim]
                        all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
                    ki, vld, dst = self._store_layout(q_t, all_store)
                    ctx_full[:, :, b.global_idx] = self._attend_dense(
                        x, q_t, b.global_idx, ki, vld, dst)
            ctx = ctx_full.reshape(B, Q, D)
            aout = xq + b.attn.W_o.linear(ctx)
        # FFN over the K query rows (one batched K-row FFN).
        fq = b._ffn_qrow(aout)                                          # [B,Q,D]
        out = x if inplace else x.clone()
        out[:, q_t] = fq
        return out


# ---------------------------------------------------------------------------
# MEGAKERNEL (#748 lever 2): CUDA-graph the passthrough-FFN chain.
#
# In every op-class live schedule the ONLY attention block is block 0 (the ingest
# CAM, S-dependent -> stays eager); EVERY subsequent live block is a passthrough
# (attention == identity), i.e. a pure query-row SwiGLU FFN that depends ONLY on
# the [1, K, D] query-row residual, NOT on S.  So the passthrough-FFN chain is
# S-INDEPENDENT: a CUDA graph keyed on (schedule, K) replays for EVERY span of that
# op-class at that K — it amortizes across the whole self-emulation (where S grows
# each span but K + the per-op live schedule are fixed).  Capture (~120ms) is paid
# ONCE per (schedule, K); replay collapses the ~185 tiny FFN launches (DIV/MOD) to
# a single graph launch, ~2x on the launch-bound chain.  BYTE-EXACT: the graphed
# arithmetic is the SAME per-block fp32/fp64 SwiGLU + residual add, verified L-inf=0
# vs the eager chain.
# ---------------------------------------------------------------------------
class GraphedFFNChain:
    """A captured CUDA graph of a run of consecutive passthrough-FFN blocks.

    ``run(xq)`` copies the [1, K, D] query-row residual into the static input,
    replays the graph, and returns the static output (the FFN chain applied)."""

    def __init__(self, kblocks, seg_block_idxs: List[int], K: int, D: int,
                 device, dtype):
        self.seg = list(seg_block_idxs)
        self.K = K
        self._captured = False
        self.static_in = torch.zeros(1, K, D, device=device, dtype=dtype)

        def _apply(inp):
            out = inp
            for bi in self.seg:
                out = kblocks[bi].b._ffn_qrow(out)
            return out

        self._apply = _apply

    def try_capture(self) -> bool:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    for _ in range(3):
                        _ = self._apply(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(self.graph):
                    self.static_out = self._apply(self.static_in)
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
# The K-batched runner: run the (live) blocks once with the K query rows.
# ---------------------------------------------------------------------------
class KBatchBoundedRunner:
    """Verify K VM steps per forward on the bounded-KV pos-sparse path.

    ``forward_span(x, ops, q_idxs)`` runs the union of the K ops' live blocks over
    the teacher-forced stream ``x`` ([1, S, D]) once, processing all ``q_idxs`` query
    rows with per-row bounded attention, and returns the block-output ``x`` (query
    rows overwritten in place).  The caller decodes each query row.
    """

    def __init__(self, model, L, window: int = 64, selective_fp64: bool = True):
        self.model = model
        self.blocks = model.blocks
        self.L = L
        self.window = int(window)
        self.live_index = build_live_index(model, L)
        self.n_blocks = len(model.blocks)
        store_dim = int(L.IS_STORE)
        self.kblocks = [KBatchBoundedBlock(b, window, store_dim) for b in model.blocks]
        # SELECTIVE fp64 (#748): by default run fp64 ONLY on the fp-fragile block(s)
        # (``lea-addr-nib``); fp32 everywhere else.  ``selective_fp64=False`` keeps
        # the a39ae2c all-fp64 baseline.
        self.fp64_idxs = None
        if selective_fp64:
            self.fp64_idxs = default_fp64_block_idxs(L, self.n_blocks)
            self.set_fp64_blocks(self.fp64_idxs)
        # INTEGER LEA-address snap (C4_LEA_INT_SNAP, #870): when the flag is on, the
        # ONLY fp64 block (``lea-addr-nib``) is computed EXACTLY in integer arithmetic
        # instead of fp64 -> ZERO fp64 in the forward.  Arms the int path on that
        # block (and drops its fp64_ffn).  Byte-exact; default OFF -> unchanged.
        self._arm_lea_int_snap()
        # MEGAKERNEL graph cache: (live-schedule tuple, K) -> GraphedFFNChain over the
        # tail passthrough-FFN segment (built lazily on first use of that shape).
        self._ffn_graphs: Dict[Tuple, object] = {}
        self._graph_disabled: set = set()

    def _arm_lea_int_snap(self) -> int:
        """Arm the integer LEA-address snap on the ``lea-addr-nib`` block(s) when
        ``C4_LEA_INT_SNAP`` is set.  Returns the number of blocks armed."""
        from .pos_sparse_forward import (lea_int_snap_enabled, resolve_lea_snap_dims,
                                         LEA_ADDR_NIB_BLOCK_NAME)
        for kb in self.kblocks:
            kb.b._lea_snap_dims = None
        if not lea_int_snap_enabled():
            return 0
        dims = resolve_lea_snap_dims(self.L)
        if dims is None:
            return 0
        names = list(getattr(self.L, "_block_names", []))
        n = 0
        for bi, kb in enumerate(self.kblocks):
            if bi < len(names) and names[bi] == LEA_ADDR_NIB_BLOCK_NAME \
                    and not kb.b.routed:
                kb.b._lea_snap_dims = dims
                kb.b.fp64_ffn = False        # exact integer path -> no fp64 needed
                n += 1
        return n

    def set_fp64_blocks(self, block_idxs) -> None:
        """SELECTIVE fp64 (#748): run the query-row FFN in fp64 ONLY on
        ``block_idxs``; fp32 on every other block.  ``None`` -> ALL blocks fp64
        (the a39ae2c all-fp64 baseline).  Byte-exactness must be re-verified."""
        keep = None if block_idxs is None else set(int(b) for b in block_idxs)
        for bi, kb in enumerate(self.kblocks):
            # a block armed with the integer LEA snap NEVER runs fp64 (it is exact).
            if getattr(kb.b, "_lea_snap_dims", None) is not None:
                kb.b.fp64_ffn = False
                continue
            kb.b.fp64_ffn = True if keep is None else (bi in keep)

    def arm_direct_cam(self, tbl) -> int:
        """SELF-EMU DIRECT-CAM (C4_SELFEMU_DIRECT_CAM): arm the global CAM heads of
        every block to DIRECT-GATHER the draft-resolved values in ``tbl``
        (``selfemu_direct_cam.DirectCamTable``) instead of softmax1 over the store
        rows.  Returns the number of blocks armed.  ``tbl=None`` disarms (back to the
        softmax path)."""
        from .selfemu_direct_cam import block_global_kinds
        block_names = list(getattr(self.L, "_block_names", []))
        n_armed = 0
        for bi, kb in enumerate(self.kblocks):
            if tbl is None or kb.b.global_idx.numel() == 0:
                kb._dcam_kinds = None
                kb._dcam_tbl = None
                continue
            kinds = block_global_kinds(bi, self.L, block_names, kb.b.global_idx)
            if kinds:
                kb._dcam_kinds = kinds
                kb._dcam_tbl = tbl
                n_armed += 1
            else:
                kb._dcam_kinds = None
                kb._dcam_tbl = None
        return n_armed

    def live_union(self, ops) -> List[int]:
        """The union of live blocks over the K ops (None -> full stack)."""
        s = set()
        for op in ops:
            live = self.live_index.get(op)
            if live is None:
                return list(range(self.n_blocks))
            s |= set(live)
        return sorted(s)

    # ------------------------------------------------------------------
    # #874 — PER-ROW block-skip (position-sparse block EXECUTION).
    #
    # ``forward_span`` runs the UNION of the K ops' live blocks for ALL K query
    # rows.  But block ``bi`` only AFFECTS the decode of query row ``j`` iff
    # ``ops[j]``'s own live set contains ``bi``.  For every OTHER row ``bi`` is a
    # provable nil passthrough (its op does not use that block; the row carries its
    # exact input residual — the SAME thing the union forward computes for it, since
    # a skipped block's output for a row is that row's input residual).  So block
    # ``bi`` need compute ONLY its LIVE rows.  The classic waste: a batch with one
    # DIV makes the whole 188-block divmod span "live" for the WHOLE K-batch, so the
    # union forward runs those 188 blocks for every ADD/PSH/LI row too — ~15x more
    # FFN work than the per-op weighted number.  Per-row block-skip runs the DIV
    # blocks ONLY on the (rare) DIV rows.
    # ------------------------------------------------------------------
    def per_block_rows(self, ops) -> Dict[int, List[int]]:
        """``{block_idx: [row j in 0..K-1 whose op uses this block]}`` — the per-block
        query-row mask.  A block absent from the map is used by NO row in this span
        (fully skipped).  If any op is unknown (``None`` live -> full stack), that row
        is live for EVERY block (safe fallback)."""
        full = None
        rows_by_block: Dict[int, List[int]] = {}
        for j, op in enumerate(ops):
            live = self.live_index.get(op)
            if live is None:
                if full is None:
                    full = list(range(self.n_blocks))
                live = full
            for bi in live:
                rows_by_block.setdefault(bi, []).append(j)
        return rows_by_block

    def forward_span_perrow(self, x: torch.Tensor, ops, q_idxs: List[int],
                            all_store: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
        """PER-ROW block-skip forward_span (#874): each live block computes ONLY the
        query rows whose op uses it, NOT the union of all K rows.

        Byte-identical to the K=1 SINGLE-STEP decode: a row that skips block ``bi``
        keeps its input residual at that block, which is EXACTLY what that row's op
        computes in its OWN single-step forward (that op's block-skip schedule never
        runs ``bi``).  The chain runs in block order over the shared in-place buffer,
        so a row's residual entering block ``bi`` is the same as under its single-step
        forward (every earlier block the row's op uses has already run for it, and
        every earlier block it does NOT use left its residual untouched).

        (NOTE: this is byte-exact to the TRUE per-step decode; the older UNION
        ``forward_span`` is NOT for a mixed batch — its SHL/EQ ax-recompose blocks run
        on, and corrupt, the AX band of the OTHER rows.  Per-row is both correct and
        skips the FFN work of blocks a row does not use.)"""
        store_dim = int(self.L.IS_STORE)
        if all_store is None:
            col = x[0, :, store_dim]
            all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
        rows_by_block = self.per_block_rows(ops)
        with torch.no_grad():
            buf = x.clone()
            # iterate blocks in order; each runs ONLY its live rows (a subset of the
            # K query rows).  Blocks used by no row are skipped entirely.
            for bi in sorted(rows_by_block.keys()):
                rows = rows_by_block[bi]
                sub_q = [q_idxs[j] for j in rows]
                sub_qt = torch.tensor(sub_q, device=x.device, dtype=torch.long)
                buf = self.kblocks[bi].forward(buf, sub_q, all_store=all_store,
                                               q_t=sub_qt, inplace=True)
        return buf

    # ------------------------------------------------------------------
    # #874 GROUPED per-row block-skip: collapse the eager per-block launch overhead.
    #
    # The naive ``forward_span_perrow`` runs the SAME #block-forwards as the union (a
    # block used by ANY row still launches once) — it only shrinks each GEMM to that
    # block's live-row subset.  On the eager path the launch count, not the FLOP, is
    # the wall, so the FLOP win does not show.  But the row-subsets are HIGHLY
    # structured: in a K-batch with one DIV, the ~179 ``alu-div`` blocks ALL share the
    # single DIV row's subset and are a consecutive PASSTHROUGH (FFN-only) run.  So we
    # COMPACT those rows into a [1, n_sub, D] buffer and run that whole consecutive
    # same-subset passthrough run as ONE FFN chain over n_sub rows (n_sub=1 for the
    # lone-DIV case) — collapsing 179 x 64-row GEMMs to 179 x 1-row GEMMs, and (with a
    # CUDA graph over the compacted chain) collapsing their launches too.  Byte-exact:
    # a passthrough block's FFN is position-independent, so running it on the compacted
    # live-row buffer and scattering back is identical to running it in place.
    # ------------------------------------------------------------------
    def _perrow_segments(self, ops):
        """Partition the live blocks into ordered SEGMENTS, each a maximal run of
        consecutive live blocks that are ALL passthrough (FFN-only) AND share the
        IDENTICAL query-row subset — plus singleton segments for attention blocks or
        subset boundaries.  Returns ``[(block_idxs, rows_tuple, is_ffn_chain)]`` in
        block order."""
        rows_by_block = self.per_block_rows(ops)
        live = sorted(rows_by_block.keys())
        segs: List[Tuple[List[int], Tuple[int, ...], bool]] = []
        cur: List[int] = []
        cur_rows: Optional[Tuple[int, ...]] = None
        for bi in live:
            rows = tuple(rows_by_block[bi])
            is_pass = self.kblocks[bi].b.is_passthrough
            if is_pass and cur and rows == cur_rows:
                cur.append(bi)
            else:
                if cur:
                    segs.append((cur, cur_rows, len(cur) >= 1 and
                                 all(self.kblocks[b].b.is_passthrough for b in cur)))
                cur = [bi]
                cur_rows = rows
        if cur:
            segs.append((cur, cur_rows, all(self.kblocks[b].b.is_passthrough
                                            for b in cur)))
        return segs

    def forward_span_perrow_graphed(self, x: torch.Tensor, ops, q_idxs: List[int],
                                    all_store: Optional[torch.Tensor] = None,
                                    graph: bool = True) -> torch.Tensor:
        """GROUPED per-row forward_span (#874): consecutive same-row-subset passthrough
        blocks run as ONE FFN chain over a COMPACTED [1, n_sub, D] buffer (n_sub = that
        subset's row count), and — when ``graph`` — the tail such run is CUDA-graphed
        (keyed on the block-run + n_sub).  Byte-exact to ``forward_span_perrow``: a
        passthrough block's FFN is position-independent, so gather-run-scatter over the
        live rows equals the in-place per-row run."""
        store_dim = int(self.L.IS_STORE)
        if all_store is None:
            col = x[0, :, store_dim]
            all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
        segs = self._perrow_segments(ops)
        with torch.no_grad():
            buf = x.clone()
            for seg_i, (blks, rows, is_ffn) in enumerate(segs):
                sub_q = [q_idxs[j] for j in rows]
                sub_qt = torch.tensor(sub_q, device=x.device, dtype=torch.long)
                if is_ffn and len(blks) >= 1:
                    # COMPACT the live rows, run the FFN chain over the small buffer,
                    # scatter back.  (Position-independent FFN -> byte-exact.)
                    xq = buf[:, sub_qt]                              # [1, n_sub, D]
                    key = (tuple(blks), len(sub_q))
                    gch = None
                    # CUDA graphs are GPU-only; on CPU run the compacted chain eagerly
                    # (still the per-row FLOP win, just no launch-collapse).
                    if graph and x.is_cuda and key not in self._graph_disabled:
                        gch = self._ffn_graphs.get(key)
                        if gch is None:
                            gch = GraphedFFNChain(self.kblocks, blks, len(sub_q),
                                                  x.shape[2], x.device, x.dtype)
                            if gch.try_capture():
                                self._ffn_graphs[key] = gch
                            else:
                                gch = None
                                self._graph_disabled.add(key)
                    if gch is not None:
                        fq = gch.run(xq)
                    else:
                        fq = xq
                        for bi in blks:
                            fq = self.kblocks[bi].b._ffn_qrow(fq)
                    buf[:, sub_qt] = fq
                else:
                    # attention (or mixed) block: run individually over its subset.
                    for bi in blks:
                        buf = self.kblocks[bi].forward(
                            buf, sub_q, all_store=all_store, q_t=sub_qt, inplace=True)
        return buf

    def forward_span(self, x: torch.Tensor, ops, q_idxs: List[int]) -> torch.Tensor:
        live_set = set(self.live_union(ops))
        # IS_STORE is a STRUCTURAL input role tag (set by the overlay, not written by
        # any block's FFN), so the store-row nonzero is block-invariant — compute it
        # ONCE for the whole span (one GPU->CPU sync instead of one per query per
        # block).  Byte-exact iff IS_STORE is unmodified across blocks (verified).
        store_dim = int(self.L.IS_STORE)
        col = x[0, :, store_dim]
        all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
        # Clone the (teacher-forced) input ONCE — the caller reuses ``x`` across
        # spans, so it must NOT be mutated; the block chain then runs IN PLACE on
        # this single buffer (each block writes ONLY its query rows), removing the
        # per-block full-stream clone (~29MB x #live-blocks at K=128).
        q_t = torch.tensor(q_idxs, device=x.device, dtype=torch.long)
        with torch.no_grad():
            buf = x.clone()
            for bi, kb in enumerate(self.kblocks):
                if bi in live_set:
                    buf = kb.forward(buf, q_idxs, all_store=all_store,
                                     q_t=q_t, inplace=True)
        return buf

    # ------------------------------------------------------------------
    def _tail_ffn_segment(self, live: List[int]) -> List[int]:
        """The trailing run of consecutive passthrough (FFN-only) live blocks — the
        S-independent, graphable segment (block 0's attention stays eager)."""
        seg: List[int] = []
        for bi in live:
            if self.kblocks[bi].b.is_passthrough:
                seg.append(bi)
            else:
                seg = []          # an attention block breaks the run; restart the tail
        return seg

    def forward_span_graphed(self, x: torch.Tensor, ops, q_idxs: List[int],
                             graph: bool = True) -> torch.Tensor:
        """MEGAKERNEL forward_span: run the leading (attention + interleaved) blocks
        eagerly, then REPLAY a CUDA graph over the trailing passthrough-FFN segment
        (S-independent, keyed on the schedule+K -> amortizes across the self-emu).
        Byte-exact to ``forward_span`` (same per-block SwiGLU + residual)."""
        live = sorted(self.live_union(ops))
        live_set = set(live)
        K = len(q_idxs)
        seg = self._tail_ffn_segment(live) if graph else []
        key = (tuple(live), K)
        gch = None
        if graph and seg and key not in self._graph_disabled:
            gch = self._ffn_graphs.get(key)
            if gch is None:
                D = x.shape[2]
                # #758 FUSED GATHER-GATE-SCATTER megakernel: replace the tail's dense
                # SwiGLU GEMV chain with the per-block PARALLEL Triton gather-scatter
                # chain (each block touches only its ~1-nnz-per-unit reads, not
                # D*Dff), CUDA-graphed as one replay.  ~2-3x on the DIV/MOD 186-block
                # tail vs the dense-graphed chain; byte-exact at DECODE (residual
                # L-inf ~1e-6 from multi-read/atomic accum order).  Falls back to the
                # dense GraphedFFNChain if unavailable / capture fails.
                from .fused_ffn_megakernel import (
                    fused_ffn_megakernel_enabled, GraphedTritonBlockGSChain)
                gch = None
                if fused_ffn_megakernel_enabled():
                    try:
                        cand = GraphedTritonBlockGSChain(
                            self.kblocks, seg, K, D, x.device, x.dtype)
                        if cand.try_capture():
                            gch = cand
                    except Exception:
                        gch = None
                if gch is None:
                    gch = GraphedFFNChain(self.kblocks, seg, K, D, x.device, x.dtype)
                    if not gch.try_capture():
                        gch = None
                if gch is not None:
                    self._ffn_graphs[key] = gch
                else:
                    self._graph_disabled.add(key)
        # eager prefix = every live block NOT in the graphed tail segment.
        eager = live if gch is None else live[:len(live) - len(seg)]
        store_dim = int(self.L.IS_STORE)
        col = x[0, :, store_dim]
        all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
        q_t = torch.tensor(q_idxs, device=x.device, dtype=torch.long)
        with torch.no_grad():
            buf = x.clone()
            for bi in eager:
                buf = self.kblocks[bi].forward(buf, q_idxs, all_store=all_store,
                                               q_t=q_t, inplace=True)
            if gch is not None:
                xq = buf[:, q_t]                                # [1,K,D]
                fq = gch.run(xq)
                buf[:, q_t] = fq
        return buf

    # -- accounting: max bounded K/V rows over the K query rows / live blocks --
    def kv_rows_max(self, x: torch.Tensor, q_idxs: List[int], ops) -> int:
        live = self.live_union(ops)
        W = self.window
        n = 0
        for bi in live:
            b = self.kblocks[bi].b
            if b.is_passthrough:
                continue
            for q in q_idxs:
                rows = 0
                if b.local_idx.numel() > 0:
                    rows = max(rows, min(W, q + 1))
                if b.global_idx.numel() > 0:
                    rows = max(rows, int(b._store_rows(x, q).numel()))
                n = max(n, rows)
        return n


__all__ = ["KBatchBoundedBlock", "KBatchBoundedRunner",
           "FP_FRAGILE_BLOCK_NAMES", "default_fp64_block_idxs"]
