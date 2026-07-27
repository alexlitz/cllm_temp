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

    # ------------------------------------------------------------------
    def _attend_multi(self, x, q_idxs, head_idx, key_rows_per_q):
        """softmax1+ALiBi context for ``head_idx`` at each query in ``q_idxs``.

        ``key_rows_per_q`` is a list (len == len(q_idxs)) of LongTensors, the key
        rows for that query (all <= its q).  Returns ``[B, Q, h, HD]`` where Q ==
        len(q_idxs).  Ragged key counts are handled by PADDING to the max key count
        and masking the pad keys to softmax1 weight 0 (score -inf)."""
        b = self.b
        B = x.shape[0]
        HD, H = b.HD, b.H
        Q = len(q_idxs)
        hset = head_idx
        nh = int(hset.numel())
        Rmax = max(int(kr.numel()) for kr in key_rows_per_q)
        # ragged per-q key layout via a [Q, Rmax] key-index matrix + a valid mask
        # (pad with row 0, mask it off so its softmax1 weight is 0).
        key_idx = x.new_zeros(Q, Rmax, dtype=torch.long)
        valid = torch.zeros(Q, Rmax, dtype=torch.bool, device=x.device)
        dist = x.new_zeros(Q, Rmax)
        q_t = torch.tensor(q_idxs, device=x.device, dtype=torch.long)
        for j, kr in enumerate(key_rows_per_q):
            r = int(kr.numel())
            key_idx[j, :r] = kr
            valid[j, :r] = True
            dist[j, :r] = (kr.float() - float(q_idxs[j])).abs()
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

    # ------------------------------------------------------------------
    def _store_rows_for(self, x, q, all_store):
        """Store rows a GLOBAL head attends for query ``q``: the precomputed
        IS_STORE rows ``<= q`` + the query row itself.  ``all_store`` is the ONE
        precomputed nonzero (shared across queries) — avoids a per-query sync."""
        rows = all_store[all_store <= q]
        if rows.numel() == 0 or int(rows[-1]) != q:
            rows = torch.cat([rows, torch.tensor([q], device=x.device)])
        return rows

    def forward(self, x: torch.Tensor, q_idxs: List[int],
                all_store: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = self.b
        B, S, D = x.shape
        HD, H = b.HD, b.H
        Q = len(q_idxs)
        q_t = torch.tensor(q_idxs, device=x.device, dtype=torch.long)
        out = x.clone()
        xq = x[:, q_t]                                                   # [B,Q,D]
        if b.is_passthrough:
            aout = xq
        else:
            ctx_full = x.new_zeros(B, Q, H, HD)
            W = b.window
            if b.local_idx.numel() > 0:
                lrows = [torch.arange(max(0, q - W + 1), q + 1, device=x.device)
                         for q in q_idxs]
                ctx_full[:, :, b.local_idx] = self._attend_multi(
                    x, q_idxs, b.local_idx, lrows)
            if b.global_idx.numel() > 0:
                if all_store is None:
                    col = x[0, :, b.store_dim]
                    all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
                srows = [self._store_rows_for(x, q, all_store) for q in q_idxs]
                ctx_full[:, :, b.global_idx] = self._attend_multi(
                    x, q_idxs, b.global_idx, srows)
            ctx = ctx_full.reshape(B, Q, D)
            aout = xq + b.attn.W_o.linear(ctx)
        out[:, q_t] = aout
        # FFN over the K query rows (one batched K-row FFN).
        fq = b._ffn_qrow(aout)                                          # [B,Q,D]
        out[:, q_t] = fq
        return out


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

    def __init__(self, model, L, window: int = 64):
        self.model = model
        self.blocks = model.blocks
        self.L = L
        self.window = int(window)
        self.live_index = build_live_index(model, L)
        self.n_blocks = len(model.blocks)
        store_dim = int(L.IS_STORE)
        self.kblocks = [KBatchBoundedBlock(b, window, store_dim) for b in model.blocks]

    def live_union(self, ops) -> List[int]:
        """The union of live blocks over the K ops (None -> full stack)."""
        s = set()
        for op in ops:
            live = self.live_index.get(op)
            if live is None:
                return list(range(self.n_blocks))
            s |= set(live)
        return sorted(s)

    def forward_span(self, x: torch.Tensor, ops, q_idxs: List[int]) -> torch.Tensor:
        live_set = set(self.live_union(ops))
        # IS_STORE is a STRUCTURAL input role tag (set by the overlay, not written by
        # any block's FFN), so the store-row nonzero is block-invariant — compute it
        # ONCE for the whole span (one GPU->CPU sync instead of one per query per
        # block).  Byte-exact iff IS_STORE is unmodified across blocks (verified).
        store_dim = int(self.L.IS_STORE)
        col = x[0, :, store_dim]
        all_store = torch.nonzero(col != 0, as_tuple=False).flatten()
        with torch.no_grad():
            for bi, kb in enumerate(self.kblocks):
                if bi in live_set:
                    x = kb.forward(x, q_idxs, all_store=all_store)
        return x

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


__all__ = ["KBatchBoundedBlock", "KBatchBoundedRunner"]
