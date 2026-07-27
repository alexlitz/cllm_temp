"""BOUNDED-KV position-sparse forward — collapse the K/V-over-S floor.

The composed position-sparse path (``pos_sparse_forward.PositionSparseRunner``)
runs Q / softmax / context / W_o / FFN at the SINGLE query row but STILL projects
K and V over ALL ``S`` stream rows (``attn.W_k.linear(x)`` is ``[B,S,D]@[D,D]``).
a39ae2c measured that this K/V-over-S GEMM is ~65 % of the ms/step and grows
LINEARLY with S — the true floor.  But only a BOUNDED set of rows is ever attended
with non-zero weight:

  * the register-INGEST heads (ALiBi slope ``INGEST_RECENCY`` / ``WIDE_INGEST_RECENCY``)
    read only the LATEST frame — measured window <= 28 tokens < one 30-token step.
    Every key older than the window has softmax1 weight EXACTLY 0 (recency ALiBi +
    the huge role-exact-match drive the tail to ZFOD).  So a LOCAL head needs K/V
    over only the last ``W`` rows.
  * the memory / stack / LEV CAM heads (ALiBi slope ``MEM_ALIBI_SLOPE``) are
    address-CAMs that key ONLY ``IS_STORE`` rows: a non-store row's store-role gate
    channel ``cR = ADDR_BITS+1`` keys ``-p`` while a load/pop/lev query keys ``+p``
    there, so a non-store row scores ``-PEN`` and its softmax1 weight is EXACTLY 0.
    So a GLOBAL head needs K/V over only the STORE rows (the ~14 live working set).
  * every OTHER head is a ZERO-VALUE head (``W_v==0`` or ``W_o==0``): its attention
    output is ``0`` no matter which keys it reads, so it needs NO K/V at all.

``BoundedBlock`` computes K/V over ONLY those bounded row subsets, per head group,
instead of over all S rows.  The softmax1 over the bounded subset equals the
softmax1 over all S rows because every dropped row's true weight is 0 — BYTE-EXACT
(verified end-to-end vs the dense-over-positions reference on the battery + a deep
nested loop).  The K/V GEMM shrinks from ``[S,D]@[D,D]`` to ``[W+n_store, D]@[D,D]``
— RUNTIME-INDEPENDENT (flat in S) once W and the store working-set are bounded.

This composes the position-sparse lever (query-row-only Q/O/FFN) with the
exact-evict / content-bound insight (bounded K/V rows) on ONE byte-exact forward.

Gate: ``C4_POS_SPARSE`` (the composed path's flag); this module is a drop-in runner
the bench selects.  The golden flag-OFF build is untouched.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .blogspec_model import softmax1
from .step_block_skip import build_live_index


# ---------------------------------------------------------------------------
# Head classification (reuses local_attention's structural rules on the compact
# streaming model).  A head is LOCAL if its ALiBi slope is an ingest slope; GLOBAL
# if it is a live value head with a non-ingest slope; DEAD (zero-V) otherwise.
# ---------------------------------------------------------------------------
def _ingest_slopes() -> Tuple[float, float]:
    from .nibble_pure_forward import INGEST_RECENCY, WIDE_INGEST_RECENCY
    return float(INGEST_RECENCY), float(WIDE_INGEST_RECENCY)


def _store_gate_channel() -> int:
    from .blogspec_memory import ADDR_BITS
    return int(ADDR_BITS) + 1


def _dense(w) -> torch.Tensor:
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


def classify_block_heads(attn, slope_tol: float = 1e-3):
    """Return ``(local_idx, global_idx, dead_idx)`` LongTensors of head indices.

    * DEAD  : ``W_v`` rows or ``W_o`` cols all zero -> output 0 regardless of keys.
    * LOCAL : live value head whose ALiBi slope is an ingest slope (proven-local,
              window <= 28 tok).
    * GLOBAL: every other live value head (memory / stack / LEV CAM, kept full-history
              but content-bounded to store rows).
    """
    H, HD = attn.n_heads, attn.head_dim
    dev = attn.alibi_slopes.device
    wv = _dense(attn.W_v).to(dev)
    wo = _dense(attn.W_o).to(dev)
    ings = _ingest_slopes()
    local: List[int] = []
    glob: List[int] = []
    dead: List[int] = []
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        live = bool((wv[sl, :] != 0).any()) and bool((wo[:, sl] != 0).any())
        if not live:
            dead.append(h)
            continue
        slope = float(attn.alibi_slopes[h])
        if any(abs(slope - s) <= slope_tol for s in ings):
            local.append(h)
        else:
            glob.append(h)
    lt = torch.tensor(local, dtype=torch.long, device=dev)
    gt = torch.tensor(glob, dtype=torch.long, device=dev)
    dt = torch.tensor(dead, dtype=torch.long, device=dev)
    return lt, gt, dt


# ---------------------------------------------------------------------------
# The bounded-KV query-row-only attention.
# ---------------------------------------------------------------------------
class BoundedBlock:
    """One block with the BOUNDED-KV query-row-only forward.

    Precomputes the per-block head classification (local / global / dead) and the
    store-gate channel column.  ``forward`` projects K/V over ONLY the last-``W``
    rows (local heads) and the store rows (global heads); dead heads are skipped.
    Byte-exact to ``_block_query_only`` (dense-over-positions K/V) for the query row.
    """

    def __init__(self, blk, window: int, store_dim: int):
        self.blk = blk
        self.attn = blk.attn
        self.ffn = blk.ffn
        self.routed = blk._routed
        self.window = int(window)
        self.store_dim = int(store_dim)          # residual dim = L.IS_STORE
        A = blk.attn
        self.H, self.HD, self.D = A.n_heads, A.head_dim, A.dim
        self.scale = A.scale
        self.alibi = A.alibi_slopes
        self.cR = _store_gate_channel()
        self.local_idx, self.global_idx, self.dead_idx = classify_block_heads(A)
        # a block is a pure passthrough (no live value head) -> attention output == x.
        self.is_passthrough = (self.local_idx.numel() == 0
                               and self.global_idx.numel() == 0)

    # -- FFN (query-row fp64, matching pos_sparse_forward) ------------------
    def _ffn_qrow(self, aout):
        if self.routed:
            return self.ffn(aout)
        F_ = self.ffn
        Wu, Wg, Wd = _dense(F_.W_up), _dense(F_.W_gate), _dense(F_.W_down)
        xq64 = aout.double()
        up = F.linear(xq64, Wu.double()) + F_.b_up.double()
        gate = F.linear(xq64, Wg.double()) + F_.b_gate.double()
        hidden = F.silu(up) * gate
        out = xq64 + F.linear(hidden, Wd.double()) + F_.b_down.double()
        return out.to(aout.dtype)

    def _store_rows(self, x: torch.Tensor, q_idx: int) -> torch.Tensor:
        """Indices of rows a GLOBAL head can attend to: the store rows (IS_STORE!=0)
        at or before the query, plus the query row itself (softmax1 self / causal)."""
        col = x[0, :, self.store_dim]                     # [S]
        rows = torch.nonzero(col != 0, as_tuple=False).flatten()
        rows = rows[rows <= q_idx]
        # always include the query row (self-attention causal row).
        if rows.numel() == 0 or int(rows[-1]) != q_idx:
            rows = torch.cat([rows, torch.tensor([q_idx], device=x.device)])
        return rows

    def _attend(self, x, xq, q_idx, head_idx, rows):
        """softmax1+ALiBi attention output for ``head_idx`` over the key ``rows``.
        Returns the context ``[B,h,1,HD]`` for those heads.  ``rows`` is a
        LongTensor of absolute key positions (all <= q_idx)."""
        B = x.shape[0]
        HD = self.HD
        xr = x[:, rows]                                          # [B,R,D]
        # project K/V over ONLY the bounded rows for these heads.
        K = self.attn.W_k.linear(xr).view(B, -1, self.H, HD)[:, :, head_idx]
        V = self.attn.W_v.linear(xr).view(B, -1, self.H, HD)[:, :, head_idx]
        K = K.transpose(1, 2)                                   # [B,h,R,HD]
        V = V.transpose(1, 2)
        Q = self.attn.W_q.linear(xq).view(B, 1, self.H, HD)[:, :, head_idx]
        Q = Q.transpose(1, 2)                                   # [B,h,1,HD]
        sc = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,h,1,R]
        dist = (rows.float() - float(q_idx)).abs()              # [R]  (rows<=q)
        sc = sc - self.alibi[head_idx].view(1, -1, 1, 1) * dist.view(1, 1, 1, -1)
        # causal: rows are all <= q_idx already, so no future mask needed.
        a = softmax1(sc, dim=-1)                                # [B,h,1,R]
        ctx = torch.matmul(a, V)                                # [B,h,1,HD]
        return ctx

    def forward(self, x: torch.Tensor, q_idx: int) -> torch.Tensor:
        B, S, D = x.shape
        HD = self.HD
        xq = x[:, q_idx:q_idx + 1]                              # [B,1,D]
        out = x.clone()
        if self.is_passthrough:
            aout = xq                                           # attention out == x
        else:
            # full ctx buffer over ALL heads; dead heads contribute 0 (W_o zero there).
            ctx_full = x.new_zeros(B, self.H, 1, HD)
            W = self.window
            lo = max(0, q_idx - W + 1)                          # window = last W rows
            win_rows = torch.arange(lo, q_idx + 1, device=x.device)
            if self.local_idx.numel() > 0:
                ctx_full[:, self.local_idx] = self._attend(
                    x, xq, q_idx, self.local_idx, win_rows)
            if self.global_idx.numel() > 0:
                srows = self._store_rows(x, q_idx)
                ctx_full[:, self.global_idx] = self._attend(
                    x, xq, q_idx, self.global_idx, srows)
            ctx = ctx_full.transpose(1, 2).contiguous().view(B, 1, D)
            aout = xq + self.attn.W_o.linear(ctx)
        out[:, q_idx:q_idx + 1] = aout
        # -- FFN (query row) ------------------------------------------------
        fq = self._ffn_qrow(aout)
        out[:, q_idx:q_idx + 1] = fq
        return out


class BoundedPosSparseRunner:
    """Position-sparse runner with BOUNDED K/V (local window + store-only global).

    Drop-in for ``PositionSparseRunner``: per decoded op, run ONLY that op's live
    blocks, and within each live block project K/V over the last-``W`` rows (local
    heads) and the store rows (global heads) instead of all S rows.  Byte-exact for
    the decoded query row; the K/V GEMM is now bounded (flat in S)."""

    def __init__(self, model, L, window: int = 64):
        self.model = model
        self.blocks = model.blocks
        self.L = L
        self.window = int(window)
        self.live_index = build_live_index(model, L)
        self.n_blocks = len(model.blocks)
        store_dim = int(L.IS_STORE)
        self.bounded = [BoundedBlock(b, window, store_dim) for b in model.blocks]

    def live_count(self, op) -> int:
        return len(self.live_index.get(op, self.live_index[None]))

    def forward(self, x: torch.Tensor, op) -> torch.Tensor:
        live = self.live_index.get(op)
        q = x.shape[1] - 1
        with torch.no_grad():
            if live is None:
                for bb in self.bounded:
                    x = bb.forward(x, q)
                return x
            live_set = set(live)
            for bi, bb in enumerate(self.bounded):
                if bi in live_set:
                    x = bb.forward(x, q)
        return x

    # -- accounting: bounded K/V row count per op (max over live blocks) -----
    def kv_rows(self, x: torch.Tensor, op) -> int:
        live = self.live_index.get(op) or list(range(self.n_blocks))
        q = x.shape[1] - 1
        W = self.window
        n = 0
        for bi in live:
            bb = self.bounded[bi]
            if bb.is_passthrough:
                continue
            rows = 0
            if bb.local_idx.numel() > 0:
                rows = max(rows, min(W, q + 1))
            if bb.global_idx.numel() > 0:
                rows = max(rows, int(bb._store_rows(x, q).numel()))
            n = max(n, rows)
        return n


__all__ = ["BoundedBlock", "BoundedPosSparseRunner", "classify_block_heads"]
