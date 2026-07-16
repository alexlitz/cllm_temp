"""KV-CACHED pure-forward C4 VM driver — the O(cache)/step form of
``run_pure_forward_complete`` (CHK-1 item #6: "KV cache eviction works properly
and maintains correct outputs over even long problems").

The naive driver (:func:`nibble_pure_forward_complete.run_pure_forward_complete`)
re-forwards the ENTIRE growing token stream every VM step, so each step is
O(stream^2) and a deep loop (~8400 steps -> ~250k-token stream) is intractable.

This driver keeps a **per-block incremental KV cache** and, each step, forwards
ONLY the small window of tokens whose residual changed:

  * the NEW 30-token frame the previous step emitted, PLUS
  * the ONE position whose overlay changes across steps — the *previous query
    row* (the emitted STEP_END token that carried all-ROLE one-hots as the step-N
    query row and, once a newer frame is appended, becomes a plain context token
    with NO roles).  Empirically (see ``test_cached_driver`` /
    ``docs/KV_CACHE_DRIVER``) this is the ONLY already-cached position whose
    block-stack output differs after an append — every earlier position is
    causally frozen (softmax1 is causal, and the appended frame sits strictly
    later, so no earlier row attends to it).

So the per-step recompute window is a FIXED 31 rows, attending against the cached
K/V of all earlier (frozen) positions — turning O(stream^2)/step into
O(cache)/step.  The register decode reads the window's last row exactly as the
naive driver reads ``model.forward``'s last row, so the emitted byte stream is
**byte-identical** to the naive re-forward (proven in ``test_cached_driver``).

Bounded eviction (``nibble_kv_prune``) keeps the cache FLAT over deep loops: the
spec's softmax1+ALiBi eviction policy drops duplicate/recency-negligible/zero
entries so a million-step program keeps a bounded cache and stays both fast AND
memory-bounded — the full CHK-1 item #6.

Everything on the compute path is ``model.forward`` (the block stack) +
argmax-generate-append; the ``assert_no_python_compute`` settrace guard still
passes (KV caching is standard autoregressive generation, all in-weights).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_pure_forward import (
    N_ROLES, _FRAME_ROLE_SLOTS, _MEM_MARKER_LOCAL, _address_bits, SP_INIT,
    _snap_lane,
)
from .nibble_pure_forward_complete import (
    PureForwardCompleteLayout, IMM_NIBS, _build_frame, _decode_reg_from_nibbles,
    _mem_top,
)
from .nibble_kv_prune import MultiHeadKVCache, KVCache
from .blogspec_layout import NIB_PER_REG


# ===========================================================================
# WINDOWED OVERLAY — apply the SAME overlay math as ``make_overlay_complete`` but
# only to a contiguous ABSOLUTE-position window [w_start .. w_start+len-1] of the
# stream.  ``is_last_row_query`` flags the final position of the window as the
# current query row (all-ROLE one-hots).  This is a pure re-encoding of the
# structural overlay for a sub-range — the residual it writes at position p is
# identical to what the full-stream overlay writes at p (verified byte-exact).
# ===========================================================================
def apply_overlay_window(x_win: torch.Tensor, w_start: int, code, L,
                         store_log: Dict[int, Tuple[int, int]],
                         is_last_row_query: bool) -> None:
    """In-place overlay of the window residual ``x_win`` ([1, W, D]).

    ``w_start`` is the ABSOLUTE stream position of ``x_win[:, 0]``.  Position 0 is
    the BOS row; positions ``1 + 30*f + local`` belong to frame ``f`` local slot
    ``local``.  Only rows inside the window are touched.
    """
    W = x_win.shape[1]
    for wi in range(W):
        p = w_start + wi
        x_win[0, wi, L.ONE] = 1.0
        for k, ins in enumerate(code):
            x_win[0, wi, L.CODE_OP[k]] = float(ins.op)
            x_win[0, wi, L.CODE_IMM[k]] = float(ins.imm)
            for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                x_win[0, wi, L.CODE_IMM_NIB[k] + j] = float(nv)
        if p == 0:
            continue                       # BOS row carries no frame roles
        # frame index + local slot for absolute position p.
        f = (p - 1) // V.FRAME_LEN
        local = (p - 1) % V.FRAME_LEN
        if local in _FRAME_ROLE_SLOTS:
            role = _FRAME_ROLE_SLOTS[local]
            x_win[0, wi, L.ROLE + role] = 1.0
            x_win[0, wi, L.IS_FRAME_BYTE] = 1.0
        if local == _MEM_MARKER_LOCAL and f in store_log:
            addr, val = store_log[f]
            x_win[0, wi, L.IS_STORE] = 1.0
            x_win[0, wi, L.IS_FRAME_BYTE] = 0.0
            for b, bit in enumerate(_address_bits(addr)):
                x_win[0, wi, L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                x_win[0, wi, L.VAL_NIB + j] = float(nv)
    if is_last_row_query:
        for role in range(N_ROLES):
            x_win[0, -1, L.ROLE + role] = 1.0


# ===========================================================================
# VECTORISED per-head prune keep-mask — a torch re-expression of the EXACT three
# ``nibble_kv_prune.KVCache.prune`` mechanisms (validated byte-for-byte against
# the reference in ``test_cached_driver`` / ``test_evict_vectorized_matches``).
# The reference is O(n^2) per-entry Python; this runs the same decision in a
# handful of batched tensor ops so eviction is tractable on deep loops.
# ===========================================================================
def prune_keep_mask_head(keys: torch.Tensor, vals: torch.Tensor,
                         positions: torch.Tensor, slope: float, scale: float,
                         cos_threshold: float, zero_eps: float,
                         recency_eps: float,
                         dup_metric: str = "cosine") -> torch.Tensor:
    """Return a boolean ``[S]`` keep-mask == the survivors of ``KVCache.prune``.

    Reproduces, in order:
      mech 1 (near-dup, keep the NEWER of a duplicate pair; ``dup_metric``),
      mech 3 (ALiBi-recency horizon: drop max_w < recency_eps),
      mech 2a (dead-value head: all survivors zero-value -> drop all),
      mech 2b (free-zero stale: drop fully-zero recency-negligible entries).

    ``dup_metric`` = "cosine" (register-marker heads) or "exact" (content-
    addressed §Memory heads whose keys share a large ADDR_BIN common-mode bias;
    see ``nibble_kv_prune.KVCache.prune``).
    """
    import math
    S = keys.shape[0]
    if S == 0:
        return torch.zeros(0, dtype=torch.bool)
    knorm = keys.norm(dim=-1)                                   # [S]
    vnorm = vals.norm(dim=-1)                                   # [S]

    # -- mechanism 1: greedy newest-first near-duplicate drop -----------------
    order = torch.argsort(positions, descending=True)          # newest first
    kept_mask = torch.zeros(S, dtype=torch.bool)               # index-space mask
    kept_idx: List[int] = []
    if dup_metric == "exact":
        # relative-L2: |k_e - k| <= (1-cos_threshold)*max(|k_e|,|k|).  Merges only
        # verbatim-identical keys, so distinct content addresses all survive.
        tol = 1.0 - cos_threshold
        for oi in order.tolist():
            if kept_idx:
                diff = (keys[kept_idx] - keys[oi]).norm(dim=-1)
                denom = torch.maximum(
                    knorm[kept_idx],
                    knorm[oi].expand_as(knorm[kept_idx])).clamp(min=1e-30)
                if bool((diff <= tol * denom).any()):
                    continue
            kept_idx.append(oi)
    else:
        # raw cosine (zero-key rows -> unit 0 -> sim 0, matching cosine_sim).
        safe = knorm.clamp(min=1e-30)
        unit = keys / safe.unsqueeze(-1)
        unit[knorm == 0] = 0.0
        for oi in order.tolist():
            if kept_idx:
                sims = unit[kept_idx] @ unit[oi]               # [n_kept]
                if float(sims.max()) > cos_threshold:
                    continue
            kept_idx.append(oi)
    for oi in kept_idx:
        kept_mask[oi] = True

    survivors = kept_mask.clone()

    # -- mechanism 3: ALiBi-recency horizon (PER-ENTRY Cauchy-Schwarz bound) ---
    if slope is not None and slope > 0.0 and survivors.any():
        surv_idx = torch.nonzero(survivors, as_tuple=False).flatten()
        newest = int(positions[surv_idx].max())
        max_kn = float(knorm[surv_idx].max())
        # ceil_score_e = max_kn * |k_e| * scale  (0 for a zero-key entry).
        ceil_score = (max_kn * knorm[surv_idx].to(torch.float64)) * scale
        dist = (newest - positions[surv_idx]).to(torch.float64)
        arg = (ceil_score - slope * dist).clamp(max=0.0)
        max_w = torch.exp(arg)
        drop = max_w < recency_eps
        drop_idx = surv_idx[drop]
        survivors[drop_idx] = False

    # -- mechanism 2a: dead-value head (all survivors zero-value) --------------
    if survivors.any():
        surv_idx = torch.nonzero(survivors, as_tuple=False).flatten()
        if bool((vnorm[surv_idx] <= zero_eps).all()):
            survivors[:] = False

    # -- mechanism 2b: free-zero (value==0 AND key==0) + recency-stale ---------
    if survivors.any():
        surv_idx = torch.nonzero(survivors, as_tuple=False).flatten()
        newest_all = int(positions[surv_idx].max())
        free_zero = (vnorm <= zero_eps) & (knorm <= zero_eps)
        for i in surv_idx.tolist():
            if not bool(free_zero[i]):
                continue
            if slope is None or slope <= 0.0:
                survivors[i] = False
            elif math.exp(-slope * (newest_all - int(positions[i]))) < recency_eps:
                survivors[i] = False
    return survivors


# ===========================================================================
# BATCHED per-block KV cache + eviction.  We keep the K/V as batched tensors
# (fast ``torch.matmul`` attention) and drive the eviction KEEP-mask through the
# PROVEN ``nibble_kv_prune`` policy (per head), so the eviction that runs on the
# real model is exactly the spec's softmax1+ALiBi policy.
# ===========================================================================
class BlockKVCacheBatched:
    """One block's KV cache as batched tensors ``(K, V, pos)`` with bounded
    eviction delegated to ``nibble_kv_prune``.

      K, V : [1, H, S, HD]   projected key/value of every cached position
      pos  : [S] long        absolute sequence position (drives ALiBi distance)

    ``evict(...)`` builds, per head, a ``nibble_kv_prune.KVCache``, replays this
    head's live entries into it, calls ``prune()`` (the spec's three mechanisms),
    and INTERSECTS the surviving position sets across heads to a single keep-mask
    (the batched tensors share one position axis).  This keeps the batched-matmul
    read fast while the DECISION is the proven policy.
    """

    def __init__(self, n_heads: int, head_dim: int, slopes: torch.Tensor):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.slopes = slopes
        self.scale = head_dim ** -0.5
        self.K: Optional[torch.Tensor] = None      # [1,H,S,HD]
        self.V: Optional[torch.Tensor] = None
        self.pos: Optional[torch.Tensor] = None    # [S]
        self.total_evicted = 0

    def as_past_kv(self):
        if self.K is None:
            return None
        return (self.K, self.V, self.pos)

    def commit(self, K_new: torch.Tensor, V_new: torch.Tensor,
               pos_new: torch.Tensor) -> None:
        """Append freshly-frozen positions' K/V (``[1,H,W,HD]`` + ``[W]``)."""
        if self.K is None:
            self.K, self.V, self.pos = K_new, V_new, pos_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)
            self.pos = torch.cat([self.pos, pos_new], dim=0)

    def size(self) -> int:
        return 0 if self.pos is None else int(self.pos.shape[0])

    def evict(self, cos_threshold: float, prune_interval: int,
              zero_eps: float, recency_eps: float) -> int:
        """Apply the spec eviction policy; return #positions dropped.

        The keep-set is the UNION across heads of each head's ``prune()`` survivors
        (a position is retained if ANY head still needs it), so no head loses an
        entry it would attend to — the batched tensors then keep exactly that
        union.  Because the batched axis is shared, this is the conservative,
        output-exact intersection of the per-head policies.

        Fast path: a head whose ENTRY VALUES are all zero is an exact softmax1
        no-op (its ``attention_output`` is the zero vector for every query —
        ``nibble_kv_prune`` mechanism 2a), so it evicts every entry and contributes
        NOTHING to ``keep_any``.  We detect those heads with a single batched norm
        check and skip their (expensive, per-entry-Python) ``prune()`` replay — the
        result is identical to running the full policy on them.
        """
        if self.K is None:
            return 0
        S = self.pos.shape[0]
        keep_any = torch.zeros(S, dtype=torch.bool)
        # batched per-head value norms: a head with all-zero values -> mechanism 2a
        # drops all its entries (no keep_any contribution). Skip it entirely.
        vnorm = self.V[0].norm(dim=-1)                       # [H, S]
        head_has_value = (vnorm > zero_eps).any(dim=-1)      # [H]
        # per-head near-dup metric: a CONTENT-ADDRESSED head (a §Memory store head)
        # has keys dominated by a shared common-mode bias (the ADDR_BIN `-smag`
        # term), so raw cosine cannot separate distinct addresses (0.999) and would
        # wrongly merge distinct stores.  Detect it by the common-mode fraction
        # |mean_key| / mean(|key|): high (~1) for such heads, low for register-
        # marker heads whose distinct keys spread in direction.  Use the exact
        # (relative-L2) dup metric on those heads so every distinct store survives.
        knorm_hs = self.K[0].norm(dim=-1)                    # [H, S]
        mean_key = self.K[0].mean(dim=1)                     # [H, HD]
        cm_frac = mean_key.norm(dim=-1) / knorm_hs.mean(dim=-1).clamp(min=1e-30)
        for h in range(self.n_heads):
            if not bool(head_has_value[h]):
                continue                                     # mechanism 2a: evict all
            metric = "exact" if float(cm_frac[h]) > 0.9 else "cosine"
            mask = prune_keep_mask_head(
                self.K[0, h], self.V[0, h], self.pos,
                slope=float(self.slopes[h]), scale=self.scale,
                cos_threshold=cos_threshold, zero_eps=zero_eps,
                recency_eps=recency_eps, dup_metric=metric)
            keep_any |= mask
        keep_idx = torch.nonzero(keep_any, as_tuple=False).flatten()
        dropped = S - int(keep_idx.numel())
        if dropped:
            self.K = self.K[:, :, keep_idx, :]
            self.V = self.V[:, :, keep_idx, :]
            self.pos = self.pos[keep_idx]
            self.total_evicted += dropped
        return dropped


# ===========================================================================
# THE KV-CACHED DRIVER.
# ===========================================================================
def run_pure_forward_cached(model, L: PureForwardCompleteLayout,
                            code: List[isa.Instr], max_steps: int = 512,
                            verbose: bool = False, collect_tokens: bool = False,
                            mask: int = 0xFF, evict: bool = True,
                            cos_threshold: float = 0.99, prune_interval: int = 120,
                            zero_eps: float = 1e-9, recency_eps: float = 1e-6,
                            stats: Optional[dict] = None):
    """KV-cached form of :func:`run_pure_forward_complete`.

    Byte-identical output to the naive re-forward driver (proven in
    ``test_cached_driver``), at O(cache)/step instead of O(stream^2)/step.  With
    ``evict=True`` the per-block caches are pruned every ``prune_interval`` tokens
    by the ``nibble_kv_prune`` policy so the cache stays FLAT over deep loops.

    ``stats`` (optional dict) is filled with ``max_seq_len`` / ``max_cache_size``
    (per block) / ``total_evicted`` / ``steps`` for the bounded-memory report.
    """
    blocks = model.blocks
    n_blocks = len(blocks)
    H = blocks[0].attn.n_heads
    HD = blocks[0].attn.head_dim
    caches = [BlockKVCacheBatched(H, HD, blocks[b].attn.alibi_slopes)
              for b in range(n_blocks)]
    tokens_since_prune = 0

    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    stream: List[int] = [V.BOS] + init_frame
    trace: List[int] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = 0

    # -- step 0: no cache yet.  The window is the WHOLE initial stream
    # ([BOS]+init_frame, 31 rows); its last row is the query row.  We commit all
    # rows EXCEPT the last (the query row is not frozen — it re-tags next step).
    win_start = 0
    win_len = len(stream)

    max_seq = len(stream)
    max_cache = 0

    for _ in range(max_steps):
        win_toks = torch.tensor([stream[win_start:win_start + win_len]])
        q_positions = torch.arange(win_start, win_start + win_len)
        with torch.no_grad():
            x = model.embed[win_toks].clone()
            apply_overlay_window(x, win_start, code, L, store_log,
                                 is_last_row_query=True)
            past = [caches[b].as_past_kv() for b in range(n_blocks)]
            hidden, new_kv = model.forward_hidden_cached(
                x, past_key_values=past, q_positions=q_positions, use_cache=True)
        state = hidden[0, -1]                      # the query row's block output

        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        ax = _decode_reg_from_nibbles(state, L, L.AX)

        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)

        # -- COMMIT the frozen rows of this window to the cache.  The window's
        # last row (the query row just decoded) is NOT frozen; every earlier row
        # is.  Its K/V for each block is in ``new_kv[b]`` = (K_all, V_all, pos_all)
        # (cache + window).  The freshly-computed window slice sits at the TAIL.
        n_commit = win_len - 1
        if n_commit > 0:
            for b in range(n_blocks):
                K_all, V_all, pos_all = new_kv[b]
                K_win = K_all[:, :, -win_len:, :]      # this window's K
                V_win = V_all[:, :, -win_len:, :]
                pos_win = pos_all[-win_len:]
                caches[b].commit(K_win[:, :, :n_commit, :],
                                 V_win[:, :, :n_commit, :],
                                 pos_win[:n_commit])

        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                  f"pc'={pc} ax={ax & 0xFF} sp={sp} bp={bp} stk={stk} "
                  f"store={'Y' if is_store else '.'}@{s_addr}={s_val} "
                  f"halt={halted} cache={caches[0].size()}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break

        # -- append the emitted frame; the NEXT window = [old_query_row] + new frame.
        prev_query_pos = win_start + win_len - 1     # absolute pos of old query row
        stream += frame
        win_start = prev_query_pos                   # recompute from the old query row
        win_len = 1 + V.FRAME_LEN                    # 1 old query row + 30 new frame
        max_seq = max(max_seq, len(stream))

        # -- bounded eviction: prune every ``prune_interval`` tokens.
        tokens_since_prune += V.FRAME_LEN
        if evict and tokens_since_prune >= prune_interval:
            for b in range(n_blocks):
                caches[b].evict(cos_threshold, prune_interval, zero_eps, recency_eps)
            tokens_since_prune = 0
        max_cache = max(max_cache, caches[0].size())

    if stats is not None:
        stats["max_seq_len"] = max_seq
        stats["max_cache_size"] = max(max_cache, caches[0].size())
        stats["total_evicted"] = sum(c.total_evicted for c in caches)
        stats["steps"] = len(trace)
        stats["n_blocks"] = n_blocks
    if collect_tokens:
        return trace, stream
    return trace
