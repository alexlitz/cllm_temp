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

import os

import torch

from . import isa
from . import blogspec_vocab as V
from . import nibble_filesys as _FS   # OPEN/READ/CLOS/PRTF via the TOOL_CALL boundary
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


# Kill-switch / A-B bench baseline for the mechanism-1 near-dup selection.  When set
# (``C4_KV_PRUNE_FULL_COSINE=1``) the batched prune uses the ORIGINAL full O(S^2)
# cosine matmul + greedy over ALL rows instead of the two-tier exact-dedup +
# cosine-on-representatives collapse.  Both give the BYTE-IDENTICAL survivor set
# (test_kv_cache_equivalence); the flag exists only to (a) A-B benchmark the prune
# speedup on identical hardware and (b) provide a byte-identical fallback.
_KV_PRUNE_FULL_COSINE = os.environ.get("C4_KV_PRUNE_FULL_COSINE", "0") == "1"


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
# VECTORISED per-head prune keep-mask — a torch re-expression of the EXACT four
# ``nibble_kv_prune.KVCache.prune`` mechanisms (validated byte-for-byte against
# the reference in ``test_kv_cache_equivalence`` / ``test_cached_driver``).
#
# The reference is O(cache^2) PER-ENTRY *PYTHON*: a greedy newest-first loop that
# compares every entry to each already-kept survivor, one Python iteration each
# (mechanism 1) plus a per-entry Python free-zero scan (mechanism 2b).  That makes
# eviction CPU-serial and leaves the GPU IDLE during it on a deep loop — a
# rec_fib(12) ~250k-token stream reaches a large live-heap cache and the pruning
# dominated wall-time.  This form runs the SAME decision as a handful of *batched
# tensor ops* ON THE INPUT DEVICE:
#   * mechanism 1's O(cache^2) pairwise near-dup test is ONE matmul / cdist
#     (GPU-busy), and the greedy newest-first selection is a fully-vectorised
#     recency-rank fixpoint (``_greedy_survivors_from_dup_matrix``) — no per-entry
#     Python loop, no per-entry device<->host ``.item()`` sync;
#   * mechanisms 3 / 2a / 2b are plain boolean masks over the whole cache.
# The keep-mask is BYTE-IDENTICAL to the per-entry policy (720-trial gate); only
# the SPEED changes, and the whole computation stays on ``keys.device``.
# ===========================================================================
def _greedy_survivors_from_dup_matrix(D: torch.Tensor,
                                      positions: torch.Tensor) -> torch.Tensor:
    """Fully-vectorised equivalent of the reference greedy newest-first near-dup
    loop, given the boolean pairwise near-dup matrix ``D`` (``D[i,j]`` = entry i
    is a near-duplicate of entry j; diagonal already cleared).

    The reference (``KVCache.prune`` mechanism 1) processes entries newest-first
    and keeps an entry iff it is NOT a near-dup of any ALREADY-KEPT (hence newer)
    entry.  That "compare only to survivors" rule differs from a naive "drop if
    ANY newer near-dup exists" whenever the near-dup relation is non-transitive
    (a rare angle-chain A~B, B~C, A!~C), so we reproduce the greedy exactly:

        keep[e] = NOT OR_f ( f strictly-newer-than-e AND D[e,f] AND keep[f] )

    Dependencies point only to strictly-newer entries (a total recency order), so
    the recurrence is an ACYCLIC fixpoint: iterating from ``keep = all True`` each
    entry reaches its final value once all newer entries have settled, in at most
    (longest non-transitive chain) passes — 1 for the transitive common case.  We
    iterate to convergence with a hard ``S``-pass cap (the DAG-depth bound) so it
    is guaranteed to terminate at the exact greedy answer, no per-entry Python.

    Tie-break: the reference uses a STABLE ``sorted(..., reverse=True)`` — equal
    positions keep their original (ascending-index) append order, i.e. among ties
    the lower index is processed first ("newer").  We encode that total order as a
    rank so "strictly newer" is exact even with duplicate positions.
    """
    S = D.shape[0]
    dev = D.device
    # Total recency order matching the reference's stable reverse-sort: primary key
    # = position DESCENDING, tie-break = index ASCENDING (processed first == the
    # "newer" survivor a duplicate defers to).  ``argsort(argsort(key, desc))``
    # gives rank 0 = newest.  ``key = position*S - index`` is a strictly-monotone
    # encoding of (position desc, index asc) for the small non-negative positions
    # and ``index < S`` used here.
    idx = torch.arange(S, device=dev)
    order_key = positions.to(torch.int64) * S - idx
    rank = torch.argsort(torch.argsort(order_key, descending=True))
    # newer[e,f] = f is strictly-newer than e AND a near-dup of e.
    newer = D & (rank.unsqueeze(1) > rank.unsqueeze(0))
    keep = torch.ones(S, dtype=torch.bool, device=dev)
    for _ in range(S + 1):
        drop = (newer & keep.unsqueeze(0)).any(dim=1)   # e drops if a KEPT newer dup
        new_keep = ~drop
        if bool(torch.equal(new_keep, keep)):
            return new_keep
        keep = new_keep
    return keep


def prune_keep_mask_head(keys: torch.Tensor, vals: torch.Tensor,
                         positions: torch.Tensor, slope: float, scale: float,
                         cos_threshold: float, zero_eps: float,
                         recency_eps: float,
                         dup_metric: str = "cosine",
                         content_addressed: Optional[bool] = None) -> torch.Tensor:
    """Return a boolean ``[S]`` keep-mask == the survivors of ``KVCache.prune``.

    Reproduces, in order:
      mech 1 (near-dup, keep the NEWER of a duplicate pair; ``dup_metric``),
      mech 3 (ALiBi-recency horizon: drop max_w < recency_eps),
      mech 2a (dead-value head: all survivors zero-value -> drop all),
      mech 2b (free-zero stale: drop fully-zero recency-negligible entries).

    ``dup_metric`` = "cosine" (register-marker heads) or "exact" (content-
    addressed §Memory heads whose keys share a large ADDR_BIN common-mode bias;
    see ``nibble_kv_prune.KVCache.prune``).

    ``content_addressed`` (``None`` ⇒ infer from ``dup_metric == "exact"``) marks
    a §Memory (address-CAM) head, whose non-zero-value store rows are the LIVE
    HEAP.  On such a head the recency HORIZON must NOT drop a live (non-zero-value,
    non-superseded) store — it is FREE-DRIVEN (evicted only by supersession or by
    freeing/zeroing), so the cache tracks the UNBOUNDED live heap with no fixed
    recency/size cap (BLOG_SPEC §Memory 410-412 + §Memory Allocation/Freeing
    689-691).  Matches ``nibble_kv_prune.KVCache.prune`` mechanism 3 exactly.

    Fully vectorised: every mechanism is a batched tensor op on ``keys``'s device
    (the O(cache^2) near-dup test is one matmul/cdist), with no per-entry Python
    loop — the eviction runs GPU-bound on the deep tail, byte-identical keep-mask.
    """
    if content_addressed is None:
        content_addressed = (dup_metric == "exact")
    S = keys.shape[0]
    dev = keys.device
    if S == 0:
        return torch.zeros(0, dtype=torch.bool, device=dev)
    positions = positions.to(dev)
    knorm = keys.norm(dim=-1)                                   # [S]
    vnorm = vals.norm(dim=-1)                                   # [S]

    # -- mechanism 1: near-duplicate supersession (latest-write-wins) ----------
    # Build the pairwise near-dup boolean matrix ``D`` in ONE batched op, then run
    # the greedy newest-first selection as a vectorised recency-rank fixpoint.
    if dup_metric == "exact":
        # relative-L2: |k_e - k_f| <= (1-cos_threshold)*max(|k_e|,|k_f|).  Merges
        # only verbatim-identical keys, so distinct content addresses all survive.
        tol = 1.0 - cos_threshold
        # ``donot_use_mm_for_euclid_dist`` forces the direct ||a-b|| formula (not
        # the matmul-expanded ||a||^2-2a.b+||b||^2 whose cancellation can drift by
        # ULPs at large S) so the boolean near-dup matrix is byte-identical to the
        # reference's per-pair ``(k_e-k_f).norm()`` — verified 0-mismatch across the
        # 720-trial + large-cache gate.  ``diff`` is the [S,S] pairwise distance.
        diff = torch.cdist(keys.unsqueeze(0), keys.unsqueeze(0),
                           compute_mode="donot_use_mm_for_euclid_dist").squeeze(0)
        denom = torch.maximum(knorm.unsqueeze(1),
                              knorm.unsqueeze(0)).clamp(min=1e-30)
        D = diff <= tol * denom                                              # [S,S]
    else:
        # raw cosine (zero-key rows -> unit 0 -> sim 0, matching cosine_sim).
        safe = knorm.clamp(min=1e-30)
        unit = keys / safe.unsqueeze(-1)
        unit[knorm == 0] = 0.0
        D = (unit @ unit.t()) > cos_threshold
    D.fill_diagonal_(False)
    survivors = _greedy_survivors_from_dup_matrix(D, positions)

    # -- mechanism 3: ALiBi-recency horizon (PER-ENTRY Cauchy-Schwarz bound) ---
    if slope is not None and slope > 0.0 and survivors.any():
        # ``newest`` / ``max_kn`` are the max over the SURVIVORS only (mask the
        # non-survivors to sentinels so a single reduction gives the same value the
        # reference computes over its ``survivors`` list).
        neg_inf_pos = torch.full_like(positions, -(1 << 62))
        newest = torch.where(survivors, positions, neg_inf_pos).max()
        surv_kn = torch.where(survivors, knorm, torch.zeros_like(knorm))
        max_kn = surv_kn.max().to(torch.float64)
        # ceil_score_e = max_kn * |k_e| * scale  (0 for a zero-key entry).
        ceil_score = (max_kn * knorm.to(torch.float64)) * scale
        # CHK-6 DECOUPLE: a ZERO-VALUE (NULL-write) entry contributes 0 to the
        # softmax1 numerator, so only its denominator term can matter; on the
        # §Memory heads it carries the store-role / load-enable GATE (``-PEN``) so
        # its ACTUAL score is <= 0 => its influence decays with DISTANCE ALONE
        # (exp(score) <= exp(-slope*dist)), independent of the EFF-inflated key
        # magnitude.  Forcing ``ceil_score = 0`` for a zero-value entry restores the
        # small, EFF-INDEPENDENT recency window that keeps the cache FLAT on deep
        # recursion / distinct-address runs, WITHOUT touching value-carrying store
        # rows (whose full content ceil keeps the EFF-sized recall horizon so a
        # deeply-nested LEV still recalls a BP/PC stored ~250k tokens ago).  Matches
        # ``nibble_kv_prune.KVCache.prune`` mechanism 3 exactly (test_cached_driver).
        zero_val = (vnorm <= zero_eps).to(torch.float64)
        ceil_score = ceil_score * (1.0 - zero_val)
        dist = (newest - positions).to(torch.float64)
        arg = (ceil_score - slope * dist).clamp(max=0.0)
        max_w = torch.exp(arg)
        drop = max_w < recency_eps
        if content_addressed:
            # LIVE HEAP: on a §Memory (address-CAM) head a NON-zero-value store is
            # retrieved by ADDRESS at an arbitrary future step, so the recency
            # horizon must be a NO-OP for it (a fixed EFF/slope window would
            # silently drop a live allocated address on a long enough program).
            # Only zero-value (freed/NULL) rows are recency-evicted here; live
            # stores are dropped ONLY by supersession (mechanism 1) or freeing.
            drop = drop & (zero_val > 0.0)
        survivors = survivors & ~drop

    # -- mechanism 2a: dead-value head (all survivors zero-value) --------------
    if survivors.any():
        surv_has_value = survivors & (vnorm > zero_eps)
        if not bool(surv_has_value.any()):
            survivors = torch.zeros_like(survivors)

    # -- mechanism 2b: free-zero (value==0 AND key==0) + recency-stale ---------
    if survivors.any():
        neg_inf_pos = torch.full_like(positions, -(1 << 62))
        newest_all = torch.where(survivors, positions, neg_inf_pos).max()
        free_zero = (vnorm <= zero_eps) & (knorm <= zero_eps)
        if slope is None or slope <= 0.0:
            stale = torch.ones_like(survivors)   # no recency model: honour the rule
        else:
            stale = torch.exp(-slope * (newest_all - positions).to(torch.float64)) \
                    < recency_eps
        survivors = survivors & ~(free_zero & stale)
    return survivors


# ===========================================================================
# BATCHED (multi-group) keep-mask — the SAME four-mechanism policy as
# ``prune_keep_mask_head`` but computed for MANY (block, head) groups at ONCE over
# a leading batch axis ``N``, all sharing ONE ``[S]`` position axis.  This is the
# fusion that de-synchronises the deep-loop eviction: instead of a Python loop of
# ``n_blocks * n_heads`` per-head calls (each launching its own kernels and — via
# ``evict_keep_index``'s ``.tolist()`` — forcing a device->host sync every block),
# the WHOLE eviction decision for every block+head runs as a handful of batched
# ``[N, S, *]`` tensor ops with ZERO per-group Python and ZERO per-group host sync.
# The per-group ``slope`` / ``dup_metric`` / ``content_addressed`` become PER-N
# vectors, so a single call reproduces the exact per-group policy byte-for-byte.
# ===========================================================================
def _recency_rank(positions: torch.Tensor,
                  valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The per-group recency ``rank`` (rank 0 = newest) used by the greedy near-dup
    selection — matching ``_greedy_survivors_batched``'s stable reverse-sort exactly.

    ``key = position*S - index`` encodes (position DESC, index ASC), so the argsort
    of the argsort gives rank 0 to the newest (largest position; ties broken by the
    smaller original index).  ``positions`` is ``[N,S]`` (or ``[S]``).
    """
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    N, S = positions.shape
    dev = positions.device
    idx = torch.arange(S, device=dev)
    order_key = positions.to(torch.int64) * S - idx.unsqueeze(0)       # [N,S]
    return torch.argsort(torch.argsort(order_key, dim=1, descending=True), dim=1)


def _greedy_survivors_batched(D: torch.Tensor,
                              positions: torch.Tensor) -> torch.Tensor:
    """Batched form of :func:`_greedy_survivors_from_dup_matrix`.

    ``D`` is ``[N, S, S]`` (per-group boolean near-dup matrix, diagonal cleared),
    ``positions`` is ``[N, S]`` (PER-GROUP — blocks that have evicted differently
    hold different absolute positions at the same padded index).  Returns the
    ``[N, S]`` greedy newest-first survivor mask, IDENTICAL per row to running the
    single-group fixpoint on ``D[n]`` with ``positions[n]``.

    Same recurrence, vectorised over N:  ``keep[e] = NOT OR_f (f strictly-newer
    AND D[e,f] AND keep[f])``.  The recency ``rank`` is per-group.  The convergence
    test is ONE batched ``torch.equal`` per pass (the common transitive case
    settles in a single pass), so the loop costs O(passes) host syncs TOTAL — not
    O(N) — and passes is ~1.
    """
    N, S, _ = D.shape
    dev = D.device
    rank = _recency_rank(positions.to(dev))
    # newer[n,e,f] = f strictly-newer than e (per group): rank[e] > rank[f]
    # (rank 0 = newest), matching ``_greedy_survivors_from_dup_matrix``.
    strictly_newer = rank.unsqueeze(2) > rank.unsqueeze(1)             # [N,S,S]
    newer = D & strictly_newer                                         # [N,S,S]
    keep = torch.ones(N, S, dtype=torch.bool, device=dev)
    for _ in range(S + 1):
        # e drops if a KEPT strictly-newer near-dup exists (per group).
        drop = (newer & keep.unsqueeze(1)).any(dim=2)                  # [N,S]
        new_keep = ~drop
        if bool(torch.equal(new_keep, keep)):
            return new_keep
        keep = new_keep
    return keep


# ===========================================================================
# TWO-TIER mechanism-1 (near-duplicate supersession) for the COSINE groups.
#
# THE INSIGHT (see docs/KV_PRUNE_TWO_TIER): the VM re-emits the SAME ~30
# register/marker frame every step, so a register-marker head's cache is dominated
# by EXACT-DUPLICATE keys — a role's key is ``W_k @ x`` and ``x`` is
# position-free (ALiBi injects position only in the attention SCORE, never the
# key), so the key of a given (role, value) content is BIT-IDENTICAL across the K
# steps that emit it.  Over a K-step verify span (S ~= K*35 ~= 1680) each of the
# ~30 distinct role/value keys appears ~K times verbatim.
#
# The old mechanism-1 ran the O(S^2) cosine matmul + O(S^2)-per-pass greedy fixpoint
# on ALL S rows — the deep-loop eviction's dominant cost (~1.5 s / prune, 30-36% of
# the fast wall, #683).  Two tiers make it O(S) for the dominant case:
#
#   TIER A (O(S) exact-duplicate collapse): sort each group's rows by key so
#   verbatim-identical keys are adjacent, giving each row a per-group CLASS id.
#   Only the NEWEST member of an exact-duplicate class can ever survive mechanism 1
#   (an older exact-dup has cosine 1.0 > threshold with its newer twin, so the
#   greedy drops it; and if the newer twin is itself dropped by an even-newer
#   near-dup, that near-dup is cosine>thr with the older twin too — SAME key
#   direction — so the older twin is dropped as well).  So collapse each class to
#   its newest representative; every non-representative is dropped for FREE.
#
#   TIER B (cosine on the small residual): build the cosine near-dup matrix and run
#   the greedy fixpoint ONLY over the S_unique representatives (~30, not ~1680),
#   then scatter the per-representative keep decision back to its representative row.
#   The final mechanism-1 survivors are ``is_representative AND cosine_greedy(repr)``.
#
# This is BYTE-IDENTICAL to ``_greedy_survivors_batched(D_cosine_full)`` (exact-dup
# is a subset of cosine near-dup, and a class shares one representative's cosine
# decision), proven in ``test_kv_cache_equivalence`` (720 + large + fused gates).
# ===========================================================================
def _mech1_cosine_survivors_dedup(keys: torch.Tensor, knorm: torch.Tensor,
                                  positions: torch.Tensor, rank: torch.Tensor,
                                  valid: torch.Tensor, cos_threshold: float):
    """Mechanism-1 survivor mask ``[N,S]`` for the COSINE metric via the two-tier
    exact-dedup + cosine-on-representatives collapse.  Byte-identical to
    ``_greedy_survivors_batched((cosine>thr) & row_valid & ~eye, positions)``.

    ``rank`` [N,S] is the recency rank (0 = newest) matching
    ``_greedy_survivors_batched`` (``positions*S - idx`` argsorted).  Invalid
    (padded) rows never survive and never partner.
    """
    N, S, HD = keys.shape
    dev = keys.device

    # -- TIER A: exact-duplicate classes per group (sort keys so equals adjacent) --
    # A stable per-group lexicographic sort by the key VECTOR groups bit-identical rows
    # together; each distinct key becomes a small integer CLASS id per group.
    # Rows that must NEVER merge into an exact-dup class (each is its own singleton —
    # kept by mechanism 1, dropped later by mech 2a/2b/3 if at all):
    #   * INVALID (padded) rows — dead, must not partner a real row;
    #   * ZERO-KEY rows — a zero key has cosine 0 with EVERYTHING (``unit`` is 0), so
    #     the reference greedy never drops a zero-key row for near-dup, EVEN against
    #     another zero-key row.  Collapsing all-zero keys into one class would wrongly
    #     drop all but the newest.
    force_unique = (~valid) | (knorm == 0)                                # [N,S]
    # lexicographic sort over the HD key columns (round-trip stable): sort by the
    # last column first ... first column last so equal-key rows end adjacent.
    order = torch.arange(S, device=dev).unsqueeze(0).expand(N, S).clone()  # [N,S]
    for c in range(HD - 1, -1, -1):
        col = torch.gather(keys[..., c], 1, order)                         # [N,S]
        s = torch.argsort(col, dim=1, stable=True)
        order = torch.gather(order, 1, s)
    # sorted keys per group
    sorted_keys = torch.gather(
        keys, 1, order.unsqueeze(-1).expand(N, S, HD))                     # [N,S,HD]
    # class boundary: a sorted row starts a new class iff its key != previous row's.
    new_class = torch.ones(N, S, dtype=torch.bool, device=dev)
    if S > 1:
        neq = (sorted_keys[:, 1:] != sorted_keys[:, :-1]).any(dim=-1)      # [N,S-1]
        new_class[:, 1:] = neq
    class_of_sorted = torch.cumsum(new_class.to(torch.int64), dim=1) - 1   # [N,S]
    # scatter class id back to ORIGINAL row order
    class_id = torch.empty(N, S, dtype=torch.int64, device=dev)
    class_id.scatter_(1, order, class_of_sorted)                           # [N,S]
    # override: each FORCE-UNIQUE row gets its OWN fresh class id (>= S, per-row
    # distinct via the row index) so it never merges with any other row — HD- and
    # value-independent (no float sentinel needed).
    if bool(force_unique.any()):
        fresh = S + torch.arange(S, device=dev, dtype=torch.int64).unsqueeze(0)
        class_id = torch.where(force_unique, fresh, class_id)             # [N,S]

    # representative of each class = the row with the MAX rank-newness (rank 0 =
    # newest, so the SMALLEST rank value).  Find, per (group, class), the min rank.
    NEG_RANK = S + 1
    rank_masked = torch.where(valid, rank, torch.full_like(rank, NEG_RANK))
    Cmax = 2 * S                                                           # class ids in [0,2S)
    # min rank per class via scatter_reduce (amin): init to NEG_RANK
    best_rank = torch.full((N, Cmax), NEG_RANK, dtype=rank.dtype, device=dev)
    best_rank.scatter_reduce_(1, class_id, rank_masked, reduce="amin",
                              include_self=True)
    my_class_best = torch.gather(best_rank, 1, class_id)                   # [N,S]
    is_repr = valid & (rank_masked == my_class_best)                      # [N,S] newest per class

    # -- TIER B: cosine greedy ONLY among representatives -----------------------
    # Restricting the cosine near-dup + greedy to representatives is exact: a dropped
    # exact-dup makes the identical cosine decision as its representative, so the
    # per-representative keep decision determines the whole class.  We COMPACT the
    # representatives to the front and pad to ``R`` = max #reps across the batch, so
    # the O(cache^2) cosine matmul + greedy fixpoint runs on ``[N, R, R]`` with
    # ``R << S`` (the ~30 distinct role/value keys, NOT the ~1680-row span) — this is
    # where the O(S^2) -> O(S) collapse actually lands.  On the deep loop R is a
    # couple dozen while S is a whole K-step span, so the matmul shrinks ~(S/R)^2.
    n_reps = is_repr.sum(dim=1)                                            # [N]
    R = int(n_reps.max().item()) if N else 0
    if R == 0:
        return torch.zeros(N, S, dtype=torch.bool, device=dev)
    # compact index: for each group, the ORIGINAL row index of its r-th representative
    # (r in [0,R)); pad slots (a group with < R reps) stay invalid.  Non-rep rows are
    # scattered to a THROWAWAY slot R (extra column) so they can never clobber a real
    # representative's slot (a scatter with a duplicated destination index picks an
    # arbitrary source — the earlier bug).
    rep_rank = torch.cumsum(is_repr.to(torch.int64), dim=1) - 1            # [N,S] 0-based slot
    comp_idx_full = torch.zeros(N, R + 1, dtype=torch.int64, device=dev)   # [N,R+1]
    slot_valid_full = torch.zeros(N, R + 1, dtype=torch.bool, device=dev)
    row_ids = torch.arange(S, device=dev).unsqueeze(0).expand(N, S)
    throwaway = torch.full_like(rep_rank, R)
    scat_slot = torch.where(is_repr, rep_rank, throwaway)                  # non-rep -> R
    comp_idx_full.scatter_(1, scat_slot, row_ids)
    slot_valid_full.scatter_(1, scat_slot, is_repr)
    comp_idx = comp_idx_full[:, :R]                                        # drop throwaway
    slot_valid = slot_valid_full[:, :R]
    # gather the representatives' keys + positions into the compact [N,R,*] tensors.
    safe = knorm.clamp(min=1e-30)
    unit = keys / safe.unsqueeze(-1)
    unit = torch.where((knorm == 0).unsqueeze(-1), torch.zeros_like(unit), unit)
    unit_c = torch.gather(unit, 1, comp_idx.unsqueeze(-1).expand(N, R, HD))  # [N,R,HD]
    unit_c = torch.where(slot_valid.unsqueeze(-1), unit_c,
                         torch.zeros_like(unit_c))
    pos_c = torch.gather(positions, 1, comp_idx)                          # [N,R]
    # The compact cosine matrix ``[N,R,R]`` is the ONLY ``O(R^2)`` allocation.  When
    # the batch barely collapses (R ~ S) this can be large, so chunk over N to a
    # memory budget — the sort/gather above were bounded by ``S*HD`` (letting the
    # CALLER pass a big N), and this keeps the ``R^2`` peak bounded regardless of R.
    eye_R = ~torch.eye(R, dtype=torch.bool, device=dev).unsqueeze(0)
    vv_all = slot_valid.unsqueeze(2) & slot_valid.unsqueeze(1)
    chunkNr = max(1, _FUSED_EVICT_MAX_DIST_ELEMS // max(R * R, 1))
    keep_c = torch.empty(N, R, dtype=torch.bool, device=dev)
    for n0 in range(0, N, chunkNr):
        n1 = min(n0 + chunkNr, N)
        Dc = torch.matmul(unit_c[n0:n1],
                          unit_c[n0:n1].transpose(1, 2)) > cos_threshold  # [c,R,R]
        Dc = Dc & vv_all[n0:n1] & eye_R
        keep_c[n0:n1] = _greedy_survivors_batched(Dc, pos_c[n0:n1]) \
            & slot_valid[n0:n1]
    # GATHER the per-representative keep decision back to the ORIGINAL rows: row e's
    # keep flag is ``keep_c[rep_rank[e]]`` iff e is a representative (else False).
    # (Gather — not scatter — so overlapping/padded slots can't clobber a real row.)
    rep_slot = rep_rank.clamp(min=0, max=max(R - 1, 0))                   # [N,S]
    keep_for_row = torch.gather(keep_c, 1, rep_slot)                      # [N,S]
    survivors = is_repr & keep_for_row
    return survivors


def _mech1_full_cosine_survivors(keys, knorm, positions, valid, cos_threshold,
                                 exact):
    """The ORIGINAL mechanism-1: full O(S^2) cosine matmul (all rows) + the exact
    (relative-L2 cdist) metric for content-addressed groups, then the greedy
    newest-first fixpoint over the whole [N,S,S] near-dup matrix.  Retained only as
    the ``C4_KV_PRUNE_FULL_COSINE=1`` A-B baseline / byte-identical fallback for the
    two-tier path — same survivor set (test_kv_cache_equivalence)."""
    N, S, HD = keys.shape
    dev = keys.device
    tol = 1.0 - cos_threshold
    safe = knorm.clamp(min=1e-30)
    unit = keys / safe.unsqueeze(-1)
    unit = torch.where((knorm == 0).unsqueeze(-1), torch.zeros_like(unit), unit)
    D = torch.matmul(unit, unit.transpose(1, 2)) > cos_threshold          # [N,S,S]
    if bool(exact.any()):
        ei = torch.nonzero(exact, as_tuple=False).flatten()
        ke = keys[ei]
        diff = torch.cdist(ke, ke,
                           compute_mode="donot_use_mm_for_euclid_dist")
        kne = knorm[ei]
        denom = torch.maximum(kne.unsqueeze(2),
                              kne.unsqueeze(1)).clamp(min=1e-30)
        D[ei] = diff <= tol * denom
    row_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    D = D & row_valid & ~torch.eye(S, dtype=torch.bool, device=dev).unsqueeze(0)
    return _greedy_survivors_batched(D, positions) & valid


def prune_keep_mask_batched(keys: torch.Tensor, vals: torch.Tensor,
                            positions: torch.Tensor, slope: torch.Tensor,
                            scale: float, cos_threshold: float, zero_eps: float,
                            recency_eps: float, exact: torch.Tensor,
                            content_addressed: torch.Tensor,
                            valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return the ``[N, S]`` boolean keep-mask for N groups at once — byte-for-byte
    the stack of ``prune_keep_mask_head`` results, with these PER-GROUP vectors:

      * ``slope``            [N]   this group's ALiBi slope (mechanism 3 decay).
      * ``exact``            [N] bool  ``dup_metric == "exact"`` for this group.
      * ``content_addressed`` [N] bool §Memory (address-CAM) head (no recency drop
        of a live non-zero store).

    ``keys`` / ``vals`` are ``[N, S, HD]`` (all groups padded to a common S);
    ``positions`` is ``[N, S]`` (PER-GROUP — blocks that already evicted differently
    hold different absolute positions at the same padded index).  ``valid``
    (``[N, S]`` bool, optional) marks the REAL rows of each group when groups were
    padded to a common S — padded rows are forced non-surviving and excluded from
    every reduction so the per-group result is identical to calling the single-group
    helper on that group's real S.

    Every mechanism is a batched op over the leading N axis; there is NO Python
    per-group loop and NO per-group host sync, so a deep-loop prune over all 306
    blocks' 23 heads is ONE fused GPU computation.  Validated byte-identical to
    the reference per-group ``prune_keep_mask_head`` in ``test_kv_cache_equivalence``.
    """
    N, S, HD = keys.shape
    dev = keys.device
    if S == 0:
        return torch.zeros(N, 0, dtype=torch.bool, device=dev)
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(N, S)               # broadcast [S]
    positions = positions.to(dev)
    if valid is None:
        valid = torch.ones(N, S, dtype=torch.bool, device=dev)
    knorm = keys.norm(dim=-1)                                   # [N,S]
    vnorm = vals.norm(dim=-1)                                   # [N,S]
    # padded rows never survive and never participate in any reduction.
    knorm = torch.where(valid, knorm, torch.zeros_like(knorm))

    # -- mechanism 1: near-duplicate supersession (per group, both metrics) -----
    # COSINE groups (the register-marker heads — the MAJORITY): the TWO-TIER
    # exact-duplicate collapse + cosine-on-representatives
    # (``_mech1_cosine_survivors_dedup``) — O(S) hashing/sort of the ~1680-row span
    # down to the ~30 distinct role/value keys, then the O(R^2) cosine matmul + greedy
    # on the tiny representative set (R << S).  This replaces the old O(S^2) full
    # cosine matmul + O(S^2)-per-pass fixpoint that dominated the deep-loop eviction
    # wall (#683).  Byte-identical survivor set (test_kv_cache_equivalence).
    #
    # EXACT groups (content-addressed §Memory heads — a SMALL subset): relative-L2
    # ``|k_e-k_f| <= (1-cos)*max(|k_e|,|k_f|)`` via the ULP-exact ``donot_use_mm``
    # cdist, computed ONLY over that subset (unchanged path — distinct heap addresses
    # are NOT verbatim-equal, so the hash collapse would not help and the ULP-exact
    # cdist is required for the address common-mode).
    if _KV_PRUNE_FULL_COSINE:
        # KILL-SWITCH (C4_KV_PRUNE_FULL_COSINE=1): the ORIGINAL full O(S^2) cosine
        # matmul + greedy on ALL rows — kept as a byte-identical fallback / A-B bench
        # baseline.  Same survivor set as the two-tier path (test_kv_cache_equivalence).
        survivors = _mech1_full_cosine_survivors(
            keys, knorm, positions, valid, cos_threshold, exact)
    else:
        rank_all = _recency_rank(positions, valid)                    # [N,S]
        survivors = _mech1_cosine_survivors_dedup(
            keys, knorm, positions, rank_all, valid, cos_threshold) & valid   # [N,S]
    if (not _KV_PRUNE_FULL_COSINE) and bool(exact.any()):
        tol = 1.0 - cos_threshold
        ei = torch.nonzero(exact, as_tuple=False).flatten()          # exact-group idx
        ke = keys[ei]                                                # [Ne,S,HD]
        diff = torch.cdist(ke, ke,
                           compute_mode="donot_use_mm_for_euclid_dist")   # [Ne,S,S]
        kne = knorm[ei]                                              # [Ne,S]
        denom = torch.maximum(kne.unsqueeze(2),
                              kne.unsqueeze(1)).clamp(min=1e-30)      # [Ne,S,S]
        De = diff <= tol * denom                                     # [Ne,S,S]
        # padded rows can never be a near-dup partner (they are dead).
        row_valid_e = valid[ei].unsqueeze(2) & valid[ei].unsqueeze(1)
        De = De & row_valid_e
        eye = torch.eye(S, dtype=torch.bool, device=dev).unsqueeze(0)
        De = De & ~eye
        surv_e = _greedy_survivors_batched(De, positions[ei]) & valid[ei]  # [Ne,S]
        survivors[ei] = surv_e

    # -- mechanism 3: ALiBi-recency horizon (per-entry Cauchy-Schwarz bound) ----
    slope = slope.to(dev).to(torch.float64)
    has_slope = slope > 0.0                                            # [N]
    any_surv = survivors.any(dim=1)                                    # [N]
    active3 = has_slope & any_surv
    NEG = -(1 << 62)
    surv_pos = torch.where(survivors, positions,
                           torch.full((N, S), NEG, dtype=positions.dtype,
                                      device=dev))
    newest = surv_pos.max(dim=1).values.to(torch.float64)             # [N]
    surv_kn = torch.where(survivors, knorm, torch.zeros_like(knorm))
    max_kn = surv_kn.max(dim=1).values.to(torch.float64)              # [N]
    ceil_score = (max_kn.unsqueeze(1) * knorm.to(torch.float64)) * scale
    zero_val = (vnorm <= zero_eps).to(torch.float64)
    ceil_score = ceil_score * (1.0 - zero_val)                        # zero-val => 0
    dist = (newest.unsqueeze(1) - positions.to(torch.float64))
    arg = (ceil_score - slope.unsqueeze(1) * dist).clamp(max=0.0)
    max_w = torch.exp(arg)
    drop3 = max_w < recency_eps
    # content-addressed groups: only zero-value rows are recency-dropped (a live
    # non-zero store is retrieved by ADDRESS at an arbitrary future step).
    ca = content_addressed.view(N, 1)
    drop3 = torch.where(ca, drop3 & (zero_val > 0.0), drop3)
    drop3 = drop3 & active3.unsqueeze(1)                              # gate inactive
    survivors = survivors & ~drop3

    # -- mechanism 2a: dead-value head (all survivors zero-value -> drop all) ----
    surv_has_value = (survivors & (vnorm > zero_eps)).any(dim=1)      # [N]
    survivors = survivors & surv_has_value.unsqueeze(1)

    # -- mechanism 2b: free-zero (value==0 AND key==0) + recency-stale -----------
    any_surv2 = survivors.any(dim=1)
    surv_pos2 = torch.where(survivors, positions,
                            torch.full((N, S), NEG, dtype=positions.dtype,
                                       device=dev))
    newest_all = surv_pos2.max(dim=1).values.to(torch.float64)       # [N]
    free_zero = (vnorm <= zero_eps) & (knorm <= zero_eps)
    stale_recency = torch.exp(
        -slope.unsqueeze(1) * (newest_all.unsqueeze(1)
                               - positions.to(torch.float64))
    ) < recency_eps
    # slope<=0 groups: honour the rule unconditionally (stale everywhere).
    stale = torch.where(has_slope.unsqueeze(1), stale_recency,
                        torch.ones(N, S, dtype=torch.bool, device=dev))
    drop2b = free_zero & stale & any_surv2.unsqueeze(1)
    survivors = survivors & ~drop2b
    return survivors


# ===========================================================================
# SHARED per-block eviction keep-index.  This is the ONE place that decides the
# spec eviction survivors for a block's ``(K, V, pos)`` cache — used by BOTH the
# single-program cached driver's ``BlockKVCacheBatched.evict`` AND the GPU-batched
# corpus runner (``nibble_pure_forward_gpu.run_batch_gpu``), so their eviction
# policy is provably byte-identical.
#
# The CRITICAL detail (the deep-loop memory-recall bug when this is skipped): a
# CONTENT-ADDRESSED §Memory store head's keys are dominated by a shared ADDR_BIN
# common-mode bias, so raw cosine similarity between DISTINCT store addresses is
# ~0.999 and mechanism-1 (near-duplicate merge) would WRONGLY drop distinct stored
# values — the LI/LC load then recalls nothing (got=0, ZFOD).  We detect such a
# head by its common-mode fraction ``|mean_key| / mean(|key|)`` (~1 for a
# content-addressed head, low for a register-marker head whose distinct keys
# spread in direction) and use the EXACT (relative-L2) dup metric on it, so every
# distinct store survives.
# ===========================================================================
def evict_keep_index(K: torch.Tensor, V: torch.Tensor, pos: torch.Tensor,
                     slopes: torch.Tensor, scale: float, n_heads: int,
                     cos_threshold: float, zero_eps: float, recency_eps: float,
                     protect_positions=None) -> torch.Tensor:
    """Return the surviving ``keep_idx`` (long tensor of index positions) for one
    block's cache ``(K [1,H,S,HD], V [1,H,S,HD], pos [S])`` under the spec
    softmax1+ALiBi eviction policy (UNION across heads, EXACT dup metric on
    content-addressed §Memory store heads).  ``protect_positions`` (absolute
    positions) pins those cached rows.  Returns ``None`` when nothing is dropped.

    GPU-BOUND: the decision runs ON ``K``'s device — the per-head near-dup
    supersession (mechanism 1) is a batched ``[H,S,S]`` matmul and the recency /
    free / dead-head masks are batched boolean ops, so a deep-loop prune keeps the
    GPU busy instead of copying the whole cache to the CPU for a per-entry Python
    loop.  The per-head decision is the SAME vectorised ``prune_keep_mask_head``
    policy (byte-identical to the reference ``KVCache.prune`` — 720-trial gate);
    only the head SELECTION (which are all-zero-value / content-addressed) is a
    handful of scalar reductions, so no O(cache) work leaves the device.
    """
    if K is None:
        return None
    S = int(pos.shape[0])
    dev = K.device
    Kh = K[0]                                           # [H, S, HD]  (on device)
    Vh = V[0]
    slopes = slopes.to(dev)
    keep_any = torch.zeros(S, dtype=torch.bool, device=dev)
    # a head whose entry VALUES are all zero is an exact softmax1 no-op
    # (mechanism 2a evicts all its entries — no keep_any contribution); skip it.
    vnorm_hs = Vh.norm(dim=-1)                           # [H, S]
    head_has_value = (vnorm_hs > zero_eps).any(dim=-1)   # [H]
    # per-head near-dup metric: content-addressed heads get the EXACT metric.
    knorm_hs = Kh.norm(dim=-1)                           # [H, S]
    mean_key = Kh.mean(dim=1)                            # [H, HD]
    cm_frac = mean_key.norm(dim=-1) / knorm_hs.mean(dim=-1).clamp(min=1e-30)
    # Move the H tiny per-head scalar decisions to host in ONE sync (H booleans /
    # floats), not per-cache-entry, so the loop itself launches only batched GPU
    # kernels through ``prune_keep_mask_head``.
    head_has_value_l = head_has_value.tolist()
    cm_frac_l = cm_frac.tolist()
    slopes_l = slopes.tolist()
    for h in range(n_heads):
        if not head_has_value_l[h]:
            continue                                     # mechanism 2a: evict all
        metric = "exact" if cm_frac_l[h] > 0.9 else "cosine"
        mask = prune_keep_mask_head(
            Kh[h], Vh[h], pos,
            slope=float(slopes_l[h]), scale=scale,
            cos_threshold=cos_threshold, zero_eps=zero_eps,
            recency_eps=recency_eps, dup_metric=metric)
        keep_any |= mask
    if protect_positions is not None and len(protect_positions) > 0:
        prot = torch.as_tensor(sorted(protect_positions),
                               dtype=pos.dtype, device=dev)
        keep_any |= torch.isin(pos, prot)          # pin the data-segment store rows
    keep_idx = torch.nonzero(keep_any, as_tuple=False).flatten()
    if int(keep_idx.numel()) >= S:
        return None                                 # nothing dropped
    return keep_idx


# ===========================================================================
# FUSED ALL-BLOCK eviction.  The deep-loop bottleneck (#667): the driver pruned
# every ``prune_interval`` tokens by a PYTHON LOOP over all ``n_blocks`` (306)
# block caches, each call (``evict_keep_index``) launching a per-head Python loop
# AND forcing MULTIPLE device->host syncs (``head_has_value.tolist()`` +
# ``cm_frac.tolist()`` + ``slopes.tolist()`` per block + a final ``.nonzero()`` /
# ``.numel()``), so the GPU STALLED waiting on the CPU every block (41% util).
#
# This fuses the WHOLE prune into a handful of batched GPU ops:
#   * ALL blocks' caches are padded to a common ``Smax`` and stacked into ONE
#     ``[n_blocks*H, Smax, HD]`` tensor, so the per-(block,head) keep decision is a
#     SINGLE ``prune_keep_mask_batched`` call (the O(cache^2) near-dup cdist for
#     every block+head at once) — no per-block, no per-head Python;
#   * head/metric selection (``head_has_value`` mechanism-2a skip, ``cm_frac`` exact
#     vs cosine) is a batched reduction with NO ``.tolist()`` — it stays on-device;
#   * the survivors are UNIONed across a block's heads on-GPU, protection positions
#     applied with ``torch.isin``, giving a ``[n_blocks, Smax]`` boolean keep-mask
#     that NEVER leaves the device before compaction.
# The keep DECISION is byte-identical to the per-block ``evict_keep_index`` (same
# ``prune_keep_mask_*`` policy, same union, same protect) — see the equivalence
# test.  Only the FUSION + de-sync changes; the survivor set does not.
# ===========================================================================
# Memory budget for the batched near-dup near-dup pass.  ``prune_keep_mask_batched``
# holds SEVERAL ``[chunkN, S, S]`` tensors AT ONCE (the cdist ``diff``, ``D_exact``,
# ``D_cos``, ``D``, and the greedy ``newer`` bool), so the peak is ~a handful times
# ``chunkN * S * S``.  We chunk the FLATTENED (block x head) group axis ``N`` so that
# ``chunkN * S * S`` stays under this budget — this bounds VRAM even at the FIRST
# (pre-flatten) prune where ``S`` is a whole verify-span (~1920) AND still fuses ALL
# 306 blocks' 23 heads into ONE call on the deep tail where ``S`` is flat (~a few
# dozen).  Peak allocation ~= (#simultaneous [chunkN,S,S] tensors) x budget-bytes.
_FUSED_EVICT_MAX_DIST_ELEMS = 24 * 1024 * 1024       # 24M elems -> ~1 GiB peak (~6x)


def evict_all_blocks_fused(caches, cos_threshold: float, zero_eps: float,
                           recency_eps: float, protect_positions=None):
    """Compute the per-block eviction keep-MASKS for EVERY block cache at once.

    ``caches`` is the list of ``BlockKVCacheBatched`` (all share H / HD; their
    ``pos`` tensors may DIVERGE after earlier prunes).  Returns a ``[n_blocks]``
    list where entry ``b`` is either ``None`` (nothing dropped for that block) or a
    boolean ``[S_b]`` keep-mask ON ``caches[b].K``'s device — the SAME survivor set
    ``caches[b].evict`` would compute, but derived by FUSED batched GPU passes with
    no per-block / per-head host sync.

    Blocks are grouped by their current cache SIZE ``S`` (they all share ``S`` in
    the lockstep-commit common case -> ONE group; they diverge only after earlier
    evictions).  Within a size-group every block's heads are stacked into ONE
    ``[Lb*H, S, HD]`` tensor, and the near-dup keep decision for that whole
    ``N = Lb*H`` set is run in ``prune_keep_mask_batched`` — chunked along ``N`` to a
    memory budget so the ``[chunkN, S, S]`` near-dup pass never blows VRAM (bounds
    the FIRST pre-flatten prune where S is a whole span, while still fusing ALL 306
    blocks x 23 heads into ONE call on the deep tail where S is flat).

    The only host transfer is ONE ``[Lb]`` "did this block drop anything" boolean
    vector per size-group (so the caller skips compaction of unchanged blocks) —
    O(#size-groups) syncs total, vs the old ~306x5 per-block-per-prune host syncs.
    """
    n_blocks = len(caches)
    if n_blocks == 0:
        return []
    live = [b for b in range(n_blocks) if caches[b].K is not None]
    result = [None] * n_blocks
    if not live:
        return result
    dev = caches[live[0]].K.device
    HD = caches[live[0]].head_dim
    scale = caches[live[0]].scale

    # In a DROP-KV split cache ``.K`` holds ONLY the global heads, so the per-group
    # head count comes from ``.K.shape[1]`` (grouped below), and ``_evict_slopes``
    # returns the matching global-head slopes so the stack/metric shapes line up.
    def _evict_slopes(cache):
        if getattr(cache, "split", False):
            return cache.slopes.index_select(0, cache._global_head_idx)
        return cache.slopes

    prot = None
    if protect_positions is not None and len(protect_positions) > 0:
        prot = torch.as_tensor(sorted(protect_positions),
                               dtype=torch.long, device=dev)

    # group the live blocks by their current cache size S (no padding within a
    # group -> exact, memory-lean; ONE group in the lockstep common case).
    by_size: Dict[Tuple[int, int], List[int]] = {}
    for b in live:
        # Group by (cache size, HEAD COUNT): a DROP-KV split cache holds only its
        # GLOBAL heads in ``.K``, and different blocks may have different global-head
        # counts (e.g. 1 vs 2), so the head axis must match within a stacked group.
        by_size.setdefault((int(caches[b].pos.shape[0]), int(caches[b].K.shape[1])),
                           []).append(b)

    # TWO memory bounds, both to ``_FUSED_EVICT_MAX_DIST_ELEMS``:
    #   * ``blk_chunk`` bounds the STACKED-cache copy (``blk_chunk*H*S*HD``), so the
    #     transient stack of block K/V never blows VRAM at the large-S first prune;
    #   * ``chunkN`` bounds the near-dup ``[chunkN,S,S]`` pass inside each block-chunk
    #     (chunks HEADS too when a single block's ``H*S*S`` exceeds budget).
    # On the deep tail S is flat so both are ALL live blocks/heads in ONE fused pass;
    # at a rare pre-flatten large-S prune they shrink — still NO per-block host sync.
    for (S, Hgrp), blocks_S in by_size.items():
        if S == 0:
            continue
        H = Hgrp                             # this group's head count (Hg if split)
        blk_chunk = max(1, _FUSED_EVICT_MAX_DIST_ELEMS // max(H * S * HD, 1))
        # ``chunkN`` bounds the per-call near-dup pass.  The FULL-cosine path holds a
        # ``[chunkN, S, S]`` matrix, so it is bounded by ``S*S``.  The DEFAULT two-tier
        # path (``_mech1_cosine_survivors_dedup``) NEVER materialises ``[*, S, S]`` —
        # its peak is the ``[chunkN, S, HD]`` sorted-key tensor + a tiny ``[chunkN,R,R]``
        # representative matmul (R = #distinct keys << S) — so it is bounded by
        # ``S*HD`` (HD=8), letting chunkN be ~S/8 * larger.  That collapses the number
        # of per-chunk kernel launches (the 8-pass lexicographic sort ran once per
        # tiny chunkN=24M/S^2 chunk before — hundreds of launches/round; now ~1), which
        # is what makes the O(S)-hash tier's win actually land at deep-loop scale.
        if _KV_PRUNE_FULL_COSINE:
            chunkN = max(1, _FUSED_EVICT_MAX_DIST_ELEMS // max(S * S, 1))
        else:
            chunkN = max(1, _FUSED_EVICT_MAX_DIST_ELEMS // max(S * HD, 1))
        Ltot = len(blocks_S)
        # accumulate EVERY block's [S] keep-mask into ONE [Ltot,S] tensor so the
        # compaction is a SINGLE batched op (one nonzero) — not 306 per-block
        # boolean-index gathers, each of which forces a host sync (that was the
        # residual deep-loop stall after the DECISION was fused).
        keep_all = torch.empty(Ltot, S, dtype=torch.bool, device=dev)
        row = 0
        for c0 in range(0, Ltot, blk_chunk):
            grp = blocks_S[c0:c0 + blk_chunk]
            Lb = len(grp)
            keys = torch.stack([caches[b].K[0] for b in grp], dim=0)   # [Lb,H,S,HD]
            vals = torch.stack([caches[b].V[0] for b in grp], dim=0)
            pos = torch.stack([caches[b].pos.to(dev) for b in grp], 0)  # [Lb,S]
            slope = torch.stack([_evict_slopes(caches[b]).to(dev) for b in grp], 0)  # [Lb,H]

            # per-(block,head) metric + dead-head selection (batched, on-device).
            vnorm = vals.norm(dim=-1)                                # [Lb,H,S]
            head_has_value = (vnorm > zero_eps).any(dim=-1)          # [Lb,H]
            knorm = keys.norm(dim=-1)                                # [Lb,H,S]
            mean_key = keys.mean(dim=2)                              # [Lb,H,HD]
            cm_frac = mean_key.norm(dim=-1) / knorm.mean(dim=-1).clamp(min=1e-30)
            exact = cm_frac > 0.9                                    # [Lb,H] bool

            N = Lb * H
            keys_n = keys.reshape(N, S, HD)
            vals_n = vals.reshape(N, S, HD)
            pos_n = pos.unsqueeze(1).expand(Lb, H, S).reshape(N, S)
            slope_n = slope.reshape(N)
            exact_n = exact.reshape(N)
            # a dead-value head contributes NOTHING to the union (mechanism 2a evicts
            # all its entries); mark it non-valid so its policy is skipped and its
            # (empty) mask never inflates the union — byte-exact.
            valid_n = head_has_value.reshape(N).unsqueeze(1).expand(N, S).contiguous()

            masks = torch.empty(N, S, dtype=torch.bool, device=dev)
            for n0 in range(0, N, chunkN):
                n1 = min(n0 + chunkN, N)
                masks[n0:n1] = prune_keep_mask_batched(
                    keys_n[n0:n1], vals_n[n0:n1], pos_n[n0:n1], slope_n[n0:n1],
                    scale, cos_threshold, zero_eps, recency_eps,
                    exact=exact_n[n0:n1], content_addressed=exact_n[n0:n1],
                    valid=valid_n[n0:n1])

            # UNION across a block's heads -> per-block keep-mask.
            keep_block = masks.reshape(Lb, H, S).any(dim=1)         # [Lb,S]
            if prot is not None:
                keep_block = keep_block | torch.isin(pos, prot)     # pin data-seg
            keep_all[row:row + Lb] = keep_block
            row += Lb

        # SINGLE batched compaction for the whole size-group: ONE nonzero gives every
        # surviving (block-row, col) pair; the per-row survivor COUNTS come from the
        # same pass.  ONE host transfer (the counts) instead of Ltot per-block syncs.
        kept_counts = keep_all.sum(dim=1)                          # [Ltot] on device
        pairs = torch.nonzero(keep_all, as_tuple=False)           # [T,2]=(row,col)
        cols = pairs[:, 1]                                        # surviving col idx
        counts_l = kept_counts.tolist()                          # 1 sync / size-group
        off = 0
        for i, b in enumerate(blocks_S):
            k = counts_l[i]
            if k < S:                                             # this block dropped
                result[b] = cols[off:off + k]                    # long keep-idx on dev
            off += k
    return result


class _SplitPastKV:
    """The past-KV a DROP-KV split cache hands to ``local_attention.windowed_forward``.

    Carries the two head-group caches that live on DIFFERENT (differently-trimmed)
    position axes: the GLOBAL heads' full-history ``(Kg, Vg, posg)`` and the LOCAL
    heads' window-trimmed ``(Kl, Vl, posl)``, plus the head-index partition so the
    forward reads each head from its own group.  A plain 3-tuple past-KV can't carry
    this (two different Sk), so the windowed forward type-checks for this object.
    ``Kg`` is ``[1,Hg,Sg,HD]`` (only the global heads); ``Kl`` is ``[1,Hl,Sl,HD]``.
    """
    __slots__ = ("Kg", "Vg", "posg", "Kl", "Vl", "posl", "g_idx", "l_idx")

    def __init__(self, Kg, Vg, posg, Kl, Vl, posl, g_idx, l_idx):
        self.Kg = Kg; self.Vg = Vg; self.posg = posg
        self.Kl = Kl; self.Vl = Vl; self.posl = posl
        self.g_idx = g_idx; self.l_idx = l_idx


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
        # -- DROP-KV local-attention split (opt-in; None = classic full-H cache) --
        # When ``set_head_groups`` is called (by ``local_attention.install`` with
        # drop_local_kv=True), this cache stores the LOCAL (sliding-window) heads'
        # K/V trimmed to only the last ``local_window`` ABSOLUTE positions and the
        # GLOBAL (memory/stack/LEV) heads' K/V full-length — two tensors with two
        # DIFFERENT position axes.  This is what actually shrinks VRAM: the ~7357
        # local head-slots keep only ~W rows instead of the whole S-row history.
        self._global_head_idx: Optional[torch.Tensor] = None   # [Hg] long
        self._local_head_idx: Optional[torch.Tensor] = None    # [Hl] long
        self._local_window: Optional[int] = None
        # -- CONTENT-BOUND global-head retention (opt-in; None = full history) -----
        # Each GLOBAL head (memory-CAM / stack-pop / LEV) is an address-CAM that
        # keys ONLY ``IS_STORE`` positions with a role-gate penalty (channel
        # ``cR = ADDR_BITS+1`` in its head slice: a store row keys ~0 there, a
        # non-store row keys ``-p`` = the huge PEN_GATE, so a load/pop/lev query
        # scores ``-PEN`` on it -> softmax1 weight EXACTLY 0).  So a global head can
        # only ever attend to STORE frames; register/code/other frames are provably
        # inert and droppable BYTE-IDENTICALLY.  When ``_content_bound`` is set, the
        # global-head commit keeps only the store rows the global heads can attend
        # to -> the global cache is bounded by the WORKING SET (distinct live store
        # addresses, after latest-write-wins eviction dedups superseded stores), NOT
        # by step count.  ``_content_cR`` is the store-role gate channel index.
        self._content_bound: bool = False
        self._content_cR: Optional[int] = None
        # local-head cache (trimmed to last W positions).  self.K/V/pos above hold
        # ONLY the GLOBAL heads when split is active (so eviction only touches them).
        self.Kl: Optional[torch.Tensor] = None     # [1,Hl,Sl,HD]
        self.Vl: Optional[torch.Tensor] = None
        self.posl: Optional[torch.Tensor] = None   # [Sl] long

    @property
    def split(self) -> bool:
        """True iff this cache is running the drop-KV local/global head split."""
        return self._local_window is not None

    def set_head_groups(self, global_head_idx, local_window: int,
                        content_bound: bool = False,
                        content_cR: Optional[int] = None) -> None:
        """Enable the DROP-KV split: ``global_head_idx`` heads keep the FULL causal
        KV, every other head is LOCAL and keeps only the last ``local_window``
        ABSOLUTE positions (OLD rows DROPPED, not masked).

        Byte-identical to full attention: a local head's true attention weight past
        the window is exactly 0 (softmax1 + huge exact-match/role scores + ALiBi
        recency drive the tail to ZFOD; measured window <= 28 tok < one 30-token VM
        step, W=64 is >2 steps), so the dropped rows contribute exactly 0.

        ``content_bound`` (default False) additionally bounds the GLOBAL cache BY
        CONTENT: each global head is an address-CAM that keys ONLY ``IS_STORE``
        positions (channel ``content_cR = ADDR_BITS+1`` in its head slice: a store
        row keys ~0 there, a non-store row keys ``-p`` = PEN_GATE, so a load query
        scores ``-PEN`` on it -> softmax1 weight EXACTLY 0).  So a non-store frame is
        provably inert for every global head and is dropped on commit, byte-
        identically.  Combined with the latest-write-wins eviction (which dedups
        superseded stores per address), the global cache is bounded by the WORKING
        SET (distinct live addresses), NOT step count -> total KV runtime-independent.
        """
        dev = self.slopes.device
        g = torch.as_tensor(sorted(int(h) for h in global_head_idx),
                            dtype=torch.long, device=dev)
        gset = set(g.tolist())
        l = torch.as_tensor([h for h in range(self.n_heads) if h not in gset],
                           dtype=torch.long, device=dev)
        self._global_head_idx = g
        self._local_head_idx = l
        self._local_window = int(local_window)
        self._content_bound = bool(content_bound)
        self._content_cR = (int(content_cR) if content_cR is not None else None)

    def _trim_local(self) -> None:
        """Drop LOCAL-head cached rows older than the window (keep last W positions
        by ABSOLUTE position).  A no-op read for the local heads past W, so dropping
        is EXACT."""
        if self.posl is None or self._local_window is None:
            return
        maxp = int(self.posl.max().item())
        cutoff = maxp - self._local_window + 1
        keep = (self.posl >= cutoff)
        if bool(keep.all()):
            return
        keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
        dropped = int(self.posl.shape[0] - keep_idx.numel())
        self.Kl = self.Kl.index_select(2, keep_idx)
        self.Vl = self.Vl.index_select(2, keep_idx)
        self.posl = self.posl.index_select(0, keep_idx)
        self.total_evicted += dropped

    def as_past_kv(self):
        if not self.split:
            if self.K is None:
                return None
            return (self.K, self.V, self.pos)
        # SPLIT: hand the windowed forward both head-group caches (global full,
        # local trimmed) plus the head partition so it reads each group's own KV.
        if self.K is None and self.Kl is None:
            return None
        return _SplitPastKV(self.K, self.V, self.pos,
                            self.Kl, self.Vl, self.posl,
                            self._global_head_idx, self._local_head_idx)

    def commit(self, K_new: torch.Tensor, V_new: torch.Tensor,
               pos_new: torch.Tensor) -> None:
        """Append freshly-frozen positions' K/V (``[1,H,W,HD]`` + ``[W]``)."""
        if self.split:
            return self._commit_split(K_new, V_new, pos_new)
        if self.K is None:
            self.K, self.V, self.pos = K_new, V_new, pos_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)
            self.pos = torch.cat([self.pos, pos_new], dim=0)

    def _commit_split(self, K_new: torch.Tensor, V_new: torch.Tensor,
                      pos_new: torch.Tensor) -> None:
        """DROP-KV commit: route the freshly-frozen K/V (``[1,H,W,HD]``) into the two
        head-group caches — GLOBAL heads appended to the full-history cache, LOCAL
        heads appended then TRIMMED to the last ``local_window`` positions (old rows
        DROPPED)."""
        g, l = self._global_head_idx, self._local_head_idx
        if l.numel() > 0:
            Kl_new = K_new.index_select(1, l)
            Vl_new = V_new.index_select(1, l)
            if self.Kl is None:
                self.Kl, self.Vl, self.posl = Kl_new, Vl_new, pos_new
            else:
                self.Kl = torch.cat([self.Kl, Kl_new], dim=2)
                self.Vl = torch.cat([self.Vl, Vl_new], dim=2)
                self.posl = torch.cat([self.posl, pos_new], dim=0)
            self._trim_local()          # keep only the last-W local rows (the win)
        if g.numel() > 0:
            Kg_new = K_new.index_select(1, g)
            Vg_new = V_new.index_select(1, g)
            if self._content_bound:
                # CONTENT-BOUND: keep ONLY the store rows the global heads can ever
                # attend to (a non-store row's role-gate key -> softmax1 weight 0,
                # so dropping it is BYTE-IDENTICAL).  This bounds the global cache by
                # the working set of live addresses instead of by step count.
                keep = self._global_content_keep(Kg_new[0])   # [W'] bool over new rows
                if not bool(keep.all()):
                    keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
                    dropped = int(keep.numel() - keep_idx.numel())
                    Kg_new = Kg_new.index_select(2, keep_idx)
                    Vg_new = Vg_new.index_select(2, keep_idx)
                    pos_new_g = pos_new.index_select(0, keep_idx)
                    self.total_evicted += dropped
                else:
                    pos_new_g = pos_new
            else:
                pos_new_g = pos_new
            if Kg_new.shape[2] == 0:
                return                       # nothing store-attendable in this span
            if self.K is None:
                self.K, self.V, self.pos = Kg_new, Vg_new, pos_new_g
            else:
                self.K = torch.cat([self.K, Kg_new], dim=2)
                self.V = torch.cat([self.V, Vg_new], dim=2)
                self.pos = torch.cat([self.pos, pos_new_g], dim=0)

    def _global_content_keep(self, Kg: torch.Tensor) -> torch.Tensor:
        """Return a ``[S]`` bool mask of the rows a GLOBAL head can EVER attend to.

        ``Kg`` is ``[Hg, S, HD]`` — the global heads' projected keys.  Each global
        head is an address-CAM whose store-role gate channel ``cR = ADDR_BITS+1``
        (``self._content_cR``) keys a STORE row at ~0 and a NON-STORE row at ``-p``
        (``p = sqrt(PEN_GATE/hs)`` >> 0): a load/pop/lev query keys ``+p`` there, so a
        non-store row's score is ``p*(-p)*hs = -PEN`` and its softmax1 weight is
        EXACTLY 0.  So a row is attendable iff its ``cR`` key is NOT the ``-p`` gate,
        i.e. ``Kg[:, :, cR] > -p/2`` — the clean bimodal split (store ~0 vs -p).

        A row is kept iff ANY global head can attend to it (all three global heads
        share the SAME store-role gate structure, so this is a union == per-head).
        ``p`` is inferred from the data (max gate magnitude) so no build constant is
        hardcoded — the threshold sits strictly between the two modes (0 and -p).
        """
        cR = self._content_cR
        if cR is None:
            # Fallback: no cR channel given -> keep everything (no content drop).
            return torch.ones(Kg.shape[1], dtype=torch.bool, device=Kg.device)
        gate = Kg[:, :, cR]                             # [Hg, S]  store~0 / nonstore~-p
        # p (the gate magnitude) = the largest -gate seen; threshold at -p/2 sits
        # strictly between the store mode (0) and the non-store mode (-p).  If NO
        # non-store row is present in this span (p ~ 0, gate all ~0), the split is
        # undecidable so we conservatively keep every row (byte-safe: at worst we
        # retain an already-store row, never drop an attendable one).
        p = (-gate).clamp(min=0.0).max()
        if float(p) <= 0.0:
            return torch.ones(Kg.shape[1], dtype=torch.bool, device=Kg.device)
        thr = -0.5 * p
        # attendable for a head iff its gate is above the threshold (store row).
        attendable = gate > thr                        # [Hg, S]
        return attendable.any(dim=0)                    # [S] union across global heads

    def size(self) -> int:
        # For a split cache this is the GLOBAL-head cache size (the only cache
        # eviction manages; the local heads are already window-bounded).  For blocks
        # with no global head it is 0 — honest: those blocks store only ~W local rows.
        return 0 if self.pos is None else int(self.pos.shape[0])

    def local_size(self) -> int:
        """#local-head cached rows (bounded by the window W). Split cache only."""
        return 0 if self.posl is None else int(self.posl.shape[0])

    def n_global_heads(self) -> int:
        return 0 if self._global_head_idx is None else int(self._global_head_idx.numel())

    def evict(self, cos_threshold: float, prune_interval: int,
              zero_eps: float, recency_eps: float,
              protect_positions=None) -> int:
        """Apply the spec eviction policy; return #positions dropped.

        The keep-set is the UNION across heads of each head's ``prune()`` survivors
        (a position is retained if ANY head still needs it), so no head loses an
        entry it would attend to — the batched tensors then keep exactly that
        union.  Because the batched axis is shared, this is the conservative,
        output-exact intersection of the per-head policies.

        ``protect_positions`` (a set/tensor of ABSOLUTE positions) PINS exactly those
        cached rows — the data-segment store tokens the program loads at ARBITRARY
        future times, so ALiBi-recency cannot bound their future usefulness (a §Memory
        store is a persistent value, not a recency-decayed register frame).  Only the
        MEM store rows are pinned (54 for the quine), NOT the whole seed frames, so the
        cache stays small: the bundled data segment (the quine's ``Q``) survives the
        whole run while every register frame (incl. the seed frames' dead marker rows)
        still evicts.

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
        # In a DROP-KV split cache ``self.K`` holds ONLY the global heads, so the
        # eviction runs over the global-head slopes / count (the local heads are
        # already window-bounded and are pruned by the W-trim on commit).
        if self.split:
            n_h = int(self._global_head_idx.numel())
            slopes = self.slopes.index_select(0, self._global_head_idx)
        else:
            n_h, slopes = self.n_heads, self.slopes
        # The keep DECISION is the SHARED spec policy (``evict_keep_index``) — the
        # SAME one the GPU-batched corpus runner uses, so eviction is byte-identical
        # across both drivers (incl. the EXACT dup metric on content-addressed
        # §Memory store heads, without which distinct stores are wrongly merged and
        # a deep-loop LI/LC recalls nothing).
        keep_idx = evict_keep_index(
            self.K, self.V, self.pos, slopes, self.scale, n_h,
            cos_threshold=cos_threshold, zero_eps=zero_eps,
            recency_eps=recency_eps, protect_positions=protect_positions)
        dropped = 0 if keep_idx is None else (S - int(keep_idx.numel()))
        if dropped:
            keep_idx_dev = keep_idx.to(self.K.device)
            self.K = self.K[:, :, keep_idx_dev, :]
            self.V = self.V[:, :, keep_idx_dev, :]
            self.pos = self.pos[keep_idx_dev]
            self.total_evicted += dropped
        return dropped

    def apply_keep_mask(self, keep) -> int:
        """Compact this block's cache to ``keep`` — the survivors
        ``evict_all_blocks_fused`` produced for this block, as EITHER a LONG index
        tensor (the fused path — already computed by ONE batched ``nonzero`` over the
        whole size-group, so this gather is a sync-FREE ``index_select``) OR a boolean
        ``[S]`` mask (falls back to boolean indexing).  Returns #positions dropped.

        The whole-group single-``nonzero`` in ``evict_all_blocks_fused`` is what
        removes the last per-block host sync: with a precomputed long index the
        compaction is ``index_select`` (no device->host round-trip), so a deep-loop
        prune of all 306 blocks stays on the GPU.  Byte-identical survivor set to
        ``evict``.
        """
        if self.K is None or keep is None:
            return 0
        S = int(self.pos.shape[0])
        keep = keep.to(self.K.device)
        if keep.dtype == torch.bool:
            self.K = self.K[:, :, keep, :]
            self.V = self.V[:, :, keep, :]
            self.pos = self.pos[keep]
        else:
            self.K = self.K.index_select(2, keep)
            self.V = self.V.index_select(2, keep)
            self.pos = self.pos.index_select(0, keep)
        dropped = S - int(self.pos.shape[0])
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
                            stats: Optional[dict] = None,
                            fio=None, data_seg=None,
                            out: Optional[List[int]] = None,
                            seed_mem: Optional[Dict[int, int]] = None):
    """KV-cached form of :func:`run_pure_forward_complete`.

    Byte-identical output to the naive re-forward driver (proven in
    ``test_cached_driver``), at O(cache)/step instead of O(stream^2)/step.  With
    ``evict=True`` the per-block caches are pruned every ``prune_interval`` tokens
    by the ``nibble_kv_prune`` policy so the cache stays FLAT over deep loops.

    ``stats`` (optional dict) is filled with ``max_seq_len`` / ``max_cache_size``
    (per block) / ``total_evicted`` / ``steps`` for the bounded-memory report.

    If ``out`` is a list, a PRTF step appends its VISIBLE output byte — the AX
    byte-0 the model itself decoded from its nibble band (a genuine LM-head argmax,
    NOT a python copy).  This is the ``printf("%c", AX)`` visible-output channel
    (§Printing / op 33), collected on the SAME KV-cached model path.
    """
    blocks = model.blocks
    n_blocks = len(blocks)
    H = blocks[0].attn.n_heads
    HD = blocks[0].attn.head_dim
    caches = [BlockKVCacheBatched(H, HD, blocks[b].attn.alibi_slopes)
              for b in range(n_blocks)]
    tokens_since_prune = 0

    # Seed the data segment (if any) as leading MEM-STORE frames so the KV memory
    # holds it before step 0 (the classic quine's "string literal in data").  Each
    # seed byte becomes one store frame recorded in ``store_log`` at frame idx k;
    # the init frame then sits at frame idx ``n_seed``.
    seed_frames: List[int] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    for k, (addr, val) in enumerate(sorted((seed_mem or {}).items())):
        seed_frames += _build_frame(0, 0, SP_INIT, SP_INIT, 0,
                                    mem_addr=addr & 0xFFFFFFFF,
                                    mem_val=val & 0xFFFFFFFF)
        store_log[k] = (addr & 0xFFFFFFFF, val & 0xFFFFFFFF)
    n_seed = len(store_log)
    # absolute positions of the seed frames' MEM store tokens (frame k occupies
    # positions 1+k*FRAME_LEN .. ; its store token is at local _MEM_MARKER_LOCAL).
    seed_store_positions = {1 + k * V.FRAME_LEN + _MEM_MARKER_LOCAL
                            for k in range(n_seed)}

    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    stream: List[int] = [V.BOS] + seed_frames + init_frame
    trace: List[int] = []
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = n_seed

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

        # -- FILE OP (OPEN/READ/CLOS/PRTF): the ONE class not computed neurally
        # (§Tool Use Mode).  The model's registers for this row are meaningless;
        # the DRIVER performs the whole op via the TOOL_CALL runner and overrides
        # them.  A READ appends its bytes as their OWN §Memory KV store frames so
        # a later LC(addr) attends to the file byte.  Multiple frames are appended
        # this step; the commit + next-window are widened to cover them all.
        if fio is not None and op in _FS.FILE_OPCODES:
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, cur_ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            pc = cur_pc + 1                     # file ops advance PC by one (no branch)
            sp = new_sp
            bp = cur_bp
            ax = new_ax & 0xFFFFFFFF
            appended = []                       # the frames this step emits
            frame_idx += 1
            trace.append(ax & mask)
            appended += _build_frame(pc, ax, sp, bp, stk)
            for (baddr, bval) in byte_stores:
                frame_idx += 1
                store_log[frame_idx] = (baddr, bval & 0xFF)
                appended += _build_frame(pc, ax, sp, bp, stk,
                                         mem_addr=baddr, mem_val=bval & 0xFF)
            # commit the CURRENT window's frozen rows (same as the normal path).
            n_commit = win_len - 1
            if n_commit > 0:
                for b in range(n_blocks):
                    K_all, V_all, pos_all = new_kv[b]
                    K_win = K_all[:, :, -win_len:, :]
                    V_win = V_all[:, :, -win_len:, :]
                    pos_win = pos_all[-win_len:]
                    caches[b].commit(K_win[:, :, :n_commit, :],
                                     V_win[:, :, :n_commit, :],
                                     pos_win[:n_commit])
            if verbose:
                print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                      f"pc'={pc} ax={ax&0xFFFFFFFF} sp={sp} (FILE, "
                      f"{len(byte_stores)} byte-stores) cache={caches[0].size()}")
            cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
            if pc < 0 or pc >= len(code):
                break
            # append all emitted frames; NEXT window = [old_query_row] + all frames.
            prev_query_pos = win_start + win_len - 1
            n_new_frames = 1 + len(byte_stores)
            stream += appended
            win_start = prev_query_pos
            win_len = 1 + n_new_frames * V.FRAME_LEN
            max_seq = max(max_seq, len(stream))
            tokens_since_prune += n_new_frames * V.FRAME_LEN
            if evict and tokens_since_prune >= prune_interval:
                for b in range(n_blocks):
                    caches[b].evict(cos_threshold, prune_interval, zero_eps, recency_eps)
                tokens_since_prune = 0
            max_cache = max(max_cache, caches[0].size())
            continue

        # --- I/O: PRTF emits a VISIBLE output byte (printf("%c", AX)) ------------
        # The byte is the model's OWN decoded AX byte-0 (a genuine LM-head argmax,
        # not a python copy).  PRTF's dispatch rule only advances PC, so AX here is
        # the value the printf prints.  Same KV-cached model path as every op.
        if op == isa.PRTF and out is not None:
            out.append(ax & 0xFF)

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

        # -- bounded eviction: prune every ``prune_interval`` tokens.  Pin ONLY the
        # data-segment STORE rows (the ``n_seed`` seed frames' MEM tokens, at local
        # offset ``_MEM_MARKER_LOCAL`` in each) so the bundled data ``Q``
        # (content-addressed, loaded at arbitrary future steps) is never recency-
        # evicted; every register frame — including the seed frames' dead marker
        # rows — still evicts, so the cache stays small.
        protect_positions = seed_store_positions
        tokens_since_prune += V.FRAME_LEN
        if evict and tokens_since_prune >= prune_interval:
            for b in range(n_blocks):
                caches[b].evict(cos_threshold, prune_interval, zero_eps, recency_eps,
                                protect_positions=protect_positions)
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
