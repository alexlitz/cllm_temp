"""DRAFT-DRIVEN, STRUCTURE-AWARE KV eviction schedule (C4_EVICT_SCHEDULE).

The perfect draft (``pf_speculative.draft_pf_program``) materialises EVERY VM
step deterministically before a single model forward runs — so the ENTIRE memory
access pattern is known up front.  This module turns that trace into a
**liveness schedule**: a single O(stores + loads) pass over the draft computes,
per store-KV row, its EXACT eviction step, and the block-verify then just DROPS
the scheduled rows instead of the O(S^2) pairwise CONTENT comparison the fused
eviction (``evict_all_blocks_fused`` -> ``prune_keep_mask_batched``) runs.

Why this is byte-identical
--------------------------
The fused content eviction drops a **§Memory store-KV row** by exactly two
mechanisms (see ``nibble_kv_prune.KVCache.prune`` / ``prune_keep_mask_batched``):

  1. **Supersession (mechanism 1, EXACT metric).**  Two store rows with a
     VERBATIM-identical key are near-duplicates; the OLDER (smaller position) is
     dropped (ALiBi latest-write-wins).  On a content-addressed §Memory head the
     store key is a pure function of the ADDRESS (``W_k @ ADDR_BIN``), so two
     stores have an identical key IFF they wrote the SAME address.  ⇒ a store to
     address A is superseded (dropped) the moment a STRICTLY-NEWER store to A
     exists in the cache.  From the draft that is: the store at frame ``fi`` to
     address A is evictable once the NEXT store to A (frame ``fi' > fi``) has been
     committed.

  2. **Free / zero-value recency (mechanism 3 + 2b).**  A store that writes VALUE
     0 (a ``free``/ZFOD zeroing) contributes 0 to the softmax1 numerator; on the
     content-addressed head its recency horizon (``ceil_score=0`` for a
     zero-value row) drops it once ``exp(-slope*dist) < recency_eps`` — i.e. once
     it is ``> -log(recency_eps)/slope`` tokens behind the newest surviving row.
     A NON-zero live store is NEVER recency-dropped on a content-addressed head
     (the horizon is a NO-OP for it — it is retrieved by ADDRESS at an arbitrary
     future step).  So a live store is dropped ONLY by supersession (1) or by
     being itself a zero-value row (2).

That is the WHOLE memory-head policy.  The schedule reproduces (1) and (2)
EXACTLY off the draft — no key comparison, no cdist, no cosine matmul.  It is
therefore byte-identical to the content path's SURVIVOR SET on the memory heads,
which is the only place the O(S^2) content comparison does real work once the
register/ingest heads are window-bounded (DROP-KV local attention) or dead-value
(mechanism 2a).

Scope / honesty
---------------
This schedule governs the **store-KV rows** (the memory / stack / LEV heads,
whose keys are content=ADDRESS).  It does NOT try to reproduce the cosine
near-dup + recency-horizon eviction of the REGISTER-marker heads — on the fast
path those heads are already window-bounded (local attention drops old rows) or
carry all-zero values (mechanism 2a evicts wholesale), so the schedule and the
content path agree there trivially (both drop the same non-store rows: none of
them are store rows, and the schedule keeps non-store rows only until the local
window / dead-value rule the cache already applies).  When the classic full-H
cache is used WITHOUT the local/dead-value split, the register heads still need
the content path; the schedule then covers only the store rows and the caller
must keep the content path for the rest (``schedule_only_memory=False`` is the
guard).  The block-verify wiring uses the schedule for the store rows and leaves
the (already cheap, window-bounded) non-store handling to the existing commit
trim — see ``evict_all_blocks_scheduled``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import c4_min.blogspec_vocab as V
from c4_min.nibble_pure_forward import _MEM_MARKER_LOCAL


# The absolute stream position of frame ``fi``'s store-KV (MEM marker) row.
# tokens = [BOS] + frame0(init) + frame1 + ... ; frame fi occupies
# [1 + FRAME_LEN*fi, 1 + FRAME_LEN*(fi+1)), and its MEM marker sits at +MEM_LOCAL.
def store_row_position(frame_idx: int) -> int:
    return 1 + V.FRAME_LEN * frame_idx + _MEM_MARKER_LOCAL


@dataclass
class StoreEntry:
    frame_idx: int          # the draft frame index that emitted this store
    position: int           # absolute stream position of its KV row
    addr: int               # the address it wrote (the content key on a mem head)
    val: int                # the value it stored (0 => a free/ZFOD zeroing row)
    evict_frame: Optional[int] = None   # frame at/after which this row is evictable
    evict_reason: str = "live"          # "superseded" | "freed" | "live"


@dataclass
class EvictionSchedule:
    """The precomputed liveness schedule.

    ``stores`` — every store-KV row, with its eviction frame + reason.
    ``drop_at_frame`` — ``frame_idx -> [absolute positions to DROP]``: at the
        eviction ROUND that first covers ``frame_idx`` (i.e. once the verify has
        committed step ``frame_idx``), these store rows are provably inert and are
        dropped.  This is the O(1)/step lookup that replaces the content compare.
    ``drop_position_at`` — ``absolute_position -> evict_frame`` (the per-row form).
    """
    stores: List[StoreEntry] = field(default_factory=list)
    drop_at_frame: Dict[int, List[int]] = field(default_factory=dict)
    drop_position_at: Dict[int, int] = field(default_factory=dict)
    # bookkeeping / report
    n_stores: int = 0
    n_superseded: int = 0
    n_freed: int = 0
    n_live: int = 0


def build_eviction_schedule(draft, *, slope_min: float = None,
                            recency_eps: float = 1e-6,
                            zero_eps: float = 1e-9,
                            supersession_only: bool = True) -> EvictionSchedule:
    """Single O(stores + loads) liveness pass over the perfect draft.

    ``supersession_only`` (default True) is the BYTE-EXACT HYBRID mode: the schedule
    encodes ONLY the SUPERSESSION drops — a store to address A is dropped when a
    strictly-newer store to A exists.  Supersession is HEAD-INDEPENDENT (a
    verbatim-same-address newer store is a near-dup on EVERY head that keys the
    address, so the older row is dropped by all of them), so it is provably
    byte-identical to the content path's mechanism-1 SURVIVOR SET.  This is the sole
    O(S^2) cost (the pairwise cdist/cosine near-dup), and it is what the schedule
    removes.  The O(S) zero-value / recency mechanisms (2a/2b/3) — whose per-head
    value embeddings the ADDRESS-only schedule cannot see (a freed PSH row is
    zero-VALUE on the memory head but may carry a live value on the stack head) — are
    left to the content path's ``skip_mech1=True`` elementwise pass, so those drops
    stay byte-exact per-head.  Set ``supersession_only=False`` for the
    ADDRESS-value-only heuristic (freed rows scheduled off ``val==0`` + recency
    horizon) — faster still but only byte-exact when NO cross-head zero-value
    ambiguity exists (pure heap programs like malloc); it can differ by a KV row on
    stack-PSH-heavy programs (loop_countdown), so it is NOT the default.

    Computes, per store-KV row, its EXACT eviction frame:

      * SUPERSEDED — the NEXT store to the SAME address (mechanism 1 exact).  A
        hash-by-address gives this in O(stores): as we scan store frames in order,
        the store currently "live" for address A is superseded the instant the
        next store to A appears.  Its eviction frame is that next store's frame.

      * FREED — a store whose VALUE is 0 (``|val| <= zero_eps`` as an integer: val
        == 0) is a zero-value row.  On the content-addressed head it is
        recency-dropped once it is ``> horizon`` tokens behind the newest row,
        where ``horizon = -log(recency_eps)/slope``.  We translate that to a frame
        offset (``ceil(horizon / FRAME_LEN)`` frames) so the schedule drops it at
        the SAME round the content recency horizon would.  ``slope_min`` is the
        SMALLEST global-head ALiBi slope (the widest horizon); passing it makes the
        freed-row drop CONSERVATIVE (never earlier than any head would).  When
        ``slope_min`` is None the freed row is dropped as soon as it is superseded
        OR at end (matches the content path only if the memory head's slope window
        is <= one frame; the caller passes the real slope for exactness).

      * LIVE — a non-zero store never re-written and never freed stays in the cache
        forever (the working-set / live-heap footprint).  This is EXACTLY the
        content path's behaviour (content-addressed head recency NO-OP on a live
        store).  Its eviction frame is None (never dropped).

    A store that is BOTH superseded and (would be) freed is dropped at the EARLIER
    of the two frames (supersession is monotone-earlier in practice, since the
    superseding store makes the old row a near-dup immediately).
    """
    store_log: Dict[int, Tuple[int, int]] = draft.store_log or {}
    sched = EvictionSchedule()

    # 1) materialise every store-KV row in FRAME order.
    entries: List[StoreEntry] = []
    for fi in sorted(store_log.keys()):
        addr, val = store_log[fi]
        entries.append(StoreEntry(frame_idx=fi,
                                  position=store_row_position(fi),
                                  addr=int(addr), val=int(val)))
    sched.stores = entries
    sched.n_stores = len(entries)

    # 2) SUPERSESSION: for each address, the previous live store is superseded at
    #    the frame of the NEXT store to that address.  One O(stores) forward scan.
    last_store_of_addr: Dict[int, StoreEntry] = {}
    for e in entries:
        prev = last_store_of_addr.get(e.addr)
        if prev is not None:
            # ``prev`` (older store to the same address) is superseded the instant
            # ``e`` (the newer verbatim-same-address key) is committed.
            prev.evict_frame = e.frame_idx
            prev.evict_reason = "superseded"
        last_store_of_addr[e.addr] = e

    # 3) FREE / zero-value recency (ONLY in the address-value heuristic mode): a
    #    store whose VALUE is 0 is a zero-value row.  Its recency horizon on the
    #    content head is ``-log(recency_eps)/slope`` tokens.  Skipped in the default
    #    byte-exact HYBRID mode (``supersession_only``) where the content path's O(S)
    #    ``skip_mech1`` pass handles zero-value/recency per-head instead.
    if (not supersession_only) and slope_min is not None and slope_min > 0.0:
        horizon_tokens = -math.log(recency_eps) / slope_min
    else:
        horizon_tokens = None
    n_steps = draft.step_count
    if horizon_tokens is not None:
        # The content path keeps a zero-value row on a content-addressed head iff
        # ``exp(-slope*(newest_pos - pos)) >= recency_eps`` for the head with the
        # SMALLEST slope (the keep is a UNION across heads, and the smallest-slope
        # head has the widest horizon), i.e. iff ``newest_pos - pos <=
        # horizon_tokens``.  ``newest_pos`` is the newest SURVIVING position in the
        # memory head's cache, which — under content-bounding — is a STORE row.  So
        # the freed row is dropped at the FIRST eviction round whose newest committed
        # store-row position exceeds ``pos + horizon_tokens``.  We compute that off
        # the store-row positions directly (positions ascend with frame index), NOT a
        # rounded frame offset, so the drop frame is EXACT vs the content recency
        # horizon (no ULP/round-up drift, no clamp-to-n_steps that would over-drop at
        # the final round — a freed row whose horizon is never reached stays LIVE,
        # exactly as the content path leaves it).
        store_positions = [e.position for e in entries]      # ascending by frame
        store_frames = [e.frame_idx for e in entries]
        import bisect
        for e in entries:
            if e.val != 0:
                continue
            thresh_pos = e.position + horizon_tokens
            # first store row STRICTLY LATER than this one whose position exceeds the
            # horizon threshold — that round's newest store position triggers the drop.
            j = bisect.bisect_right(store_positions, thresh_pos)
            if j >= len(store_positions):
                freed_frame = None                           # horizon never reached
            else:
                freed_frame = store_frames[j]
            if freed_frame is not None:
                if e.evict_frame is None or freed_frame < e.evict_frame:
                    e.evict_frame = freed_frame
                    e.evict_reason = "freed"

    # 4) tally + build the round-indexed drop map.
    for e in entries:
        if e.evict_frame is None:
            sched.n_live += 1
            continue
        if e.evict_reason == "superseded":
            sched.n_superseded += 1
        elif e.evict_reason == "freed":
            sched.n_freed += 1
        sched.drop_at_frame.setdefault(e.evict_frame, []).append(e.position)
        sched.drop_position_at[e.position] = e.evict_frame
    return sched


# ===========================================================================
# PART B: DRAFT-DIRECT CAM READ — per-read address -> exact resolved KV row.
# ===========================================================================
@dataclass
class ResolvedRead:
    """One CAM read the draft performs, resolved to the exact KV store row it reads.

    ``head``       — "mem" (§Memory LI/LC), "pop" (stack head, incl. LEV's MEM[BP]),
                     or "lev" (LEV return-PC head, MEM[BP+4]).
    ``read_frame`` — the draft frame index whose step performs the read.
    ``addr``       — the address queried (the CAM binary-address key).
    ``store_frame``— the frame that stored the LATEST value to ``addr`` before the read
                     (latest-write-wins, exactly the softmax1-CAM+ALiBi winner), or
                     None when the address is UNWRITTEN (softmax1 +1 sink -> ZFOD 0).
    ``store_position`` — the absolute stream position of that store's KV (MEM) row, or
                     None for a ZFOD read.
    ``value``      — the value the direct gather returns (the store's val, or 0 ZFOD).
    """
    head: str
    read_frame: int
    addr: int
    store_frame: Optional[int]
    store_position: Optional[int]
    value: int


def resolve_load_rows(draft) -> Dict[int, List[ResolvedRead]]:
    """PART B: resolve every drafted CAM read to the exact KV row its address reads.

    For each read ``(head, addr)`` at frame ``F`` (from ``draft.read_log``), find the
    store to ``addr`` with the LARGEST frame ``< F`` (latest-write-wins).  That is
    EXACTLY the row the softmax1-CAM+ALiBi head selects: the address CAM gives every
    store to ``addr`` an identical top score, ALiBi's recency then picks the most
    recent, and softmax1's +1 sink returns 0 when no store to ``addr`` exists (ZFOD).

    A single O(stores + reads) pass: walk frames in order, keep the running "latest
    store frame per address"; at each read frame, snapshot the current latest store
    for its address.  Because the read at frame ``F`` sees only stores committed at
    frames ``< F`` (the store's KV row precedes the read's query row in the causal
    stream), we advance the running map to just-before ``F`` before resolving.

    Returns ``{read_frame: [ResolvedRead, ...]}``.  The direct gather then reads
    ``store_log[store_frame]``'s value (or 0) — no O(K) attention score at all.
    """
    store_log: Dict[int, Tuple[int, int]] = draft.store_log or {}
    read_log: Dict[int, List[Tuple[str, int]]] = draft.read_log or {}
    # latest store frame per address, built incrementally as we sweep frames.
    latest: Dict[int, int] = {}
    out: Dict[int, List[ResolvedRead]] = {}
    # the union of all frames that either store or read, in ascending order.
    all_frames = sorted(set(store_log) | set(read_log))
    # a store at frame S becomes visible to any read at frame > S.  We process frames
    # in order; at frame F we FIRST resolve reads (they see stores at frames < F), THEN
    # commit F's own store (visible to later reads).  Since a frame is EITHER a store
    # OR a read in this VM (a store frame carries no CAM read and vice-versa, except SI
    # which reads then stores — handled: the pop read of the store ADDRESS at frame F
    # resolves against stores < F, and F's value store is committed after), this order
    # is exact.
    for f in all_frames:
        for (head, addr) in read_log.get(f, []):
            sf = latest.get(addr)
            if sf is None:
                out.setdefault(f, []).append(
                    ResolvedRead(head, f, addr, None, None, 0))
            else:
                sval = store_log[sf][1]
                out.setdefault(f, []).append(
                    ResolvedRead(head, f, addr, sf, store_row_position(sf), sval))
        if f in store_log:
            addr, _val = store_log[f]
            latest[addr] = f
    return out


def positions_to_drop_through(sched: EvictionSchedule, up_to_frame: int) -> List[int]:
    """All store-row absolute positions whose eviction frame is ``<= up_to_frame``
    — the cumulative drop set the block-verify applies at the eviction round that
    has committed through frame ``up_to_frame``.  O(#dropped-so-far)."""
    out: List[int] = []
    for frame, positions in sched.drop_at_frame.items():
        if frame <= up_to_frame:
            out.extend(positions)
    return out


def sorted_evict_frames(sched: EvictionSchedule) -> List[int]:
    """The ASCENDING list of distinct eviction frames — lets the driver walk a
    FRONTIER pointer and emit only the NEW dead positions each round (true
    O(steps) total, no cumulative rescan).  Paired with ``positions_new_at``."""
    return sorted(sched.drop_at_frame.keys())


def positions_new_at(sched: EvictionSchedule, frame: int) -> List[int]:
    """The store-row positions that become dead EXACTLY at eviction frame ``frame``
    (the incremental drop set — already-dropped rows are not in the cache, so an
    incremental drop is byte-identical to the cumulative one)."""
    return sched.drop_at_frame.get(frame, [])
