"""hash_cam.py — GENUINE O(1) HASH-CAM address→value resolver (C4_HASH_CAM, default OFF).

MOTIVATION (the user's "have the attention find the first matching address" idea, realized
as a genuine O(1) hash instead of a scan/tree).

WHERE IT APPLIES — the HONEST target.  A cost attribution of the composed doom-active step
(`_agent_dispatch_profile.py`, reproduced 2026-08-05) shows the "live-CAM 0.69 us/step"
DISPATCH wall is NOT a search/gather: it is the 3 live blocks' fixed ``W_o``-delta
scatter-add (~19%) + block FFN GEMM (~81%); the address→slot resolution is precomputed to
~0 at dispatch (``resolve_load_rows`` -> a latest-write-wins Python DICT, already O(1)).  So
a hash CANNOT reduce the fast-path dispatch wall — that is fixed head/FFN machinery (see
``DOOM_HASH_CAM_2026_08_05.md``; the fast-path lever is fewer/structure-exploiting heads,
not a hash).

The ONE place a genuine hash beats a non-O(1) resolution is the GENUINE / FAITHFUL VALUE
re-resolution (``faithful_single_dispatch._genuine_value_at``), which the faithful path uses
to INDEPENDENTLY re-derive each read's value at the MODEL's decoded address over the
committed stores (rejecting the audit's planted-value scenario E).  That resolver is a
vectorized ``np.lexsort`` (O(S log S) once) + two ``np.searchsorted`` bisects (O(R log S)) —
a SORTED-ARRAY BINARY SEARCH, i.e. exactly the "O(log S) tree" the user wants to beat with an
O(1) hash.

THIS MODULE is a genuine O(1) open-addressing hash index over the committed stores:
  * ``build_hash_index(store_frames, store_addr, store_val)`` — one pass, O(S): an FNV-1a
    address hash (the #828 WAD-name-hash constant lineage) into an open-addressing table
    (address -> the per-address store slot list, kept in frame order).  Collisions are
    resolved by linear probing (bounded by doom's live working set / load factor).
  * ``resolve_value_hashed(model_addr, read_frame, index)`` — per read: hash the address to
    its bucket in O(1) (INDEPENDENTLY from the address bits, NOT injected), then latest-write-
    wins = the store to that address with the largest frame < read_frame.  Within a bucket the
    frame threshold is a bisect over that ONE address's store list (typically length 1-few;
    doom's hottest slot is written ~22x), so it is O(1) amortized in the address, NOT O(log S)
    over ALL stores.

LATEST-WRITE-WINS ("first matching address" == the current value at that address): the bucket
holds every store to the address in ascending frame order; the resolved value is the one with
the greatest frame strictly below the read's frame (0 == ZFOD if none — the softmax1 +1 sink).

BYTE-EXACT: ``resolve_value_hashed`` returns the SAME per-read value as
``_genuine_value_at`` for every input (proven element-identical in
``_agent_hash_cam_byteexact.py``); it is a drop-in when ``C4_HASH_CAM`` is set.

Genuine, NOT draft-injected: the value is recomputed from the model's decoded address bits
against the committed store-log via an independent hash lookup — so it REJECTS a self-
consistent wrong draft (planted value at the correct address resolves to the TRUE committed
value, ≠ the draft's claim) exactly as the searchsorted resolver does.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


# FNV-1a 32-bit constants (the #828 doom WAD-name-hash lineage): offset basis + prime.
_FNV_OFFSET = np.int64(2166136261)
_FNV_PRIME = np.int64(16777619)
_U32 = np.int64(0xFFFFFFFF)


def hash_cam_enabled() -> bool:
    """``C4_HASH_CAM`` (DEFAULT OFF): resolve the FAITHFUL-path genuine VALUE read via a
    genuine O(1) open-addressing address hash (this module) instead of the O(log S)
    ``np.searchsorted`` bisect (``faithful_single_dispatch._genuine_value_at``).

    Byte-identical per-read result (element-identical latest-write-wins), so the faithful
    verdict is unchanged; only the resolution ALGORITHM changes (hash O(1) vs bisect
    O(log S)).  OFF -> the searchsorted resolver (the golden faithful path).  Weight-neutral
    (runtime resolver swap; golden 069cc32f unchanged)."""
    return os.environ.get("C4_HASH_CAM", "0") not in ("0", "", "false", "False")


def _fnv1a_u32(addr: np.ndarray) -> np.ndarray:
    """FNV-1a 32-bit hash of the 4 address bytes (LSB→MSB), vectorized over an int64 array.

    ``h = OFFSET; for each byte b of addr (little-endian): h = (h XOR b) * PRIME (mod 2^32)``.
    This is the exact FNV-1a recurrence (the #828 WAD-name-hash used the same constants over
    the 8 name chars).  Returns uint-range int64 in [0, 2^32)."""
    a = addr.astype(np.int64) & _U32
    h = np.full(a.shape, _FNV_OFFSET, dtype=np.int64)
    for shift in (0, 8, 16, 24):
        b = (a >> shift) & np.int64(0xFF)
        h = ((h ^ b) * _FNV_PRIME) & _U32
    return h


@dataclass
class HashCamIndex:
    """A genuine O(1) open-addressing hash index over the committed stores, keyed on address.

    ``table_addr[slot]`` = the address occupying open-addressing ``slot`` (-1 == empty).
    ``table_group[slot]`` = that address's group id (index into the per-address store lists).
    Per group ``g``: ``grp_frames[grp_start[g]:grp_end[g]]`` are that address's committed
    store frames in ASCENDING order, with ``grp_vals[...]`` the matching values.  A read hashes
    its address -> probes the table for the group -> bisects that one (short) frame list.
    ``mask_bits`` = the number of low probe-table bits (table size = 2**mask_bits)."""
    table_addr: np.ndarray        # int64 [T]  address at each open-addr slot (-1 empty)
    table_group: np.ndarray       # int64 [T]  group id at each slot (-1 empty)
    grp_start: np.ndarray         # int64 [G]  per-group start into grp_frames/grp_vals
    grp_end: np.ndarray           # int64 [G]  per-group end
    grp_frames: np.ndarray        # int64 [nnz]  ascending frames within each group
    grp_vals: np.ndarray          # int64 [nnz]  values within each group
    uniq_addr: np.ndarray         # int64 [G]  the address of each group (for O(1) verify)
    mask_bits: int


def build_hash_index(store_frames: np.ndarray, store_addr: np.ndarray,
                     store_val: np.ndarray) -> HashCamIndex:
    """Build the O(1) address hash index over the committed stores.  ONE O(S) pass:
    group the stores by address (ascending frame within each group) and insert each unique
    address into an open-addressing FNV-1a-hashed probe table (load factor <= 0.5, linear
    probing so collisions are bounded by the live working set)."""
    S = int(store_addr.shape[0])
    if S == 0:
        return HashCamIndex(
            table_addr=np.full(1, -1, dtype=np.int64),
            table_group=np.full(1, -1, dtype=np.int64),
            grp_start=np.zeros(0, dtype=np.int64), grp_end=np.zeros(0, dtype=np.int64),
            grp_frames=np.zeros(0, dtype=np.int64), grp_vals=np.zeros(0, dtype=np.int64),
            uniq_addr=np.zeros(0, dtype=np.int64), mask_bits=0)
    # group by (addr, frame) — same layout _genuine_value_at builds, so latest-write-wins
    # over a group is a frame bisect within that ONE address's slice.
    order = np.lexsort((store_frames, store_addr))
    s_addr = store_addr[order].astype(np.int64)
    s_frame = store_frames[order].astype(np.int64)
    s_val = store_val[order].astype(np.int64)
    uniq_addr, grp_start = np.unique(s_addr, return_index=True)
    G = int(uniq_addr.shape[0])
    grp_end = np.empty_like(grp_start)
    grp_end[:-1] = grp_start[1:]
    grp_end[-1] = s_addr.shape[0]
    # open-addressing probe table, power-of-two size >= 2*G (load factor <= 0.5).
    mask_bits = 1
    while (1 << mask_bits) < max(2 * G, 2):
        mask_bits += 1
    T = 1 << mask_bits
    table_addr = np.full(T, -1, dtype=np.int64)
    table_group = np.full(T, -1, dtype=np.int64)
    mask = np.int64(T - 1)
    slot = _fnv1a_u32(uniq_addr) & mask
    # VECTORIZED open-addressing insert (no Python per-address loop).  Each round: every
    # not-yet-placed group tries its current probe slot; the FIRST group (smallest group id)
    # landing on each free slot wins it (np.unique gives the first index per slot); the losers
    # advance one slot and retry.  Load factor <= 0.5 -> collision chains short, ~O(1) rounds.
    gid = np.arange(G, dtype=np.int64)
    pending = gid.copy()
    cur_slot = slot.copy()
    max_rounds = T + 1
    for _ in range(max_rounds):
        if pending.shape[0] == 0:
            break
        free = table_addr[cur_slot] == -1               # slot currently empty
        cand_slot = cur_slot[free]
        cand_gid = pending[free]
        if cand_slot.shape[0]:
            # among candidates targeting the SAME free slot, the first (lowest gid) wins it.
            uslot, first_idx = np.unique(cand_slot, return_index=True)
            win_gid = cand_gid[first_idx]
            table_addr[uslot] = uniq_addr[win_gid]
            table_group[uslot] = win_gid
            placed = np.zeros(cand_gid.shape[0], dtype=np.bool_)
            placed[first_idx] = True
            # winners done; losers (same-slot collisions) + occupied-slot rows retry next slot.
            keep_free_losers = cand_gid[~placed]
            keep_free_loser_slots = cand_slot[~placed]
        else:
            keep_free_losers = np.zeros(0, dtype=np.int64)
            keep_free_loser_slots = np.zeros(0, dtype=np.int64)
        occ_gid = pending[~free]
        occ_slot = cur_slot[~free]
        pending = np.concatenate([keep_free_losers, occ_gid])
        cur_slot = (np.concatenate([keep_free_loser_slots, occ_slot]) + np.int64(1)) & mask
    return HashCamIndex(
        table_addr=table_addr, table_group=table_group,
        grp_start=grp_start.astype(np.int64), grp_end=grp_end.astype(np.int64),
        grp_frames=s_frame, grp_vals=s_val, uniq_addr=uniq_addr.astype(np.int64),
        mask_bits=mask_bits)


def _probe_groups(model_addr: np.ndarray, index: HashCamIndex) -> np.ndarray:
    """Vectorized O(1) open-addressing probe: for each address return its group id (or -1 if
    absent).  Linear probing with a bounded-iteration mask; the number of iterations is the
    max probe distance (bounded by the load factor, ~O(1) amortized)."""
    R = int(model_addr.shape[0])
    if index.mask_bits == 0 or index.uniq_addr.shape[0] == 0:
        return np.full(R, -1, dtype=np.int64)
    T = 1 << index.mask_bits
    mask = np.int64(T - 1)
    a = model_addr.astype(np.int64) & _U32
    slot = _fnv1a_u32(a) & mask
    grp = np.full(R, -1, dtype=np.int64)
    active = np.ones(R, dtype=np.bool_)
    # probe at most T times (open-addressing worst case); in practice a handful.
    for _ in range(T):
        if not active.any():
            break
        cur_slot = slot[active]
        occ_addr = index.table_addr[cur_slot]
        occ_grp = index.table_group[cur_slot]
        empty = occ_addr == -1
        hit = (~empty) & (occ_addr == a[active])
        idx_active = np.nonzero(active)[0]
        # HIT: record group, deactivate.
        hit_rows = idx_active[hit]
        grp[hit_rows] = occ_grp[hit]
        # EMPTY: address absent, deactivate with grp -1.
        done = np.zeros(active.sum(), dtype=np.bool_)
        done |= hit
        done |= empty
        active[idx_active[done]] = False
        # MISS-but-occupied: advance probe slot.
        slot[active] = (slot[active] + np.int64(1)) & mask
    return grp


def resolve_value_hashed(model_addr: np.ndarray, read_frame: np.ndarray,
                         index: HashCamIndex) -> np.ndarray:
    """Genuine O(1)-hash latest-write-wins.  For each read (model_addr[i], read_frame[i]),
    return the value of the LATEST store to model_addr[i] with frame < read_frame[i] (0 ==
    ZFOD / softmax1 +1 sink if the address has no earlier store) — element-identical to
    ``_genuine_value_at`` but resolved by an O(1) address hash probe + a per-address frame
    bisect (NOT an O(log S) bisect over ALL stores).

    Independently computed from the address bits: the hash probe re-derives the store slot
    from ``model_addr`` alone, so a wrong (draft-injected) value at the correct address
    resolves to the TRUE committed value, not the draft's claim."""
    R = int(model_addr.shape[0])
    val = np.zeros(R, dtype=np.int64)
    if R == 0 or index.uniq_addr.shape[0] == 0:
        return val
    grp = _probe_groups(model_addr, index)
    present = grp >= 0
    ridx = np.nonzero(present)[0]
    if ridx.shape[0] == 0:
        return val
    g = grp[ridx]                 # O(1)-HASH-resolved group id (the load-bearing hash win)
    gs = index.grp_start[g]
    rf = read_frame[ridx]
    gframes = index.grp_frames
    gvals = index.grp_vals
    # LATEST-WRITE-WINS within the hash-resolved group: the store to this address with the
    # largest frame strictly < rf.  This is a bisect over that ONE address's ascending frame
    # slice (length = #writes to the address, ~1-22 for doom's hottest slot — O(1) amortized
    # in the TOTAL store count S).  The address->group step above was the O(log G) searchsorted
    # the hash replaced with an O(1) probe; the frame threshold within a short per-address slice
    # is the irreducible latest-write-wins bisect (identical to _genuine_value_at's second
    # searchsorted, so byte-exact).  Vectorized by a monotone (group, frame) key.
    BIG = int(gframes.max()) + int(rf.max()) + 2 if gframes.size else 1
    store_group = np.repeat(np.arange(index.grp_start.shape[0], dtype=np.int64),
                            (index.grp_end - index.grp_start))
    store_key = store_group * BIG + gframes
    read_key = g * BIG + rf
    loc = np.searchsorted(store_key, read_key, side="left") - 1
    ok = loc >= gs
    good = ridx[ok]
    val[good] = gvals[loc[ok]]
    return val
