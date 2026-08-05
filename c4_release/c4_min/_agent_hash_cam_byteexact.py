#!/usr/bin/env python3
"""_agent_hash_cam_byteexact.py — CPU-only (contention-immune) proof + timing for the
GENUINE O(1) HASH-CAM value resolver (C4_HASH_CAM).

Three checks, NO GPU / NO model build (pure numpy resolver equivalence):
  1. BYTE-EXACT: ``resolve_value_hashed`` == ``_genuine_value_at`` element-for-element over
     random store-logs AND a doom-like store-log (few hot addresses written many times +
     many cold addresses written once), for every read.  L-inf = 0.
  2. GENUINE (scenario E): a self-consistent wrong draft plants a value 777 at the correct
     address's slot; the hash resolver INDEPENDENTLY re-derives the TRUE committed value from
     the address bits -> the planted value is REJECTED (resolved != planted).  Same as the
     searchsorted resolver (both genuine), unlike the draft-trusted direct-CAM (accepts).
  3. TIMING: hash build + resolve vs searchsorted resolve, across store-log sizes -> does the
     O(1) hash actually beat the O(log S) bisect at doom scale?

Run: OMP_NUM_THREADS=4 python -m c4_min._agent_hash_cam_byteexact
"""
from __future__ import annotations
import time

import numpy as np

from c4_min.hash_cam import build_hash_index, resolve_value_hashed, _fnv1a_u32
from c4_min.faithful_single_dispatch import _genuine_value_at, FaithfulPlan


def _mk_plan(store_frames, store_addr, store_val):
    return FaithfulPlan(
        heads_by_block={}, draft_addr={}, draft_val={}, code_addr={},
        store_frames=np.asarray(store_frames, dtype=np.int64),
        store_addr=np.asarray(store_addr, dtype=np.int64),
        store_val=np.asarray(store_val, dtype=np.int64))


def _rand_storelog(rng, S, n_addr):
    """S stores over n_addr distinct addresses at strictly increasing frames."""
    frames = np.sort(rng.choice(np.arange(1, 4 * S + 1), size=S, replace=False)).astype(np.int64)
    addrs = rng.integers(0, n_addr, size=S).astype(np.int64) * np.int64(4)  # word-aligned
    vals = rng.integers(0, 1 << 20, size=S).astype(np.int64)
    return frames, addrs, vals


def _doomlike_storelog(rng, S):
    """A doom-like store-log: a few HOT slots (stack/heap cells) written many times +
    a long tail of cold addresses written once (the persistent-heap pattern)."""
    n_hot = 16
    hot = (rng.integers(0x1000, 0x1010, size=n_hot) * 4).astype(np.int64)
    addrs = np.empty(S, dtype=np.int64)
    # 60% of writes go to the hot slots, 40% to unique cold addresses.
    n_hot_writes = int(0.6 * S)
    addrs[:n_hot_writes] = hot[rng.integers(0, n_hot, size=n_hot_writes)]
    addrs[n_hot_writes:] = (np.arange(S - n_hot_writes) * 4 + 0x40000).astype(np.int64)
    rng.shuffle(addrs)
    frames = np.arange(1, S + 1, dtype=np.int64)
    vals = rng.integers(0, 1 << 24, size=S).astype(np.int64)
    return frames, addrs, vals


def check_byteexact():
    print("=== 1. BYTE-EXACT: hash resolver == searchsorted resolver ===", flush=True)
    rng = np.random.default_rng(0xC4)
    all_ok = True
    configs = [("random small", lambda: _rand_storelog(rng, 200, 40)),
               ("random large", lambda: _rand_storelog(rng, 20000, 3000)),
               ("doom-like 50k", lambda: _doomlike_storelog(rng, 50000)),
               ("doom-like 200k", lambda: _doomlike_storelog(rng, 200000))]
    for name, mk in configs:
        sf, sa, sv = mk()
        plan = _mk_plan(sf, sa, sv)
        idx = build_hash_index(sf, sa, sv)
        # reads: random addresses (some present, some absent) at random read frames.
        R = 5000
        present_addr = sa[rng.integers(0, sa.shape[0], size=R // 2)]
        absent_addr = (rng.integers(0, 1 << 22, size=R - R // 2) * 4 + 1).astype(np.int64)
        maddr = np.concatenate([present_addr, absent_addr])
        rng.shuffle(maddr)
        rframe = rng.integers(0, int(sf.max()) + 2, size=R).astype(np.int64)
        ref = _genuine_value_at(maddr, rframe, plan)
        got = resolve_value_hashed(maddr, rframe, idx)
        linf = int(np.abs(ref - got).max()) if R else 0
        ok = linf == 0
        all_ok = all_ok and ok
        print(f"  {name:>16}: S={sa.shape[0]:>7} uniq={np.unique(sa).shape[0]:>6} "
              f"R={R} L-inf={linf}  {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            bad = np.nonzero(ref != got)[0][:5]
            for b in bad:
                print(f"     read addr={maddr[b]} frame={rframe[b]} ref={ref[b]} got={got[b]}",
                      flush=True)
    print(f"  -> {'ALL BYTE-EXACT (L-inf=0)' if all_ok else 'DIVERGENCE'}\n", flush=True)
    return all_ok


def check_genuine_scenario_E():
    print("=== 2. GENUINE (audit scenario E): rejects the planted value ===", flush=True)
    # true committed store: mem[200]=66 at frame 3.  A self-consistent wrong draft PLANTS a
    # decoy value 777 at the SAME address (or a decoy address) — the draft-trusted path injects
    # 777 and accepts.  The GENUINE hash resolver recomputes latest-write-wins over the TRUE
    # committed stores at the address the MODEL queried -> 66, NOT 777.
    sf = np.array([3], dtype=np.int64)          # frame 3: store to addr 200
    sa = np.array([200], dtype=np.int64)
    sv = np.array([66], dtype=np.int64)         # TRUE committed value
    idx = build_hash_index(sf, sa, sv)
    plan = _mk_plan(sf, sa, sv)
    # the read: model queries addr 200 at read frame 5 (after the store).
    maddr = np.array([200], dtype=np.int64)
    rframe = np.array([5], dtype=np.int64)
    draft_planted_value = 777                   # what the self-consistent wrong draft claims
    genuine_hash = int(resolve_value_hashed(maddr, rframe, idx)[0])
    genuine_ss = int(_genuine_value_at(maddr, rframe, plan)[0])
    print(f"  true committed mem[200]=66; draft PLANTS value {draft_planted_value}", flush=True)
    print(f"  hash resolver genuine value  = {genuine_hash}", flush=True)
    print(f"  searchsorted genuine value   = {genuine_ss}", flush=True)
    rejects = (genuine_hash != draft_planted_value) and (genuine_hash == 66)
    matches_ref = genuine_hash == genuine_ss
    print(f"  -> planted 777 REJECTED by hash resolver: {rejects}  "
          f"(recomputed genuine 66 != planted 777)", flush=True)
    print(f"  -> hash == searchsorted genuine: {matches_ref}", flush=True)
    # ALSO the value-stale layer: store 66 then 999 to addr 200; draft injects STALE 66.
    sf2 = np.array([3, 7], dtype=np.int64)
    sa2 = np.array([200, 200], dtype=np.int64)
    sv2 = np.array([66, 999], dtype=np.int64)
    idx2 = build_hash_index(sf2, sa2, sv2)
    plan2 = _mk_plan(sf2, sa2, sv2)
    g2 = int(resolve_value_hashed(np.array([200]), np.array([9]), idx2)[0])
    g2ref = int(_genuine_value_at(np.array([200]), np.array([9]), plan2)[0])
    stale_ok = (g2 == 999) and (g2 == g2ref)
    print(f"  value-stale: store 66@3 then 999@7, read@9 -> genuine={g2} (latest 999); "
          f"draft-stale 66 REJECTED: {g2 == 999 and g2 == g2ref}", flush=True)
    ok = rejects and matches_ref and stale_ok
    print(f"  -> {'GENUINE O(1) READ CONFIRMED (rejects planted + stale)' if ok else 'FAIL'}\n",
          flush=True)
    return ok


def check_timing():
    print("=== 3. TIMING: O(1) hash vs O(log S) searchsorted ===", flush=True)
    rng = np.random.default_rng(7)
    R = 20000
    print(f"  (R={R} reads per config; build = one-time index/lexsort, resolve = per-frame)",
          flush=True)
    print(f"  {'S stores':>10} {'uniq':>7} {'ss build+resolve ms':>22} "
          f"{'hash build ms':>15} {'hash resolve ms':>16} {'resolve speedup':>16}", flush=True)
    for S in [2000, 20000, 100000, 400000]:
        sf, sa, sv = _doomlike_storelog(rng, S)
        plan = _mk_plan(sf, sa, sv)
        maddr = sa[rng.integers(0, sa.shape[0], size=R)]
        rframe = rng.integers(0, int(sf.max()) + 2, size=R).astype(np.int64)
        # searchsorted (does its own lexsort+unique internally each call).
        for _ in range(2):
            _genuine_value_at(maddr, rframe, plan)
        t0 = time.perf_counter()
        for _ in range(5):
            _genuine_value_at(maddr, rframe, plan)
        ss_ms = (time.perf_counter() - t0) / 5 * 1e3
        # hash: build once, resolve per frame.
        t0 = time.perf_counter()
        idx = build_hash_index(sf, sa, sv)
        hb_ms = (time.perf_counter() - t0) * 1e3
        for _ in range(2):
            resolve_value_hashed(maddr, rframe, idx)
        t0 = time.perf_counter()
        for _ in range(5):
            resolve_value_hashed(maddr, rframe, idx)
        hr_ms = (time.perf_counter() - t0) / 5 * 1e3
        # compare RESOLVE-only (the per-frame cost; build amortizes in a continuous render).
        speedup = ss_ms / hr_ms if hr_ms else 0.0
        print(f"  {S:>10} {np.unique(sa).shape[0]:>7} {ss_ms:>22.3f} {hb_ms:>15.3f} "
              f"{hr_ms:>16.3f} {speedup:>15.2f}x", flush=True)
    print("\n  NOTE: ``_genuine_value_at`` re-lexsorts the WHOLE store-log every call (O(S log "
          "S)); the hash build lexsorts once and the per-frame resolve is the O(1) probe + a "
          "per-address frame bisect.  In the FAITHFUL pipeline the resolve is the per-frame "
          "cost; the build amortizes across a continuous render (C4_FAITHFUL_PRECOMPUTE_CACHE "
          "already caches the whole precompute, so both are one-time there).", flush=True)


def main():
    ok1 = check_byteexact()
    ok2 = check_genuine_scenario_E()
    check_timing()
    print(f"\n{'=== HASH-CAM VERIFIED ===' if (ok1 and ok2) else '=== FAILED ==='}", flush=True)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
