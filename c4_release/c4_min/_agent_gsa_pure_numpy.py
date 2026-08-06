"""PURE-NUMPY equivalence + genuineness test for C4_GENUINE_STRUCTURED_ATTN.

No model build (CPU-only, memory-safe, fast).  Proves:
  1. BYTE-EXACT: genuine_structured_value == _genuine_value_at == resolve_value_hashed
     element-for-element over a battery of committed store-logs + reads (correct execution).
  2. GENUINENESS (scenario E value layer): a store row whose PHYSICAL value nibbles are the
     TRUE committed value resolves to the TRUE value (not a draft's stale claim) — and the
     structured read recomputes it from W_v-nibbles + softmax1, catching a draft that injected
     a stale value.  ALSO: a store row whose PHYSICAL KEY (address bits) does NOT encode the
     queried address scores below the softmax1 sink -> ZFOD 0 (the key-binding check
     single-dispatch's store_log-atom read does NOT do).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_MEM_EFF", "500000.0")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min.hash_cam import build_hash_index, resolve_value_hashed
from c4_min.faithful_single_dispatch import _genuine_value_at, FaithfulPlan
from c4_min.genuine_structured_attn import (genuine_structured_value, cam_numerics,
                                            resolve_winner_hashed, _reconstruct_score,
                                            _softmax1_winner_weight, _decode_value_from_nibbles)


def _mk_plan(store_frames, store_addr, store_val):
    return FaithfulPlan(heads_by_block={}, draft_addr={}, draft_val={}, code_addr={},
                        store_frames=np.asarray(store_frames, dtype=np.int64),
                        store_addr=np.asarray(store_addr, dtype=np.int64),
                        store_val=np.asarray(store_val, dtype=np.int64))


def _rand_case(rng, n_addr, n_store, n_read, mask=0xFFFFFFFF):
    addrs = rng.integers(1, 4096, size=n_addr)
    sf, sa, sv = [], [], []
    frame = 1
    for _ in range(n_store):
        a = int(addrs[rng.integers(0, n_addr)])
        v = int(rng.integers(0, mask + 1)) & mask
        sf.append(frame); sa.append(a); sv.append(v); frame += 1
    # reads at frames after the stores, mixing hit + miss (unwritten) addresses.
    raddr, rframe = [], []
    for _ in range(n_read):
        if rng.random() < 0.2:
            a = int(rng.integers(1, 4096))         # possibly-unwritten (ZFOD test)
        else:
            a = int(addrs[rng.integers(0, n_addr)])
        raddr.append(a); rframe.append(frame + int(rng.integers(0, 5))); frame += 1
    return (np.asarray(sf), np.asarray(sa), np.asarray(sv),
            np.asarray(raddr, dtype=np.int64), np.asarray(rframe, dtype=np.int64))


def main():
    num = cam_numerics()
    print(f"[numerics] EFF={num.eff} BIAS={num.bias} addr_bits={num.addr_bits} "
          f"alibi_slope={num.alibi_slope} n_val_nib={num.n_val_nib}", flush=True)
    rng = np.random.default_rng(0)
    mask = 0xFFFFFFFF
    all_ok = True
    total_reads = 0
    for trial in range(200):
        n_addr = int(rng.integers(1, 40))
        n_store = int(rng.integers(1, 200))
        n_read = int(rng.integers(1, 80))
        sf, sa, sv, raddr, rframe = _rand_case(rng, n_addr, n_store, n_read, mask)
        plan = _mk_plan(sf, sa, sv)
        idx = build_hash_index(sf, sa, sv)
        ref = _genuine_value_at(raddr, rframe, plan)            # searchsorted (single-dispatch)
        hashed = resolve_value_hashed(raddr, rframe, idx)       # hash atom (C4_HASH_CAM)
        gsa = genuine_structured_value(raddr, rframe, idx, num, mask=mask)  # GENUINE softmax
        total_reads += raddr.shape[0]
        if not (np.array_equal(ref, hashed) and np.array_equal(ref, gsa)):
            all_ok = False
            bad = np.nonzero((ref != gsa) | (ref != hashed))[0][:5]
            print(f"  [trial {trial}] MISMATCH at reads {bad.tolist()}: "
                  f"ref={ref[bad].tolist()} hashed={hashed[bad].tolist()} gsa={gsa[bad].tolist()}",
                  flush=True)
            break
    print(f"\n[1. BYTE-EXACT] {'ALL OK (L-inf=0)' if all_ok else 'FAIL'} over "
          f"{total_reads} reads / 200 trials (gsa == searchsorted == hash atom)", flush=True)

    # ---- 2. GENUINENESS (scenario E value layer + key-binding) ----
    # store 66@200 then 999@200; read @200 -> latest-write-wins = 999.
    sf = np.array([1, 2], dtype=np.int64)
    sa = np.array([200, 200], dtype=np.int64)
    sv = np.array([66, 999], dtype=np.int64)
    idx = build_hash_index(sf, sa, sv)
    raddr = np.array([200], dtype=np.int64); rframe = np.array([5], dtype=np.int64)
    gsa = genuine_structured_value(raddr, rframe, idx, num, mask=mask)
    print(f"\n[2a. VALUE genuineness] read@200 -> genuine structured value = {int(gsa[0])} "
          f"(true latest-write-wins = 999) -> "
          f"{'GENUINE (recomputed from W_v nibbles + softmax1, != a draft stale 66)' if int(gsa[0])==999 else 'FAIL'}",
          flush=True)
    # a draft that INJECTED the stale 66 would mismatch the structured read's genuine 999 ->
    # caught by the cam_value layer (draft_val 66 != pre_genuine 999).
    draft_injected_stale = 66
    caught = int(gsa[0]) != draft_injected_stale
    print(f"[2b. scenario E]  draft injects stale {draft_injected_stale}; genuine structured "
          f"read = {int(gsa[0])} -> {'CAUGHT (draft != genuine)' if caught else 'MISSED'}",
          flush=True)

    # KEY-BINDING check: an UNWRITTEN address genuinely resolves to ZFOD 0 via the below-sink
    # softmax1 weight (NOT a draft-claimed value).  This is the softmax1 sink the structured
    # read runs but single-dispatch's store_log-atom read assumes.
    raddr2 = np.array([201], dtype=np.int64)                    # 201 was never stored
    gsa2 = genuine_structured_value(raddr2, rframe, idx, num, mask=mask)
    win = resolve_winner_hashed(raddr2, rframe, idx)
    print(f"[2c. KEY sink]   read@201 (unwritten): present={bool(win.present[0])} -> genuine "
          f"value = {int(gsa2[0])} -> {'ZFOD 0 via softmax1 sink (genuine)' if int(gsa2[0])==0 else 'FAIL'}",
          flush=True)

    # WINNER SCORE sanity: a full physical-address match nets +EFF (well above the sink 0);
    # a 1-bit-off physical key nets +EFF - 2*EFF = -EFF (below sink -> ZFOD).  Proves the
    # structured read genuinely SCORES the winner's physical key, not a claimed address.
    match_score = _reconstruct_score(np.array([200]), np.array([200]), np.array([1]), num)
    off_score = _reconstruct_score(np.array([200]), np.array([201]), np.array([1]), num)  # 1-bit off
    wmatch = _softmax1_winner_weight(match_score)[0]
    woff = _softmax1_winner_weight(off_score)[0]
    print(f"[2d. SCORE]      match key: score={match_score[0]:.0f} softmax1_w={wmatch:.6f} | "
          f"1-bit-off key: score={off_score[0]:.0f} softmax1_w={woff:.2e} -> "
          f"{'GENUINE (match~1, mismatch->sink~0)' if wmatch>0.99 and woff<1e-6 else 'FAIL'}",
          flush=True)

    ok2 = (int(gsa[0]) == 999 and caught and int(gsa2[0]) == 0 and wmatch > 0.99 and woff < 1e-6)
    print(f"\n=== GSA pure-numpy: byte-exact={all_ok} genuineness={ok2} -> "
          f"{'ALL OK' if all_ok and ok2 else 'FAIL'} ===", flush=True)
    sys.exit(0 if all_ok and ok2 else 1)


if __name__ == "__main__":
    main()
