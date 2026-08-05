"""EQUIVALENCE + GENUINENESS of the CACHED faithful value-precompute (CPU-only).

Proves, with NO GPU / model (pure numpy over synthetic plans + model_addrs):
  1. build_faithful_precompute with C4_FAITHFUL_PRECOMPUTE_CACHE=1 returns an ELEMENT-
     IDENTICAL FaithfulPrecompute to the no-cache path (steps / draft_addr / draft_val /
     pre_genuine / code_steps / code_addr all np.array_equal), so verify_faithful_fast gives
     the byte-identical verdict — the cache is pure reuse, not a recompute shortcut.
  2. A cache HIT (same draft object + same store-log length) returns the SAME object (O(1)),
     while a genuinely-changed store-log (new committed store appended) MISSES and RECOMPUTES
     the correct new genuine value — the cache never returns a stale genuine value.
  3. The cached path STILL REJECTS the wrong draft at BOTH layers (address + value): caching
     the value re-resolution does NOT weaken the genuine latest-write-wins check.

Run: python -m c4_min._agent_faithful_precompute_cache_equiv
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
import numpy as np

from c4_min.faithful_single_dispatch import (
    FaithfulPlan, verify_faithful, build_faithful_precompute, verify_faithful_fast,
    _genuine_value_at, build_faithful_plan)


class _Draft:
    def __init__(self, n, win_starts, store_log):
        self.step_count = n
        self.win_starts = win_starts
        self.store_log = store_log
        self.frames = [{} for _ in range(n)]


def _mk_plan(store_log):
    sf = sorted(store_log)
    sframe = np.asarray(sf, dtype=np.int64)
    saddr = np.fromiter((store_log[f][0] & 0xFFFFFFFF for f in sf), dtype=np.int64, count=len(sf))
    sval = np.fromiter((store_log[f][1] & 0xFFFFFFFF for f in sf), dtype=np.int64, count=len(sf))
    return sframe, saddr, sval


def _plan(store_log, draft_addr, draft_val, code_addr):
    sframe, saddr, sval = _mk_plan(store_log)
    return FaithfulPlan(
        heads_by_block={}, draft_addr=draft_addr, draft_val=draft_val,
        code_addr=code_addr, store_frames=sframe, store_addr=saddr, store_val=sval)


def _pre_equal(a, b):
    """Element-wise equality of two FaithfulPrecompute objects."""
    if a.n != b.n or a.mask != b.mask:
        return False
    if set(a.steps) != set(b.steps):
        return False
    for k in a.steps:
        for fld in ("steps", "draft_addr", "draft_val", "pre_genuine"):
            if not np.array_equal(getattr(a, fld)[k], getattr(b, fld)[k]):
                return False
    return (np.array_equal(a.code_steps, b.code_steps)
            and np.array_equal(a.code_addr, b.code_addr))


def main():
    n = 6
    ws = np.arange(n, dtype=np.int64) * 10
    store_log = {0: (0, 0), 1: (0, 0), 2: (200, 66), 5: (200, 999)}
    da = {"mem": {int(ws[4]): 200}, "pop": {}, "lev": {}, "uni": {}}
    dv = {"mem": {int(ws[4]): 999}, "pop": {}, "lev": {}, "uni": {}}
    ca = {int(ws[s]): s for s in range(n)}
    plan = _plan(store_log, da, dv, ca)
    ma = {(7, "mem"): np.zeros(n, dtype=np.int64), (2, "code"): np.arange(n, dtype=np.int64)}
    ma[(7, "mem")][4] = 200

    print("=== 1. CACHED == NO-CACHE (element-identical FaithfulPrecompute) ===", flush=True)
    os.environ.pop("C4_FAITHFUL_PRECOMPUTE_CACHE", None)
    d_nocache = _Draft(n, ws, dict(store_log))
    pre_nocache = build_faithful_precompute(d_nocache, plan, ws, n)
    assert not hasattr(d_nocache, "_faithful_precompute_cache"), "cache should NOT populate when OFF"

    os.environ["C4_FAITHFUL_PRECOMPUTE_CACHE"] = "1"
    d_cache = _Draft(n, ws, dict(store_log))
    pre_c1 = build_faithful_precompute(d_cache, plan, ws, n)          # cold -> compute + cache
    pre_c2 = build_faithful_precompute(d_cache, plan, ws, n)          # warm -> HIT
    assert _pre_equal(pre_nocache, pre_c1), "cold cached != no-cache"
    assert pre_c2 is pre_c1, "warm call must return the SAME cached object (O(1) hit)"
    print(f"  cold-cached == no-cache element-wise: True", flush=True)
    print(f"  warm call returns SAME object (id match): {pre_c2 is pre_c1}", flush=True)
    # verdict identical.
    v_nocache = verify_faithful_fast(pre_nocache, ma)
    v_cache = verify_faithful_fast(pre_c2, ma)
    assert (v_nocache.ok == v_cache.ok and v_nocache.first_bad_step == v_cache.first_bad_step
            and v_nocache.kind == v_cache.kind), "verdict differs cached vs no-cache"
    print(f"  verdict cached==no-cache: ok={v_cache.ok} (clean accepts)", flush=True)

    print("\n=== 2. changed store-log MISSES + recomputes correct new genuine value ===", flush=True)
    # append a NEWER committed store to addr 200 (value 777) BEFORE the read frame of step 4.
    # read_frame(step4) with n_seed=len(store_log).  Original store_log has 4 entries -> n_seed=4
    # -> rf[4]=4+1+4=9.  A store at frame 6 (<9) to addr200=777 becomes the new latest write.
    store_log2 = dict(store_log); store_log2[6] = (200, 777)
    d_cache.store_log = store_log2                                     # same draft OBJECT, new log
    plan2 = build_plan_from_log(store_log2, da, dv, ca)
    pre_c3 = build_faithful_precompute(d_cache, plan2, ws, n)          # MISS (len changed) -> recompute
    assert pre_c3 is not pre_c1, "changed store-log must MISS (new object), not return stale"
    gv_new = int(pre_c3.pre_genuine["mem"][0])                         # single mem read (step 4)
    print(f"  genuine value after appending 777@f6: {gv_new} (== 777, the new latest write)", flush=True)
    assert gv_new == 777, f"cache returned stale genuine value {gv_new} != 777"
    # cross-check vs a fresh no-cache draft.
    os.environ.pop("C4_FAITHFUL_PRECOMPUTE_CACHE", None)
    d_fresh = _Draft(n, ws, dict(store_log2))
    pre_fresh = build_faithful_precompute(d_fresh, plan2, ws, n)
    assert _pre_equal(pre_c3, pre_fresh), "recomputed cached != fresh no-cache on new log"
    print(f"  recomputed cached == fresh no-cache on new log: True", flush=True)

    print("\n=== 3. cached path STILL rejects the wrong draft at BOTH layers ===", flush=True)
    os.environ["C4_FAITHFUL_PRECOMPUTE_CACHE"] = "1"
    # wrong ADDRESS.
    da_wa = {"mem": {int(ws[4]): 201}, "pop": {}, "lev": {}, "uni": {}}
    dv_wa = {"mem": {int(ws[4]): 66}, "pop": {}, "lev": {}, "uni": {}}
    plan_wa = _plan(store_log, da_wa, dv_wa, ca)
    d_wa = _Draft(n, ws, dict(store_log))
    pre_wa = build_faithful_precompute(d_wa, plan_wa, ws, n)
    v_wa = verify_faithful_fast(pre_wa, ma)
    assert (not v_wa.ok) and v_wa.kind == "cam_addr" and v_wa.first_bad_step == 4, v_wa
    print(f"  address layer: REJECTED at step {v_wa.first_bad_step} kind={v_wa.kind}", flush=True)
    # stale VALUE.
    da_sv = {"mem": {int(ws[4]): 200}, "pop": {}, "lev": {}, "uni": {}}
    dv_sv = {"mem": {int(ws[4]): 66}, "pop": {}, "lev": {}, "uni": {}}   # stale (genuine=999)
    plan_sv = _plan(store_log, da_sv, dv_sv, ca)
    d_sv = _Draft(n, ws, dict(store_log))
    pre_sv = build_faithful_precompute(d_sv, plan_sv, ws, n)
    v_sv = verify_faithful_fast(pre_sv, ma)
    assert (not v_sv.ok) and v_sv.kind == "cam_value" and v_sv.first_bad_step == 4, v_sv
    print(f"  value   layer: REJECTED at step {v_sv.first_bad_step} kind={v_sv.kind}", flush=True)

    print("\nRESULT: the CACHED value-precompute is ELEMENT-IDENTICAL to the no-cache path, "
          "MISSES + recomputes correctly on a changed store-log (never stale), and STILL "
          "REJECTS the wrong draft at BOTH the address and value layers.  Genuineness preserved.",
          flush=True)


def build_plan_from_log(store_log, da, dv, ca):
    return _plan(store_log, da, dv, ca)


if __name__ == "__main__":
    main()
