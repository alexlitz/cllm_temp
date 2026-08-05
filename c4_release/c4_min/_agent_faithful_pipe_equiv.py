"""EQUIVALENCE + GENUINENESS of the PIPELINED/VECTORIZED faithful verify (CPU-only).

Proves, with NO GPU / model (pure numpy over synthetic plans + model_addrs):
  1. verify_faithful_fast(build_faithful_precompute(...)) == verify_faithful(...) in the
     reported (ok, first_bad_step, kind) verdict on a battery of scenarios (clean, wrong
     address, stale value, unarmed, routing) — the pipelined split is byte-identical.
  2. The pipelined path STILL REJECTS the wrong draft at BOTH layers (address + value),
     i.e. the pipelining did not weaken the genuine check.

Run: python -m c4_min._agent_faithful_pipe_equiv
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
import numpy as np

from c4_min.faithful_single_dispatch import (
    FaithfulPlan, verify_faithful, build_faithful_precompute, verify_faithful_fast,
    _genuine_value_at)


class _Draft:
    """Minimal draft stub carrying only what _read_frame_of_step needs."""
    def __init__(self, n, win_starts, store_log):
        self.step_count = n
        self.win_starts = win_starts
        self.store_log = store_log
        # non-file frames -> _read_frame_of_step gives n_seed+1+step.
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


def _check(name, plan, draft, model_addrs, ws, mask=0xFFFFFFFF):
    v_old = verify_faithful(draft, plan, model_addrs, ws, mask=mask)
    pre = build_faithful_precompute(draft, plan, ws, draft.step_count, mask=mask)
    v_new = verify_faithful_fast(pre, model_addrs)
    same = (v_old.ok == v_new.ok and v_old.first_bad_step == v_new.first_bad_step
            and v_old.kind == v_new.kind)
    print(f"  [{name:>22}] old(ok={v_old.ok},step={v_old.first_bad_step},kind={v_old.kind})"
          f"  new(ok={v_new.ok},step={v_new.first_bad_step},kind={v_new.kind})  "
          f"{'MATCH' if same else 'MISMATCH!!'}", flush=True)
    assert same, f"{name}: verdict mismatch"
    # also assert the check COUNTS match (addr/value/routing).
    assert v_old.n_addr_checked == v_new.n_addr_checked, name
    assert v_old.n_value_checked == v_new.n_value_checked, name
    assert v_old.n_routing_checked == v_new.n_routing_checked, name
    return v_old, v_new


def main():
    # A tiny synthetic program: 3 mem reads over stores {addr 200: 66@f2, 999@f5}, plus a
    # code fetch each step.  n_seed=2 (two leading seed frames 0,1).  win_starts = 10*step.
    n = 6
    ws = np.arange(n, dtype=np.int64) * 10
    store_log = {0: (0, 0), 1: (0, 0), 2: (200, 66), 5: (200, 999)}
    draft = _Draft(n, ws, store_log)
    from c4_min.faithful_single_dispatch import _read_frame_of_step
    rf = _read_frame_of_step(draft, n)
    # read step 4 reads addr 200 (its read_frame rf[4]); latest store < rf[4]?  rf = n_seed+1+step.
    # n_seed=2 -> rf[step]=3+step: rf[4]=7 -> latest store frame<7 to addr200 = f5(999).
    print(f"[setup] read_frame per step = {rf.tolist()}", flush=True)
    print("\n=== EQUIVALENCE (pipelined/vectorized == original verify_faithful) ===", flush=True)

    # ---- clean: draft reads 200, injects genuine 999, model queries 200. ----
    da = {"mem": {int(ws[4]): 200}, "pop": {}, "lev": {}, "uni": {}}
    dv = {"mem": {int(ws[4]): 999}, "pop": {}, "lev": {}, "uni": {}}
    ca = {int(ws[s]): s for s in range(n)}       # code fetch pc==step, benign
    plan = _plan(store_log, da, dv, ca)
    ma = {(7, "mem"): np.zeros(n, dtype=np.int64), (2, "code"): np.arange(n, dtype=np.int64)}
    ma[(7, "mem")][4] = 200
    v_old, _ = _check("clean", plan, draft, ma, ws)
    assert v_old.ok, "clean should accept"

    # ---- WRONG ADDRESS: model queries 200 but draft claims 201 (scenario E addr layer). ----
    da2 = {"mem": {int(ws[4]): 201}, "pop": {}, "lev": {}, "uni": {}}
    dv2 = {"mem": {int(ws[4]): 66}, "pop": {}, "lev": {}, "uni": {}}
    plan2 = _plan(store_log, da2, dv2, ca)
    v_old2, v_new2 = _check("wrong-address", plan2, draft, ma, ws)
    assert (not v_old2.ok) and v_old2.kind == "cam_addr" and v_old2.first_bad_step == 4
    assert (not v_new2.ok) and v_new2.kind == "cam_addr" and v_new2.first_bad_step == 4

    # ---- STALE VALUE: address agrees (200==200) but draft injects stale 66; genuine=999. ----
    da3 = {"mem": {int(ws[4]): 200}, "pop": {}, "lev": {}, "uni": {}}
    dv3 = {"mem": {int(ws[4]): 66}, "pop": {}, "lev": {}, "uni": {}}   # STALE
    plan3 = _plan(store_log, da3, dv3, ca)
    v_old3, v_new3 = _check("stale-value", plan3, draft, ma, ws)
    assert (not v_old3.ok) and v_old3.kind == "cam_value" and v_old3.first_bad_step == 4
    assert (not v_new3.ok) and v_new3.kind == "cam_value" and v_new3.first_bad_step == 4

    # ---- UNARMED (model_addr==0 at the read): conservatively skipped -> accept. ----
    ma_un = {(7, "mem"): np.zeros(n, dtype=np.int64), (2, "code"): np.arange(n, dtype=np.int64)}
    v_old4, v_new4 = _check("unarmed-skip", plan3, draft, ma_un, ws)
    assert v_old4.ok and v_new4.ok, "unarmed read is skipped -> accept in both"

    # ---- ROUTING: model decodes a different PC than the draft routed at step 3. ----
    ma_rt = {(7, "mem"): np.zeros(n, dtype=np.int64), (2, "code"): np.arange(n, dtype=np.int64)}
    ma_rt[(2, "code")][3] = 999      # model PC 999 != draft pc 3
    v_old5, v_new5 = _check("routing-mismatch", plan, draft, ma_rt, ws)
    assert (not v_old5.ok) and v_old5.kind == "routing" and v_old5.first_bad_step == 3
    assert (not v_new5.ok) and v_new5.kind == "routing" and v_new5.first_bad_step == 3

    # ---- ADDRESS-DOMINATES-AT-TIE: same step has BOTH wrong addr AND (would-be) wrong val. ----
    da6 = {"mem": {int(ws[4]): 201}, "pop": {}, "lev": {}, "uni": {}}
    dv6 = {"mem": {int(ws[4]): 12345}, "pop": {}, "lev": {}, "uni": {}}  # value also wrong
    plan6 = _plan(store_log, da6, dv6, ca)
    v_old6, v_new6 = _check("addr-dominates-tie", plan6, draft, ma, ws)
    assert v_old6.kind == "cam_addr" and v_new6.kind == "cam_addr", "addr must dominate at tie"

    print("\n=== GENUINENESS (pipelined path STILL rejects both layers) ===", flush=True)
    print(f"  address layer: wrong-address draft REJECTED at step {v_new2.first_bad_step} "
          f"(kind={v_new2.kind}, {v_new2.detail})", flush=True)
    print(f"  value   layer: stale-value  draft REJECTED at step {v_new3.first_bad_step} "
          f"(kind={v_new3.kind}, {v_new3.detail})", flush=True)
    gv = _genuine_value_at(np.array([200]), np.array([rf[4]]), plan3)
    print(f"  genuine latest-write-wins @200 = {int(gv[0])} (==999) != draft injected 66", flush=True)

    print("\nRESULT: pipelined+vectorized verify is BYTE-IDENTICAL in verdict to "
          "verify_faithful across all scenarios, AND still REJECTS the wrong draft at BOTH "
          "the address and the value layer.  Genuineness preserved.", flush=True)


if __name__ == "__main__":
    main()
