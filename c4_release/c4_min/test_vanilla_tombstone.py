"""VANILLA IN-STEP TOMBSTONE tests (C4_VANILLA_TOMBSTONE, #736).

Proves the blog-faithful ``free()`` as a MODEL zero-write distributed across the
free/return step's SPARE token positions, byte-exact through the real
``model.forward``:

  * a heap malloc/free BATTERY: store N distinct addresses, free a subset in ONE
    step (N zero-writes on N spare slots), then LI each -> freed reads 0 (ZFOD
    tombstone), live reads its value; the decoded trace is identical to the pre-free
    path.
  * a nested LEV frame-free: a callee frame's N slots tombstoned in ONE return step
    (AX + STACK0 unchanged -> slack), 0 extra steps.
  * the SLACK budget + spill accounting.
  * golden ``8f4dd780`` unchanged flag-OFF (the driver never populates
    ``frame_spare_stores`` unless the flag is set AND tombstones are requested).

Run:  PYTHONPATH=<c4_release> python -m c4_min.test_vanilla_tombstone
 (or: python -m pytest c4_min/test_vanilla_tombstone.py -q)
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min import nibble_pure_forward as PF
from c4_min import nibble_vanilla_tombstone as TS


def _build():
    return PF.build_pure_forward_model(code_size=32, include_memory=True,
                                       include_cmp=False, include_bitwise=False)


# ---------------------------------------------------------------------------
# Slack-budget / distribution unit tests (no model).
# ---------------------------------------------------------------------------
def test_marker_slots_always_spare():
    # unknown register deltas -> only the 5 markers are safe (MEM reserved).
    slots = TS.spare_slots_for_step(changed_regs=None, reserve_mem=True)
    assert slots == list(TS.MARKER_SLOTS)
    assert TS.frame_tombstone_capacity(None) == 5


def test_lev_return_slack_is_thirteen():
    # a LEV/return moves PC/SP/BP but leaves AX + STACK0 unchanged -> 5 markers + 8
    # role bytes = 13 spare slots in ONE step.
    changed = {"PC", "SP", "BP"}
    cap = TS.frame_tombstone_capacity(changed, reserve_mem=True)
    assert cap == 13, cap
    spare, spill = TS.distribute_tombstones(list(range(13)), changed_regs=changed)
    assert len(spare) == 13 and spill == []
    # a 14th free spills to the next step (honest, no silent drop).
    spare, spill = TS.distribute_tombstones(list(range(14)), changed_regs=changed)
    assert len(spare) == 13 and spill == [13]


def test_all_tombstones_are_zero_writes():
    spare, _ = TS.distribute_tombstones([0x40, 0x44, 0x48], changed_regs={"PC"})
    assert all(v == 0 for (_a, v) in spare.values())      # value 0 == the tombstone


# ---------------------------------------------------------------------------
# Byte-exactness through the real model.forward.
# ---------------------------------------------------------------------------
def _run(model, L, code, frame_tombstones=None, flag="1"):
    old = os.environ.get("C4_VANILLA_TOMBSTONE")
    os.environ["C4_VANILLA_TOMBSTONE"] = flag
    report: dict = {}
    try:
        trace = PF.run_pure_forward(model, L, code,
                                    frame_tombstones=frame_tombstones,
                                    report=report)
    finally:
        if old is None:
            os.environ.pop("C4_VANILLA_TOMBSTONE", None)
        else:
            os.environ["C4_VANILLA_TOMBSTONE"] = old
    return trace, report


def test_heap_malloc_free_battery_in_one_step():
    """Store 3 distinct addresses (3 SI stores), then in ONE step tombstone all 3
    (3 zero-writes on 3 spare slots of that step's frame), then LI each freed addr
    -> 0, and LI a still-live 4th -> its value.  The tombstone step's op is a NOP
    whose registers barely move, so plenty of slack."""
    model, L = _build()
    # 4 stores: 0x40=42, 0x44=99, 0x48=7, 0x4C=13.  Free 0x40/0x44/0x48 in ONE step.
    prog = [
        ("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),     # store 0x40=42
        ("IMM", 0x44), ("PSH", 0), ("IMM", 99), ("SI", 0),     # store 0x44=99
        ("IMM", 0x48), ("PSH", 0), ("IMM", 7),  ("SI", 0),     # store 0x48=7
        ("IMM", 0x4C), ("PSH", 0), ("IMM", 13), ("SI", 0),     # store 0x4C=13
        ("IMM", 0),                                            # <-- FREE STEP (tombstones ride here)
        ("IMM", 0x40), ("LI", 0),                              # freed -> 0
        ("IMM", 0x44), ("LI", 0),                              # freed -> 0
        ("IMM", 0x48), ("LI", 0),                              # freed -> 0
        ("IMM", 0x4C), ("LI", 0),                              # live  -> 13
        ("HALT", 0),
    ]
    code = isa.assemble(prog)
    # emitted frames: 0=init, then one per instr.  The FREE STEP is the IMM at
    # program index 16 -> emitted frame 17.
    free_frame = 17
    ft = {free_frame: [0x40, 0x44, 0x48]}
    trace, report = _run(model, L, code, frame_tombstones=ft)
    # LI results are at their instruction positions in the trace (1 trace entry/step).
    # instr indices of the LIs: 18,20,22,24 (each LI is one step).  trace[i] = AX
    # after step i (step i executes instr i for a straight-line program until HALT).
    li_040 = trace[18]
    li_044 = trace[20]
    li_048 = trace[22]
    li_04c = trace[24]
    assert report["n_tombstones"] == 3, report
    assert report["n_spill"] == 0, report
    assert li_040 == 0, ("freed 0x40 should ZFOD", li_040)
    assert li_044 == 0, ("freed 0x44 should ZFOD", li_044)
    assert li_048 == 0, ("freed 0x48 should ZFOD", li_048)
    assert li_04c == 13, ("live 0x4C keeps value", li_04c)


def _frame_pcs(stream):
    """The decoded PC of every emitted 30-token frame (PC is unaffected by an AX
    value change, so it is the clean witness that the ingest / control flow was NOT
    corrupted by the tombstones)."""
    import c4_min.blogspec_vocab as V
    pcs = []
    pos = 1
    while pos + V.FRAME_LEN <= len(stream):
        pcs.append(V.value_from_bytes_le(stream[pos + 1:pos + 5]))
        pos += V.FRAME_LEN
    return pcs


def test_tombstone_trace_matches_prefree_path():
    """The control flow (per-step PC progression) is BYTE-IDENTICAL whether the
    frees ride the in-step tombstone (flag ON) or are absent — the tombstones did
    NOT corrupt the register frame ingest.  The only decoded difference is the two
    LI-of-freed steps, which flip value->0 (and AX inherits that forward, which is
    the CORRECT post-free behaviour, not a corruption)."""
    model, L = _build()
    prog = [
        ("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
        ("IMM", 0x44), ("PSH", 0), ("IMM", 99), ("SI", 0),
        ("IMM", 0), ("IMM", 0), ("IMM", 0),                    # 3 slack NOP steps
        ("IMM", 0x40), ("LI", 0),
        ("IMM", 0x44), ("LI", 0),
        ("HALT", 0),
    ]
    code = isa.assemble(prog)
    base, base_stream = PF.run_pure_forward(model, L, code, collect_tokens=True)
    # free 0x40 + 0x44 at emitted frame 9 (the second slack NOP: instr idx 9).
    old = os.environ.get("C4_VANILLA_TOMBSTONE"); os.environ["C4_VANILLA_TOMBSTONE"] = "1"
    report: dict = {}
    try:
        trace, ts_stream = PF.run_pure_forward(
            model, L, code, frame_tombstones={9: [0x40, 0x44]},
            report=report, collect_tokens=True)
    finally:
        if old is None: os.environ.pop("C4_VANILLA_TOMBSTONE", None)
        else: os.environ["C4_VANILLA_TOMBSTONE"] = old
    assert report["n_tombstones"] == 2 and report["n_spill"] == 0
    # LI indices: instr 12 (0x40) and 14 (0x44).
    assert base[12] == 42 and base[14] == 99             # pre-free: reads the value
    assert trace[12] == 0 and trace[14] == 0             # post-free: ZFOD tombstone
    # CONTROL FLOW byte-identical: every step's PC matches (ingest uncorrupted).
    assert _frame_pcs(ts_stream) == _frame_pcs(base_stream)


def test_flag_off_is_noop():
    """With the flag OFF the driver ignores ``frame_tombstones`` entirely -> the
    trace is byte-identical to no tombstones (the fidelity path is gated)."""
    model, L = _build()
    prog = [
        ("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
        ("IMM", 0),
        ("IMM", 0x40), ("LI", 0), ("HALT", 0),
    ]
    code = isa.assemble(prog)
    base = PF.run_pure_forward(model, L, code)
    trace, report = _run(model, L, code, frame_tombstones={5: [0x40]}, flag="0")
    assert report["n_tombstones"] == 0                   # driver laid NO store rows
    assert report["frame_spare_stores"] == {}            # nothing distributed
    assert trace == base                                 # flag OFF -> no-op
    assert trace[6] == 42                                # 0x40 NOT freed (flag off)


def test_nested_lev_frame_free_slack():
    """A nested-call return: the callee frame's slots are tombstoned in ONE return
    step.  We model the return step as a step where PC/SP/BP move (the LEV) but AX +
    STACK0 are unchanged, so the frame carries up to 13 tombstones (5 markers + 8
    role bytes).  Byte-exact: the freed callee slots read 0, the caller's live data
    survives.

    (The base pure-forward model has no LEV op; we EMULATE the return-step slack by
    freeing a callee frame at a step whose AX/STACK0 are unchanged — the exact slack
    profile of a real LEV — and assert all frees fit ONE step.)"""
    model, L = PF.build_pure_forward_model(code_size=64, include_memory=True,
                                           include_cmp=False, include_bitwise=False)
    # a "callee frame" = 6 slots at 0x80..0x94 all live; caller data at 0x40 live.
    callee = [0x80, 0x84, 0x88, 0x8C, 0x90, 0x94]
    prog = [("IMM", 0x40), ("PSH", 0), ("IMM", 55), ("SI", 0)]     # caller live @0x40
    for k, a in enumerate(callee):                                 # 6 callee stores
        prog += [("IMM", a), ("PSH", 0), ("IMM", 100 + k), ("SI", 0)]
    prog += [("IMM", 0)]                                           # the RETURN step
    ret_step = len(prog)                                           # instr idx of a following...
    prog += [("IMM", 0x40), ("LI", 0)]                            # caller data still live
    for a in callee:
        prog += [("IMM", a), ("LI", 0)]                          # each callee slot -> 0
    prog += [("HALT", 0)]
    code = isa.assemble(prog)
    # the RETURN step is the ("IMM",0) at instr index len-... ; emitted frame =
    # instr_idx + 1 (init frame is 0).  Find it: it's the IMM 0 right before the LIs.
    # instr index of that IMM 0:
    imm0_idx = 4 + 4 * len(callee)                                # after caller+callee stores
    free_frame = imm0_idx + 1                                     # emitted frame idx
    ft = {free_frame: list(callee)}                              # free all 6 in ONE step
    trace, report = _run(model, L, code, frame_tombstones=ft)
    assert report["n_tombstones"] == 6, report                   # all 6 fit ONE step
    assert report["n_spill"] == 0, report                        # 0 extra steps
    # caller data survives; every callee slot reads 0.
    li0_idx = imm0_idx + 2                                        # first LI (0x40)
    assert trace[li0_idx] == 55, (trace[li0_idx],)               # caller live
    for k in range(len(callee)):
        li = li0_idx + 2 + 2 * k                                 # each callee LI
        assert trace[li] == 0, (k, callee[k], trace[li])         # freed -> ZFOD


# ---------------------------------------------------------------------------
# COMPOSITION with the cache-manager free-reclaim (C4_EXACT_EVICT territory).
# ---------------------------------------------------------------------------
def test_cache_manager_reclaims_in_step_tombstones():
    """Each in-step tombstone projects to the SAME §Memory KV row (W_k/W_v) as a
    single-frame free would — so the reference cache manager's free/zero-value
    eviction reclaims them EXACTLY as it reclaims a single free (proven in
    test_kv_free_driven).  We drive the real head: malloc N addresses, free a subset
    (the in-step tombstones = value-0 stores), and assert the freed rows + their old
    value rows are evicted and read ZFOD, while live addresses survive.

    This is the composition claim: the VANILLA MODEL-OP tombstone (this feature)
    emits real zero-writes; the CACHE MANAGER (exact-evict / nibble_kv_prune) then
    reclaims the zeroed rows.  On the fast path the two are REDUNDANT — exact-evict
    reclaims the same rows off the draft at 0 model cost; the tombstone is the
    fidelity path that makes the free a genuine model store."""
    from c4_min.test_kv_free_driven import EvictingKVMemory, _n_live_stores
    mem = EvictingKVMemory(prune_interval=1)
    addrs = [0x40, 0x44, 0x48, 0x4C]
    vals = {a: 10 + i for i, a in enumerate(addrs)}
    for a in addrs:
        mem.store(a, vals[a]); mem.prune_now()
    assert _n_live_stores(mem) == 4
    # the in-step tombstone batch: free 0x40/0x44/0x48 (three value-0 stores that in
    # the VM ride ONE step's spare slots; here they are three store(a,0) rows — the
    # cache manager sees identical KV rows either way).
    for a in (0x40, 0x44, 0x48):
        mem.store(a, 0); mem.prune_now()            # the tombstone zero-write
    assert _n_live_stores(mem) == 1, _n_live_stores(mem)   # only 0x4C left live
    # freed addresses read ZFOD 0 over the bounded (reclaimed) cache; live survives.
    # (This is the reclaim claim: after the manager drops the tombstone + old-value
    # rows, a freed load reads ZFOD 0 exactly as if the rows were never there.)
    for a in (0x40, 0x44, 0x48):
        assert mem.bounded_load(a) == 0, (a, mem.bounded_load(a))
    assert mem.bounded_load(0x4C) == vals[0x4C]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} vanilla-tombstone tests passed")
