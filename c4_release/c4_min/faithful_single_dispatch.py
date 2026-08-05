"""FAITHFUL SINGLE-DISPATCH (C4_FAITHFUL_SINGLE_DISPATCH, DEFAULT OFF).

The FAST single-dispatch (``precomputed_schedule``) runs the whole DIV-free doom
K-batch as ONE fused per-row map (block-0 ingest -> mega dead-FFN chain -> live CAM
blocks -> decode), ~1.9 us/step device.  It is byte-exact on CORRECT execution, but
every CAM read's ADDRESS + VALUE + the code-fetch ROUTING op is INJECTED from the
draft (the ``cam_tables`` scatter) and TRUSTED: a self-consistent wrong draft (audit
scenario E) is ACCEPTED because the injected value makes the decode match ``want``.

This lever makes the single-dispatch GENUINE without abandoning it.  It does NOT
fall back to the slow ``verify_blocks`` (~6-36 ms/step) -- it keeps the fast fused
per-row map (block-0 + mega + fused FFN, byte-exact A-class) and adds an INDEPENDENT
verification of the three draft-derived quantities the schedule injects, each derived
from the MODEL's OWN genuine computation, with a first-divergence terminal FAIL
(mirroring ``C4_DIRECT_CAM_VERIFY_ADDR``):

  1. ROUTING (opcode).  The code-fetch CAM injects ``(op, imm)`` at the register-
     verified PC.  We decode the model's OWN code-fetch query address (the sign of
     ``W_q(x)`` on the ``CODE_QRY_BIN`` bits, exactly ``_decode_model_query_addr``)
     from the code-select block's genuine input residual and confirm it equals the
     PC the draft's schedule routed on.  The register compare already verifies the PC
     transition; this additionally pins the ROUTING op to the model's own fetch.

  2. ADDRESS.  The mem/pop/lev CAM reads inject a draft-resolved address.  We decode
     the model's OWN queried address (the sign of ``W_q(x)`` on ``QRY_BIN`` /
     ``SP_QRY_BIN`` / ``LEV_QRY_BIN``) from the mem-cam / stack-pop-cam block's
     genuine input residual and compare to the draft's ``resolve_load_rows`` address.
     O(n_bits) per read, NOT O(n_store).

  3. VALUE (the genuine softmax read).  The injected value is TRUSTED by the fast
     decode.  We INDEPENDENTLY re-derive each read's value the way the real
     softmax1+ALiBi CAM head would: latest-write-wins over the COMMITTED stores at
     the address the MODEL queried (step 2's decoded address), scored over the
     EVICTED / bounded store set (``exact_evict`` -- evicted rows contribute ~0 by the
     latest-write-wins / ALiBi + ZFOD identity, so real-attention-over-evicted ==
     real-attention-over-full-log).  This is the ``C4_FAITHFUL_ATTN_EVICT`` mechanism
     wired into the single-dispatch: the value is resolved from the MODEL's address
     against the actual committed KV, NOT from the draft's claimed address.  A wrong
     value AT a genuine address that the committed stores do not support is CAUGHT; a
     wrong ADDRESS is caught in step 2; a wrong ROUTING op in step 1.

Because the model's genuine query address is decoded from the SAME fused residual the
fast map already computes (``W_q(x)`` at the two live CAM blocks -- one tiny extra GEMM
+ sign-decode per chunk), the verify rides on the fast path.  The value re-resolution
is a single vectorized latest-write-wins over the model's decoded addresses (O(reads),
same routine ``_resolve_reads_vec`` the schedule uses -- but keyed on the MODEL's
address, not the draft's).  On a wrong draft it REJECTS at the first divergent read
(byte-exact first-mismatch stop); on correct execution it is byte-exact == the fast
single-dispatch == the draft == the reference.

DEFAULT OFF -> the fast draft-trusted single-dispatch (golden 069cc32f unchanged).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .direct_cam_batched import (ADDR_BITS, CODE_ADDR_BITS, cam_head_map,
                                  build_resolved_table, _n_seed)


def faithful_single_dispatch_enabled() -> bool:
    """``C4_FAITHFUL_SINGLE_DISPATCH`` (DEFAULT OFF): the GENUINELY-COMPUTING fast
    path.  Keeps the fast fused single-dispatch (``C4_PRECOMPUTED_SCHEDULE``) but makes
    every draft-derived quantity (routing op / read address / read value) INDEPENDENTLY
    verified against the MODEL's own genuine computation, with a first-divergence
    terminal FAIL -- so a self-consistent wrong draft is REJECTED (unlike the plain fast
    path).  OFF -> the draft-trusted fast single-dispatch (byte-identical golden path).

    Implies ``C4_EXACT_EVICT`` (the value re-resolution scores the evicted / bounded
    committed store set -- the ``C4_FAITHFUL_ATTN_EVICT`` mechanism)."""
    return os.environ.get("C4_FAITHFUL_SINGLE_DISPATCH", "0") not in ("0", "", "false", "False")


def _faithful_precompute_cache_enabled() -> bool:
    """``C4_FAITHFUL_PRECOMPUTE_CACHE`` (DEFAULT OFF).  BUILD-REDUCTION for the CONTINUOUS
    render.  The faithful value-verify PRECOMPUTE (``build_faithful_precompute`` — the heavy
    ``_genuine_value_at`` latest-write-wins over the committed store-log at every draft read
    address, ~1.1 us/step on the build thread) is a PURE function of the IMMUTABLE draft
    (store-log + frames + win_starts + the draft's resolved read addr/val + code routing).  In
    a continuous render the SAME draft is re-drafted every frame, so the store-log PREFIX is
    unchanged and the genuine re-resolution is IDENTICAL every frame.  Cache the resolved
    ``FaithfulPrecompute`` ONCE on the draft (keyed by the reused plan identity + store-log
    identity + mask + n — see ``_precompute_cache_key``) and REUSE it per frame instead of
    re-resolving.

    This is the value-verify analog of ``C4_SCHED_CACHE_RESOLVED`` for the schedule build: it
    turns the per-frame value-precompute into a one-time cost, dropping the faithful build
    thread below the dispatch (dispatch-bound).  Byte-identical (the SAME immutable arrays —
    ``FaithfulPrecompute`` is read-only in ``verify_faithful_fast``, which only gathers/compares
    and never mutates it).  DEFAULT OFF -> the per-frame re-resolve."""
    return os.environ.get("C4_FAITHFUL_PRECOMPUTE_CACHE", "0") not in ("0", "", "false", "False")


def _precompute_cache_key(draft, plan: "FaithfulPlan", n: int, mask: int) -> tuple:
    """A cheap identity key for ALL inputs the precompute depends on.  The result is a pure
    function of ``(plan.draft_addr/val/code_addr, plan committed stores, store_log, frames,
    win_starts, mask, n)`` — so the key must capture the PLAN (which carries the draft's resolved
    read addr/val/code + the committed store arrays) AND the store-log, not the store-log alone.

    We key on ``id(plan)`` + the store-log identity + n/mask/step_count.  In the steady-state
    continuous harness ``PipelinedScheduleBuilder`` builds the plan ONCE (``__init__``) and reuses
    that same object every frame, so ``id(plan)`` is stable -> cache hits; and the plan is a pure
    function of the immutable draft.  A DIFFERENT plan (different resolved reads / a new draft) is
    a different object -> new id -> MISS -> recompute (never a stale hit).  ``id(store_log)`` +
    ``len`` additionally guards against a same-object plan whose backing store-log was mutated in
    place (a genuinely new committed prefix)."""
    sl = draft.store_log or {}
    return (id(plan), id(sl), len(sl), int(n), int(mask),
            int(getattr(draft, "step_count", n)))


# ===========================================================================
# 1. The band each live CAM block's QUERY is decoded from, per head kind.
# ===========================================================================
def _qry_band(L, kind: str) -> Optional[int]:
    """The residual band holding the model's binary-address QUERY for a CAM head kind.
    ``_bake_cam_head`` reads ``qry_band`` into ``W_q`` per bit, so the model's decoded
    query address = sign of ``W_q(x)`` at the head's per-bit query columns == the
    ``direct_cam_batched._decode_model_query_addr`` sign decode of that band."""
    if kind == "mem":
        return int(L.QRY_BIN)
    if kind == "pop":
        return int(L.SP_QRY_BIN)
    if kind == "lev":
        return int(L.LEV_QRY_BIN)
    if kind == "uni":
        return int(getattr(L, "UNI_QRY_BIN", L.QRY_BIN))
    if kind == "code":
        cq = getattr(L, "CODE_QRY_BIN", None)
        return int(cq) if cq is not None else None
    return None


@dataclass
class FaithfulPlan:
    """The per-live-block verification plan.

    ``heads_by_block[block_idx]`` -> list of (head_idx, kind, qry_base_col).
    ``qry_base_col`` = head*HD; the model's query bit b sits at ``Q[.., head, b]``
    (``_bake_cam_head`` writes ``W_q[base+b, qry_band+b]``), so decoding
    ``sign(Qh[:, :n_bits])`` recovers the model's queried address."""
    heads_by_block: Dict[int, List[Tuple[int, str, int]]]
    draft_addr: Dict[str, Dict[int, int]]              # kind -> {qpos: draft addr}
    draft_val: Dict[str, Dict[int, int]]               # kind -> {qpos: draft value}
    code_addr: Dict[int, int]                          # {qpos: PC the draft routed}
    # committed stores for the model-address value re-resolution (exact-evict bounded).
    store_frames: np.ndarray
    store_addr: np.ndarray
    store_val: np.ndarray


def build_faithful_plan(model, L, code, draft) -> FaithfulPlan:
    """Assemble the verification plan from the draft + the CAM head map.

    Reuses ``build_resolved_table`` (the draft's per-read resolved addr+value, keyed by
    absolute query-row position) and ``cam_head_map`` (which block/head is each CAM
    read).  The committed store arrays feed the MODEL-address value re-resolution."""
    tbl = build_resolved_table(draft, code)
    chm = cam_head_map(model, L)
    heads_by_block: Dict[int, List[Tuple[int, str, int]]] = {}
    for bi, heads in chm.items():
        HD = int(model.blocks[bi].attn.head_dim)
        heads_by_block[bi] = [(int(h), kind, int(h) * HD) for (h, kind) in heads]
    sl = draft.store_log or {}
    sf = sorted(sl)
    store_frames = np.asarray(sf, dtype=np.int64)
    store_addr = np.fromiter((sl[f][0] & 0xFFFFFFFF for f in sf), dtype=np.int64,
                             count=len(sf))
    store_val = np.fromiter((sl[f][1] & 0xFFFFFFFF for f in sf), dtype=np.int64,
                            count=len(sf))
    return FaithfulPlan(
        heads_by_block=heads_by_block,
        draft_addr={k: dict(tbl.addr[k]) for k in ("mem", "pop", "lev", "uni")},
        draft_val={k: dict(getattr(tbl, k)) for k in ("mem", "pop", "lev", "uni")},
        code_addr=dict(tbl.code_addr),
        store_frames=store_frames, store_addr=store_addr, store_val=store_val)


# ===========================================================================
# 2. THE GENUINE VALUE at the MODEL's decoded address (latest-write-wins over the
#    committed / evicted stores).  This is the softmax1+ALiBi winner value the real
#    CAM head would retrieve for the address the MODEL queried.
# ===========================================================================
def _genuine_value_at(model_addr: np.ndarray, read_frame: np.ndarray,
                      plan: FaithfulPlan) -> np.ndarray:
    """For each read (model_addr[i], read_frame[i]) return the value of the LATEST store
    to model_addr[i] with frame < read_frame[i] (0 if none -- the softmax1 +1 sink /
    ZFOD).  This is EXACTLY what a real softmax1+ALiBi CAM head retrieves for the
    address the MODEL queried, over the committed stores.  Vectorized latest-write-wins
    (same algebra as ``precomputed_schedule._resolve_reads_vec``, keyed on the MODEL's
    decoded address instead of the draft's)."""
    R = model_addr.shape[0]
    val = np.zeros(R, dtype=np.int64)
    sframe, saddr, sval = plan.store_frames, plan.store_addr, plan.store_val
    if R == 0 or sframe.shape[0] == 0:
        return val
    order = np.lexsort((sframe, saddr))
    s_addr_s = saddr[order]; s_frame_s = sframe[order]; s_val_s = sval[order]
    uniq_addr, grp_start = np.unique(s_addr_s, return_index=True)
    grp_end = np.empty_like(grp_start)
    grp_end[:-1] = grp_start[1:]; grp_end[-1] = s_addr_s.shape[0]
    gi = np.searchsorted(uniq_addr, model_addr)
    in_range = gi < uniq_addr.shape[0]
    matched = np.zeros(R, dtype=np.bool_)
    matched[in_range] = uniq_addr[gi[in_range]] == model_addr[in_range]
    ridx = np.nonzero(matched)[0]
    if ridx.shape[0] == 0:
        return val
    g = gi[ridx]; gs = grp_start[g]
    rfr = read_frame[ridx]
    BIG = int(s_frame_s.max()) + int(rfr.max()) + 2 if s_frame_s.size else 1
    store_group = np.searchsorted(uniq_addr, s_addr_s)
    store_key = store_group * BIG + s_frame_s
    read_key = g * BIG + rfr
    loc = np.searchsorted(store_key, read_key, side="left") - 1
    ok = loc >= gs
    good = ridx[ok]
    val[good] = s_val_s[loc[ok]]
    return val


# ===========================================================================
# 3. THE VERIFY.  Given the model's decoded query addresses per (block, head) at every
#    query row, compare ROUTING / ADDRESS / VALUE to the draft; return first divergence.
# ===========================================================================
@dataclass
class FaithfulVerdict:
    ok: bool
    first_bad_step: Optional[int]
    kind: Optional[str]          # "routing" | "cam_addr" | "cam_value"
    detail: Optional[dict]
    n_addr_checked: int = 0
    n_value_checked: int = 0
    n_routing_checked: int = 0


def _read_frame_of_step(draft, n: int) -> np.ndarray:
    """The primary draft frame index of each step (the read frame its CAM reads use).
    Mirrors the frame counter in ``_build_cam_sparse_gpu`` / ``build_resolved_table``."""
    frames = draft.frames
    nbs = np.zeros(n, dtype=np.int64)
    for s in range(n):
        f = frames[s]
        if f.get("is_file"):
            nbs[s] = int(f.get("n_byte_stores", 0) or 0)
    prefix = np.zeros(n, dtype=np.int64)
    if n > 1:
        prefix[1:] = np.cumsum(nbs[:-1])
    return _n_seed(draft) + 1 + np.arange(n, dtype=np.int64) + prefix


def verify_faithful(draft, plan: FaithfulPlan,
                    model_addrs: Dict[Tuple[int, str], np.ndarray],
                    win_starts: np.ndarray, mask: int = 0xFFFFFFFF) -> FaithfulVerdict:
    """Independent verify of routing / address / value from the model's decoded query
    addresses.  ``model_addrs[(block, kind)]`` : int64 [n_steps] the model's decoded
    query address at each step's query row for that CAM head (0 == the query is not
    asserted in this residual view; conservatively skipped -- never a false positive,
    the documented ``ma==0`` limit of ``C4_DIRECT_CAM_VERIFY_ADDR``).

    Returns the FIRST divergent step (lowest index) across all three checks, or ok."""
    n = int(win_starts.shape[0])
    read_frame = _read_frame_of_step(draft, n)
    first_bad = n
    bad_kind = None
    bad_detail = None
    n_addr = n_val = n_rt = 0

    def _note(step, kind, detail):
        nonlocal first_bad, bad_kind, bad_detail
        if step < first_bad:
            first_bad = step
            bad_kind = kind
            bad_detail = detail

    ws = win_starts.astype(np.int64)
    for (bi, kind), maddr in model_addrs.items():
        maddr = np.asarray(maddr, dtype=np.int64)
        if kind == "code":
            n_bits = CODE_ADDR_BITS
            m = maddr & ((1 << n_bits) - 1)
            for s in range(n):
                da = plan.code_addr.get(int(ws[s]))
                if da is None:
                    continue
                ma = int(m[s]); dab = int(da) & ((1 << n_bits) - 1)
                n_rt += 1
                if ma != 0 and ma != dab:
                    _note(s, "routing", {"head": bi, "model_pc": ma, "draft_pc": dab})
            continue
        n_bits = ADDR_BITS
        addr_d = plan.draft_addr.get(kind, {})
        val_d = plan.draft_val.get(kind, {})
        if not addr_d:
            continue
        m = maddr & ((1 << n_bits) - 1)
        sel = [s for s in range(n) if int(ws[s]) in addr_d]
        if not sel:
            continue
        sel_np = np.asarray(sel, dtype=np.int64)
        m_sel = m[sel_np]
        # ADDRESS check.
        for i, s in enumerate(sel):
            ma = int(m_sel[i])
            dab = int(addr_d[int(ws[s])]) & ((1 << n_bits) - 1)
            n_addr += 1
            if ma != 0 and ma != dab:
                _note(s, "cam_addr", {"head": bi, "kind": kind,
                                      "model_addr": ma, "draft_addr": dab})
        # VALUE check (genuine latest-write-wins at the MODEL's decoded address).
        armed = m_sel != 0
        if not armed.any():
            continue
        arm_steps = sel_np[armed]
        arm_addr = m_sel[armed]
        arm_rf = read_frame[arm_steps]
        genuine = _genuine_value_at(arm_addr, arm_rf, plan)
        for i, s in enumerate(arm_steps.tolist()):
            dv = val_d.get(int(ws[s]))
            if dv is None:
                continue
            gv = int(genuine[i]) & mask
            n_val += 1
            if gv != (int(dv) & mask):
                _note(s, "cam_value", {"head": bi, "kind": kind,
                                       "model_value": gv, "draft_value": int(dv) & mask,
                                       "model_addr": int(arm_addr[i])})

    if first_bad >= n:
        return FaithfulVerdict(ok=True, first_bad_step=None, kind=None, detail=None,
                               n_addr_checked=n_addr, n_value_checked=n_val,
                               n_routing_checked=n_rt)
    return FaithfulVerdict(ok=False, first_bad_step=int(first_bad), kind=bad_kind,
                           detail=bad_detail, n_addr_checked=n_addr,
                           n_value_checked=n_val, n_routing_checked=n_rt)


# ===========================================================================
# 4. THE PIPELINED + VECTORIZED VERIFY.  Split ``verify_faithful`` into a heavy
#    DRAFT-ONLY precompute (the latest-write-wins value re-resolution) that is a PURE
#    function of the store-log — knowable BEFORE the dispatch, so it runs on the
#    GIL-releasing build thread CONCURRENT with the previous frame's graph replay — and a
#    cheap fully-VECTORIZED (no Python per-read loop) model-vs-precomputed compare on the
#    critical path.
#
# WHY the pipelined/vectorized verdict is byte-identical to ``verify_faithful``:
#   The genuine value re-resolution is ``latest_write_wins(addr, committed stores)``.
#   ``verify_faithful`` keys it on the MODEL's decoded address; here we precompute it at
#   the DRAFT's claimed address (known pre-dispatch).  The VALUE check only contributes at
#   a step where the query is ARMED (model_addr != 0), and:
#     * armed & address MATCHES (model_addr == draft_addr): the two addresses are the SAME,
#       so ``genuine(model_addr) == genuine(draft_addr) == pre_genuine`` — identical value.
#     * armed & address MISMATCHES: the ADDRESS check ``_note``s that step (address is
#       scored before value per head), and ``_note`` keeps the LOWEST step and the FIRST
#       kind at a tie, so the reported (step, kind) is the address one in BOTH forms — the
#       value check at that already-failing step is dominated either way.
#   So the reported (first_bad_step, kind) verdict is identical; only a dominated
#   value-DETAIL at an already-address-failing step could differ (never the verdict).
#   GENUINENESS is preserved: ``pre_genuine`` is the independent latest-write-wins over the
#   COMMITTED stores (rejects the value-stale draft — scenario E value layer), and the
#   address layer (rejects the wrong-address draft — scenario E address layer) is unchanged.
# ===========================================================================
@dataclass
class FaithfulPrecompute:
    """DRAFT-ONLY precomputed verify state (build-thread, pre-dispatch).

    Per CAM kind (mem/pop/lev/uni) dense int64 arrays over the READS the draft declares
    for that kind (one entry per query step that reads that kind, in step order):
      ``steps[kind]``       : step index of each read.
      ``draft_addr[kind]``  : draft's claimed read address (the address-check target).
      ``draft_val[kind]``   : draft's injected read value (the fast decode trusts this).
      ``pre_genuine[kind]`` : GENUINE latest-write-wins value at ``draft_addr`` over the
                              COMMITTED stores (== ``genuine(model_addr)`` once the address
                              check passes — the value-check target).  This latest-write-wins
                              re-resolution is the heavy (~90%) numpy work, done ONCE here on
                              the build thread instead of on the critical path.
    Plus routing: ``code_steps`` / ``code_addr`` (the PC the draft routed).  ``n`` = steps."""
    n: int
    mask: int
    steps: Dict[str, np.ndarray]
    draft_addr: Dict[str, np.ndarray]
    draft_val: Dict[str, np.ndarray]
    pre_genuine: Dict[str, np.ndarray]
    code_steps: np.ndarray
    code_addr: np.ndarray


def build_faithful_precompute(draft, plan: FaithfulPlan, win_starts: np.ndarray,
                              n: int, mask: int = 0xFFFFFFFF) -> FaithfulPrecompute:
    """Precompute the per-read genuine VALUE re-resolution + the compare targets from the
    DRAFT + store-log ALONE (no model output).  A PURE function of the store-log, so it
    runs on the GIL-releasing build thread concurrent with the previous frame's graph
    replay.

    The heavy work is the single vectorized ``_genuine_value_at`` latest-write-wins over the
    committed stores at every draft read address (the ~1.9 us/step the critical path used to
    pay AFTER the dispatch).  The critical path then only gathers + compares
    (``verify_faithful_fast``).

    ``C4_FAITHFUL_PRECOMPUTE_CACHE`` (DEFAULT OFF): in a CONTINUOUS render the result is a pure
    function of the IMMUTABLE store-log, so cache it ONCE on the draft (keyed by store-log
    identity + mask + n) and reuse it per frame — dropping this ~1.1 us/step build-thread cost
    to a one-time cost after frame 0.  Byte-identical (the returned object is immutable + read
    only in ``verify_faithful_fast``)."""
    _cache = _faithful_precompute_cache_enabled()
    if _cache:
        key = _precompute_cache_key(draft, plan, n, mask)
        cached = getattr(draft, "_faithful_precompute_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]                          # hit: reuse the immutable precompute
    ws = np.asarray(win_starts[:n], dtype=np.int64)
    read_frame = _read_frame_of_step(draft, n)
    amask = (1 << ADDR_BITS) - 1
    # C4_HASH_CAM: build the GENUINE O(1) address hash index over the committed stores ONCE,
    # and resolve every read's latest-write-wins value via an O(1) hash probe (+ a per-address
    # frame bisect) instead of the O(log S) searchsorted over uniq_addr in _genuine_value_at.
    # Byte-identical per-read result; only the address->slot resolution ALGORITHM changes.
    from .hash_cam import hash_cam_enabled, build_hash_index, resolve_value_hashed
    _hash_cam = hash_cam_enabled()
    _hash_index = (build_hash_index(plan.store_frames, plan.store_addr, plan.store_val)
                   if _hash_cam else None)

    def _resolve(addr_arr, rf_arr):
        if _hash_cam:
            return resolve_value_hashed(addr_arr, rf_arr, _hash_index)
        return _genuine_value_at(addr_arr, rf_arr, plan)
    steps: Dict[str, np.ndarray] = {}
    draft_addr: Dict[str, np.ndarray] = {}
    draft_val: Dict[str, np.ndarray] = {}
    pre_genuine: Dict[str, np.ndarray] = {}
    # position -> step (each step has a unique query row win_start).
    pos_to_step = {int(ws[s]): s for s in range(n)}
    for kind in ("mem", "pop", "lev", "uni"):
        addr_d = plan.draft_addr.get(kind, {})
        if not addr_d:
            continue
        val_d = plan.draft_val.get(kind, {})
        pos_arr = np.fromiter((p for p in addr_d.keys() if p in pos_to_step),
                              dtype=np.int64, count=-1)
        if pos_arr.shape[0] == 0:
            continue
        step_arr = np.fromiter((pos_to_step[int(p)] for p in pos_arr),
                               dtype=np.int64, count=pos_arr.shape[0])
        order = np.argsort(step_arr, kind="stable")
        step_arr = step_arr[order]; pos_arr = pos_arr[order]
        da = np.fromiter((int(addr_d[int(p)]) & amask for p in pos_arr),
                         dtype=np.int64, count=pos_arr.shape[0])
        dv = np.fromiter((int(val_d.get(int(p), 0)) & mask for p in pos_arr),
                         dtype=np.int64, count=pos_arr.shape[0])
        rf = read_frame[step_arr]
        # THE HEAVY PART (latest-write-wins at the DRAFT address).  C4_HASH_CAM routes this
        # through the O(1) hash resolver; else the O(log S) searchsorted (_genuine_value_at).
        pg = _resolve(da, rf)
        steps[kind] = step_arr
        draft_addr[kind] = da
        draft_val[kind] = dv
        pre_genuine[kind] = (pg & mask).astype(np.int64)
    # ROUTING (code) targets — dense over the steps with a code fetch.
    ca = plan.code_addr
    if ca:
        cmask = (1 << CODE_ADDR_BITS) - 1
        cpos = np.fromiter((p for p in ca.keys() if p in pos_to_step),
                           dtype=np.int64, count=-1)
        cstep = np.fromiter((pos_to_step[int(p)] for p in cpos),
                            dtype=np.int64, count=cpos.shape[0])
        corder = np.argsort(cstep, kind="stable")
        cstep = cstep[corder]; cpos = cpos[corder]
        caddr = np.fromiter((int(ca[int(p)]) & cmask for p in cpos),
                            dtype=np.int64, count=cpos.shape[0])
    else:
        cstep = np.zeros(0, dtype=np.int64); caddr = np.zeros(0, dtype=np.int64)
    result = FaithfulPrecompute(n=n, mask=mask, steps=steps, draft_addr=draft_addr,
                                draft_val=draft_val, pre_genuine=pre_genuine,
                                code_steps=cstep, code_addr=caddr)
    if _cache:
        try:
            draft._faithful_precompute_cache = (key, result)     # one-time; reused per frame
        except Exception:
            pass
    return result


def verify_faithful_fast(pre: FaithfulPrecompute,
                         model_addrs: Dict[Tuple[int, str], np.ndarray]) -> FaithfulVerdict:
    """CRITICAL-PATH verify: a fully-VECTORIZED (no per-read Python loop) compare of the
    model's decoded query addresses against the DRAFT-precomputed targets (``pre``).

    Byte-identical first-divergence verdict to ``verify_faithful`` (see the module-4
    equivalence note).  ``pre`` already carries the heavy value re-resolution done on the
    build thread; here we only gather the model addresses at the read steps and compare."""
    n = pre.n
    mask = pre.mask
    first_bad = n
    bad_kind = None
    bad_detail = None
    n_addr = n_val = n_rt = 0

    def _note(step, kind, detail):
        nonlocal first_bad, bad_kind, bad_detail
        if step < first_bad:
            first_bad = step
            bad_kind = kind
            bad_detail = detail

    amask = (1 << ADDR_BITS) - 1
    cmask = (1 << CODE_ADDR_BITS) - 1
    for (bi, kind), maddr in model_addrs.items():
        maddr = np.asarray(maddr, dtype=np.int64)
        if kind == "code":
            cs = pre.code_steps
            if cs.shape[0] == 0:
                continue
            m = maddr[cs] & cmask
            da = pre.code_addr
            n_rt += cs.shape[0]
            bad = (m != 0) & (m != da)
            if bad.any():
                j = int(np.argmax(bad))       # first bad in step order (cs is sorted)
                _note(int(cs[j]), "routing", {"head": bi, "model_pc": int(m[j]),
                                              "draft_pc": int(da[j])})
            continue
        step_arr = pre.steps.get(kind)
        if step_arr is None or step_arr.shape[0] == 0:
            continue
        da = pre.draft_addr[kind]
        dv = pre.draft_val[kind]
        pg = pre.pre_genuine[kind]
        m = maddr[step_arr] & amask
        armed = m != 0
        n_addr += step_arr.shape[0]
        n_val += int(armed.sum())
        # ADDRESS: armed AND model != draft.
        addr_bad = armed & (m != da)
        if addr_bad.any():
            j = int(np.argmax(addr_bad))
            _note(int(step_arr[j]), "cam_addr", {"head": bi, "kind": kind,
                                                 "model_addr": int(m[j]),
                                                 "draft_addr": int(da[j])})
        # VALUE: armed AND genuine(draft_addr) != draft's injected value.
        val_bad = armed & (pg != dv)
        if val_bad.any():
            j = int(np.argmax(val_bad))
            _note(int(step_arr[j]), "cam_value", {"head": bi, "kind": kind,
                                                  "model_value": int(pg[j]),
                                                  "draft_value": int(dv[j]),
                                                  "model_addr": int(m[j])})

    if first_bad >= n:
        return FaithfulVerdict(ok=True, first_bad_step=None, kind=None, detail=None,
                               n_addr_checked=n_addr, n_value_checked=n_val,
                               n_routing_checked=n_rt)
    return FaithfulVerdict(ok=False, first_bad_step=int(first_bad), kind=bad_kind,
                           detail=bad_detail, n_addr_checked=n_addr,
                           n_value_checked=n_val, n_routing_checked=n_rt)


__all__ = ["faithful_single_dispatch_enabled", "FaithfulPlan", "build_faithful_plan",
           "FaithfulVerdict", "verify_faithful", "_qry_band", "_genuine_value_at",
           "FaithfulPrecompute", "build_faithful_precompute", "verify_faithful_fast",
           "_faithful_precompute_cache_enabled"]
