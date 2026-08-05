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


__all__ = ["faithful_single_dispatch_enabled", "FaithfulPlan", "build_faithful_plan",
           "FaithfulVerdict", "verify_faithful", "_qry_band", "_genuine_value_at"]
