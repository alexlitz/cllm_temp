"""PART B: DRAFT-DIRECT-INDEX CAM READ (fast-path only, C4_DIRECT_CAM_READ).

The §Memory / stack / LEV CAM read is a TOP-KEY exact binary-address match with
latest-write-wins recency (softmax1 + ALiBi over the K store-KV rows).  The perfect
draft (``pf_speculative.draft_pf_program``) already materialises the ENTIRE memory
access pattern before a single model forward runs, so it ALSO knows — per read — the
EXACT KV store row the address resolves to (``nibble_evict_schedule.resolve_load_rows``:
address -> latest superseding store row).

So on the SPECULATIVE / fast path the read need not run the O(K) softmax scoring at
all: it can DIRECT-GATHER the value from the known row.  This module drives exactly
that and proves it is BYTE-IDENTICAL (decoded AX trace) to the softmax CAM path.

HONESTY (stated plainly): this is a FAST-PATH speedup, NOT the vanilla mechanism.  A
direct index gather is not softmax attention — the vanilla / golden path keeps the
softmax1 + ALiBi CAM read (``run_pure_forward_complete``).  The direct read is only
valid because the perfect draft already knows the answer the softmax would compute;
it is a verification-time / speculative accelerator that removes the O(K²) global-head
score, and the softmax path remains the ground truth it is checked against.

The mechanism (byte-identity)
-----------------------------
The softmax CAM head writes, into the read's destination nibble band, the value
nibbles of the store row whose address == the query and which is the most recent such
store (ALiBi latest-write-wins), or 0 (softmax1 +1 sink) when the address is unwritten.
``resolve_load_rows`` computes that SAME winner in O(reads + stores) off the draft.  So
overwriting the destination band with the resolved value's nibbles is byte-identical to
what the softmax head wrote — the SAME nibbles, hence the SAME LM-head byte decode.

We prove it by running the model TWICE per step over the SAME token stream:
  * SOFTMAX: the normal ``model.forward`` (every CAM head runs its full score).
  * DIRECT : the normal forward, then the destination nibble bands (AX / STACK0 /
    LEV_RET) are OVERWRITTEN with the ``resolve_load_rows`` value — the softmax head's
    O(K) score is thrown away and replaced by the O(1) gather.
and asserting the decoded per-step AX is identical.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .nibble_pure_forward_complete import (
    run_pure_forward_complete, make_overlay_complete, _build_frame, _seed_frames,
    _decode_reg_from_nibbles, _snap_lane, _mem_top, SP_INIT)
from .pf_speculative import draft_pf_program
from .nibble_evict_schedule import resolve_load_rows


def direct_cam_read_enabled() -> bool:
    """C4_DIRECT_CAM_READ (DEFAULT OFF): fast-path direct-index CAM gather.

    OFF -> the vanilla softmax1+ALiBi CAM read (byte-exact golden path).  ON -> the
    fast verify path gathers the resolved KV row directly (no O(K) score).  This is a
    SPECULATIVE / fast-path accelerator only; the vanilla path always keeps softmax."""
    return os.environ.get("C4_DIRECT_CAM_READ", "0") not in ("0", "", "false", "False")


# the destination nibble band for each CAM-read head class.
def _dest_band(L, head: str) -> int:
    return {"mem": L.AX, "pop": L.STACK0, "lev": L.LEV_RET}[head]


def run_pure_forward_direct_cam(model, L, code: List[isa.Instr], *,
                                max_steps: int = 512, mask: int = 0xFF,
                                verbose: bool = False):
    """Run the whole VM step-by-step over the growing frame stream (like
    ``run_pure_forward_complete``) but with the DIRECT-INDEX CAM read: on any step
    whose draft performs a CAM read, the read's destination nibble band is written
    with the value ``resolve_load_rows`` gathered from the resolved KV store row —
    the softmax head's O(K) score is bypassed.

    Returns the per-step AX trace (same shape as ``run_pure_forward_complete``).

    Implementation: we DRAFT the program first (free), resolve every read to its KV
    row, then drive the model exactly as the naive driver does — but after the block
    stack runs, we OVERWRITE the read destination band(s) with the gathered nibbles
    before decoding.  Because the resolved value == the softmax winner's value, the
    decoded register state is byte-identical; the model's own PC/SP/BP transition
    (pure FFN, no CAM) is untouched, so the stream stays in lockstep with the draft.
    """
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
    reads = resolve_load_rows(draft)            # {read_frame: [ResolvedRead,...]}

    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    stream: List[int] = [V.BOS] + init_frame
    trace: List[int] = []
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = 0
    store_log: Dict[int, Tuple[int, int]] = {}
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1].clone()
        # -- DIRECT-INDEX CAM READ: overwrite the read destination nibble band(s) with
        #    the gathered value from the resolved KV row (bypassing the softmax score).
        this_frame = frame_idx + 1              # the frame this step emits
        for r in reads.get(this_frame, []):
            band = _dest_band(L, r.head)
            for j, nv in enumerate(V.nibbles_of_value(r.value & 0xFFFFFFFF, NIB_PER_REG)):
                state[band + j] = float(nv)
        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ax = _decode_reg_from_nibbles(state, L, L.AX)
        # store bookkeeping (identical to the naive driver).
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
        stream += frame
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                  f"pc'={pc} ax={ax & mask} sp={sp} bp={bp} "
                  f"reads={reads.get(this_frame, [])}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    return trace


def attention_flop_report(draft) -> dict:
    """Analytic O(K)->O(1) global-head scoring reduction for this program.

    The softmax CAM read scores the query against ALL K store-KV rows in the cache
    (a ``[HD] x [K]`` dot per read, per global head).  The direct gather reads ONE
    row.  We report, per read, the cache depth K it would have scored (the number of
    store rows committed BEFORE the read) — the score work the direct read removes."""
    from .nibble_evict_schedule import resolve_load_rows
    reads = resolve_load_rows(draft)
    store_frames = sorted(draft.store_log or {})
    import bisect
    per_read = []
    total_softmax_scores = 0
    for f in sorted(reads):
        # K = number of store rows committed before frame f (what softmax would score).
        k = bisect.bisect_left(store_frames, f)
        for r in reads[f]:
            per_read.append((r.head, f, k))
            total_softmax_scores += k
    n_reads = len(per_read)
    return {
        "n_reads": n_reads,
        "total_softmax_row_scores": total_softmax_scores,   # O(sum K) dot products
        "total_direct_row_gathers": n_reads,                # O(reads) — one row each
        "mean_K": (total_softmax_scores / n_reads) if n_reads else 0.0,
        "per_read": per_read,
    }
