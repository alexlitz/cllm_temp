"""CPU byte-exact gate for the EXACT O(steps) KV eviction (C4_EXACT_EVICT).

The exact schedule REPLACES the O(S^2) content-eviction (``evict_all_blocks_fused``,
cdist/cosine) ENTIRELY: a store-KV row's evict-frame is the EARLIEST of
{supersession, last-read+1 (the read log's final resolved read + 1)}, which subsumes
the pop-free and heap-tombstone frees EXACTLY (a pop / tombstone consumer IS a
resolved read).  No content comparison runs at all on the exact path.

This differs from the plain ``C4_EVICT_SCHEDULE`` HYBRID (supersession-off-draft +
an O(S) content ``skip_mech1`` pass) in that the exact path drops a row the moment it
is past its final reader, so the KV survivor SET is strictly TIGHTER than the content
path (which keeps a live non-zero store forever).  The gate is therefore NOT
"identical KV positions" — it is:

  * DECODED byte-exactness: per-step AX trace, accept, final AX, visible output all
    identical to BOTH the content path AND the draft (a dropped row is provably never
    read again, so the model output cannot change);
  * ZERO read-after-free: the schedule's own audit (a read resolving to a row at/after
    its evict-frame) is 0 — the a02e92ec dropped-live-row bug guard;
  * BOUNDED KV: the exact path's max cache size is <= the content path's (tighter).

Run with pytest, or standalone:  python -m c4_min.test_exact_evict
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa

_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.bench_fast_path import (
    build_loop_countdown, build_malloc, build_malloc_free, build_nested)


_MODEL = None
_L = None
_DEVICE = None
_K = 32
_EVICT_EVERY = 8


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _model():
    global _MODEL, _L
    if _MODEL is None:
        # code_size must cover the LONGEST battery program (malloc_free is ~151
        # instrs); the model's CODE_OP table is sized to code_size, so an undersized
        # table IndexErrors on the longer program's opcode lookup.
        _MODEL, _L, _ = build_lib_model_streaming(
            code_size=192, recurrent_divmod=True, addr32=True,
            compute_mode="dense_kernel")
        if _device() != "cpu":
            _MODEL = _MODEL.to(_device())
    return _MODEL, _L


def _programs() -> List[Tuple[str, object]]:
    return [
        ("loop_countdown", build_loop_countdown(10)[0]),   # stack-PSH register frames
        ("nested_call", build_nested(3, 12)[0]),           # stack head (call/return)
        ("malloc_heap", build_malloc(16)[0]),              # §Memory store/load
        ("malloc_free", build_malloc_free(12)[0]),         # heap free->tombstone->reuse
    ]


def _install_content_bound(model):
    from c4_min.local_attention import install_local_attention
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)


def _run(model, L, code, draft, *, mode: str, block_steps: int):
    """mode in {content, schedule, exact}."""
    stats: dict = {}
    out: List[int] = []
    kw = dict(block_steps=block_steps, device=_device(), evict=True,
              mask=0xFFFFFFFF, stats=stats, fast=True, collect_out=out,
              evict_interval_steps=_EVICT_EVERY)
    if mode == "content":
        kw.update(evict_schedule=False, exact_evict=False)
    elif mode == "schedule":
        kw.update(evict_schedule=True, exact_evict=False)
    elif mode == "exact":
        kw.update(evict_schedule=True, exact_evict=True)
    else:
        raise ValueError(mode)
    vr = verify_blocks(model, L, code, draft, **kw)
    return vr, stats, out


def check_program(label: str, code, *, block_steps: int = _K, verbose=True):
    model, L = _model()
    _install_content_bound(model)
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    assert draft.halted, f"{label}: draft did not halt"

    vr_c, st_c, out_c = _run(model, L, code, draft, mode="content", block_steps=block_steps)
    vr_e, st_e, out_e = _run(model, L, code, draft, mode="exact", block_steps=block_steps)

    # DECODED byte-exactness: the model matched the draft in BOTH runs, and the
    # final AX / output bytes agree with the content path and the draft.
    ax_ok = vr_c.all_matched and vr_e.all_matched
    accept_ok = (vr_c.accepted_steps == vr_e.accepted_steps == draft.step_count)
    final_ok = (vr_c.decoded_final_ax == vr_e.decoded_final_ax == draft.final_ax_masked)
    out_ok = (out_c == out_e)
    # ZERO read-after-free — the exactness proof (a02e92ec guard).
    raf = st_e.get("sched_read_after_free", -1)
    raf_ok = (raf == 0)
    # BOUNDED: the exact cache never exceeds the content cache (strictly tighter).
    bound_ok = (vr_e.max_cache_size <= vr_c.max_cache_size)

    if verbose:
        print(f"  [{label}] steps={draft.step_count} stores={len(draft.store_log)} "
              f"reads={sum(len(v) for v in (draft.read_log or {}).values())}")
        print(f"       matched c/e={vr_c.all_matched}/{vr_e.all_matched} "
              f"final c/e/draft={vr_c.decoded_final_ax}/{vr_e.decoded_final_ax}/"
              f"{draft.final_ax_masked} out={out_ok}")
        print(f"       maxcache content={vr_c.max_cache_size} exact={vr_e.max_cache_size} "
              f"(bounded={bound_ok}) evicted c/e={vr_c.total_evicted}/{vr_e.total_evicted}")
        print(f"       exact-schedule: sup={st_e.get('sched_superseded')} "
              f"dead_unread={st_e.get('sched_dead_unread')} live={st_e.get('sched_live')} "
              f"| READ-AFTER-FREE={raf} ({'OK' if raf_ok else 'BUG'})")

    ok = ax_ok and accept_ok and final_ok and out_ok and raf_ok and bound_ok
    return ok, dict(label=label, ax_ok=ax_ok, accept_ok=accept_ok, final_ok=final_ok,
                    out_ok=out_ok, raf_ok=raf_ok, bound_ok=bound_ok,
                    read_after_free=raf, steps=draft.step_count,
                    maxcache_content=vr_c.max_cache_size,
                    maxcache_exact=vr_e.max_cache_size)


def test_exact_evict_byte_exact():
    for label, code in _programs():
        ok, info = check_program(label, code)
        assert info["ax_ok"], f"{label}: verify did not match draft (exact or content)"
        assert info["final_ok"], f"{label}: final AX mismatch"
        assert info["out_ok"], f"{label}: visible output differs content vs exact"
        assert info["raf_ok"], (f"{label}: READ-AFTER-FREE = {info['read_after_free']} "
                                f"(exact eviction dropped a still-read row — a02e92ec bug)")
        assert info["bound_ok"], f"{label}: exact cache NOT bounded by content cache"
        assert ok


if __name__ == "__main__":
    print(f"=== C4_EXACT_EVICT byte-exact battery (device={_device()}) ===")
    allok = True
    for label, code in _programs():
        ok, info = check_program(label, code)
        allok = allok and ok
        print(f"  {label}: {'OK' if ok else 'MISMATCH'}  {info}")
    print("RESULT:", "ALL OK" if allok else "MISMATCH")
    raise SystemExit(0 if allok else 1)
