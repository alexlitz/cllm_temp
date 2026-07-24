"""CPU byte-identity gate for the DRAFT-DRIVEN eviction schedule (C4_EVICT_SCHEDULE).

Proves that the schedule-driven ``verify_blocks`` (liveness precompute + O(1)/step
position drop) is byte-for-byte identical to the proven content-bound path
(``evict_all_blocks_fused`` — the O(S^2) pairwise content comparison), on a battery
of programs that exercise the memory KV head (SI/LI store-load, a store/load loop),
the stack head (nested calls), and pure arithmetic (register frames):

  * decoded per-step AX trace identical
  * accept decisions + final AX identical
  * the KV CACHE CONTENTS at the end match (same absolute positions retained per
    block) — the strong gate (not just the decoded bytes).

Run with pytest, or standalone:  python -m c4_min.test_evict_schedule
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
    build_loop_countdown, build_malloc, build_nested)


_MODEL = None
_L = None
_DEVICE = None
# block-verify K + eviction cadence — small enough to force MANY eviction rounds
# (so the schedule vs content decision is exercised repeatedly) yet big enough to
# keep the pytest wall reasonable on GPU.
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
        _MODEL, _L, _ = build_lib_model_streaming(
            code_size=128, recurrent_divmod=True, addr32=True,
            compute_mode="dense_kernel")
        if _device() != "cpu":
            _MODEL = _MODEL.to(_device())
    return _MODEL, _L


# Battery: (label, code) via the validated bench builders — a pure-arithmetic loop
# (register frames), a heap malloc/memset/memcmp (the §Memory KV head store/load —
# the critical test), and a nested-call loop (the stack head).  Small sizes so the
# CPU pytest is tractable; the deeper GPU numbers are in ``bench_evict_schedule``.
def _programs() -> List[Tuple[str, object]]:
    return [
        ("loop_countdown", build_loop_countdown(10)[0]),      # stack-PSH register frames
        ("malloc_heap", build_malloc(16)[0]),                 # §Memory store/load (critical)
        ("nested_call", build_nested(3, 12)[0]),              # stack head (call/return)
    ]


def _install_content_bound(model):
    """Enable DROP-KV local attention + content-bound global (the memory-heavy fast
    path the schedule targets)."""
    from c4_min.local_attention import install_local_attention
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)


def _cache_positions(caches) -> List[List[int]]:
    """The absolute positions retained in each block's GLOBAL cache (the store-row
    cache the schedule/content path evicts).  This is the KV-contents fingerprint."""
    out = []
    for c in caches:
        if c.pos is None:
            out.append([])
        else:
            out.append(sorted(int(p) for p in c.pos.tolist()))
    return out


def _run_once(model, L, code, draft, *, evict_schedule: bool, block_steps: int):
    """Run verify_blocks and return (result, per-block cache positions).  We reach
    into verify_blocks by re-implementing the tiny driver here is not needed — we
    just run it and read the returned result + reconstruct the caches via a second
    verify that RETAINS them.  Instead we monkey-capture the caches by running the
    verify and using its stats + a direct cache-position probe."""
    # verify_blocks builds its own caches internally; to fingerprint the retained
    # rows we run it and rely on the fact that the schedule and content path are
    # deterministic, then compare the decoded traces + accept + final AX + cache
    # sizes reported.  For the STRONG cache-contents gate we call the lower-level
    # path below in _run_capture.
    stats: dict = {}
    out: List[int] = []
    vr = verify_blocks(model, L, code, draft, block_steps=block_steps,
                       device=_device(), evict=True, mask=0xFFFFFFFF, stats=stats,
                       fast=True, collect_out=out, evict_schedule=evict_schedule,
                       evict_interval_steps=_EVICT_EVERY)
    return vr, stats, out


def _run_capture(model, L, code, draft, *, evict_schedule, block_steps):
    """Run the verify AND capture the final per-block cache positions by wrapping the
    caches.  We patch pf_speculative.BlockKVCacheBatched to record the caches list
    into a holder via evict_all_blocks — simplest: re-run verify with a hook."""
    import c4_min.pf_speculative as PS
    holder = {}
    orig = PS.BlockKVCacheBatched

    class _Capturing(orig):
        pass

    # Capture caches by intercepting the eviction functions (both share the caches
    # list as first arg).
    captured = {"caches": None}
    orig_fused = PS.evict_all_blocks_fused
    orig_sched = PS.evict_all_blocks_scheduled

    def _cap_fused(caches, *a, **k):
        captured["caches"] = caches
        return orig_fused(caches, *a, **k)

    def _cap_sched(caches, *a, **k):
        captured["caches"] = caches
        return orig_sched(caches, *a, **k)

    PS.evict_all_blocks_fused = _cap_fused
    PS.evict_all_blocks_scheduled = _cap_sched
    try:
        vr, stats, out = _run_once(model, L, code, draft,
                                   evict_schedule=evict_schedule,
                                   block_steps=block_steps)
    finally:
        PS.evict_all_blocks_fused = orig_fused
        PS.evict_all_blocks_scheduled = orig_sched
    positions = _cache_positions(captured["caches"]) if captured["caches"] else []
    return vr, stats, out, positions


def _decoded_trace(draft):
    return [f["ax"] & 0xFFFFFFFF for f in draft.frames]


def check_program(label: str, code, *, block_steps: int = _K, verbose=True):
    model, L = _model()
    # (re)install the content-bound split fresh each time is idempotent per attr set.
    _install_content_bound(model)
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    assert draft.halted, f"{label}: draft did not halt"

    vr_c, st_c, out_c, pos_c = _run_capture(
        model, L, code, draft, evict_schedule=False, block_steps=block_steps)
    vr_s, st_s, out_s, pos_s = _run_capture(
        model, L, code, draft, evict_schedule=True, block_steps=block_steps)

    ax_ok = (vr_c.all_matched and vr_s.all_matched)
    accept_ok = (vr_c.accepted_steps == vr_s.accepted_steps ==
                 draft.step_count)
    final_ok = (vr_c.decoded_final_ax == vr_s.decoded_final_ax ==
                draft.final_ax_masked)
    out_ok = (out_c == out_s)
    # KV-contents gate: the retained absolute positions per block must match on the
    # GLOBAL (store-row) caches the schedule governs.  Content path may additionally
    # drop recency-negligible NON-store rows the schedule keeps; on the content-bound
    # split there are NO non-store rows in the global cache (dropped on commit), so
    # the position sets must be IDENTICAL.
    kv_ok = (pos_c == pos_s)
    n_store_blocks = sum(1 for p in pos_c if p)

    if verbose:
        print(f"  [{label}] steps={draft.step_count} stores={len(draft.store_log)} "
              f"loads={len(draft.load_log)} "
              f"| all_matched c/s={vr_c.all_matched}/{vr_s.all_matched} "
              f"final c/s={vr_c.decoded_final_ax}/{vr_s.decoded_final_ax} "
              f"draft={draft.final_ax_masked}")
        print(f"           evicted c/s={vr_c.total_evicted}/{vr_s.total_evicted} "
              f"maxcache c/s={vr_c.max_cache_size}/{vr_s.max_cache_size} "
              f"| store-blocks={n_store_blocks} KV-contents-match={kv_ok} "
              f"out-match={out_ok}")
        if not kv_ok:
            # show first divergent block
            for b, (a, bb) in enumerate(zip(pos_c, pos_s)):
                if a != bb:
                    print(f"           block {b} DIVERGES: content={a} schedule={bb}")
                    break
    ok = ax_ok and accept_ok and final_ok and out_ok and kv_ok
    return ok, dict(label=label, ax_ok=ax_ok, accept_ok=accept_ok,
                    final_ok=final_ok, out_ok=out_ok, kv_ok=kv_ok,
                    steps=draft.step_count, stores=len(draft.store_log),
                    evicted_c=vr_c.total_evicted, evicted_s=vr_s.total_evicted)


def test_evict_schedule_byte_identical():
    results = []
    for label, code in _programs():
        ok, info = check_program(label, code)
        results.append((ok, info))
        assert info["ax_ok"], f"{label}: verify did not match draft"
        assert info["final_ok"], f"{label}: final AX mismatch"
        assert info["out_ok"], f"{label}: output byte mismatch schedule vs content"
        assert info["kv_ok"], f"{label}: KV cache contents differ schedule vs content"
    assert all(ok for ok, _ in results)


if __name__ == "__main__":
    print(f"=== C4_EVICT_SCHEDULE byte-identity battery (device={_device()}) ===")
    allok = True
    for label, code in _programs():
        ok, info = check_program(label, code)
        allok = allok and ok
        print(f"  {label}: {'OK' if ok else 'MISMATCH'}  {info}")
    print("RESULT:", "ALL OK" if allok else "MISMATCH")
    raise SystemExit(0 if allok else 1)
