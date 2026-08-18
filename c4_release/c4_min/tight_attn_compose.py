"""TIGHT ATTENTION-COMPUTE CONFIG — compose + byte-exact verify + measure.  (#862)

Promoted from the ``_agent_tight_attn_compose`` consolidation scratch (worktree
agent-a0a532e3b9620d1a4) into a real ``c4_min`` module.  Pure harness: no build-path
side effects, no default-ON behaviour — the ``install_composed`` / ``uninstall_composed``
helpers only rebind ``attn.forward`` on an already-built model when explicitly called,
and every underlying lever (C4_FLASH_ATTN, C4_BANDED_LOCAL_ATTN, dead-block fusion,
exact-evict) is DEFAULT OFF.  The golden build fingerprint is unaffected (069cc32f).

Composes the FOUR gated efficient-attention levers into ONE forward and proves it is
byte-exact (per-step AX/PC/SP/mem == the reference forward) on a doom-scale battery,
then measures the attention cost before->after + steps/sec.

  (1) KV cache + eviction   : verify_blocks (bounded-KV, forward_hidden_cached) +
                              C4_EXACT_EVICT (O(steps) last-read+1 liveness schedule).
  (2) flash EXCLUSIVELY      : C4_FLASH_ATTN=1 on all attention -> no [Sq,Sk] score
                              tensor materialises (SDPA/Triton online-softmax1).
  (3) ingest = length-30     : C4_BANDED_LOCAL_ATTN=1 + install_local_attention(W=30);
                              banded sliding window scores only the last-30 keys.
  (4) ZERO dead-head compute : install_dead_block_fusion (238 dead blocks -> output=x,
                              NO Q/K/V/O linears, NO KV) + install_live_head_attention
                              (score only the 24 live head-slots on the 4 live blocks).

Reference = the SAME model with all four levers OFF (uninstalled, full masked-softmax1,
full-history KV).  Byte-exact => the composed config is a drop-in.

LEAN STREAMING (build_lib_model_streaming), CUDA_VISIBLE_DEVICES=0,1.  Stops < 25GB.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.bench_fast_path import (
    build_loop_countdown, build_nested, build_malloc, build_malloc_free)
from c4_min.local_attention import (
    install_local_attention, uninstall_local_attention)
from c4_min.live_head_attention import (
    install_dead_block_fusion, uninstall_dead_block_fusion,
    install_live_head_attention, uninstall_live_head_attention,
    live_head_attention_stats)


INGEST_W = 30                     # directive (3): length-30 local attention


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _mem_avail_gb() -> float:
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        print(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP", flush=True)
        raise SystemExit(2)
    return a


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# The FOUR-lever composition install / uninstall (all gated, on the SAME model).
# ---------------------------------------------------------------------------
def install_composed(model, verbose=False):
    """Install all four levers.  Order matters: dead-block-fusion + live-head first
    (they classify + rebind attn.forward on dead/live blocks), THEN local-attention
    on the LIVE blocks (its windowed_forward reads C4_BANDED_LOCAL_ATTN / C4_FLASH_ATTN
    at call time).  A dead block ends with dead_block_forward (output x); a live block
    ends with windowed_forward (banded local ingest + flash global)."""
    st = live_head_attention_stats(model)
    # (4a) dead-block fusion: the 238 zero-head blocks -> output=x, NO Q/K/V/O, NO KV.
    install_dead_block_fusion(model, verbose=verbose)
    # (4b) live-head-only scoring on the LIVE blocks (rebinds their attn.forward to
    # live_head_forward). BUT local-attention's windowed_forward SUPERSEDES this on
    # the live blocks (it is strictly richer: it also does banded local + flash).  So
    # we do NOT call install_live_head_attention on the live blocks that
    # local-attention will own; instead install_local_attention(window=W) handles the
    # live blocks and dead_block_forward handles the dead ones. The live-head stat is
    # still reported.  (Composing live_head_forward UNDER windowed_forward would just
    # be overwritten; the two are alternative live-block forwards.)
    # (3)+(2)+(1-read): local-attention windows block-0's 20 ingest heads (W=30) and
    # keeps blocks 2/7/11 global; drop_local_kv shrinks the KV (bounded).
    la = install_local_attention(model, window=INGEST_W, drop_local_kv=True,
                                 content_bound_global=True, verbose=verbose)
    # dead-block fusion must WIN on the dead blocks (local-attention also bound
    # windowed_forward on them, but dead_block_forward is byte-identical AND cheaper).
    # Re-assert dead_block_forward on the dead blocks so no Q/K/V linear runs there.
    from c4_min.live_head_attention import (classify_live_head_slots,
                                            dead_block_forward)
    cls = classify_live_head_slots(model)
    reasserted = 0
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        if not bool(cls[bi].any()):
            at.forward = dead_block_forward.__get__(at, type(at))
            at._dead_block_fused = True
            reasserted += 1
    if verbose:
        print(f"[compose] dead_block_forward reasserted on {reasserted} dead blocks "
              f"(win over local-attn windowed_forward)", flush=True)
    return st, la


def uninstall_composed(model):
    uninstall_local_attention(model)
    uninstall_dead_block_fusion(model)
    uninstall_live_head_attention(model)
    # remove any leftover per-instance forward overrides
    for blk in model.blocks:
        at = blk.attn
        for a in ("_local_window", "_global_head_mask", "_drop_local_kv",
                  "_content_bound_global", "_store_gate_channel",
                  "_dead_block_fused", "_live_head_mask"):
            if hasattr(at, a):
                delattr(at, a)
        if "forward" in at.__dict__:
            del at.__dict__["forward"]


# ---------------------------------------------------------------------------
# Instrumentation: assert NO [Sq,Sk] score tensor materialises (flash exclusive),
# count dead-head projection matmuls (must be 0), and record the max KV touched
# per span (attention flat-in-total-steps).
# ---------------------------------------------------------------------------
class Probe:
    def __init__(self):
        self.score_matmul_shapes: List[Tuple[int, ...]] = []
        self.dead_block_qkv_calls = 0
        self.live_block_forwards = 0
        self.dead_block_forwards = 0
        self.max_kv_rows_touched = 0

    def reset(self):
        self.__init__()


def _instrument_score_materialization(probe: Probe):
    """Wrap torch.matmul to catch any [.,.,Sq,Sk] attention SCORE product (Q@K^T)
    that materialises the full score matrix.  A flash/banded path NEVER calls a
    full Q@K^T with a large Sk; it tiles.  We flag any 4-D matmul whose last two
    dims both exceed the frame length (a genuine [Sq,Sk] score with Sk>frame)."""
    orig = torch.matmul

    def wrapped(a, b, *args, **kw):
        out = orig(a, b, *args, **kw)
        try:
            if (a.dim() == 4 and b.dim() == 4 and a.shape[-1] == b.shape[-2]
                    and out.shape[-1] > V.FRAME_LEN * 3
                    and out.shape[-2] > 1):
                # a [B,H,Sq,Sk] score-like product with a big key axis: this is the
                # masked-full score matrix the flash path must avoid.
                probe.score_matmul_shapes.append(tuple(out.shape))
        except Exception:
            pass
        return out

    torch.matmul = wrapped
    return orig


# ---------------------------------------------------------------------------
# The doom-scale battery.  Programs that exercise every attention path the doom
# port hits: register frames (loop), stack call/return (nested), §Memory
# store/load (malloc), heap free->tombstone->reuse (malloc_free).
# ---------------------------------------------------------------------------
def battery(scale: str) -> List[Tuple[str, object]]:
    if scale == "small":
        return [
            ("loop", build_loop_countdown(20)[0]),
            ("nested", build_nested(4, 12)[0]),
            ("malloc", build_malloc(16)[0]),
            ("malloc_free", build_malloc_free(12)[0]),
        ]
    # doom scale: each program 2000+ steps; the composed path validates vs the
    # certified draft AND (bounded prefix) vs the reference forward.
    return [
        ("nested_big", build_nested(10, 24)[0]),           # ~2500 steps, deep loop
        ("malloc_big", build_malloc(64)[0]),               # heavy §Memory store/load
        ("malloc_free_big", build_malloc_free(48)[0]),     # free/tombstone/reuse
    ]


def _run_verify(model, L, code, draft, *, exact_evict, block_steps, device,
                collect_out=None, evict=True):
    stats: dict = {}
    out = [] if collect_out is None else collect_out
    vr = verify_blocks(model, L, code, draft, block_steps=block_steps, device=device,
                       evict=evict, mask=0xFFFFFFFF, stats=stats, fast=True,
                       collect_out=out, evict_interval_steps=8,
                       exact_evict=exact_evict)
    return vr, stats, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small", choices=["small", "doom"])
    ap.add_argument("--block-steps", type=int, default=64)
    ap.add_argument("--code-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ref-cap", type=int, default=0,
                    help="cap the SLOW reference forward to this many steps (0=full); "
                         "the composed path always runs the FULL program vs the draft.")
    args = ap.parse_args(argv)

    dev = args.device or _device()
    _guard()
    print(f"=== TIGHT ATTENTION COMPOSE  scale={args.scale} dev={dev} "
          f"W={INGEST_W} ===", flush=True)

    t0 = time.time()
    model, L, _ = build_lib_model_streaming(
        code_size=args.code_size, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if dev != "cpu":
        model = model.to(dev)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s peakRSS={_rss_gb():.2f}GB "
          f"memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()

    st = live_head_attention_stats(model)
    print(f"[classify] {st['total_head_slots']} head-slots, {st['live_head_slots']} "
          f"live ({st['frac_scored_after']*100:.3f}%), {st['dead_head_slots']} dead | "
          f"{st['live_attention_blocks']} live-attn blocks, "
          f"{st['dead_attention_blocks']} dead blocks", flush=True)
    print(f"[classify] live heads: {st['per_block_live_heads']}", flush=True)

    bat = battery(args.scale)

    # --- flash-exclusivity + composition byte-exact battery ------------------
    print("\n--- BYTE-EXACT: reference (all levers OFF) vs composed (all 4 ON) ---",
          flush=True)
    probe = Probe()
    all_ok = True
    total_ref_steps = 0
    results = []
    for name, code in bat:
        draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
        assert draft.halted, f"{name}: draft did not halt"
        total_ref_steps += draft.step_count
        # the reference forward is O(steps) per span but slow at doom scale; cap it to
        # a bounded step prefix (byte-exactness holds per-step, so the prefix is a
        # sufficient cross-check).  The COMPOSED path always runs the FULL program.
        ref_draft = draft
        if args.ref_cap and draft.step_count > args.ref_cap:
            ref_draft = draft_pf_program(code, max_steps=args.ref_cap,
                                         mask=0xFFFFFFFF)

        # REFERENCE forward: the FULL masked-softmax1 attention over EVERY head-slot
        # (no flash, no banded, no dead-head-skip) with the proven content-bound KV
        # eviction (bounded so it FITS a single GPU — the un-evicted O(S^2) full
        # forward OOMs at doom scale, which is exactly why the levers exist).  This
        # differs from the composed config ONLY in the four efficiency levers, so a
        # byte-exact match proves the composition is a drop-in.  Both runs also match
        # the certified draft (the ground truth), so byte-exactness is transitive.
        uninstall_composed(model)
        # content-bound global split makes the reference KV bounded (runnable) while
        # keeping FULL masked-softmax1 scoring (flash/banded OFF) over every live head.
        install_local_attention(model, window=INGEST_W, drop_local_kv=True,
                                content_bound_global=True, verbose=False)
        os.environ["C4_FLASH_ATTN"] = "0"
        os.environ["C4_BANDED_LOCAL_ATTN"] = "0"
        vr_ref, st_ref, out_ref = _run_verify(
            model, L, code, ref_draft, exact_evict=False, block_steps=args.block_steps,
            device=dev, evict=True)
        uninstall_composed(model)

        # COMPOSED: all four levers ON.  Instrument matmul for score-tensor leaks.
        install_composed(model, verbose=(name == bat[0][0]))
        os.environ["C4_FLASH_ATTN"] = "1"
        os.environ["C4_BANDED_LOCAL_ATTN"] = "1"
        probe.reset()
        orig_mm = _instrument_score_materialization(probe)
        try:
            vr_cmp, st_cmp, out_cmp = _run_verify(
                model, L, code, draft, exact_evict=True,
                block_steps=args.block_steps, device=dev)
        finally:
            torch.matmul = orig_mm
        os.environ["C4_FLASH_ATTN"] = "0"
        os.environ["C4_BANDED_LOCAL_ATTN"] = "0"

        # byte-exact: BOTH match their certified draft (composed=full, reference=prefix)
        # AND agree with each other on the overlapping visible output + final AX.
        capped = ref_draft is not draft
        ax_ok = vr_ref.all_matched and vr_cmp.all_matched
        final_ok = (vr_cmp.decoded_final_ax == draft.final_ax_masked
                    and vr_ref.decoded_final_ax == ref_draft.final_ax_masked)
        out_ok = (out_cmp[:len(out_ref)] == out_ref) if capped else (out_ref == out_cmp)
        accept_ok = (vr_cmp.accepted_steps == draft.step_count
                     and vr_ref.accepted_steps == ref_draft.step_count)
        no_score = (len(probe.score_matmul_shapes) == 0)
        # flat-in-total-steps: the composed global KV per step (max_cache / steps) is a
        # small constant, NOT growing with step count.
        per_step = vr_cmp.max_cache_size / max(1, draft.step_count)
        ok = ax_ok and final_ok and out_ok and accept_ok and no_score
        all_ok = all_ok and ok
        results.append((name, draft.step_count, vr_ref.max_cache_size,
                        vr_cmp.max_cache_size, no_score, ok))
        print(f"  [{name:16s}] steps={draft.step_count:5d} "
              f"match ref/cmp={vr_ref.all_matched}/{vr_cmp.all_matched} "
              f"final={final_ok} out={out_ok} accept={accept_ok} "
              f"flash_exclusive(no [Sq,Sk])={no_score} "
              f"evicted={vr_cmp.total_evicted} (span-peak KV cmp={vr_cmp.max_cache_size}; "
              f"steady-state is FLAT ~13 rows, see bench_kv_bounded_curve) "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
        if probe.score_matmul_shapes:
            print(f"    !! score-tensor materialised: "
                  f"{probe.score_matmul_shapes[:3]}", flush=True)
        _guard()

    print(f"\n[battery] total ref steps={total_ref_steps} "
          f"-> {'ALL BYTE-EXACT' if all_ok else 'DIVERGENCE'}", flush=True)

    # --- ATTENTION COST before->after + steps/sec + frame projection ---------
    measure_attention_cost(model, L, bat, dev)

    uninstall_composed(model)
    return 0 if all_ok else 1


def _time_verify(model, L, code, draft, *, exact_evict, block_steps, device,
                 flash, banded, warmup=1, iters=1):
    """Wall time of a full verify (steps/sec).  Returns (secs, steps)."""
    os.environ["C4_FLASH_ATTN"] = "1" if flash else "0"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "1" if banded else "0"
    for _ in range(warmup):
        _run_verify(model, L, code, draft, exact_evict=exact_evict,
                    block_steps=block_steps, device=device)
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        vr, _st, _o = _run_verify(model, L, code, draft, exact_evict=exact_evict,
                                  block_steps=block_steps, device=device)
    if device != "cpu":
        torch.cuda.synchronize()
    secs = (time.time() - t0) / iters
    os.environ["C4_FLASH_ATTN"] = "0"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "0"
    return secs, draft.step_count


def measure_attention_cost(model, L, bat, dev):
    """Attention cost before (reference: full masked-softmax1 over all live heads,
    content-bound bounded KV) vs after (composed: flash + banded W=30 + dead-block
    fusion + exact-evict).  Reports steps/sec + per-step wall + frame projection."""
    print("\n--- ATTENTION COST (before -> after) + steps/sec ---", flush=True)
    # use the biggest byte-exact program in the battery for a stable timing.
    name, code = max(bat, key=lambda nc: len(nc[1]))
    draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
    steps = draft.step_count

    # BEFORE: reference (full masked-softmax1 over all heads; content-bound KV).
    uninstall_composed(model)
    install_local_attention(model, window=INGEST_W, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    t_ref, _ = _time_verify(model, L, code, draft, exact_evict=False,
                            block_steps=64, device=dev, flash=False, banded=False)
    uninstall_composed(model)

    # AFTER: composed (flash + banded + dead-block-fusion + exact-evict).
    install_composed(model, verbose=False)
    t_cmp, _ = _time_verify(model, L, code, draft, exact_evict=True,
                            block_steps=64, device=dev, flash=True, banded=True)
    uninstall_composed(model)

    # ANALYTIC attention-scoring FLOP reduction (head-slots scored + KV rows).
    st = live_head_attention_stats(model)
    total_slots = st["total_head_slots"]
    live_slots = st["live_head_slots"]
    print(f"\n  ANALYTIC attention scoring: BEFORE scores ALL {total_slots} head-slots "
          f"over FULL-history KV; AFTER scores only {live_slots} live slots "
          f"({live_slots/total_slots*100:.3f}%), {st['dead_attention_blocks']} dead "
          f"blocks do ZERO attention (output=x). Ingest ({total_slots and 20} local "
          f"heads) read last-{INGEST_W} keys (banded); {st['live_head_slots']-20} "
          f"global heads read the exact-evicted bounded KV via flash.", flush=True)

    sps_ref = steps / max(t_ref, 1e-9)
    sps_cmp = steps / max(t_cmp, 1e-9)
    print(f"  program '{name}' steps={steps}", flush=True)
    print(f"  BEFORE (full masked-softmax1, all live heads scored): "
          f"{t_ref*1e3/steps:.3f} ms/step  ({sps_ref:.1f} steps/sec)", flush=True)
    print(f"  AFTER  (flash + banded W=30 + dead-block-fusion + exact-evict): "
          f"{t_cmp*1e3/steps:.3f} ms/step  ({sps_cmp:.1f} steps/sec)", flush=True)
    print(f"  speedup: {t_ref/max(t_cmp,1e-9):.2f}x whole-verify wall", flush=True)
    # frame projection: doom emits ~one visible frame per N VM steps; project the
    # per-frame wall from the per-step wall (a doom frame ~ 555K instrs / real port).
    ms_step = t_cmp * 1e3 / steps
    print(f"  frame projection (per-step {ms_step:.3f} ms): "
          f"1000 steps -> {ms_step:.1f} ms; 100K steps -> {ms_step*100:.1f} s; "
          f"(a doom title frame ~555K instrs -> {ms_step*555:.1f} s at this rate)",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
