#!/usr/bin/env python3
"""AGGRESSIVE-SPECULATION, CHECKPOINT-RESUMABLE full-1096 verify-through-the-model.

Every program is verified THROUGH THE ACTUAL MODEL — ``verify_blocks`` confirms the
model's argmax register state equals the perfect draft at EVERY step-query row (NO
sampling, greedy).  The draft (the logical VM) is PERFECT (100% acceptance, never a
re-draft), so small verify blocks are pure waste.  This runner drives the four
speedup levers to their aggressive limit and makes the sweep survive a wall-clock
kill:

AGGRESSIVE SPECULATION (minimise the forward COUNT — the expensive thing)
  1. WHOLE-PROGRAM DRAFT up front — the logical VM drafts the entire token stream
     for every program (pure Python, instant), so the whole verify depth is known
     before any forward.
  2. MAX-CHUNK VERIFY — each verify forward is as LARGE as VRAM allows: a shallow
     program's whole stream verifies in ONE forward; a deep stream (rec_fib ~8369
     steps -> ~250k tokens) is chunked by the incremental KV cache into the FEWEST
     forwards that fit VRAM (``--lean-block-steps`` big -> a handful, not 8369).
  3. CROSS-PROGRAM BATCH — many programs' whole drafted streams are stacked
     (depth-bucketed) into ONE batched forward, so a single forward verifies many
     programs' many steps at once (``speculative_run_batch``).

Two model lanes (each a pre-saved SPARSE artifact, ``--load-sparse``, EFF=500000):
  * LEAN lane   (42 blocks)  — every non-divmod cluster, incl. the deep rec_fib
    tail.  Light per-token, so a BIG block-steps chunk fits (few forwards).
  * DIVMOD lane (300+ blocks) — div / mod / gcd / expr_mod / expr_mul_div (the
    32-bit long-division blocks).  Heavy per-token (300 blocks), so a smaller
    block-steps; but its deepest program is only ~253 steps (gcd), so still few
    forwards.

CHECKPOINT-RESUMABLE
  Each program's verdict is written to a persistent JSON (``--checkpoint``) as it is
  produced, flushed after every batch.  On restart the runner SKIPS every program
  already in the checkpoint, so a wall-clock kill never loses progress — the sweep
  accumulates to 1096/1096 across restarts.

CORRECTNESS (non-negotiable)
  ``verify_blocks`` argmax == draft at every position; PASS iff the model accepts
  the full stream AND the decoded final AX == the byte-exact reference.  SP_INIT=0xFC
  + EFF=500000 -> 1096/1096.  ``--spotcheck-n`` proves byte-identity vs the
  token-by-token KV-cached driver.

Usage
-----
    OMP_NUM_THREADS=4 python c4_min/run_corpus_resumable.py \
        --device cuda:0 \
        --lean-sparse /tmp/fs_lean_sparse.pt \
        --divmod-sparse /tmp/fs_divmod_sparse.pt \
        --checkpoint /tmp/fastspec_1096.json \
        --lean-block-steps 2000 --divmod-block-steps 96 \
        --lean-batch 64 --divmod-batch 24 --vram-budget 6.0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# DEEP-RECURSION constants MUST be pinned before importing the driver modules.
import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

import torch  # noqa: E402

from c4_min import blogspec_memory as _MEM  # noqa: E402
from c4_min.run_corpus_stacked import bytecode_to_isa, cluster_of  # noqa: E402
from c4_min.compact_alloc import load_sparse_transformer  # noqa: E402
from c4_min.pf_speculative import draft_pf_program, spotcheck_vs_cached  # noqa: E402
from c4_min.batched_speculative import _bucket, _verify_batch  # noqa: E402

# Clusters that need the 32-bit long-division blocks -> the DIVMOD lane.
_DIVMOD_CLUSTERS = frozenset({
    "div", "mod", "gcd", "expr_mod", "expr_mul_div",
    "edge_1div", "edge_div_one", "edge_zero_div", "edge_zero_mod",
})


# ---------------------------------------------------------------------------
# Checkpoint I/O (atomic, resumable).
# ---------------------------------------------------------------------------
def _load_checkpoint(path: str) -> Dict[int, dict]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 — a truncated write; start fresh but keep .bak
        try:
            os.replace(path, path + ".corrupt")
        except OSError:
            pass
        return {}
    return {int(r["idx"]): r for r in data.get("results", [])}


def _flush_checkpoint(path: str, done: Dict[int, dict], meta: dict) -> None:
    if not path:
        return
    results = [done[i] for i in sorted(done)]
    counts = Counter(r["status"] for r in results)
    payload = dict(meta)
    payload["n_done"] = len(results)
    payload["status_counts"] = dict(counts)
    payload["n_pass"] = counts.get("PASS", 0)
    payload["results"] = results
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)   # atomic: a kill mid-flush leaves the prior good file


# ---------------------------------------------------------------------------
def _prepare(indexed):
    """Compile + draft each program (both free) so we know the exact depth and can
    depth-bucket for the aggressive cross-program batch."""
    from src.compiler import compile_c
    items = []
    errors = []
    for idx, (source, expected, description) in indexed:
        cl = cluster_of(description)
        try:
            code = bytecode_to_isa(compile_c(source)[0])
        except Exception as exc:  # noqa: BLE001
            errors.append(dict(idx=idx, cluster=cl, status="ERROR",
                               expected=expected & 0xFFFFFFFF, got=None,
                               detail=f"compile: {exc!r}", description=description,
                               lane="-"))
            continue
        items.append(dict(idx=idx, cluster=cl, description=description,
                          expected=expected, code=code))
    return items, errors


def _run_lane(lane_name, model, L, items, *, block_steps, batch_cap, device,
              evict, prune_interval, vram_budget, fast, checkpoint, done, meta,
              t0, flush_every, rss_abort_gb):
    """Draft (free) + depth-bucket + aggressive batched verify a lane's items,
    checkpointing after every batch.  Byte-identical to per-program verify_blocks."""
    import resource
    mask = 0xFFFFFFFF
    # draft everything (free); non-halting -> TIMEOUT (should be none on this corpus).
    verifiable = []
    for it in items:
        draft = draft_pf_program(it["code"], max_steps=300000, mask=mask)
        if not draft.halted:
            r = dict(idx=it["idx"], cluster=it["cluster"], status="TIMEOUT",
                     expected=it["expected"] & 0xFFFFFFFF, got=None, steps=draft.step_count,
                     forwards=0, detail="draft did not HALT", description=it["description"],
                     lane=lane_name)
            done[it["idx"]] = r
            continue
        it2 = dict(it)
        it2["draft"] = draft
        verifiable.append(it2)

    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    bytes_per_prog_row = (n_blocks * H * HD * 4 * 2
                          if device.startswith("cuda") else 0.0)
    batches = _bucket(verifiable, batch_cap, block_steps=block_steps,
                      bytes_per_prog_row=bytes_per_prog_row,
                      vram_budget_gb=vram_budget, evict=evict,
                      prune_interval=prune_interval)
    lane_forwards = 0
    lane_naive = 0
    bi = 0
    since_flush = 0
    while bi < len(batches):
        batch = batches[bi]
        try:
            nf, sn = _verify_batch(model, L, batch, block_steps, device, evict,
                                   prune_interval, mask, fast)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                # split the DEEP single program across FEWER-but-smaller chunks by
                # halving block_steps for just this item (still few forwards).
                it = batch[0]
                if block_steps > 8:
                    bs2 = max(8, block_steps // 2)
                    try:
                        nf, sn = _verify_batch(model, L, batch, bs2, device, evict,
                                               prune_interval, mask, fast)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        r = dict(idx=it["idx"], cluster=it["cluster"], status="ERROR",
                                 expected=it["expected"] & 0xFFFFFFFF, got=None,
                                 steps=it["draft"].step_count, forwards=0,
                                 detail="OOM even at min block-steps",
                                 description=it["description"], lane=lane_name)
                        done[it["idx"]] = r
                        bi += 1
                        continue
                else:
                    r = dict(idx=it["idx"], cluster=it["cluster"], status="ERROR",
                             expected=it["expected"] & 0xFFFFFFFF, got=None,
                             steps=it["draft"].step_count, forwards=0,
                             detail="OOM at batch=1 min block-steps",
                             description=it["description"], lane=lane_name)
                    done[it["idx"]] = r
                    bi += 1
                    continue
            else:
                half = (len(batch) + 1) // 2
                batches[bi:bi + 1] = [batch[:half], batch[half:]]
                continue
        lane_forwards += nf
        lane_naive += sn
        for it in batch:
            r = dict(idx=it["idx"], cluster=it["cluster"], status=it["status"],
                     expected=it["expected"] & 0xFFFFFFFF, got=it["got"],
                     steps=it["steps"], forwards=nf, accepted=it.get("accepted"),
                     detail=it["detail"], description=it["description"], lane=lane_name)
            done[it["idx"]] = r
        bi += 1
        since_flush += len(batch)
        meta["lane_forwards"][lane_name] = lane_forwards
        meta["lane_naive"][lane_name] = lane_naive
        if since_flush >= flush_every or bi >= len(batches):
            _flush_checkpoint(checkpoint, done, meta)
            since_flush = 0
        npass = sum(1 for r in done.values() if r["status"] == "PASS")
        el = time.monotonic() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"[resumable:{lane_name}] batch {bi}/{len(batches)} | done {len(done)} "
              f"(pass {npass}) | fwd {lane_forwards} | {el:.0f}s | RSS {rss:.1f}GB",
              file=sys.stderr, flush=True)
        if rss > rss_abort_gb:
            print(f"[resumable] RSS {rss:.1f}GB > abort {rss_abort_gb}GB — flushing "
                  f"+ exiting cleanly (resume to continue).", file=sys.stderr, flush=True)
            _flush_checkpoint(checkpoint, done, meta)
            raise SystemExit(3)
    return lane_forwards, lane_naive


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--compute-mode", type=str, default="dense_kernel",
                    choices=["dense_kernel", "sparse_mm"])
    ap.add_argument("--lean-sparse", type=str, required=True,
                    help="pre-saved LEAN sparse artifact (42 blocks, no divmod).")
    ap.add_argument("--divmod-sparse", type=str, required=True,
                    help="pre-saved DIVMOD sparse artifact (300+ blocks).")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="persistent JSON — verdicts flushed here per batch; resumed.")
    ap.add_argument("--lean-block-steps", type=int, default=64,
                    help="VM steps per verify forward on the LEAN lane.  ONE forward "
                         "covers a whole program up to this depth (most of the corpus "
                         "is <=64 steps -> 1 forward each).  A DEEP stream (rec_fib "
                         "8369) chunks into ~depth/bs forwards.  The per-forward span "
                         "is a W=bs*30-wide attention (QUADRATIC in the span), so this "
                         "is the largest chunk that stays well under the VRAM budget "
                         "on the deepest program (bs=64 -> ~3GB; bs>=256 OOMs rec_fib).")
    ap.add_argument("--divmod-block-steps", type=int, default=48,
                    help="VM steps per verify forward on the DIVMOD lane.  The 300+ "
                         "block model is HEAVY per-token, so the W=bs*30 quadratic "
                         "span must be smaller: the deepest divmod program (gcd, 253 "
                         "steps) verifies in 6 forwards at bs=48 (~13GB); bs>=96 OOMs "
                         "it.  Shallow div/mod (~5 steps) still verify in 1 forward.")
    ap.add_argument("--lean-batch", type=int, default=64)
    ap.add_argument("--divmod-batch", type=int, default=24)
    ap.add_argument("--vram-budget", type=float, default=6.0)
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--no-fast-overlay", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clusters", type=str, default=None)
    ap.add_argument("--flush-every", type=int, default=8,
                    help="flush the checkpoint after this many programs.")
    ap.add_argument("--spotcheck-n", type=int, default=0,
                    help="byte-identity spot-check vs the token-by-token driver on "
                         "the N shortest passing programs.")
    ap.add_argument("--rss-abort-gb", type=float, default=30.0)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[resumable] CUDA unavailable; cpu", file=sys.stderr)
        device = "cpu"
    assert abs(_MEM.EFF - 500000.0) < 1e-6, f"EFF must be 500000, got {_MEM.EFF}"

    from tests.test_suite_1000 import generate_test_programs
    all_tests = generate_test_programs()
    indexed = list(enumerate(all_tests))
    if args.clusters:
        want = {c.strip() for c in args.clusters.split(",")}
        indexed = [(i, tp) for i, tp in indexed if cluster_of(tp[2]) in want]
    if args.limit is not None:
        indexed = indexed[:args.limit]

    items, errors = _prepare(indexed)
    total = len(items) + len(errors)

    # resume: load prior verdicts, skip programs already done.
    done: Dict[int, dict] = _load_checkpoint(args.checkpoint)
    n_resumed = len(done)
    for e in errors:
        done.setdefault(e["idx"], e)
    remaining = [it for it in items if it["idx"] not in done]
    lean_items = [it for it in remaining if it["cluster"] not in _DIVMOD_CLUSTERS]
    divmod_items = [it for it in remaining if it["cluster"] in _DIVMOD_CLUSTERS]

    print(f"[resumable:{device}] corpus={total} resumed={n_resumed} "
          f"remaining={len(remaining)} (lean {len(lean_items)} / divmod {len(divmod_items)})",
          file=sys.stderr, flush=True)

    meta = {
        "device": device, "compute_mode": args.compute_mode,
        "sp_init": _PFC.SP_INIT, "eff": _MEM.EFF,
        "lean_block_steps": args.lean_block_steps,
        "divmod_block_steps": args.divmod_block_steps,
        "lean_batch": args.lean_batch, "divmod_batch": args.divmod_batch,
        "evict": not args.no_evict, "total": total,
        "lane_forwards": {}, "lane_naive": {},
    }
    evict = not args.no_evict
    fast = not args.no_fast_overlay
    t0 = time.monotonic()

    # LEAN lane first (the bulk + the deep rec_fib tail).
    if lean_items:
        print(f"[resumable:{device}] loading LEAN sparse {args.lean_sparse} ...",
              file=sys.stderr, flush=True)
        lean, Ll = load_sparse_transformer(args.lean_sparse, compute_mode=args.compute_mode)
        lean = lean.to(device)
        print(f"[resumable:{device}] LEAN loaded: {len(lean.blocks)} blocks dim={lean.dim}",
              file=sys.stderr, flush=True)
        _run_lane("lean", lean, Ll, lean_items, block_steps=args.lean_block_steps,
                  batch_cap=args.lean_batch, device=device, evict=evict,
                  prune_interval=args.prune_interval, vram_budget=args.vram_budget,
                  fast=fast, checkpoint=args.checkpoint, done=done, meta=meta, t0=t0,
                  flush_every=args.flush_every, rss_abort_gb=args.rss_abort_gb)
        if args.spotcheck_n > 0:
            _spotcheck(lean, Ll, lean_items, done, args, device, evict, "lean")
        del lean
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # DIVMOD lane.
    if divmod_items:
        print(f"[resumable:{device}] loading DIVMOD sparse {args.divmod_sparse} ...",
              file=sys.stderr, flush=True)
        dm, Ld = load_sparse_transformer(args.divmod_sparse, compute_mode=args.compute_mode)
        dm = dm.to(device)
        print(f"[resumable:{device}] DIVMOD loaded: {len(dm.blocks)} blocks dim={dm.dim}",
              file=sys.stderr, flush=True)
        _run_lane("divmod", dm, Ld, divmod_items, block_steps=args.divmod_block_steps,
                  batch_cap=args.divmod_batch, device=device, evict=evict,
                  prune_interval=args.prune_interval, vram_budget=args.vram_budget,
                  fast=fast, checkpoint=args.checkpoint, done=done, meta=meta, t0=t0,
                  flush_every=args.flush_every, rss_abort_gb=args.rss_abort_gb)
        if args.spotcheck_n > 0:
            _spotcheck(dm, Ld, divmod_items, done, args, device, evict, "divmod")
        del dm
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    wall = time.monotonic() - t0
    meta["wall_seconds"] = wall
    _flush_checkpoint(args.checkpoint, done, meta)
    _report(done, meta, wall, device, total)
    return 0


def _spotcheck(model, L, items, done, args, device, evict, lane):
    """Byte-identity spot-check vs the token-by-token KV-cached driver."""
    by_idx = {it["idx"]: it for it in items}
    passing = sorted([done[i] for i in by_idx
                      if i in done and done[i]["status"] == "PASS"],
                     key=lambda r: r.get("steps", 0))[:args.spotcheck_n]
    print(f"[resumable:{lane}] byte-identity spot-check vs token-by-token driver "
          f"on {len(passing)} programs ...", file=sys.stderr, flush=True)
    all_id = True
    for r in passing:
        it = by_idx.get(r["idx"])
        if it is None:
            continue
        sc = spotcheck_vs_cached(model, L, it["code"],
                                 max_steps=(r.get("steps", 0) or 0) + 8,
                                 device=device, evict=evict,
                                 prune_interval=args.prune_interval,
                                 block_steps=args.divmod_block_steps if lane == "divmod"
                                 else args.lean_block_steps)
        all_id = all_id and sc["identical"]
        print(f"  idx={r['idx']} [{r['cluster']}] IDENTICAL={sc['identical']} "
              f"fast==slow={sc['fast_matches_slow']} driver={sc['driver_final_ax']} "
              f"spec={sc['spec_final_ax']} steps={sc['n_steps_driver']}",
              file=sys.stderr, flush=True)
    print(f"[resumable:{lane}] spot-check: {'ALL IDENTICAL' if all_id else 'MISMATCH!'}",
          file=sys.stderr, flush=True)


def _report(done, meta, wall, device, total):
    results = [done[i] for i in sorted(done)]
    counts = Counter(r["status"] for r in results)
    n_pass = counts.get("PASS", 0)
    per_cluster = defaultdict(Counter)
    for r in results:
        per_cluster[r["cluster"]][r["status"]] += 1
    lf = meta.get("lane_forwards", {})
    ln = meta.get("lane_naive", {})
    tot_fwd = sum(lf.values())
    tot_naive = sum(ln.values())
    print("\n" + "=" * 78)
    print(f"AGGRESSIVE-SPECULATION RESUMABLE FULL-1096 VERIFY on {device}")
    print("=" * 78)
    print(f"  SP_INIT={meta['sp_init']:#x} EFF={meta['eff']} evict={meta['evict']} "
          f"compute={meta['compute_mode']}")
    print(f"  lean block-steps={meta['lean_block_steps']} batch={meta['lean_batch']} | "
          f"divmod block-steps={meta['divmod_block_steps']} batch={meta['divmod_batch']}")
    print(f"  scored {len(results)}/{total} | wall {wall:.1f}s ({wall/60:.2f} min)")
    print(f"  status: " + " ".join(f"{s}={counts.get(s,0)}"
                                    for s in ("PASS", "FAIL", "TIMEOUT", "ERROR")))
    print(f"  TOTAL MODEL FORWARDS: {tot_fwd}  (naive token-by-token would be "
          f"{tot_naive} = {tot_naive/max(tot_fwd,1):.1f}x fewer)")
    for lane in sorted(lf):
        print(f"    {lane:8s} forwards={lf[lane]:6d}  naive={ln.get(lane,0):8d}")
    print(f"  SCORE: {n_pass}/{len(results)} ({100.0*n_pass/max(len(results),1):.2f}%)")
    print("-" * 78)
    for cl in sorted(per_cluster):
        c = per_cluster[cl]
        n = sum(c.values())
        line = (f"    {cl:18s} n={n:3d} PASS={c.get('PASS',0):3d}")
        if c.get("FAIL") or c.get("ERROR") or c.get("TIMEOUT"):
            line += (f" FAIL={c.get('FAIL',0)} TIMEOUT={c.get('TIMEOUT',0)} "
                     f"ERROR={c.get('ERROR',0)}")
        print(line)
    genuine = [r for r in results if r["status"] in ("FAIL", "ERROR", "TIMEOUT")]
    if genuine:
        print("-" * 78)
        print("  NON-PASS (report, not hide):")
        for r in genuine[:60]:
            print(f"    id={r['idx']} [{r['cluster']}] {r['status']}: {r.get('detail','')[:80]}")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
