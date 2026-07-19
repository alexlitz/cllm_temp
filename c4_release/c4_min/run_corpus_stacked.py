#!/usr/bin/env python3
"""STACKED FAST CORPUS RUNNER — sparse + speculation + O(1)-decode (+ cross-program
batching), with a per-lever ABLATION mode.

The four speedup levers, each previously measured only in ISOLATION, STACKED:

  1. SPARSE compute       — SparseTransformer(compute_mode); sparse_mm is the CPU
                            compute win (work ~ nnz at 99.998% sparsity),
                            argmax-identical; dense_kernel is bit-identical (L-inf=0)
                            and the GPU default.
  2. SPECULATION          — perfect-draft (the logical VM drafts the whole stream,
                            zero forwards) + block-verify (a handful of batched
                            forwards); 32-61x fewer forwards on deep programs.
  3. O(1)-DECODE          — the row-invariant program-in-data overlay is broadcast
                            ONCE per block instead of re-written into every row
                            (build_code_vec + apply_overlay_window_fast); 137x
                            faster overlay, byte-identical.  This is what unstalls
                            the deepest programs (O(stream*C) -> O(stream + blocks*C)).
  4. CROSS-PROGRAM BATCH  — stack many INDEPENDENT programs' block-verify spans into
                            ONE [B, W, D] batched forward (they are causally
                            independent), amortising the 300-block forward launch.

A program PASSes iff the model ACCEPTS the full drafted stream (every step-query
row's decoded register state matches the draft — byte-for-byte what greedy
token-by-token autoregression emits) AND the decoded final AX == expected.

Runs the FULL 1096 corpus (or a subset / clusters).  The deep-recursion constants
(EFF=500000, SP_INIT=0xFC) are applied so rec_fib / rec_sum / gcd / loop_* pass.

Usage
-----
    OMP_NUM_THREADS=4 python c4_min/run_corpus_stacked.py \
        --device cuda:0 --compute-mode dense_kernel --output /tmp/stacked.json
    # CPU, sparse compute:
    OMP_NUM_THREADS=4 python c4_min/run_corpus_stacked.py \
        --device cpu --compute-mode sparse_mm --clusters add,loop_sum,gcd
    # ablation on a subset:
    ... --ablation --clusters add,sub,var_simple,loop_sum,gcd,rec_fib --per-cluster 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

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

from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model, ref_interpret,
)
from c4_min.sparse_forward import SparseTransformer  # noqa: E402
from c4_min.pf_speculative import (  # noqa: E402
    speculative_run, draft_pf_program, spotcheck_vs_cached,
)
from c4_min.batched_speculative import speculative_run_batch  # noqa: E402

_WORD = 8
_SLOT_SCALED_OPS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _sign32(imm: int) -> int:
    return imm if imm < (1 << 31) else imm - (1 << 32)


def bytecode_to_isa(bytecode):
    out = []
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8
        if op in _SLOT_SCALED_OPS:
            simm = _sign32(imm)
            assert simm % _WORD == 0, f"unaligned {isa.NAMES.get(op)} imm {simm}"
            out.append(isa.Instr(op, simm // _WORD))
        else:
            out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out


def cluster_of(description: str) -> str:
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


def build_model(device: str, compute_mode: str, include_divmod: bool,
                include_bitwise: bool, code_size: int, verbose: bool = True):
    t = time.monotonic()
    if verbose:
        print(f"[stacked:{device}] building sparse model "
              f"(divmod={include_divmod} bitwise={include_bitwise} "
              f"compute={compute_mode}) ...", file=sys.stderr, flush=True)
    if include_divmod:
        base, L, _cs = build_compact_pure_forward_model(
            code_size=code_size, include_bitwise=include_bitwise, include_divmod=True)
    else:
        base, L = build_pure_forward_complete_model(
            code_size=code_size, include_bitwise=include_bitwise, include_divmod=False)
    sparse = SparseTransformer(base, compute_mode=compute_mode)
    st = sparse.stats()
    del base
    sparse = sparse.to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
        vram = torch.cuda.memory_allocated(torch.device(device)) / 1e6
    else:
        vram = 0.0
    build_dt = time.monotonic() - t
    if verbose:
        print(f"[stacked:{device}] built in {build_dt:.0f}s | blocks={len(sparse.blocks)} "
              f"dim={sparse.dim} storage={st.sparse_mb:.1f}MB VRAM={vram:.0f}MB",
              file=sys.stderr, flush=True)
    return sparse, L, st, build_dt, vram


def select_programs(args, all_tests):
    indexed = list(enumerate(all_tests))[args.offset:]
    if args.limit is not None:
        indexed = indexed[:args.limit]
    if args.clusters:
        want = {c.strip() for c in args.clusters.split(",")}
        indexed = [(i, tp) for i, tp in indexed if cluster_of(tp[2]) in want]
    if args.exclude_clusters:
        drop = {c.strip() for c in args.exclude_clusters.split(",")}
        indexed = [(i, tp) for i, tp in indexed if cluster_of(tp[2]) not in drop]
    if args.per_cluster is not None:
        seen: Counter = Counter()
        out = []
        for i, tp in indexed:
            cl = cluster_of(tp[2])
            if seen[cl] < args.per_cluster:
                seen[cl] += 1
                out.append((i, tp))
        indexed = out
    return indexed


def prepare(indexed):
    """Compile each program + draft it (free) to get the exact step count so we can
    depth-bucket for cross-program batching."""
    from src.compiler import compile_c
    prepared = []
    for idx, (source, expected, description) in indexed:
        try:
            code = bytecode_to_isa(compile_c(source)[0])
        except Exception as exc:  # noqa: BLE001
            prepared.append(dict(idx=idx, cluster=cluster_of(description),
                                 description=description, expected=expected,
                                 code=None, error=f"compile: {exc!r}"))
            continue
        prepared.append(dict(idx=idx, cluster=cluster_of(description),
                             description=description, expected=expected,
                             code=code, error=None))
    return prepared


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compute-mode", type=str, default="sparse_mm",
                    choices=["dense_kernel", "sparse_mm"])
    ap.add_argument("--no-divmod", action="store_true")
    ap.add_argument("--no-bitwise", action="store_true", default=True)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--block-steps", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--no-fast-overlay", action="store_true",
                    help="disable the O(1)-decode overlay (for the ablation).")
    ap.add_argument("--batch", type=int, default=1,
                    help="cross-program batch size for the block-verify (1 = per "
                         "program).  Programs are depth-bucketed so a batch runs "
                         "~its members' depth with no idle slots.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--clusters", type=str, default=None)
    ap.add_argument("--exclude-clusters", type=str, default=None)
    ap.add_argument("--per-cluster", type=int, default=None)
    ap.add_argument("--spotcheck-n", type=int, default=0,
                    help="byte-identity spot-check vs the token-by-token driver on "
                         "the N shortest passing programs (proves AR-identity).")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--progress", type=int, default=25)
    ap.add_argument("--rss-abort-gb", type=float, default=40.0)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[stacked] CUDA unavailable; cpu", file=sys.stderr)
        device = "cpu"

    from tests.test_suite_1000 import generate_test_programs
    all_tests = generate_test_programs()
    indexed = select_programs(args, all_tests)

    include_divmod = not args.no_divmod
    include_bitwise = not args.no_bitwise
    sparse, L, st, build_dt, vram = build_model(
        device, args.compute_mode, include_divmod, include_bitwise, args.code_size)

    prepared = prepare(indexed)
    print(f"[stacked:{device}] running {len(prepared)} programs | "
          f"batch={args.batch} fast_overlay={not args.no_fast_overlay} "
          f"block_steps={args.block_steps} evict={not args.no_evict}",
          file=sys.stderr, flush=True)

    evict = not args.no_evict
    fast = not args.no_fast_overlay
    results: List[dict] = []
    per_cluster: Dict[str, Counter] = defaultdict(Counter)
    sum_naive = sum_spec = 0
    t0 = time.monotonic()

    ok = [p for p in prepared if p["code"] is not None]
    errs = [p for p in prepared if p["code"] is None]
    for p in errs:
        results.append(dict(idx=p["idx"], cluster=p["cluster"], status="ERROR",
                            detail=p["error"], description=p["description"]))
        per_cluster[p["cluster"]]["ERROR"] += 1

    if args.batch <= 1:
        # per-program speculative (steps batched within a program).
        for k, p in enumerate(ok):
            try:
                r = speculative_run(sparse, L, p["code"], p["expected"],
                                    block_steps=args.block_steps,
                                    max_steps=args.max_steps, device=device,
                                    evict=evict, prune_interval=args.prune_interval,
                                    fast=fast)
            except Exception as exc:  # noqa: BLE001
                results.append(dict(idx=p["idx"], cluster=p["cluster"],
                                    status="ERROR", detail=f"spec: {exc!r}",
                                    description=p["description"]))
                per_cluster[p["cluster"]]["ERROR"] += 1
                continue
            per_cluster[p["cluster"]][r.status] += 1
            sum_naive += r.naive_forwards
            sum_spec += r.forwards
            results.append(dict(
                idx=p["idx"], cluster=p["cluster"], status=r.status,
                expected=r.expected, got=r.decoded_final_ax, steps=r.step_count,
                forwards=r.forwards, naive_forwards=r.naive_forwards,
                speedup=round(r.speedup, 2), max_cache=r.max_cache_size,
                detail=r.detail, description=p["description"]))
            if args.progress and (k + 1) % args.progress == 0:
                npass = sum(1 for x in results if x["status"] == "PASS")
                _report_progress(device, k + 1, len(ok), npass, t0, args)
    else:
        # cross-program batched speculative (block-verify spans stacked across
        # depth-bucketed programs).
        rows, sn, ss = speculative_run_batch(
            sparse, L, ok, block_steps=args.block_steps, max_steps=args.max_steps,
            device=device, evict=evict, prune_interval=args.prune_interval,
            fast=fast, batch_cap=args.batch,
            progress=lambda done, npass: _report_progress(
                device, done, len(ok), npass, t0, args))
        sum_naive += sn
        sum_spec += ss
        for r in rows:
            per_cluster[r["cluster"]][r["status"]] += 1
            results.append(r)

    wall = time.monotonic() - t0
    counts = Counter(r["status"] for r in results)
    n_pass = counts.get("PASS", 0)
    total = len(results)
    total_steps = sum(r.get("steps", 0) or 0 for r in results)
    agg_speedup = (sum_naive / sum_spec) if sum_spec else float("inf")

    # -- byte-identity spot-check vs the token-by-token KV-cached driver ----------
    spotchecks = []
    if args.spotcheck_n > 0:
        passing = sorted([r for r in results if r["status"] == "PASS"],
                         key=lambda r: r.get("steps", 0))[:args.spotcheck_n]
        print(f"[stacked:{device}] byte-identity spot-check vs token-by-token "
              f"driver on {len(passing)} programs ...", file=sys.stderr, flush=True)
        by_idx = {p["idx"]: p for p in ok}
        for r in passing:
            p = by_idx.get(r["idx"])
            if p is None:
                continue
            sc = spotcheck_vs_cached(sparse, L, p["code"],
                                     max_steps=(r.get("steps", 0) or 0) + 8,
                                     device=device, evict=evict,
                                     prune_interval=args.prune_interval,
                                     block_steps=args.block_steps)
            spotchecks.append(dict(idx=r["idx"], cluster=r["cluster"], **sc))
            print(f"  idx={r['idx']} [{r['cluster']}] IDENTICAL={sc['identical']} "
                  f"fast==slow={sc['fast_matches_slow']} "
                  f"driver_final={sc['driver_final_ax']} "
                  f"spec_final={sc['spec_final_ax']} steps={sc['n_steps_driver']}",
                  file=sys.stderr, flush=True)

    # -- report -------------------------------------------------------------------
    _report(results, per_cluster, wall, build_dt, st, vram, device, args,
            counts, n_pass, total, total_steps, sum_naive, sum_spec, agg_speedup,
            spotchecks, fast, evict)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "device": device, "compute_mode": args.compute_mode,
                "wall_seconds": wall, "build_seconds": build_dt,
                "include_divmod": include_divmod, "batch": args.batch,
                "fast_overlay": fast, "evict": evict,
                "block_steps": args.block_steps,
                "storage_mb": st.sparse_mb, "vram_load_mb": vram,
                "n_programs": total, "total_steps": total_steps,
                "steps_per_sec": round(total_steps / wall, 1) if wall else 0,
                "status_counts": dict(counts),
                "forward_speedup": {"naive": sum_naive, "spec": sum_spec,
                                    "speedup": round(agg_speedup, 2)},
                "per_cluster": {cl: dict(c) for cl, c in per_cluster.items()},
                "spotchecks": spotchecks,
                "results": results,
            }, fh, indent=2)
        print(f"[stacked:{device}] wrote {args.output}", file=sys.stderr, flush=True)
    return 0


def _report_progress(device, done, total, npass, t0, args):
    el = time.monotonic() - t0
    rate = done / el if el else 0
    eta = (total - done) / rate if rate else 0
    print(f"[stacked:{device}] {done}/{total} (pass {npass}) "
          f"[{el:.0f}s, {rate:.2f} prog/s, ETA {eta:.0f}s]",
          file=sys.stderr, flush=True)


def _report(results, per_cluster, wall, build_dt, st, vram, device, args,
            counts, n_pass, total, total_steps, sum_naive, sum_spec, agg_speedup,
            spotchecks, fast, evict):
    print("\n" + "=" * 78)
    print(f"STACKED FAST CORPUS RUNNER — sparse + speculation + O(1)-decode"
          f"{' + batch' if args.batch > 1 else ''} on {device}")
    print("=" * 78)
    print(f"  compute: {args.compute_mode} | fast_overlay: {fast} | "
          f"batch: {args.batch} | evict: {evict} | block_steps: {args.block_steps}")
    print(f"  model: {st.sparse_mb:.1f}MB storage, {vram:.0f}MB VRAM, "
          f"build {build_dt:.0f}s")
    print(f"  ran {total} programs | wall {wall:.1f}s ({wall/60:.2f} min) "
          f"| {total_steps} VM steps | {total_steps/wall:.0f} steps/s")
    print(f"  status: " + " ".join(f"{s}={counts.get(s,0)}"
                                    for s in ("PASS", "FAIL", "TIMEOUT", "ERROR")))
    print(f"  forward speedup (naive token-by-token vs speculative): "
          f"{sum_naive} -> {sum_spec} = {agg_speedup:.1f}x")
    print(f"  SCORE: {n_pass}/{total} ({100.0*n_pass/max(total,1):.2f}%)")
    if spotchecks:
        all_id = all(s["identical"] for s in spotchecks)
        print("-" * 78)
        print(f"  BYTE-IDENTITY vs token-by-token KV-cached driver: "
              f"{sum(1 for s in spotchecks if s['identical'])}/{len(spotchecks)} "
              f"identical {'(ALL MATCH)' if all_id else '(MISMATCH!)'}")
    print("-" * 78)
    print("  PER-CLUSTER:")
    for cl in sorted(per_cluster):
        c = per_cluster[cl]
        n = sum(c.values())
        print(f"    {cl:18s} n={n:3d} PASS={c.get('PASS',0):3d} "
              f"FAIL={c.get('FAIL',0):3d} TIMEOUT={c.get('TIMEOUT',0):3d} "
              f"ERROR={c.get('ERROR',0):3d}")
    genuine = [r for r in results if r["status"] in ("FAIL", "ERROR")]
    if genuine:
        print("-" * 78)
        print("  GENUINE FAILS / ERRORS (report, not hide):")
        for r in genuine[:40]:
            print(f"    id={r['idx']} [{r['cluster']}] {r['status']}: "
                  f"{r.get('detail','')}")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
