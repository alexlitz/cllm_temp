#!/usr/bin/env python3
"""c4_min PURE-FORWARD VM — FULL 1096 SCOREBOARD, KV-CACHED (CHK-1 item #6).

Same corpus / verdict / categorisation as ``run_1096_pure_forward.py``, but every
VM step runs through the **KV-cached** driver
(:func:`nibble_pure_forward_cached.run_pure_forward_cached`): each step forwards
ONLY the small window of tokens whose residual changed (the new 30-token frame +
the one re-tagged previous query row) against a per-block incremental KV cache,
turning the naive O(stream^2)/step re-forward into O(cache)/step.  Bounded
eviction (``nibble_kv_prune``, the spec softmax1+ALiBi policy) keeps the cache FLAT
over deep loops, so long problems are BOTH fast AND memory-bounded.

The output is byte-identical to the naive re-forward driver (proven in
``test_cached_driver`` / ``--verify``), so the pass fraction is the SAME verdict —
this runner just makes the full 1096 (deep loops INCLUDED) tractable.

MEMORY DISCIPLINE
-----------------
The window forward touches only ~31 rows, so activations are TINY (31 x dim x
blocks) regardless of stream length — the naive driver's ~118GB dense-divmod
activation blow-up on deep loops CANNOT happen here.  Weights are stored SPARSE
where possible.  Runs SEQUENTIALLY.  Set ``OMP_NUM_THREADS=4``.

Usage
-----
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_pure_forward_cached.py \
        --output /tmp/pf_1096_cached.json
    # bound wall-time with a per-cluster stratified sample (coverage reported):
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_1096_pure_forward_cached.py \
        --per-cluster 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import c4_min.nibble_pure_forward as _PF        # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model, run_pure_forward_complete, ref_interpret,
)
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached  # noqa: E402
from c4_min.nibble_pure_forward import assert_no_python_compute  # noqa: E402

# Re-use the exact corpus->isa translation + cluster keying from the naive runner.
from c4_min.run_1096_pure_forward import (  # noqa: E402
    bytecode_to_isa, cluster_of, Result, _STATUSES,
    _cluster_table, _print_cluster_table,
)


# ---------------------------------------------------------------------------
# The scoreboard core — KV-cached.
# ---------------------------------------------------------------------------
def score_program(idx: int, source: str, expected: int, description: str,
                  *, model, L, compile_c, step_cap: int, guard: bool,
                  ref_steps: Optional[int] = None,
                  evict: bool = True, prune_interval: int = 120,
                  agg: Optional[dict] = None) -> Result:
    cluster = cluster_of(description)
    exp = expected & 0xFFFFFFFF
    base = dict(idx=idx, description=description, cluster=cluster, expected=exp)
    try:
        bytecode, _data = compile_c(source)
        code = bytecode_to_isa(bytecode)
    except Exception as exc:  # noqa: BLE001
        return Result(got_exit=None, got_steps=None, status="ERROR",
                      detail=f"compile/translate: {exc!r}", **base)

    n_instrs = len(code)
    if ref_steps is not None:
        step_cap = min(step_cap, ref_steps + 6)
    stats: dict = {}
    try:
        if guard:
            gclean = True
            try:
                trace = assert_no_python_compute(
                    run_pure_forward_cached, model, L, code, max_steps=step_cap,
                    mask=0xFFFFFFFF, evict=evict, prune_interval=prune_interval,
                    stats=stats)
            except AssertionError as gexc:
                trace = run_pure_forward_cached(
                    model, L, code, max_steps=step_cap, mask=0xFFFFFFFF,
                    evict=evict, prune_interval=prune_interval, stats=stats)
                return Result(got_exit=(trace[-1] if trace else None),
                              got_steps=len(trace), status="ERROR",
                              guard_clean=False, n_instrs=n_instrs,
                              detail=f"GUARD LEAK: {gexc}", **base)
        else:
            gclean = None
            trace = run_pure_forward_cached(
                model, L, code, max_steps=step_cap, mask=0xFFFFFFFF,
                evict=evict, prune_interval=prune_interval, stats=stats)
    except Exception as exc:  # noqa: BLE001
        return Result(got_exit=None, got_steps=None, status="ERROR",
                      guard_clean=(guard and False), n_instrs=n_instrs,
                      detail=f"run: {exc!r}", **base)

    if agg is not None and stats:
        agg["max_seq_len"] = max(agg.get("max_seq_len", 0), stats.get("max_seq_len", 0))
        agg["max_cache_size"] = max(agg.get("max_cache_size", 0),
                                    stats.get("max_cache_size", 0))
        agg["total_evicted"] = agg.get("total_evicted", 0) + stats.get("total_evicted", 0)

    if not trace:
        return Result(got_exit=None, got_steps=0, status="ERROR",
                      guard_clean=gclean, n_instrs=n_instrs,
                      detail="no frame emitted", **base)

    got = int(trace[-1]) & 0xFFFFFFFF
    steps = len(trace)
    detail = (f"maxseq={stats.get('max_seq_len')} "
              f"cache={stats.get('max_cache_size')} "
              f"evicted={stats.get('total_evicted')}")
    if steps >= step_cap:
        return Result(got_exit=got, got_steps=steps, status="TIMEOUT",
                      guard_clean=gclean, n_instrs=n_instrs,
                      detail=f"no HALT within {step_cap} steps; " + detail, **base)
    if got == exp:
        return Result(got_exit=got, got_steps=steps, status="PASS",
                      guard_clean=gclean, n_instrs=n_instrs, detail=detail, **base)
    return Result(got_exit=got, got_steps=steps, status="FAIL",
                  guard_clean=gclean, n_instrs=n_instrs,
                  detail=f"exit mismatch: exp {exp} got {got}; " + detail, **base)


def _print_summary(results, wall, step_cap, coverage, guard, agg, speedup,
                   fh=sys.stdout):
    counts = Counter(r.status for r in results)
    total = len(results)
    n_pass = counts.get("PASS", 0)
    print("\n" + "=" * 72, file=fh)
    print("c4_min PURE-FORWARD VM SCOREBOARD — KV-CACHED (100% model.forward)",
          file=fh)
    print("=" * 72, file=fh)
    print(f"  coverage: {coverage}", file=fh)
    print(f"  step cap: {step_cap}   wall: {wall:.1f}s", file=fh)
    if guard:
        n_guard = sum(1 for r in results if r.guard_clean is True)
        n_leak = sum(1 for r in results if r.guard_clean is False)
        print(f"  purity guard (assert_no_python_compute): "
              f"{n_guard} clean, {n_leak} leaked", file=fh)
    print(f"  KV-cache: max_seq_len={agg.get('max_seq_len')} "
          f"max_cache_size(block0)={agg.get('max_cache_size')} "
          f"total_evicted={agg.get('total_evicted')}", file=fh)
    if speedup is not None:
        print(f"  speedup vs naive (sampled): {speedup}", file=fh)
    print("-" * 72, file=fh)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}", file=fh)
    print("-" * 72, file=fh)
    print(f"  SCORE:  {n_pass}/{total}  ({100.0 * n_pass / total:.2f}%)  "
          f"[KV-cached pure-forward VM, 32-bit]", file=fh)
    print("=" * 72, file=fh)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--per-cluster", type=int, default=None)
    ap.add_argument("--step-cap", type=int, default=10000)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--no-divmod", action="store_true")
    ap.add_argument("--no-evict", action="store_true",
                    help="Disable bounded eviction (cache grows; still cached).")
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--speedup-sample", type=int, default=0,
                    help="Run N programs ALSO through the naive driver to report a "
                         "cached-vs-naive speedup ratio. Default 0 (skip).")
    ap.add_argument("--verify", action="store_true",
                    help="Assert cached==naive byte-identity on the speedup sample.")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--print-nonpass", action="store_true")
    ap.add_argument("--progress", type=int, default=25)
    args = ap.parse_args(argv)

    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    all_tests = generate_test_programs()

    indexed = list(enumerate(all_tests))[args.offset:]
    if args.limit is not None:
        indexed = indexed[:args.limit]
    if args.per_cluster is not None:
        seen: Counter = Counter()
        sampled = []
        for idx, tp in indexed:
            cl = cluster_of(tp[2])
            if seen[cl] < args.per_cluster:
                seen[cl] += 1
                sampled.append((idx, tp))
        indexed = sampled

    # reference step counts (HARNESS sizing; not model compute).
    ref_steps_by_idx: Dict[int, Optional[int]] = {}
    for idx, (source, _exp, _desc) in indexed:
        try:
            bc, _d = compile_c(source)
            tr = ref_interpret(bytecode_to_isa(bc), max_steps=200000, mask=0xFFFFFFFF)
            ref_steps_by_idx[idx] = len(tr)
        except Exception:  # noqa: BLE001
            ref_steps_by_idx[idx] = None

    coverage = (f"{len(indexed)}/{len(all_tests)} run (deep loops INCLUDED)"
                + (f" (stratified: <= {args.per_cluster}/cluster)"
                   if args.per_cluster is not None else ""))

    include_divmod = not args.no_divmod
    evict = not args.no_evict
    t_build = time.monotonic()
    print(f"[pf-1096-cached] building pure-forward model (code_size={args.code_size}, "
          f"include_divmod={include_divmod}, include_bitwise=False) ...",
          file=sys.stderr, flush=True)
    model, L = build_pure_forward_complete_model(
        code_size=args.code_size, include_bitwise=False, include_divmod=include_divmod)
    print(f"[pf-1096-cached] model: dim={L.D} blocks={len(model.blocks)} "
          f"heads={model.blocks[0].attn.n_heads} ({time.monotonic()-t_build:.1f}s)",
          file=sys.stderr, flush=True)
    print(f"[pf-1096-cached] scoring {coverage} | step_cap={args.step_cap} | "
          f"evict={'ON' if evict else 'off'} (interval={args.prune_interval}) | "
          f"guard={'ON' if args.guard else 'off'}", file=sys.stderr, flush=True)

    # optional cached-vs-naive speedup + byte-identity on a small sample.
    speedup = None
    if args.speedup_sample > 0:
        sample = indexed[:args.speedup_sample]
        t_c = t_n = 0.0
        n_ident = 0
        for idx, (source, _exp, _desc) in sample:
            try:
                code = bytecode_to_isa(compile_c(source)[0])
            except Exception:  # noqa: BLE001
                continue
            cap = min(args.step_cap, (ref_steps_by_idx.get(idx) or 0) + 6)
            t0 = time.monotonic()
            tc = run_pure_forward_cached(model, L, code, max_steps=cap,
                                         mask=0xFFFFFFFF, evict=evict,
                                         prune_interval=args.prune_interval)
            t_c += time.monotonic() - t0
            t0 = time.monotonic()
            tn = run_pure_forward_complete(model, L, code, max_steps=cap,
                                           mask=0xFFFFFFFF)
            t_n += time.monotonic() - t0
            if tc == tn:
                n_ident += 1
            elif args.verify:
                print(f"[VERIFY] byte-identity MISMATCH idx={idx}: "
                      f"cached={tc[-3:]} naive={tn[-3:]}", file=sys.stderr, flush=True)
        speedup = (f"{t_n/t_c:.1f}x (naive {t_n:.0f}s vs cached {t_c:.0f}s over "
                   f"{len(sample)} progs; byte-identical {n_ident}/{len(sample)})")
        print(f"[pf-1096-cached] {speedup}", file=sys.stderr, flush=True)

    t0 = time.monotonic()
    results: List[Result] = []
    agg: dict = {}
    for i, (idx, (source, expected, description)) in enumerate(indexed):
        r = score_program(idx, source, expected, description,
                          model=model, L=L, compile_c=compile_c,
                          step_cap=args.step_cap, guard=args.guard,
                          ref_steps=ref_steps_by_idx.get(idx),
                          evict=evict, prune_interval=args.prune_interval, agg=agg)
        results.append(r)
        if args.progress and (i + 1) % args.progress == 0:
            npass = sum(1 for x in results if x.status == "PASS")
            print(f"[pf-1096-cached] {i + 1}/{len(indexed)} done "
                  f"(pass so far: {npass}) [{time.monotonic()-t0:.0f}s] "
                  f"cache~{agg.get('max_cache_size')}", file=sys.stderr, flush=True)
    wall = time.monotonic() - t0

    results.sort(key=lambda r: r.idx)
    _print_summary(results, wall, args.step_cap, coverage, args.guard, agg, speedup)
    table = _cluster_table(results)
    _print_cluster_table(table)

    if args.print_nonpass:
        print("\nNON-PASS PROGRAMS", file=sys.stdout)
        for r in results:
            if r.status != "PASS":
                print(f"  id={r.idx:04d} [{r.status:7s}] {r.cluster:16s} "
                      f"exp={r.expected} got={r.got_exit} {r.detail}  "
                      f"({r.description})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "wall_seconds": wall,
                "coverage": coverage,
                "step_cap": args.step_cap,
                "include_divmod": include_divmod,
                "evict": evict,
                "prune_interval": args.prune_interval,
                "guard": args.guard,
                "kv_cache": agg,
                "speedup": speedup,
                "summary": {s: sum(1 for r in results if r.status == s)
                            for s in _STATUSES} | {"total": len(results)},
                "clusters": {k: v for k, v in table.items()},
                "results": [asdict(r) for r in results],
            }, fh, indent=2)
        print(f"[pf-1096-cached] wrote {args.output}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
