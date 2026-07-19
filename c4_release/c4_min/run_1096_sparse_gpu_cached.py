#!/usr/bin/env python3
"""CHK-1 headline runner — the SPARSE pure-forward whole-VM model, KV-cached, on GPU.

Composes the two proven CHK-1 wins into one full-1096 scoreboard:

  * **small/sparse weights** — the c4_min pure-forward VM is ~99.998 % sparse, so
    ``sparse_forward.SparseTransformer`` stores it as a few MB of CSR and (in
    ``dense_kernel`` mode) runs the SAME ``F.linear`` GEMMs -> **L-inf=0**
    byte-identical to the dense forward.  The SINGLE full-op-set model (every
    opcode incl. DIV/MOD) is built via the STREAMING sparse builder
    (``compact_alloc.build_compact_sparse_streaming``), which NEVER materialises
    the ~130 GB full-op dense / ~48 GB dense-compact intermediate: peak build RSS
    is ~one dense block, and the result fits a 24 GB card in a few MB.
  * **O(cache)/step** — every VM step is driven by
    ``nibble_pure_forward_cached.run_pure_forward_cached``: a fixed 31-row window
    against a per-block incremental KV cache with bounded (softmax1+ALiBi)
    eviction, so deep loops are BOTH fast AND memory-bounded.  The cached output
    is byte-identical to the naive whole-VM re-forward (``test_cached_driver``),
    with ``SP_INIT`` pinned to ``0xF0`` (as the reference runner does).

The SPARSE model's per-program PASS/FAIL is the authoritative pure-forward VM
result.  ``--load-sparse`` reloads a pre-saved streamed sparse model with no
rebuild.

Shard the corpus across the two idle A5000s (one worker per GPU):
    OMP_NUM_THREADS=4 python c4_min/run_1096_sparse_gpu_cached.py \
        --device cuda:0 --shard-of 2 --shard-idx 0 --output /tmp/pf_shard0.json
    OMP_NUM_THREADS=4 python c4_min/run_1096_sparse_gpu_cached.py \
        --device cuda:1 --shard-of 2 --shard-idx 1 --output /tmp/pf_shard1.json
then merge the shard JSONs (see ``--merge``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import asdict
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402

# Pin SP_INIT to 0xF0 for BOTH the reference interpreter and the model driver so
# the cached-window overlay matches the naive whole-VM forward (else SP would be
# the un-masked 32-bit 0x10000 image and the frame arithmetic diverges).
import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.nibble_pure_forward_complete import ref_interpret  # noqa: E402
from c4_min.sparse_forward import SparseTransformer  # noqa: E402
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached  # noqa: E402

from c4_min.run_1096_pure_forward import (  # noqa: E402
    bytecode_to_isa, cluster_of, Result, _STATUSES,
    _cluster_table, _print_cluster_table,
)


# ---------------------------------------------------------------------------
def _to_device_model(sp: SparseTransformer, device: str) -> SparseTransformer:
    return sp.to(device)


def _run_one(model, L, code, cap, evict, prune_interval, device, stats):
    """Drive one program through the KV-cached driver on ``device``."""
    tr = run_pure_forward_cached(
        model, L, code, max_steps=cap, mask=0xFFFFFFFF,
        evict=evict, prune_interval=prune_interval, stats=stats)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    return tr


def score_program(idx, source, expected, description, *, model, L, compile_c,
                  step_cap, ref_steps, evict, prune_interval, device,
                  compact=None, Lc=None, agg=None) -> Result:
    cluster = cluster_of(description)
    exp = expected & 0xFFFFFFFF
    base = dict(idx=idx, description=description, cluster=cluster, expected=exp)
    try:
        code = bytecode_to_isa(compile_c(source)[0])
    except Exception as exc:  # noqa: BLE001
        return Result(got_exit=None, got_steps=None, status="ERROR",
                      detail=f"compile/translate: {exc!r}", **base)
    n_instrs = len(code)
    cap = step_cap if ref_steps is None else min(step_cap, ref_steps + 6)
    stats: dict = {}
    try:
        tr = _run_one(model, L, code, cap, evict, prune_interval, device, stats)
    except Exception as exc:  # noqa: BLE001
        return Result(got_exit=None, got_steps=None, status="ERROR",
                      n_instrs=n_instrs, detail=f"run: {exc!r}", **base)

    if agg is not None and stats:
        agg["max_seq_len"] = max(agg.get("max_seq_len", 0), stats.get("max_seq_len", 0))
        agg["max_cache_size"] = max(agg.get("max_cache_size", 0),
                                    stats.get("max_cache_size", 0))
        agg["total_evicted"] = agg.get("total_evicted", 0) + stats.get("total_evicted", 0)
        agg["total_steps"] = agg.get("total_steps", 0) + len(tr)

    if not tr:
        return Result(got_exit=None, got_steps=0, status="ERROR",
                      n_instrs=n_instrs, detail="no frame emitted", **base)

    got = int(tr[-1]) & 0xFFFFFFFF
    steps = len(tr)

    # OPTIONAL empirical-liveness backstop: run the compact model on the SAME
    # program (same cached driver) and record if it disagrees with the sparse
    # per-program result. compact != sparse => an unsound dim-share.
    compact_note = ""
    if compact is not None:
        try:
            tc = _run_one(compact, Lc, code, cap, evict, prune_interval,
                          device, {})
            got_c = int(tc[-1]) & 0xFFFFFFFF if tc else None
            if tc != tr:
                compact_note = (f" [COMPACT_DIFFERS trace_ident=False "
                                f"sparse_final={got} compact_final={got_c}]")
        except Exception as exc:  # noqa: BLE001
            compact_note = f" [COMPACT_ERR {exc!r}]"

    detail = (f"maxseq={stats.get('max_seq_len')} cache={stats.get('max_cache_size')} "
              f"evicted={stats.get('total_evicted')}{compact_note}")
    if steps >= cap:
        return Result(got_exit=got, got_steps=steps, status="TIMEOUT",
                      n_instrs=n_instrs, detail=f"no HALT within {cap} steps; " + detail,
                      **base)
    if got == exp:
        return Result(got_exit=got, got_steps=steps, status="PASS",
                      n_instrs=n_instrs, detail=detail, **base)
    return Result(got_exit=got, got_steps=steps, status="FAIL",
                  n_instrs=n_instrs,
                  detail=f"exit mismatch: exp {exp} got {got}; " + detail, **base)


def _write_output(path, results, agg, device, args, st,
                  vram_load, wall, partial=False, vram_peak=0.0):
    """Serialise the (possibly partial) scoreboard to ``path`` as JSON."""
    counts = Counter(r.status for r in results)
    n_pass = counts.get("PASS", 0)
    total = len(results)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "device": device, "shard_of": args.shard_of,
            "shard_idx": args.shard_idx, "wall_seconds": wall,
            "steps_total": agg.get("total_steps", 0),
            "partial": partial,
            "step_cap": args.step_cap, "op_set": "full",
            "compute_mode": args.compute_mode, "evict": not args.no_evict,
            "vram_load_mb": vram_load, "vram_peak_mb": vram_peak,
            "storage_mb": st.sparse_mb,
            "kv_cache": {k: agg.get(k) for k in
                         ("max_seq_len", "max_cache_size", "total_evicted")},
            "summary": {s: counts.get(s, 0) for s in _STATUSES}
            | {"total": total, "pass": n_pass},
            "clusters": dict(_cluster_table(sorted(results, key=lambda r: r.idx))),
            "results": [asdict(r) for r in sorted(results, key=lambda r: r.idx)],
        }, fh, indent=2)


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--shard-of", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-cluster", type=int, default=None,
                    help="stratified sample <= N/cluster (coverage reported).")
    ap.add_argument("--clusters", type=str, default=None,
                    help="comma-list of clusters to INCLUDE (others skipped).")
    ap.add_argument("--exclude-clusters", type=str, default=None,
                    help="comma-list of clusters to EXCLUDE (e.g. divmod-only ones "
                         "when this shard runs the LEAN model).")
    ap.add_argument("--step-cap", type=int, default=12000)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--compute-mode", type=str, default="dense_kernel",
                    choices=["dense_kernel", "sparse_mm"],
                    help="dense_kernel: L-inf=0. sparse_mm: argmax-identical, faster.")
    ap.add_argument("--load-sparse", type=str, default=None,
                    help="Reload a saved streamed sparse model (save_sparse_transformer "
                         "artifact) INSTEAD of building — no rebuild, no peak.")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--progress", type=int, default=20)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[sparse-gpu] CUDA not available; falling back to cpu", file=sys.stderr)
        device = "cpu"

    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    all_tests = generate_test_programs()

    indexed = list(enumerate(all_tests))
    if args.clusters:
        keep = {c.strip() for c in args.clusters.split(",") if c.strip()}
        indexed = [(i, tp) for i, tp in indexed if cluster_of(tp[2]) in keep]
    if args.exclude_clusters:
        drop = {c.strip() for c in args.exclude_clusters.split(",") if c.strip()}
        indexed = [(i, tp) for i, tp in indexed if cluster_of(tp[2]) not in drop]
    if args.per_cluster is not None:
        seen: Counter = Counter()
        sampled = []
        for idx, tp in indexed:
            cl = cluster_of(tp[2])
            if seen[cl] < args.per_cluster:
                seen[cl] += 1
                sampled.append((idx, tp))
        indexed = sampled
    # deterministic contiguous-stride shard so each GPU worker gets an even mix.
    if args.shard_of > 1:
        indexed = [it for i, it in enumerate(indexed)
                   if i % args.shard_of == args.shard_idx]
    if args.limit is not None:
        indexed = indexed[:args.limit]

    t_build = time.monotonic()

    # The SINGLE full-op-set interpreter (every opcode incl. DIV/MOD + bitwise) is
    # ALWAYS built via a memory-safe path: reload a saved streamed model, or STREAM
    # build directly SPARSE block-at-a-time (peak build RSS ~one dense block, never
    # the ~130 GB full-op dense / ~48 GB dense-compact intermediate).

    # -- FAST PATH: reload a pre-saved streamed sparse model (no rebuild, no peak).
    if args.load_sparse:
        from c4_min.compact_alloc import load_sparse_transformer
        print(f"[sparse-gpu:{device}] reloading sparse model from "
              f"{args.load_sparse} ...", file=sys.stderr, flush=True)
        sparse, L = load_sparse_transformer(args.load_sparse,
                                            compute_mode=args.compute_mode)
        st = sparse.stats()
    else:
        from c4_min.compact_alloc import build_compact_sparse_streaming
        print(f"[sparse-gpu:{device}] STREAM building full-op-set pure-forward "
              f"model ...", file=sys.stderr, flush=True)
        sparse, L, cstats = build_compact_sparse_streaming(
            code_size=args.code_size, compute_mode=args.compute_mode)
        st = sparse.stats()
        print(f"[sparse-gpu:{device}] STREAM compact: dim {cstats.dim_before}->"
              f"{cstats.dim_after} blocks={cstats.n_blocks} "
              f"nnz={cstats.nonzero_params}", file=sys.stderr, flush=True)

    sparse = sparse.to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
        vram_load = torch.cuda.memory_allocated(torch.device(device)) / 1e6
    else:
        vram_load = 0.0
    print(f"[sparse-gpu:{device}] sparse model built in {time.monotonic()-t_build:.0f}s "
          f"| dim={sparse.dim} blocks={len(sparse.blocks)} "
          f"heads={sparse.blocks[0].attn.n_heads} "
          f"| storage {st.sparse_mb:.1f}MB (dense-equiv {st.dense_gb:.1f}GB) nnz={st.total_nnz} "
          f"| VRAM-load {vram_load:.0f}MB | mode={args.compute_mode}",
          file=sys.stderr, flush=True)

    # There is ONE canonical full-op-set model now; no separate compact model to
    # cross-check against (the streaming build IS the compact model).
    compact_check = None
    Lc = None

    coverage = (f"shard {args.shard_idx+1}/{args.shard_of}: "
                f"{len(indexed)}/{len(all_tests)} programs (deep loops INCLUDED, "
                f"full op set incl. divmod)"
                + (f" [<= {args.per_cluster}/cluster]" if args.per_cluster else ""))
    print(f"[sparse-gpu:{device}] scoring {coverage} | step_cap={args.step_cap} | "
          f"evict={'off' if args.no_evict else 'ON'}",
          file=sys.stderr, flush=True)

    # reference step counts (pure-python harness sizing; not model compute).
    ref_steps_by_idx: Dict[int, Optional[int]] = {}
    for idx, (source, _e, _d) in indexed:
        try:
            tr = ref_interpret(bytecode_to_isa(compile_c(source)[0]),
                               max_steps=200000, mask=0xFFFFFFFF)
            ref_steps_by_idx[idx] = len(tr)
        except Exception:  # noqa: BLE001
            ref_steps_by_idx[idx] = None

    evict = not args.no_evict
    t0 = time.monotonic()
    results: List[Result] = []
    agg: dict = {}
    for i, (idx, (source, expected, description)) in enumerate(indexed):
        r = score_program(
            idx, source, expected, description, model=sparse, L=L,
            compile_c=compile_c, step_cap=args.step_cap,
            ref_steps=ref_steps_by_idx.get(idx), evict=evict,
            prune_interval=args.prune_interval, device=device,
            compact=compact_check, Lc=Lc, agg=agg)
        results.append(r)
        if args.progress and (i + 1) % args.progress == 0:
            npass = sum(1 for x in results if x.status == "PASS")
            steps_done = agg.get("total_steps", 0)
            el = time.monotonic() - t0
            print(f"[sparse-gpu:{device}] {i+1}/{len(indexed)} done "
                  f"(pass {npass}) [{el:.0f}s, {steps_done/max(el,1e-6):.1f} st/s] "
                  f"cache~{agg.get('max_cache_size')}", file=sys.stderr, flush=True)
            # Incremental checkpoint: write partial results so a long deep-loop
            # tail never loses the completed portion (JSON is rewritten each time).
            if args.output:
                _write_output(args.output + ".partial", results, agg, device,
                              args, st, vram_load,
                              time.monotonic() - t0, partial=True)
    wall = time.monotonic() - t0

    if device.startswith("cuda"):
        vram_peak = torch.cuda.max_memory_allocated(torch.device(device)) / 1e6
    else:
        vram_peak = 0.0

    results.sort(key=lambda r: r.idx)
    counts = Counter(r.status for r in results)
    n_pass = counts.get("PASS", 0)
    total = len(results)
    steps_total = agg.get("total_steps", 0)

    print("\n" + "=" * 74)
    print(f"c4_min SPARSE PURE-FORWARD VM — KV-CACHED on {device} "
          f"(shard {args.shard_idx+1}/{args.shard_of})")
    print("=" * 74)
    print(f"  coverage: {coverage}")
    print(f"  wall: {wall:.0f}s ({wall/60:.1f} min)  steps: {steps_total}  "
          f"steps/sec: {steps_total/max(wall,1e-6):.1f}")
    print(f"  VRAM: load {vram_load:.0f}MB peak {vram_peak:.0f}MB  "
          f"storage {st.sparse_mb:.1f}MB")
    print(f"  KV-cache: max_seq_len={agg.get('max_seq_len')} "
          f"max_cache(block0)={agg.get('max_cache_size')} "
          f"total_evicted={agg.get('total_evicted')}")
    print("-" * 74)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}")
    print("-" * 74)
    print(f"  SHARD SCORE: {n_pass}/{total} ({100.0*n_pass/max(total,1):.2f}%)")
    print("=" * 74)
    _print_cluster_table(_cluster_table(results))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "device": device, "shard_of": args.shard_of,
                "shard_idx": args.shard_idx, "wall_seconds": wall,
                "steps_total": steps_total, "coverage": coverage,
                "step_cap": args.step_cap, "op_set": "full",
                "compute_mode": args.compute_mode, "evict": evict,
                "vram_load_mb": vram_load, "vram_peak_mb": vram_peak,
                "storage_mb": st.sparse_mb,
                "kv_cache": {k: agg.get(k) for k in
                             ("max_seq_len", "max_cache_size", "total_evicted")},
                "summary": {s: counts.get(s, 0) for s in _STATUSES}
                | {"total": total, "pass": n_pass},
                "clusters": dict(_cluster_table(results)),
                "results": [asdict(r) for r in results],
            }, fh, indent=2)
        print(f"[sparse-gpu:{device}] wrote {args.output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
