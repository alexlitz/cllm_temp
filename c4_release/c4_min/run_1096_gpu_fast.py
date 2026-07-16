#!/usr/bin/env python3
"""c4_min PURE-FORWARD VM — FULL 1096 SCOREBOARD, GPU-BATCHED + STEP-BUCKETED.

The FAST corpus runner on the NOW-SMALL model.  The single-program CPU cached
driver runs ONE program per VM step; this runner:

  1. Builds the SMALL model once per device: ``compact_alloc`` (never
     materialises the padded FFN) + ``sparse_forward.SparseTransformer``
     (dense_kernel -> L-inf=0).  Loads in ~0.034 GB VRAM.
  2. **Batches across independent programs** — stacks ``B`` programs' 31-row
     windows into one ``[B, W, D]`` batched forward and steps them together
     (:func:`nibble_pure_forward_gpu.run_batch_gpu`), masking/terminating each
     slot as its program HALTs.
  3. **STEP-COUNT BUCKETS the corpus** — the logical VM gives each program's
     EXACT step count for free (``ref_interpret``, no forwards), so programs are
     grouped into HOMOGENEOUS-DEPTH batches: a batch runs ~its members' step
     count with NO idle slots (a shallow bucket clears in a few steps; a deep
     bucket runs together).  Without this a ragged batch would idle every shallow
     program for thousands of steps waiting on the one deep program in it.
  4. Uses BOTH GPUs data-parallel (round-robin shard across ``cuda:0`` /
     ``cuda:1``, one worker process per device).

The GPU byte trace of each program is byte-identical to the CPU cached driver's
(same per-step arithmetic; only the block-stack eval is batched + device-resident),
up to the documented saturated-tie fp handful.

DEEP LOOPS: the deep-loop clusters have a known KV-memory divergence being fixed
separately (chk1-memfix-deeploop) — they still diverge here (same divergence as
the CPU driver).  They are run (bounded by ``--step-cap`` + ref_steps headroom)
and reported, not blocked on.  ``--max-ref-steps`` reports programs above a step
threshold as DEEP (not run) to bound the deep-tail wall-time.

Usage
-----
    OMP_NUM_THREADS=4 python c4_min/run_1096_gpu_fast.py \\
        --devices cuda:0,cuda:1 --output /tmp/gpu_fast_1096.json
    # single GPU / non-divmod (lean) fast path:
    ... --devices cuda:0 --no-divmod --clusters add,sub,if_gt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


_STATUSES = ("PASS", "FAIL", "TIMEOUT", "ERROR", "DEEP")


# ---------------------------------------------------------------------------
# Step-count BUCKETING.  Groups programs of similar ref_steps so a batch runs
# ~its members' depth with no idle slots.  Bucket edges grow geometrically (a
# 5-step add and a 5000-step loop should NOT share a batch), and each bucket is
# further split into sub-batches of at most ``batch_cap`` programs.
# ---------------------------------------------------------------------------
def bucket_by_steps(items: List[dict], batch_cap: int,
                    bytes_per_prog_row: float = 0.0, vram_budget_gb: float = 12.0,
                    frame_len: int = 30, edges: Optional[List[int]] = None
                    ) -> List[List[dict]]:
    """Return a list of batches (each a list of item dicts) STEP-COUNT-BUCKETED.

    Programs are first partitioned into depth buckets by ``ref_steps`` (geometric
    edges), then each bucket is chunked into sub-batches.  The sub-batch size is
    the KEY scheduling win: the batched-forward VRAM peak is dominated by the
    per-program KV cache ``B * n_blocks * cache_rows * H * HD * 4 * 2``, whose
    ``cache_rows ~ frame_len * max_depth`` grows with the DEEPEST program in the
    batch.  So a SHALLOW bucket (cache tiny) runs at B up to ``batch_cap``, while a
    DEEP bucket auto-shrinks to keep the predicted cache under ``vram_budget_gb``
    — a batch runs ~its members' depth with NO idle slots AND no OOM.

    ``bytes_per_prog_row`` = ``n_blocks * H * HD * 4 * 2`` (K+V, fp32, all blocks);
    when 0 the budget-sizing is skipped and only ``batch_cap`` applies."""
    if edges is None:
        # geometric-ish depth bands: shallow programs dominate the corpus, so the
        # low bands are fine-grained; deep loops get their own coarse bands.
        edges = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    buckets: "OrderedDict[int, List[dict]]" = OrderedDict()
    for it in items:
        rs = it.get("ref_steps")
        key = len(edges)  # overflow band for None / very deep
        if rs is not None:
            for bi, e in enumerate(edges):
                if rs <= e:
                    key = bi
                    break
        buckets.setdefault(key, []).append(it)
    batches: List[List[dict]] = []
    for key in sorted(buckets):
        band = sorted(buckets[key], key=lambda r: (r.get("ref_steps") or 0))
        # depth-adaptive sub-batch size for THIS band (uses the band's MAX depth).
        max_depth = max((r.get("ref_steps") or 1) for r in band)
        cache_rows = frame_len * (max_depth + 2)          # +2 query-row slack
        if bytes_per_prog_row > 0:
            per_prog_gb = cache_rows * bytes_per_prog_row / 1e9
            b_budget = max(1, int(vram_budget_gb / max(per_prog_gb, 1e-9)))
        else:
            b_budget = batch_cap
        sub = max(1, min(batch_cap, b_budget))
        for s in range(0, len(band), sub):
            batches.append(band[s:s + sub])
    return batches


# ---------------------------------------------------------------------------
# One worker: builds the model on its device, buckets + batches its shard.
# ---------------------------------------------------------------------------
def _worker(device: str, shard: List[dict], batch_cap: int, code_size: int,
            include_divmod: bool, include_bitwise: bool, compute_mode: str,
            step_cap: int, evict: bool, prune_interval: int,
            edges: Optional[List[int]], out_q, verbose: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = (device.split(":")[-1]
                                          if device.startswith("cuda") else "")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    from c4_min.nibble_pure_forward_complete import build_pure_forward_complete_model
    from c4_min.compact_alloc import build_compact_pure_forward_model
    from c4_min.sparse_forward import SparseTransformer
    from c4_min.nibble_pure_forward_gpu import run_batch_gpu
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from src.compiler import compile_c

    dev = "cuda:0" if device.startswith("cuda") else "cpu"
    t_build = time.monotonic()
    if include_divmod:
        base, L, _cs = build_compact_pure_forward_model(
            code_size=code_size, include_bitwise=include_bitwise, include_divmod=True)
    else:
        base, L = build_pure_forward_complete_model(
            code_size=code_size, include_bitwise=include_bitwise, include_divmod=False)
    sparse = SparseTransformer(base, compute_mode=compute_mode)
    st = sparse.stats()
    del base
    sparse = sparse.to(dev)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        vram_load = torch.cuda.memory_allocated() / 1e9
    else:
        vram_load = 0.0
    build_dt = time.monotonic() - t_build
    if verbose:
        print(f"[gpu-fast {device}] model built {build_dt:.1f}s | "
              f"blocks={len(sparse.blocks)} heads={sparse.blocks[0].attn.n_heads} "
              f"storage={st.sparse_mb:.1f}MB VRAM-load={vram_load*1000:.0f}MB",
              file=sys.stderr, flush=True)

    # compile + translate each program (harness-side; not model compute).
    prepared = []
    for item in shard:
        try:
            code = bytecode_to_isa(compile_c(item["source"])[0])
            prepared.append({**item, "code": code})
        except Exception as exc:  # noqa: BLE001
            out_q.put({"idx": item["idx"], "description": item["description"],
                       "expected": item["expected"] & 0xFFFFFFFF, "status": "ERROR",
                       "got_exit": None, "got_steps": None,
                       "detail": f"compile/translate: {exc!r}"})

    # STEP-COUNT BUCKET the shard, then run each batch.  The per-batch size is
    # depth-adaptive: the KV-cache VRAM peak is B * n_blocks * cache_rows * H * HD *
    # 4 * 2, so we hand the bucketer the model's per-(program,row) cache bytes and a
    # VRAM budget — shallow buckets run wide, deep buckets auto-shrink.
    import c4_min.blogspec_vocab as _V
    nblk = len(sparse.blocks)
    Hh = sparse.blocks[0].attn.n_heads
    HDh = sparse.blocks[0].attn.head_dim
    bytes_per_prog_row = nblk * Hh * HDh * 4 * 2   # K+V, fp32, all blocks
    vram_budget = float(os.environ.get("C4_GPU_VRAM_BUDGET_GB", "11.0"))
    batches = bucket_by_steps(prepared, batch_cap,
                              bytes_per_prog_row=bytes_per_prog_row,
                              vram_budget_gb=vram_budget, frame_len=_V.FRAME_LEN,
                              edges=edges)
    max_vram = 0.0
    n_oom = 0
    n_done = 0
    t0 = time.monotonic()
    bi = 0
    while bi < len(batches):
        batch = batches[bi]
        codes = [it["code"] for it in batch]
        caps = [it["step_cap"] for it in batch]
        stats: dict = {}
        try:
            traces = run_batch_gpu(sparse, L, codes, caps, device=dev,
                                   mask=0xFFFFFFFF, evict=evict,
                                   prune_interval=prune_interval, stats=stats)
            if dev.startswith("cuda"):
                peak = torch.cuda.max_memory_allocated() / 1e9
                max_vram = max(max_vram, peak)
                torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_oom += 1
            if len(batch) == 1:
                for it in batch:
                    out_q.put({"idx": it["idx"], "description": it["description"],
                               "expected": it["expected"] & 0xFFFFFFFF,
                               "status": "ERROR", "got_exit": None,
                               "got_steps": None, "detail": "OOM at batch=1"})
                bi += 1
                continue
            # halve this batch and retry (log; no silent drop).
            half = (len(batch) + 1) // 2
            batches[bi:bi + 1] = [batch[:half], batch[half:]]
            print(f"[gpu-fast {device}] OOM on batch of {len(batch)} "
                  f"-> split to {half}+{len(batch) - half}, retrying",
                  file=sys.stderr, flush=True)
            continue
        for it, tr in zip(batch, traces):
            _emit_result(it, tr, out_q)
            n_done += 1
        bi += 1
        if verbose and (n_done % 100 < len(batch)):
            print(f"[gpu-fast {device}] {n_done}/{len(prepared)} "
                  f"[{time.monotonic() - t0:.0f}s] vram~{max_vram*1000:.0f}MB "
                  f"batch={len(batch)} fwds={stats.get('n_forwards')} "
                  f"maxcache={stats.get('max_cache_size')} ooms={n_oom}",
                  file=sys.stderr, flush=True)
    out_q.put({"__meta__": device, "build_dt": build_dt, "vram_load_gb": vram_load,
               "storage_mb": st.sparse_mb, "max_vram_gb": max_vram,
               "wall": time.monotonic() - t0, "n": n_done, "ooms": n_oom,
               "n_batches": len(batches)})


def _emit_result(item: dict, trace: List[int], out_q) -> None:
    base = {"idx": item["idx"], "description": item["description"],
            "expected": item["expected"] & 0xFFFFFFFF}
    exp = item["expected"] & 0xFFFFFFFF
    cap = item["step_cap"]
    if not trace:
        out_q.put({**base, "status": "ERROR", "got_exit": None, "got_steps": 0,
                   "detail": "no frame emitted"})
        return
    got = int(trace[-1]) & 0xFFFFFFFF
    steps = len(trace)
    if steps >= cap:
        out_q.put({**base, "status": "TIMEOUT", "got_exit": got, "got_steps": steps,
                   "detail": f"no HALT within {cap} steps"})
    elif got == exp:
        out_q.put({**base, "status": "PASS", "got_exit": got, "got_steps": steps,
                   "detail": ""})
    else:
        out_q.put({**base, "status": "FAIL", "got_exit": got, "got_steps": steps,
                   "detail": f"exit mismatch: exp {exp} got {got}"})


def _cluster_table_from_rows(rows):
    table: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for r in rows:
        row = table.setdefault(r.get("cluster", "misc"),
                               {"n": 0, **{s: 0 for s in _STATUSES}})
        row["n"] += 1
        row[r["status"]] += 1
    return table


def _report(rows, wall, args, meta, devices, evict, n_run, n_deep, n_total):
    counts = Counter(r["status"] for r in rows)
    total = len(rows)
    n_pass = counts.get("PASS", 0)
    print("\n" + "=" * 74)
    print("c4_min PURE-FORWARD VM SCOREBOARD — GPU-BATCHED + STEP-BUCKETED")
    print("=" * 74)
    print(f"  devices: {devices}   batch-cap: {args.batch}   "
          f"model: {'DIVMOD' if not args.no_divmod else 'LEAN'} (sparse)")
    print(f"  coverage: {n_run} run / {n_deep} DEEP (not run) / {n_total} corpus")
    print(f"  eviction: {'ON' if evict else 'OFF (byte-identical to naive)'}")
    print(f"  wall: {wall:.1f}s ({wall/60:.1f} min)")
    for m in meta:
        print(f"    [{m['__meta__']}] build={m['build_dt']:.1f}s "
              f"storage={m['storage_mb']:.1f}MB vram_load={m['vram_load_gb']*1000:.0f}MB "
              f"peak_vram={m['max_vram_gb']:.2f}GB shard_wall={m['wall']:.1f}s "
              f"n={m['n']} batches={m.get('n_batches')} ooms={m.get('ooms', 0)}")
    print("-" * 74)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}")
    print("-" * 74)
    print(f"  SCORE:  {n_pass}/{total}  ({100.0 * n_pass / max(total,1):.2f}%)  "
          f"[GPU-batched pure-forward VM, 32-bit]")
    print("=" * 74)
    table = _cluster_table_from_rows(rows)
    print("\nPER-CLUSTER BREAKDOWN")
    hdr = (f"  {'cluster':18s} {'n':>4s} {'PASS':>5s} {'FAIL':>5s} "
           f"{'TMOUT':>6s} {'ERR':>4s} {'DEEP':>5s} {'pass%':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cl, row in sorted(table.items()):
        n = row["n"]
        pct = (100.0 * row["PASS"] / n) if n else 0.0
        print(f"  {cl:18s} {n:4d} {row['PASS']:5d} {row['FAIL']:5d} "
              f"{row['TIMEOUT']:6d} {row['ERROR']:4d} {row['DEEP']:5d} {pct:6.1f}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", type=str, default=None,
                    help="Comma list, e.g. 'cuda:0,cuda:1' (default: all CUDA).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--per-cluster", type=int, default=None)
    ap.add_argument("--clusters", type=str, default=None,
                    help="Comma-separated cluster names to include.")
    ap.add_argument("--exclude-clusters", type=str, default=None)
    ap.add_argument("--step-cap", type=int, default=10000)
    ap.add_argument("--max-ref-steps", type=int, default=None,
                    help="Report programs whose ref step count exceeds this as DEEP "
                         "(not run); bounds deep-loop wall-time.")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--no-divmod", action="store_true",
                    help="LEAN model (no div/mod). Default: compact DIVMOD (superset).")
    ap.add_argument("--no-bitwise", action="store_true")
    ap.add_argument("--compute-mode", type=str, default="dense_kernel",
                    choices=["dense_kernel", "sparse_mm"])
    ap.add_argument("--batch", type=int, default=256,
                    help="Per-batch program cap (each depth bucket is chunked to this).")
    ap.add_argument("--evict", action="store_true",
                    help="Enable bounded eviction (deep loops); diverges from naive.")
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--print-nonpass", action="store_true")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args(argv)

    import torch
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        n = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n)] if n else ["cpu"]
    print(f"[gpu-fast] devices: {devices}", file=sys.stderr, flush=True)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min.run_1096_pure_forward import bytecode_to_isa, cluster_of
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
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
        sampled = []
        for idx, tp in indexed:
            cl = cluster_of(tp[2])
            if seen[cl] < args.per_cluster:
                seen[cl] += 1
                sampled.append((idx, tp))
        indexed = sampled

    # reference step counts (harness sizing; NOT model compute) + DEEP split.
    print(f"[gpu-fast] sizing {len(indexed)} programs (ref_interpret) ...",
          file=sys.stderr, flush=True)
    items: List[dict] = []
    deep: List[dict] = []
    for idx, (source, expected, description) in indexed:
        try:
            bc, _d = compile_c(source)
            rs = len(ref_interpret(bytecode_to_isa(bc), max_steps=200000,
                                   mask=0xFFFFFFFF))
        except Exception:  # noqa: BLE001
            rs = None
        cap = args.step_cap if rs is None else min(args.step_cap, rs + 6)
        rec = dict(idx=idx, source=source, expected=expected,
                   description=description, cluster=cluster_of(description),
                   step_cap=cap, ref_steps=rs)
        if (args.max_ref_steps is not None and rs is not None
                and rs > args.max_ref_steps):
            deep.append(rec)
        else:
            items.append(rec)

    # split across devices data-parallel (round-robin on the STEP-SORTED corpus so
    # each device gets a balanced depth mix — else one device eats every deep loop).
    items.sort(key=lambda r: (r["ref_steps"] or 0))
    shards: List[List[dict]] = [[] for _ in devices]
    for k, rec in enumerate(items):
        shards[k % len(devices)].append(rec)

    include_bitwise = not args.no_bitwise
    include_divmod = not args.no_divmod
    edges = None
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    t0 = time.monotonic()
    for dev, shard in zip(devices, shards):
        if not shard:
            continue
        p = ctx.Process(target=_worker, args=(
            dev, shard, args.batch, args.code_size, include_divmod,
            include_bitwise, args.compute_mode, args.step_cap, args.evict,
            args.prune_interval, edges, out_q, args.verbose))
        p.start()
        procs.append(p)

    results: List[dict] = []
    meta: List[dict] = []
    expected_n = sum(len(s) for s in shards)
    n_workers = sum(1 for s in shards if s)
    got = 0
    # Drain until every result AND every worker's __meta__ (one per worker) has
    # arrived — a worker sends its meta AFTER its last result, so a naive
    # "stop at expected_n results" loop can exit before the final meta is read.
    while got < expected_n or len(meta) < n_workers:
        r = out_q.get()
        if "__meta__" in r:
            meta.append(r)
            continue
        results.append(r)
        got += 1
    for p in procs:
        p.join()
    wall = time.monotonic() - t0

    by_idx = {r["idx"]: r for r in results}
    rows = []
    for rec in items:
        r = by_idx.get(rec["idx"], {"idx": rec["idx"],
                                    "description": rec["description"],
                                    "expected": rec["expected"] & 0xFFFFFFFF,
                                    "status": "ERROR", "got_exit": None,
                                    "got_steps": None, "detail": "lost"})
        r["cluster"] = rec["cluster"]
        r["ref_steps"] = rec["ref_steps"]
        rows.append(r)
    for rec in deep:
        rows.append({"idx": rec["idx"], "description": rec["description"],
                     "expected": rec["expected"] & 0xFFFFFFFF, "cluster": rec["cluster"],
                     "status": "DEEP", "got_exit": None, "got_steps": None,
                     "ref_steps": rec["ref_steps"],
                     "detail": f"ref_steps={rec['ref_steps']} > {args.max_ref_steps}"})
    rows.sort(key=lambda r: r["idx"])

    evict = args.evict
    _report(rows, wall, args, meta, devices, evict, expected_n, len(deep),
            len(all_tests))

    if args.output:
        table = _cluster_table_from_rows(rows)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "wall_seconds": wall, "devices": devices, "batch": args.batch,
                "divmod": include_divmod, "bitwise": include_bitwise,
                "compute_mode": args.compute_mode, "evict": evict,
                "step_cap": args.step_cap, "max_ref_steps": args.max_ref_steps,
                "meta": meta,
                "summary": {s: sum(1 for r in rows if r["status"] == s)
                            for s in _STATUSES} | {"total": len(rows)},
                "clusters": table, "results": rows,
            }, fh, indent=2)
        print(f"[gpu-fast] wrote {args.output}", file=sys.stderr, flush=True)

    if args.print_nonpass:
        for r in rows:
            if r["status"] != "PASS":
                print(f"  id={r['idx']:04d} [{r['status']:7s}] "
                      f"{r.get('cluster',''):16s} exp={r['expected']} "
                      f"got={r['got_exit']} rs={r.get('ref_steps')} "
                      f"{r.get('detail','')} ({r['description']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
