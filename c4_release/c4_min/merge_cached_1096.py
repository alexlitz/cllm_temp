#!/usr/bin/env python3
"""Merge the KV-cached full-1096 scoreboard shards into ONE scoreboard.

The full 1096 is scored in shards so it fits the wall-time / memory budget:
  * LEAN halves (``--no-divmod``) cover every NON-divmod cluster, and
  * divmod-model runs cover the div/mod/expr_mul_div/expr_mod/edge-div|mod
    clusters (those need the 32-bit long-division blocks).
For a program present in BOTH a LEAN shard and a divmod shard, the DIVMOD result
wins (it is the capable model for that cluster).  Deep loops are INCLUDED; a
program exceeding the step cap is a TIMEOUT (transparently reported, no silent
truncation).

Usage:
    python c4_min/merge_cached_1096.py shard1.json shard2.json ... [--divmod-first f.json ...]
The ``--divmod-first`` shards take precedence per-idx over the plain shards.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict

_STATUSES = ("PASS", "FAIL", "TIMEOUT", "ERROR", "DEEP")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="plain (LEAN) shard JSONs")
    ap.add_argument("--divmod", action="append", default=[],
                    help="divmod-model shard JSON (wins per-idx over plain shards)")
    ap.add_argument("--total", type=int, default=1096)
    args = ap.parse_args(argv)

    by_idx = {}          # idx -> result dict
    provenance = {}      # idx -> 'lean' | 'divmod'
    wall = 0.0
    cache = {"max_seq_len": 0, "max_cache_size": 0, "total_evicted": 0}

    def ingest(path, tag, override):
        nonlocal wall
        d = _load(path)
        wall_local = d.get("wall_seconds", 0.0)
        wall += wall_local
        kv = d.get("kv_cache", {})
        cache["max_seq_len"] = max(cache["max_seq_len"], kv.get("max_seq_len", 0) or 0)
        cache["max_cache_size"] = max(cache["max_cache_size"],
                                      kv.get("max_cache_size", 0) or 0)
        cache["total_evicted"] += kv.get("total_evicted", 0) or 0
        for r in d["results"]:
            i = r["idx"]
            if override or i not in by_idx:
                by_idx[i] = r
                provenance[i] = tag

    for p in args.shards:
        ingest(p, "lean", override=False)
    for p in args.divmod:
        ingest(p, "divmod", override=True)   # divmod wins

    results = [by_idx[i] for i in sorted(by_idx)]
    counts = Counter(r["status"] for r in results)
    total_scored = len(results)
    n_pass = counts.get("PASS", 0)

    # per-cluster table
    table = OrderedDict()
    for r in results:
        row = table.setdefault(r["cluster"], {"n": 0, **{s: 0 for s in _STATUSES}})
        row["n"] += 1
        row[r["status"]] += 1

    print("=" * 78)
    print("c4_min PURE-FORWARD VM — FULL 1096 KV-CACHED SCOREBOARD (merged shards)")
    print("=" * 78)
    print(f"  scored: {total_scored}/{args.total}  "
          f"(missing {args.total - total_scored})")
    print(f"  aggregate shard wall: {wall:.0f}s ({wall/60:.0f} min)")
    print(f"  KV-cache: max_seq_len={cache['max_seq_len']} "
          f"max_cache_size(block0, flat)={cache['max_cache_size']} "
          f"total_evicted={cache['total_evicted']}")
    print("-" * 78)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}")
    print("-" * 78)
    denom = total_scored if total_scored else 1
    print(f"  SCORE (of scored):    {n_pass}/{total_scored}  "
          f"({100.0 * n_pass / denom:.2f}%)")
    print(f"  SCORE (of full 1096): {n_pass}/{args.total}  "
          f"({100.0 * n_pass / args.total:.2f}%)")
    print("=" * 78)

    print("\nPER-CLUSTER BREAKDOWN")
    hdr = (f"  {'cluster':18s} {'n':>4s} {'PASS':>5s} {'FAIL':>5s} "
           f"{'TMOUT':>6s} {'ERR':>4s} {'src':>7s} {'pass%':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    # find a representative provenance per cluster
    cl_src = {}
    for r in results:
        cl_src.setdefault(r["cluster"], provenance[r["idx"]])
    for cluster, row in sorted(table.items()):
        n = row["n"]
        pct = (100.0 * row["PASS"] / n) if n else 0.0
        print(f"  {cluster:18s} {n:4d} {row['PASS']:5d} {row['FAIL']:5d} "
              f"{row['TIMEOUT']:6d} {row['ERROR']:4d} {cl_src.get(cluster,''):>7s} "
              f"{pct:6.1f}")

    # dump merged
    out = {
        "scored": total_scored, "total": args.total,
        "summary": {s: counts.get(s, 0) for s in _STATUSES} | {"pass": n_pass},
        "kv_cache": cache,
        "clusters": {k: v for k, v in table.items()},
    }
    with open("/tmp/cached_full1096_merged.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[merge] wrote /tmp/cached_full1096_merged.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
