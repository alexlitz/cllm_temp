#!/usr/bin/env python3
"""Merge the disjoint SPARSE-GPU shard JSONs into ONE full-1096 CHK-1 scoreboard.

The divmod lane scores the DIV/MOD/gcd clusters (the 32-bit long-division blocks);
the LEAN lane scores every OTHER cluster.  The two shards are DISJOINT by cluster
(each program appears in exactly one), so the merge is a union.  Any program that
appears in both (should be none) prefers the DIVMOD result (the capable model).

Usage:
    python c4_min/merge_sparse_gpu_1096.py --divmod pf_divmod_lane.json \
        --lean pf_lean_lane.json [--total 1096] [--out /tmp/chk1_full1096.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict

_STATUSES = ("PASS", "FAIL", "TIMEOUT", "ERROR", "DEEP")


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--divmod", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--total", type=int, default=1096)
    ap.add_argument("--out", default="/tmp/chk1_full1096.json")
    args = ap.parse_args(argv)

    dm = _load(args.divmod)
    ln = _load(args.lean)

    by_idx = {}
    provenance = {}
    for r in ln["results"]:
        by_idx[r["idx"]] = r
        provenance[r["idx"]] = "lean"
    for r in dm["results"]:                    # divmod wins per-idx
        by_idx[r["idx"]] = r
        provenance[r["idx"]] = "divmod"

    results = [by_idx[i] for i in sorted(by_idx)]
    counts = Counter(r["status"] for r in results)
    scored = len(results)
    n_pass = counts.get("PASS", 0)

    # aggregate wall / kv / vram
    wall_each = {"divmod": dm.get("wall_seconds", 0.0), "lean": ln.get("wall_seconds", 0.0)}
    wall_concurrent = max(wall_each.values())     # lanes run concurrently
    steps_total = dm.get("steps_total", 0) + ln.get("steps_total", 0)
    kv = {"max_seq_len": max(dm["kv_cache"].get("max_seq_len", 0) or 0,
                             ln["kv_cache"].get("max_seq_len", 0) or 0),
          "max_cache_size": max(dm["kv_cache"].get("max_cache_size", 0) or 0,
                                ln["kv_cache"].get("max_cache_size", 0) or 0),
          "total_evicted": (dm["kv_cache"].get("total_evicted", 0) or 0)
          + (ln["kv_cache"].get("total_evicted", 0) or 0)}

    table = OrderedDict()
    for r in results:
        row = table.setdefault(r["cluster"], {"n": 0, **{s: 0 for s in _STATUSES}})
        row["n"] += 1
        row[r["status"]] += 1

    print("=" * 80)
    print("CHK-1 (#2/#10) — SPARSE PURE-FORWARD WHOLE-VM, KV-CACHED, FULL 1096 (GPU)")
    print("=" * 80)
    print(f"  scored: {scored}/{args.total}  (missing {args.total - scored})")
    print(f"  divmod lane: {dm['device']} wall {wall_each['divmod']:.0f}s  |  "
          f"lean lane: {ln['device']} wall {wall_each['lean']:.0f}s")
    print(f"  concurrent wall (2-GPU): {wall_concurrent:.0f}s ({wall_concurrent/60:.1f} min)  "
          f"| total steps: {steps_total}  | steps/sec (aggregate): "
          f"{steps_total/max(wall_concurrent,1e-6):.1f}")
    print(f"  compute_mode: divmod={dm.get('compute_mode')} lean={ln.get('compute_mode')}"
          f"  (dense_kernel => L-inf=0)")
    print(f"  VRAM peak: divmod {dm.get('vram_peak_mb',0):.0f}MB  lean {ln.get('vram_peak_mb',0):.0f}MB"
          f"  | sparse storage: divmod {dm.get('storage_mb',0):.1f}MB lean {ln.get('storage_mb',0):.1f}MB")
    print(f"  KV-cache: max_seq_len={kv['max_seq_len']} "
          f"max_cache(block0,flat)={kv['max_cache_size']} total_evicted={kv['total_evicted']}")
    cc = []
    for src in (dm, ln):
        cc += src.get("compact_disagreements", [])
    if dm.get("check_compact") or ln.get("check_compact"):
        print(f"  compact-vs-sparse disagreements: {len(cc)} "
              f"{'(empirical-liveness VALIDATED)' if not cc else '(see ids: '+str(cc)+')'}")
    print("-" * 80)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}")
    print("-" * 80)
    print(f"  CHK-1 SCORE: {n_pass}/{scored}  ({100.0*n_pass/max(scored,1):.2f}%)")
    if scored != args.total:
        print(f"  CHK-1 SCORE (of full {args.total}): {n_pass}/{args.total} "
              f"({100.0*n_pass/args.total:.2f}%)")
    print("=" * 80)

    print("\nPER-CLUSTER BREAKDOWN")
    hdr = (f"  {'cluster':18s} {'n':>4s} {'PASS':>5s} {'FAIL':>5s} "
           f"{'TMOUT':>6s} {'ERR':>4s} {'src':>7s} {'pass%':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    cl_src = {}
    for r in results:
        cl_src.setdefault(r["cluster"], provenance[r["idx"]])
    for cluster, row in sorted(table.items()):
        n = row["n"]
        pct = (100.0 * row["PASS"] / n) if n else 0.0
        print(f"  {cluster:18s} {n:4d} {row['PASS']:5d} {row['FAIL']:5d} "
              f"{row['TIMEOUT']:6d} {row['ERROR']:4d} {cl_src.get(cluster,''):>7s} {pct:6.1f}")

    nonpass = [r for r in results if r["status"] != "PASS"]
    if nonpass:
        print(f"\nNON-PASS ({len(nonpass)}):")
        for r in sorted(nonpass, key=lambda r: (r["status"], r["cluster"], r["idx"])):
            print(f"  id={r['idx']:4d} [{r['status']:7s}] {r['cluster']:16s} "
                  f"exp={r['expected']} got={r['got_exit']} steps={r['got_steps']}  "
                  f"{r.get('detail','')[:70]}")

    out = {"scored": scored, "total": args.total,
           "summary": {s: counts.get(s, 0) for s in _STATUSES} | {"pass": n_pass},
           "concurrent_wall_seconds": wall_concurrent, "steps_total": steps_total,
           "kv_cache": kv, "compact_disagreements": cc,
           "clusters": {k: v for k, v in table.items()},
           "results": results}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[merge] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
