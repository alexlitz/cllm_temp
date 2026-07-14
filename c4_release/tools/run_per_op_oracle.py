#!/usr/bin/env python3
"""PARALLEL runner for the per-op decode oracle (tests/oracles/per_op_decode.py).

The per-op oracle decodes each op-class's representative programs through the
BIT-EXACT CPU faithful forward (``FaithfulAutoregressiveRunner`` /
``ModelExactForward``), whose per-program ``full_trace`` verdict is byte-identical
to the 30-min GPU gate (``tools/run_1096_canonical.py --criterion full_trace
--spec-k 0``) — but with NO GPU. A single program's decode is ~20-45s (a growing-
tape O(steps^2) CPU forward), so the WHOLE 30-op-class suite run serially is
~25-35 min. This runner SHARDS the op-classes across ``--workers`` CPU processes
(each builds the cached model once, ~4s, then decodes its shard), collapsing the
wall to ~= (serial_time / workers) + build. On an 8-core box the full suite lands
in ~5-8 min — fast enough to be the per-op-class gate.

It records a BASELINE (op-class -> pass/fail + per-program verdicts) so a
decode-preserving change can be diffed against the known-good reference: any
op-class that was ``pass`` and is now ``fail`` BLOCKS the change.

Usage
-----
    # Run the full suite in parallel, print the per-op-class table:
    OMP_NUM_THREADS=2 python tools/run_per_op_oracle.py --workers 8

    # Record the current tree as the baseline reference:
    OMP_NUM_THREADS=2 python tools/run_per_op_oracle.py --workers 8 \
        --record-baseline tools/per_op_oracle_baseline.json

    # Gate a change: compare against the recorded baseline (exit 1 on any
    # op-class regression pass->fail):
    OMP_NUM_THREADS=2 python tools/run_per_op_oracle.py --workers 8 \
        --baseline tools/per_op_oracle_baseline.json

    # Narrow while iterating:
    OMP_NUM_THREADS=4 python tools/run_per_op_oracle.py --op-classes ADD,SUB,MUL

    # Print the op -> corpus-cluster derivation map (how each op is covered):
    python tools/run_per_op_oracle.py --map

MEMORY NOTE: each worker resident-set is ~1.2-1.5GB (one CPU model build). Size
``--workers`` to keep ``workers * 1.5GB`` under the box's free RAM, and cap
``OMP_NUM_THREADS`` (2-4) so the workers don't oversubscribe cores.

Tooling only: never writes weights (reads the SAME cached baked model the smoke
gate builds), so the golden model is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


def _all_op_classes() -> List[str]:
    from tests.oracles.per_op_decode import ALL_OP_CLASSES

    return list(ALL_OP_CLASSES)


def _shard(ops: List[str], n: int) -> List[List[str]]:
    """Distribute op-classes into ``n`` shards, CO-LOCATING op-classes that share
    a representative program so the per-process decode cache is exploited.

    Op-classes that decode the SAME program (var_simple_0 covers LI/SI/LEA/JSR/
    ENT; func_identity_0 covers ADJ/LEV) are grouped onto ONE worker, so the
    expensive shared decode runs ONCE (via the ``per_op_decode._DECODE_CACHE``)
    instead of once per worker. Groups are then greedily packed onto the ``n``
    shards (largest group first) to balance the load.
    """
    from tests.oracles.per_op_decode import op_class_specs

    specs = op_class_specs()
    # Build a signature for each op = the set of representative programs it runs.
    def sig(op: str):
        s = specs[op]
        return frozenset([("c", i) for i in s.corpus_ids]
                         + [("r", r.label) for r in s.raw])

    # Union-find on shared programs so all ops sharing ANY program cluster.
    parent = {op: op for op in ops}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    prog_owner: dict = {}
    for op in ops:
        for p in sig(op):
            if p in prog_owner:
                union(op, prog_owner[p])
            else:
                prog_owner[p] = op
    groups: dict = {}
    for op in ops:
        groups.setdefault(find(op), []).append(op)
    group_list = sorted(groups.values(), key=len, reverse=True)

    # Greedy pack groups onto n shards (smallest-current-shard first).
    shards: List[List[str]] = [[] for _ in range(max(1, n))]
    for grp in group_list:
        shards.sort(key=len)
        shards[0].extend(grp)
    return [s for s in shards if s]


def _run_shard(ops: List[str], max_steps_cap: int) -> subprocess.Popen:
    """Spawn one worker: ``per_op_decode.py --op-classes <shard> --json <tmp>``."""
    import tempfile

    out_json = tempfile.NamedTemporaryFile(
        prefix="peroporacle_", suffix=".json", delete=False
    ).name
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "2"))
    env["PYTHONPATH"] = _PKG + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-u",
        os.path.join(_PKG, "tests", "oracles", "per_op_decode.py"),
        "--op-classes", ",".join(ops),
        "--max-steps-cap", str(max_steps_cap),
        "--json", out_json,
    ]
    log = out_json + ".log"
    fh = open(log, "w")
    proc = subprocess.Popen(cmd, env=env, cwd=_PKG, stdout=fh, stderr=subprocess.STDOUT)
    proc._out_json = out_json  # type: ignore[attr-defined]
    proc._log = log  # type: ignore[attr-defined]
    proc._ops = ops  # type: ignore[attr-defined]
    proc._fh = fh  # type: ignore[attr-defined]
    return proc


def _collect(proc: subprocess.Popen) -> Dict[str, dict]:
    rc = proc.wait()
    try:
        proc._fh.close()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    out_json = proc._out_json  # type: ignore[attr-defined]
    if rc not in (0, 1) or not os.path.exists(out_json):
        # rc 0 = all pass, 1 = some op-class fail (both are valid results);
        # anything else (or missing json) is a worker crash.
        sys.stderr.write(
            f"[per-op-oracle] worker for {proc._ops} exited {rc}; "  # type: ignore[attr-defined]
            f"see {proc._log}\n"  # type: ignore[attr-defined]
        )
        return {op: {"ok": False, "programs": [], "worker_error": True}
                for op in proc._ops}  # type: ignore[attr-defined]
    data = json.load(open(out_json))
    return data["op_classes"]


def _print_map() -> None:
    from tests.oracles.per_op_decode import op_class_specs

    specs = op_class_specs()
    print("op-class -> representatives (corpus id | raw program):")
    for op, s in specs.items():
        reps = [f"corpus#{i}" for i in s.corpus_ids] + [r.label for r in s.raw]
        kind = "corpus" if s.corpus_ids else "raw-only"
        print(f"  {op:5s} [{kind:9s}]: {reps}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--op-classes", default=None,
                    help="comma-separated op-classes (default: all 30).")
    ap.add_argument("--max-steps-cap", type=int, default=60)
    ap.add_argument("--record-baseline", default=None,
                    help="write the current run's verdicts as the baseline JSON.")
    ap.add_argument("--baseline", default=None,
                    help="compare against this baseline JSON; exit 1 on any "
                         "op-class regression (was pass, now fail).")
    ap.add_argument("--map", action="store_true",
                    help="print the op->representative map and exit.")
    ap.add_argument("--json", default=None, help="write the merged verdicts JSON.")
    args = ap.parse_args(argv)

    if args.map:
        _print_map()
        return 0

    ops = (
        [o.strip().upper() for o in args.op_classes.split(",") if o.strip()]
        if args.op_classes else _all_op_classes()
    )
    shards = _shard(ops, args.workers)
    print(f"[per-op-oracle] {len(ops)} op-classes across {len(shards)} worker(s) "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}); "
          f"decoding via BIT-EXACT CPU faithful forward (no GPU).", flush=True)

    t0 = time.monotonic()
    procs = [_run_shard(s, args.max_steps_cap) for s in shards]
    merged: Dict[str, dict] = {}
    for p in procs:
        merged.update(_collect(p))

    # Order the merged result by the canonical op-class order.
    order = _all_op_classes()
    merged = {op: merged[op] for op in order if op in merged}

    # Report.
    print("\n" + "=" * 78)
    print("PER-OP DECODE ORACLE  (faithful full_trace verdict; == GPU gate, no GPU)")
    print("=" * 78)
    n_ok = 0
    for op, r in merged.items():
        progs = r.get("programs", [])
        n_pass = sum(1 for p in progs if p.get("status") == "pass")
        n_skip = sum(1 for p in progs if p.get("status") == "skipped")
        ok = r.get("ok", False)
        n_ok += 1 if ok else 0
        badge = "PASS" if ok else ("ERR " if r.get("worker_error") else "FAIL")
        fails = [
            f"{p['label']}={p['status']}"
            for p in progs if p.get("status") not in ("pass", "skipped")
        ]
        extra = f"  <-- {fails}" if fails else (" (all skipped)" if n_skip == len(progs) and progs else "")
        print(f"  [{badge}] {op:5s} {n_pass}/{len(progs)} progs pass{extra}")

    print("-" * 78)
    print(f"OP-CLASS PASS: {n_ok}/{len(merged)}   "
          f"(wall {time.monotonic() - t0:.0f}s, {len(shards)} workers)")

    if args.record_baseline:
        json.dump(
            {"op_classes": {op: {"ok": r.get("ok", False),
                                 "programs": r.get("programs", [])}
                            for op, r in merged.items()}},
            open(args.record_baseline, "w"), indent=1,
        )
        print(f"[per-op-oracle] recorded baseline -> {args.record_baseline}")

    if args.json:
        json.dump({"op_classes": merged}, open(args.json, "w"), indent=1)

    rc = 0 if n_ok == len(merged) else 1
    if args.baseline:
        base = json.load(open(args.baseline))["op_classes"]
        regressions = []
        gains = []
        for op, r in merged.items():
            was = base.get(op, {}).get("ok", None)
            now = r.get("ok", False)
            if was is True and not now:
                regressions.append(op)
            elif was is False and now:
                gains.append(op)
        print("\nBASELINE DIFF:")
        print(f"  regressions (pass -> fail): {regressions or 'none'}")
        print(f"  gains       (fail -> pass): {gains or 'none'}")
        if regressions:
            print("  *** BLOCK the change: op-class decode regressed. ***")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
