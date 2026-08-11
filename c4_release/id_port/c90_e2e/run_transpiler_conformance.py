#!/usr/bin/env python3
"""Run the broad C90 corpus through the SOURCE-path transpiler harness and emit
a conformance matrix (pass/fail by construct class).

  python3 run_transpiler_conformance.py [--filter CAT] [--json OUT.json]

For each case: gcc -m32 is the ORACLE (exit mod 256 + stdout).  The port path
(transpile.py -> compile_c -> c4vm) must byte-match it.  sizeof-gap cases are
tagged, not failed.

CPU-only, no model load.  Prints a per-category matrix + a per-failing-case
detail block for bug attribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, OrderedDict

sys.path.insert(0, "/tmp/wt_transpiler_c90/c4_release/id_port/c90_e2e")
from transpiler_harness import run_case, Result
from transpiler_corpus import CORPUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None, help="only run this category")
    ap.add_argument("--name", default=None, help="only run cases whose name contains this")
    ap.add_argument("--json", default=None, help="write full results JSON here")
    ap.add_argument("--cycles", type=int, default=40_000_000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results: list[Result] = []
    cats = OrderedDict()
    for name, cat, src, expect, sizeof_gap in CORPUS:
        if args.filter and cat != args.filter:
            continue
        if args.name and args.name not in name:
            continue
        r = run_case(name, cat, src, expect, sizeof_gap=sizeof_gap,
                     timeout_cycles=args.cycles)
        results.append(r)
        cats.setdefault(cat, [])
        cats[cat].append(r)
        status = r.stage
        if status == "PASS" and r.bug_class == "c4_x86_sizeof_gap":
            status = "PASS(sizeof)"
        mark = {"PASS": "ok  ", "PASS(sizeof)": "ok~ ", "MISMATCH": "FAIL",
                "transpile_error": "TERR", "compile_error": "CERR",
                "vm_error": "VMER", "gcc_fail": "GCC?"}.get(status, "??? ")
        line = f"  [{mark}] {name:26s} ({cat})"
        if r.stage not in ("PASS",):
            line += f"  -- {r.detail[:120]}"
        print(line)

    # ---- matrix ----
    print("\n" + "=" * 72)
    print("CONFORMANCE MATRIX BY CONSTRUCT CLASS")
    print("=" * 72)
    hdr = f"{'category':12s} {'pass':>5s} {'sizeof':>6s} {'MISMATCH':>8s} {'compileE':>8s} {'transpE':>7s} {'vmE':>4s} {'total':>5s}"
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(int)
    for cat, rs in cats.items():
        c = defaultdict(int)
        for r in rs:
            if r.stage == "PASS" and r.bug_class == "c4_x86_sizeof_gap":
                c["sizeof"] += 1
            elif r.stage == "PASS":
                c["pass"] += 1
            elif r.stage == "MISMATCH":
                c["mismatch"] += 1
            elif r.stage == "compile_error":
                c["compileE"] += 1
            elif r.stage == "transpile_error":
                c["transpE"] += 1
            elif r.stage == "vm_error":
                c["vmE"] += 1
            elif r.stage == "gcc_fail":
                c["gccfail"] += 1
        for k, v in c.items():
            tot[k] += v
        print(f"{cat:12s} {c['pass']:>5d} {c['sizeof']:>6d} {c['mismatch']:>8d} "
              f"{c['compileE']:>8d} {c['transpE']:>7d} {c['vmE']:>4d} {len(rs):>5d}")
    print("-" * len(hdr))
    total = len(results)
    print(f"{'TOTAL':12s} {tot['pass']:>5d} {tot['sizeof']:>6d} {tot['mismatch']:>8d} "
          f"{tot['compileE']:>8d} {tot['transpE']:>7d} {tot['vmE']:>4d} {total:>5d}")
    clean = tot['pass'] + tot['sizeof']
    print(f"\nPASS (incl sizeof-gap as expected): {clean}/{total} = {100.0*clean/max(total,1):.1f}%")
    print(f"Bug-revealing failures (mismatch+compileE+transpE+vmE): "
          f"{tot['mismatch']+tot['compileE']+tot['transpE']+tot['vmE']}/{total}")

    # ---- failing-case details for bug attribution ----
    fails = [r for r in results if r.stage in ("MISMATCH", "compile_error",
                                               "transpile_error", "vm_error")]
    if fails:
        print("\n" + "=" * 72)
        print(f"FAILING CASES ({len(fails)}) -- for bug attribution")
        print("=" * 72)
        for r in fails:
            print(f"\n### {r.name} [{r.category}] stage={r.stage}")
            print(f"    gcc: exit={r.gcc_exit} stdout={r.gcc_stdout!r}")
            print(f"    port: exit={r.port_exit} stdout={r.port_stdout!r}")
            print(f"    detail: {r.detail}")

    if args.json:
        payload = {
            "summary": dict(tot),
            "pass_rate": clean / max(total, 1),
            "total": total,
            "results": [
                {k: v for k, v in asrec(r).items() if k != "transpiled"
                 or r.stage in ("MISMATCH", "compile_error", "transpile_error", "vm_error")}
                for r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")


def asrec(r: Result):
    from dataclasses import asdict
    return asdict(r)


if __name__ == "__main__":
    main()
