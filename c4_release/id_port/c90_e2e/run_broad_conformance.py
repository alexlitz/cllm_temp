#!/usr/bin/env python3
"""Run the BROAD C90 corpus (hand-written 117 + c-testsuite single-exec) through
the SOURCE-path transpiler harness and emit a comprehensive conformance matrix.

  python3 run_broad_conformance.py [--json OUT.json] [--only cts|hand]
                                   [--filter CAT] [--name SUBSTR] [--cycles N]

Oracle = gcc -m32 -std=c90 (exit mod 256 + stdout).  Port path = transpile.py ->
compile_c -> c4vm.  A case gcc REJECTS (c99/c11-only, needs 64-bit, etc.) is
counted ``gcc_skip`` and excluded from the in-subset denominator (we cannot judge
the port against a broken oracle).

OUT-OF-SUBSET cases (name suffix ``__B3_fnptr`` / ``__B9_varargs`` / ``__float`` /
``__longlong``) are bucketed separately: a fail there is a DOCUMENTED subset
boundary, not a transpiler bug.  The in-subset conformance % is the headline.

CPU-only, no model load.  Pure ADDITION; golden 174ece66 unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, OrderedDict
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transpiler_harness import run_case, Result           # noqa: E402
from transpiler_corpus import CORPUS as HAND_CORPUS        # noqa: E402
from ctestsuite_loader import load_cases                   # noqa: E402


OOS_SUFFIXES = ("B3_fnptr", "B9_varargs", "float", "longlong")

# The hand corpus documents fnptr (B3) and user-varargs (B9) as OUT-OF-SUBSET by
# design (Doom uses the FN_ID __actions__ dispatch + builtin printf).  A case in
# those categories is a by-design boundary, not a transpiler bug.
_HAND_OOS_CATS = {"fnptr": "B3_fnptr", "varargs": "B9_varargs"}


def oos_of(name: str, category: str, suite: str):
    if "__" in name:
        suf = name.split("__", 1)[1]
        if suf in OOS_SUFFIXES:
            return suf
    if suite == "hand" and category in _HAND_OOS_CATS:
        return _HAND_OOS_CATS[category]
    return None


def build_corpus(only: str | None):
    cases = []
    if only in (None, "hand"):
        for t in HAND_CORPUS:
            cases.append(("hand", *t))
    if only in (None, "cts"):
        for t in load_cases():
            cases.append(("cts", *t))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", choices=["cts", "hand"], default=None)
    ap.add_argument("--filter", default=None, help="only this category")
    ap.add_argument("--name", default=None, help="only names containing SUBSTR")
    ap.add_argument("--cycles", type=int, default=60_000_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    corpus = build_corpus(args.only)
    results = []          # (source_suite, oos_reason, Result)
    for suite, name, cat, src, expect, sizeof_gap in corpus:
        if args.filter and cat != args.filter:
            continue
        if args.name and args.name not in name:
            continue
        r = run_case(name, cat, src, expect, sizeof_gap=sizeof_gap,
                     timeout_cycles=args.cycles)
        oos = oos_of(name, cat, suite)
        results.append((suite, oos, r))
        if not args.quiet:
            status = r.stage
            if status == "PASS" and r.bug_class == "c4_x86_sizeof_gap":
                status = "PASS(sizeof)"
            mark = {"PASS": "ok  ", "PASS(sizeof)": "ok~ ", "MISMATCH": "FAIL",
                    "transpile_error": "TERR", "compile_error": "CERR",
                    "vm_error": "VMER", "gcc_fail": "gcc?"}.get(status, "??? ")
            tag = f" [{oos}]" if oos else ""
            line = f"  [{mark}] {name:30s} ({cat}){tag}"
            if r.stage not in ("PASS",):
                line += f"  -- {r.detail[:100]}"
            print(line)

    # ---------------- matrix ----------------
    # buckets:
    #   in-subset PASS / sizeof / FAIL(mismatch/compileE/transpE/vmE)
    #   out-of-subset (by-design) by reason
    #   gcc_skip (oracle rejected -> excluded)
    def is_fail(r):
        return r.stage in ("MISMATCH", "compile_error", "transpile_error", "vm_error")

    def is_pass(r):
        return r.stage == "PASS"

    insub = [(s, o, r) for (s, o, r) in results if o is None and r.stage != "gcc_fail"]
    outsub = [(s, o, r) for (s, o, r) in results if o is not None and r.stage != "gcc_fail"]
    gcc_skip = [(s, o, r) for (s, o, r) in results if r.stage == "gcc_fail"]

    in_pass = sum(1 for _, _, r in insub if is_pass(r))
    in_fail = sum(1 for _, _, r in insub if is_fail(r))
    in_total = len(insub)

    print("\n" + "=" * 78)
    print("BROAD C90 CONFORMANCE MATRIX  (oracle: gcc -m32 -std=c90)")
    print("=" * 78)
    print(f"corpus total (attempted):        {len(results)}")
    print(f"  gcc-rejected (excluded/skip):  {len(gcc_skip)}  "
          f"(c99/c11-only / needs-64bit — no valid oracle)")
    print(f"  out-of-subset (by-design):     {len(outsub)}")
    print(f"  IN-SUBSET (judged):            {in_total}")
    print("-" * 78)
    print(f"  IN-SUBSET PASS:  {in_pass}/{in_total} = "
          f"{100.0*in_pass/max(in_total,1):.1f}%")
    print(f"  IN-SUBSET FAIL (transpiler bugs): {in_fail}")

    # per-category in-subset
    print("\nIN-SUBSET by construct class:")
    hdr = (f"{'category':12s} {'pass':>5s} {'sizeof':>6s} {'MISM':>5s} "
           f"{'cErr':>5s} {'tErr':>5s} {'vmE':>4s} {'total':>5s}")
    print(hdr); print("-" * len(hdr))
    bycat = OrderedDict()
    for s, o, r in insub:
        bycat.setdefault(r.category, []).append(r)
    catgrand = defaultdict(int)
    for cat in sorted(bycat):
        c = defaultdict(int)
        for r in bycat[cat]:
            if r.stage == "PASS" and r.bug_class == "c4_x86_sizeof_gap":
                c["sizeof"] += 1
            elif r.stage == "PASS":
                c["pass"] += 1
            elif r.stage == "MISMATCH":
                c["mism"] += 1
            elif r.stage == "compile_error":
                c["cErr"] += 1
            elif r.stage == "transpile_error":
                c["tErr"] += 1
            elif r.stage == "vm_error":
                c["vmE"] += 1
        for k, v in c.items():
            catgrand[k] += v
        print(f"{cat:12s} {c['pass']:>5d} {c['sizeof']:>6d} {c['mism']:>5d} "
              f"{c['cErr']:>5d} {c['tErr']:>5d} {c['vmE']:>4d} {len(bycat[cat]):>5d}")

    # out-of-subset breakdown
    print("\nOUT-OF-SUBSET (by-design boundary — fails EXPECTED):")
    oosb = defaultdict(lambda: defaultdict(int))
    for s, o, r in outsub:
        oosb[o]["total"] += 1
        oosb[o]["pass" if is_pass(r) else "fail"] += 1
    for reason in sorted(oosb):
        b = oosb[reason]
        print(f"  {reason:12s}  total={b['total']:>3d}  "
              f"pass={b['pass']:>3d}  fail(expected)={b['fail']:>3d}")

    # failing in-subset detail (bug attribution)
    fails = [(s, r) for s, o, r in insub if is_fail(r)]
    if fails and not args.quiet:
        print("\n" + "=" * 78)
        print(f"IN-SUBSET FAILING CASES ({len(fails)}) — transpiler bugs to fix")
        print("=" * 78)
        for s, r in fails:
            print(f"\n### {r.name} [{r.category}] suite={s} stage={r.stage}")
            print(f"    gcc:  exit={r.gcc_exit} stdout={r.gcc_stdout!r}")
            print(f"    port: exit={r.port_exit} stdout={r.port_stdout!r}")
            print(f"    detail: {r.detail[:300]}")

    if args.json:
        payload = {
            "in_subset_total": in_total,
            "in_subset_pass": in_pass,
            "in_subset_fail": in_fail,
            "in_subset_pass_rate": in_pass / max(in_total, 1),
            "out_of_subset_total": len(outsub),
            "gcc_skipped": len(gcc_skip),
            "corpus_total": len(results),
            "results": [
                {"suite": s, "oos": o,
                 **{k: v for k, v in asdict(r).items()
                    if k != "transpiled" or is_fail(r)}}
                for s, o, r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
