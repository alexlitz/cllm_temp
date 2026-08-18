#!/usr/bin/env python3
"""Run the gcc c-torture corpus through the COMBINED 32-BIT machine harness.

This is the ``run_broad_conformance.py`` sibling wired to ``transpiler_harness32``
(compiler32 WORD=4 + c4vm32u STRIDE=4 + transpile(preprocess,native32,fnptr) with
the wave-2 nested/packed/gnumisc passes default-active).  It measures the COMBINED
32-bit-machine in-subset pass rate over the c-torture corpus.

  python3 run_ctorture32.py [--json OUT.json] [--limit N] [--cycles N]

Oracle = gcc -m32 -std=c90 (exit mod 256 + stdout), rebuilt per case inside the
harness.  A case gcc REJECTS (needs a header/feature gcc -std=c90 -m32 won't take,
c99/c11-only, etc.) is ``gcc_fail`` and EXCLUDED from the in-subset denominator --
we cannot judge the port against a missing/broken oracle.

OUT-OF-SUBSET cases (fnptr / varargs / float / long-long, detected by
ctestsuite_loader.out_of_subset_reason -> name suffix) are bucketed separately: a
fail there is a DOCUMENTED subset boundary, not a transpiler/machine bug.  With
fnptr=True the B3_fnptr sub-bucket is now (partly) IN reach, but we keep the
loader's by-design classification for an honest headline.

CPU-only, no model load.  Pure ADDITION under id_port/c90_e2e/; golden 174ece66
untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, OrderedDict
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transpiler_harness32 import run_case, Result       # noqa: E402  (32-bit harness)
from ctestsuite_loader import load_cases                 # noqa: E402


def is_fail(r):
    return r.stage in ("MISMATCH", "compile_error", "transpile_error", "vm_error")


def is_pass(r):
    return r.stage == "PASS"


def oos_of(name: str):
    if "__" in name:
        return name.split("__", 1)[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cycles", type=int, default=40_000_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cases = load_cases(limit=args.limit)
    results = []          # (oos_reason, Result)
    t0 = time.time()
    for i, (name, cat, src, expect, sizeof_gap) in enumerate(cases):
        r = run_case(name, cat, src, expect, sizeof_gap=sizeof_gap,
                     timeout_cycles=args.cycles)
        oos = oos_of(name)
        results.append((oos, r))
        if not args.quiet and (i % 50 == 0):
            print(f"  ...{i}/{len(cases)}  ({time.time()-t0:.0f}s)", file=sys.stderr)

    insub = [(o, r) for (o, r) in results if o is None and r.stage != "gcc_fail"]
    outsub = [(o, r) for (o, r) in results if o is not None and r.stage != "gcc_fail"]
    gcc_skip = [(o, r) for (o, r) in results if r.stage == "gcc_fail"]

    in_pass = sum(1 for _, r in insub if is_pass(r))
    in_fail = sum(1 for _, r in insub if is_fail(r))
    in_total = len(insub)

    print("\n" + "=" * 78)
    print("COMBINED 32-BIT MACHINE — c-torture CONFORMANCE  (oracle: gcc -m32 -std=c90)")
    print("machine: compiler32(WORD=4) + c4vm32u(STRIDE=4) + "
          "transpile(preprocess,native32,fnptr + nested/packed/gnumisc)")
    print("=" * 78)
    print(f"corpus total (attempted):        {len(results)}")
    print(f"  gcc-rejected (excluded/skip):  {len(gcc_skip)}")
    print(f"  out-of-subset (by-design):     {len(outsub)}")
    print(f"  IN-SUBSET (judged):            {in_total}")
    print("-" * 78)
    print(f"  IN-SUBSET PASS:  {in_pass}/{in_total} = "
          f"{100.0*in_pass/max(in_total,1):.1f}%")
    print(f"  IN-SUBSET FAIL:  {in_fail}")

    # per-stage breakdown of in-subset fails
    stage_ct = defaultdict(int)
    for _, r in insub:
        stage_ct[r.stage] += 1
    print("\nIN-SUBSET stage breakdown:")
    for st in ("PASS", "MISMATCH", "compile_error", "transpile_error", "vm_error"):
        print(f"  {st:16s} {stage_ct.get(st,0)}")

    # out-of-subset breakdown
    print("\nOUT-OF-SUBSET (by-design boundary):")
    oosb = defaultdict(lambda: defaultdict(int))
    for o, r in outsub:
        oosb[o]["total"] += 1
        oosb[o]["pass" if is_pass(r) else "fail"] += 1
    for reason in sorted(oosb):
        b = oosb[reason]
        print(f"  {reason:12s}  total={b['total']:>3d}  "
              f"pass={b['pass']:>3d}  fail(expected)={b['fail']:>3d}")

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
                {"oos": o,
                 **{k: v for k, v in asdict(r).items()
                    if k != "transpiled" or is_fail(r)}}
                for o, r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
