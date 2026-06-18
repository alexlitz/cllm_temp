#!/usr/bin/env python
"""Fast GPU regression tripwire — the truthful, ~5-min gate for any fix.

The full 1096 run (~30 min) is the only thing that catches cross-cluster
regressions (a "local" fix perturbing a distant cluster via shared
exact-cancellation weights — e.g. byte-0's operand-gather Q/K slot silently
breaking var_simple + expr_mod). This samples 2 ok + 2 fail programs per
cluster (covers all 56 clusters, both states) so the same regression is a
5-min catch. Run it flag-ON (env) before believing ANY fix.

Usage:
    C4_MY_FIX=1 CUDA_VISIBLE_DEVICES=0 python tools/gpu_tripwire.py [baseline.json]

Exits non-zero if any ok-tripwire regresses (ok->fail). Prints flips (fail->ok)
per cluster so you see the actual delta. The baseline json is the recorded
per-id status of the build you're comparing against (default: the committed
tools/tripwire_baseline.json).
"""
import json, os, re, sys, subprocess, collections

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tripwire_baseline.json")


def cluster(idx, progs):
    return re.sub(r"_\d+$", "", progs[idx][2].split(":")[0])


def main():
    spec = json.load(open(BASELINE))
    expected = {int(k): v for k, v in spec["expected"].items()}
    ids = sorted(expected)
    idstr = ",".join(map(str, ids))
    out = "/tmp/tripwire_result.json"
    print(f"[tripwire] running {len(ids)} programs (baseline={spec.get('baseline')}) ...", flush=True)
    rc = subprocess.call([
        sys.executable, os.path.join(HERE, "run_1096_canonical.py"),
        "--ids", idstr, "--criterion", "full_trace", "--spec-k", "0",
        "--max-steps-cap", "40", "--output", out,
    ])
    if rc != 0:
        print(f"[tripwire] runner exited {rc}", file=sys.stderr)
        return 2

    from tests.test_suite_1000 import generate_test_programs
    progs = generate_test_programs()
    got = {r["idx"]: r["status"] for r in json.load(open(out))["results"]}

    regressions = collections.defaultdict(list)  # ok -> not-ok
    flips = collections.defaultdict(list)         # not-ok -> ok
    for i in ids:
        e, g = expected[i], got.get(i, "MISSING")
        if e == "ok" and g != "ok":
            regressions[cluster(i, progs)].append((i, g))
        elif e != "ok" and g == "ok":
            flips[cluster(i, progs)].append(i)

    nreg = sum(len(v) for v in regressions.values())
    nflip = sum(len(v) for v in flips.values())
    print(f"\n[tripwire] REGRESSIONS (ok->fail): {nreg}")
    for c, v in sorted(regressions.items()):
        print(f"    {c}: {v}")
    print(f"[tripwire] FLIPS (fail->ok): {nflip}")
    for c, v in sorted(flips.items()):
        print(f"    {c}: {len(v)} {v}")
    print(f"[tripwire] net on sample: {nflip - nreg:+d}  ({'CLEAN' if nreg == 0 else 'REGRESSION — do NOT land'})")
    return 1 if nreg else 0


if __name__ == "__main__":
    sys.exit(main())
