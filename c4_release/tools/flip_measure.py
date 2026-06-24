#!/usr/bin/env python3
"""Robust campaign-flip measurement — the authoritative ``campaign vs golden``
gate, hardened against the silent ``pass=0`` corruption.

WHY THIS EXISTS
---------------
The naive ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python
tools/run_1096_canonical.py --ids 0-1095`` reports a *number* even when the
underlying model build went degenerate — and a degenerate build fails EVERY
program (observed: ``pass=0 fail=264`` across add/var/func/expr clusters that
each pass 25-50/50 in isolation). Two structural hazards make this easy:

  1. The disk-cache key is ``SHA256(source bytes + kwargs JSON + format ver)``
     and DOES NOT include the campaign env flags. A campaign build and a golden
     build collide on the key, so a cache dir shared across flag configs hands
     back the wrong model. (Mitigation: a dedicated, freshly-cleared cache dir
     per flag config — enforced here.)
  2. Building the (auto-widened) campaign model under GPU memory starvation
     (an 8-agent fleet) can silently produce wrong weights → ``pass=0``.
     (Mitigation: wait for memory headroom, then a CANARY GATE.)

THE CANARY GATE
---------------
Before trusting an 863-program sweep, build the model and run four programs that
are *guaranteed to pass* in any correct campaign build — one per cluster family:

    id 0   = add_*          (campaign add 50/50)
    id 250 = var_simple_*   (campaign 25/25)
    id 550 = func_identity_*(campaign 25/25)
    id 875 = expr_mod_*     (campaign 25/25)

If all four pass, the cached weights are sound and the full sweep (which
cache-hits the SAME verified build) is trustworthy. If any canary fails, the
build is corrupt: we ABORT with a non-zero exit and print ``MEASUREMENT
INVALID`` — never a fake number. The full sweep itself re-includes the canary
ids and is re-checked post-hoc, so a fresh corruption in the sweep process is
also caught.

USAGE
-----
    python tools/flip_measure.py                       # full campaign measure
    python tools/flip_measure.py --gpu 1 --golden 468  # device + flip baseline
    python tools/flip_measure.py --full-ids 0-499      # a sub-corpus

Exit codes: 0 = measured OK (see the FLIP verdict), 2 = build corrupt /
canary failed (number is INVALID), 1 = a runner subprocess crashed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One guaranteed-pass program per cluster family. A correct campaign build
# passes all four; a degenerate build fails them.
CANARY_IDS = [0, 250, 550, 875]
CANARY_LABEL = {0: "add", 250: "var_simple", 550: "func_identity", 875: "expr_mod"}


def _free_gb() -> float:
    """Available RAM in GiB from /proc/meminfo (MemAvailable)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("inf")


def _wait_for_memory(min_free_gb: float, timeout_s: float) -> None:
    """Block until free RAM clears the threshold (so the build isn't starved)."""
    t0 = time.monotonic()
    while True:
        free = _free_gb()
        if free >= min_free_gb:
            print(f"[flip] memory OK: {free:.1f}GB free (>= {min_free_gb}GB)", flush=True)
            return
        waited = time.monotonic() - t0
        if waited >= timeout_s:
            print(
                f"[flip] WARNING: {free:.1f}GB free < {min_free_gb}GB after "
                f"{waited:.0f}s wait — proceeding anyway (build may be at risk)",
                flush=True,
            )
            return
        print(
            f"[flip] waiting for memory: {free:.1f}GB free < {min_free_gb}GB "
            f"({waited:.0f}/{timeout_s:.0f}s)",
            flush=True,
        )
        time.sleep(20)


def _run_canonical(ids: str, out_path: str, env: dict, chunk: int, tag: str) -> dict:
    """Invoke run_1096_canonical for ``ids``; return {idx: status}. Raises on
    a subprocess crash."""
    cmd = [
        sys.executable, os.path.join(REPO, "tools", "run_1096_canonical.py"),
        "--ids", ids, "--criterion", "full_trace", "--spec-k", "0",
        "--chunk", str(chunk), "--output", out_path,
    ]
    log_path = out_path + ".log"
    print(f"[flip] {tag}: {' '.join(cmd[1:])}", flush=True)
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(out_path):
        # Surface the tail of the log so a crash is debuggable.
        tail = ""
        try:
            with open(log_path) as fh:
                tail = "".join(fh.readlines()[-15:])
        except OSError:
            pass
        raise RuntimeError(f"{tag} runner exited rc={rc}; log tail:\n{tail}")
    with open(out_path) as fh:
        data = json.load(fh)
    return {x["idx"]: x["status"] for x in data["results"]}


def _cluster_of(idx: int) -> str:
    """Cluster name for a corpus id (strips the trailing _N)."""
    try:
        from tests.test_suite_1000 import generate_test_programs
        progs = generate_test_programs()
        desc = progs[idx][2].split(":")[0]
        return re.sub(r"_\d+$", "", desc)
    except Exception:
        return f"id{idx}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/tmp/c4cache_flip")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--full-ids", default="0-1095")
    ap.add_argument("--golden", type=int, default=468, help="golden pass count to compare against")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--mem-timeout", type=float, default=900.0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--output", default="/tmp/flip_full.json")
    ap.add_argument("--skip-canary", action="store_true", help="(debug) skip the canary gate")
    args = ap.parse_args()

    # Campaign config + isolated, flag-consistent caches. The disk cache key is
    # NOT flag-aware, so a fresh dedicated dir is mandatory for a clean build.
    env = dict(os.environ)
    env["C4_NO_STACK0_EMIT"] = "1"
    env["C4_OPERAND_FROM_MEMSP"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["C4_VM_CACHE_DIR"] = args.cache_dir
    env["TORCHINDUCTOR_CACHE_DIR"] = args.cache_dir + "_inductor"

    print("=" * 70)
    print(f"[flip] CAMPAIGN flip measurement  gpu={args.gpu}  cache={args.cache_dir}")
    print(f"[flip] golden baseline = {args.golden}")
    print("=" * 70, flush=True)

    # Fresh caches — never reuse a possibly-corrupt build.
    shutil.rmtree(args.cache_dir, ignore_errors=True)
    shutil.rmtree(args.cache_dir + "_inductor", ignore_errors=True)

    _wait_for_memory(args.min_free_gb, args.mem_timeout)

    # ---- Phase 1: CANARY GATE (builds + caches the model) -------------------
    if not args.skip_canary:
        canary_ids = ",".join(str(i) for i in CANARY_IDS)
        st = _run_canonical(canary_ids, "/tmp/flip_canary.json", env,
                            chunk=len(CANARY_IDS), tag="CANARY")
        bad = [i for i in CANARY_IDS if st.get(i) != "ok"]
        for i in CANARY_IDS:
            print(f"[flip] canary {CANARY_LABEL[i]:14s} (id {i}): {st.get(i, 'MISSING')}")
        if bad:
            print("=" * 70)
            print(f"❌ MEASUREMENT INVALID — build corrupt: canary FAILED for "
                  f"{[CANARY_LABEL[i] for i in bad]}.")
            print("   (A correct campaign build passes all four. Re-run in an "
                  "uncontended window / with more --min-free-gb.)")
            print("=" * 70)
            return 2
        print("[flip] ✅ canary PASSED — build is sound, proceeding to full sweep", flush=True)

    # ---- Phase 2: FULL SWEEP (cache-hits the verified build) ----------------
    st = _run_canonical(args.full_ids, args.output, env, chunk=args.chunk, tag="FULL")

    # Post-hoc re-check: the canary ids are inside the full sweep; if they
    # regressed here the sweep process corrupted independently → INVALID.
    full_canary_bad = [i for i in CANARY_IDS
                       if i in st and st.get(i) != "ok"]
    if full_canary_bad:
        print("=" * 70)
        print(f"❌ MEASUREMENT INVALID — full-sweep build corrupt: canary ids "
              f"{[CANARY_LABEL[i] for i in full_canary_bad]} failed in the sweep.")
        print("=" * 70)
        return 2

    # ---- Report -------------------------------------------------------------
    total = sum(1 for v in st.values() if v == "ok")
    err = sum(1 for v in st.values() if v == "error")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for idx, status in st.items():
        cl = _cluster_of(idx)
        agg[cl][0 if status == "ok" else 1] += 1

    print("=" * 70)
    delta = total - args.golden
    verdict = (f"🟢 CROSSED golden by +{delta}" if delta > 0
               else f"🔴 below golden by {-delta}" if delta < 0
               else "🟡 EXACTLY at golden")
    print(f"[flip] CAMPAIGN = {total}/{len(st)}   golden = {args.golden}   →  {verdict}")
    if err:
        print(f"[flip] (errors: {err} — counted as fail)")
    print("[flip] failing clusters (campaign):")
    for cl, (ok, f) in sorted(agg.items()):
        if f > 0:
            print(f"   {cl:18s} {ok:3d} ok / {f:3d} fail")
    print("=" * 70)
    print(f"[flip] FLIP DECISION: {'SHIP IT (campaign > golden → flip net-positive)' if delta > 0 else 'HOLD (campaign <= golden — keep closing the gap)'}")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
