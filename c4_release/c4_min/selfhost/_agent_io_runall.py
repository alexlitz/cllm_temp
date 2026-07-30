#!/usr/bin/env python3
"""_agent_io_runall.py — run the pre-built MODE-1/MODE-2 I/O binaries (from the
manifest _agent_io_bakeall / the fast-build step writes), compare RAW bytes to the
clean-room reference + expected, and MEASURE wall + VM steps.  One binary per call
(so each gets its own timeout under contention).  Byte-exact comparison in Python
(no shell $() newline stripping)."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time


def run_one(exe, io_burst, stdin_bytes, timeout=240):
    env = dict(os.environ)
    if io_burst:
        env["C4_IO_BURST"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run([exe, "--allc"], input=stdin_bytes,
                           capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return None, time.time() - t0, None, b"TIMEOUT"
    wall = time.time() - t0
    steps = None
    m = re.search(rb"allc: (\d+) VM steps", p.stderr)
    if m:
        steps = int(m.group(1))
    return p.stdout, wall, steps, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/tmp/io_bakeall/manifest.json")
    ap.add_argument("--only", default="", help="comma list of names to run")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out-json", default="/tmp/io_bakeall/runresults.json")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    only = set(x for x in args.only.split(",") if x)
    results = []
    for m in manifest:
        if only and m["name"] not in only:
            continue
        stdin_bytes = m["stdin"].encode("latin-1")
        expected = m["expected"].encode("latin-1")
        refout = m["refout"].encode("latin-1")
        out, wall, steps, stderr = run_one(
            m["exe"], io_burst=(m["io"] == "burst"),
            stdin_bytes=stdin_bytes, timeout=args.timeout)
        ok_exp = out == expected
        ok_ref = out == refout
        results.append(dict(name=m["name"], io=m["io"], out=None if out is None
                            else out.decode("latin-1"), wall=wall, steps=steps,
                            expected=m["expected"], ok_exp=ok_exp, ok_ref=ok_ref))
        print(f"{m['name']:6s} {m['io']:6s}: steps={steps} wall={wall:6.1f}s  "
              f"out={out!r}  exp={expected!r}  byte_exact={ok_exp} ref_ok={ok_ref}",
              flush=True)
    json.dump(results, open(args.out_json, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
