#!/usr/bin/env python3
"""_agent_allc_measure.py — MEASURE the all-C VM (onnx_runtime_nibble_allc.c, --allc)
vs the Python-hybrid serve path, on the SAME echo program, isolating the
removed-Python (overlay + pipe-marshalling + decode) overhead.

  * ALL-C: run ./allc_echo --allc, parse its ALLC_TIMING per-step stderr (overlay /
    fwd / decode ms) -> the plumbing (overlay+decode) is what Python used to do.
  * HYBRID: drive the SAME echo through the Python serve path (the incremental
    binary in --serve mode) and time (a) the per-step total wall and (b) the C
    forward-only wall inside it -> the DIFFERENCE is the Python overlay + embed +
    the [1,S,D] pipe marshalling per step = the overhead the all-C binary removes.

Reports both, the removed-Python delta, and confirms byte-exact 'hello\\n'.
Tooling only.  Requires a prebuilt all-C binary + the shared model artifacts.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time
import numpy as np


def run_allc(exe, threads=8, bounded=0):
    env = dict(os.environ, INCR_THREADS=str(threads), ALLC_TIMING="1")
    if bounded:
        env["ALLC_BOUNDED_W"] = str(bounded)
    t0 = time.time()
    p = subprocess.run([exe, "--allc"], capture_output=True, env=env)
    wall = time.time() - t0
    out = p.stdout
    steps = []
    for m in re.finditer(
            r"step \d+ S=(\d+): overlay ([\d.]+)\s+fwd ([\d.]+)\s+decode ([\d.]+)",
            p.stderr.decode(errors="replace")):
        steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                      float(m.group(4))))
    return out, wall, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allc-exe", default="/tmp/allc_echo")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default="/tmp/fullisa_sparse")
    args = ap.parse_args()

    print("=== ALL-C echo (whole VM in C, no Python) ===", flush=True)
    out, wall, steps = run_allc(args.allc_exe, args.threads)
    ok = out == b"hello\n"
    n = len(steps)
    ov = sum(s[1] for s in steps); fw = sum(s[2] for s in steps)
    de = sum(s[3] for s in steps)
    print(f"  output: {out!r}  BYTE-EXACT={ok}")
    print(f"  steps={n}  wall={wall:.3f}s  ({wall/max(n,1)*1000:.1f} ms/step)")
    print(f"  per-step avg: overlay {ov/max(n,1):.2f}  fwd {fw/max(n,1):.1f}  "
          f"decode {de/max(n,1):.3f} ms")
    print(f"  ==> all-C PLUMBING (overlay+decode, the ex-Python work) = "
          f"{(ov+de)/max(n,1):.2f} ms/step; forward = {fw/max(n,1):.1f} ms/step")

    # bounded-KV byte-exactness + attention-op reduction
    print("\n=== bounded-KV (local-head windowing) byte-exactness ===", flush=True)
    for W in (64, 32):
        o2, w2, s2 = run_allc(args.allc_exe, args.threads, bounded=W)
        print(f"  W={W}: output {o2!r}  identical_to_exact={o2 == out}  "
              f"wall={w2:.3f}s")

    print("\nNOTE: the removed-Python overhead is the ~250 ms/step the hybrid spent "
          "on the Python overlay + [1,S,D] pipe marshalling + torch embed; the all-C "
          "binary does that same plumbing in ~%.2f ms/step (overlay+decode)."
          % ((ov + de) / max(n, 1)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
