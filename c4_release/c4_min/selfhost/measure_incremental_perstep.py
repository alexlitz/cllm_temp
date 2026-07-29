#!/usr/bin/env python3
"""measure_incremental_perstep.py — MODEL-FREE per-step wall curve for the windowed
KV-cached incremental C runtime vs the full-recompute sparse runtime.

The per-step COST does not depend on the residual VALUES — only on S (and the graph).
So we time the C binaries directly on SYNTHETIC append-only frames (real overlay-shaped
residual is not needed to measure wall time), avoiding the 137s model rebuild.  A frame
grows S by 30 each step exactly as the corpus driver does.

Reports, for each binary and each S:
  * full-recompute  [O(S^2)]  — capped at a modest S (the tail is ~minutes/step)
  * incremental o2  [windowed O(S)]  — FLAT in S
  * incremental simd[windowed O(S) + AVX] — the SIMD gain

Run:  python -m c4_min.selfhost.measure_incremental_perstep --out-dir /tmp/wt_fullisa
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import time
from typing import List

import numpy as np


def _feed(exe, binp, incremental, S_list, D, reps=1, warmup=True):
    """Spawn a serve process; feed frames of increasing S (append-only); return
    [(S, ms)].  Frames are random-but-fixed float32 (cost is value-independent)."""
    args = [exe, binp, "-", "--residual-in", "--serve"]
    if incremental:
        args.append("--incremental")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    rng = np.random.default_rng(0)
    out = []
    prev = None
    for S in S_list:
        # build an append-only frame: reuse the prefix rows, append fresh tail.
        if prev is None or prev.shape[0] >= S:
            x = rng.standard_normal((S, D)).astype("<f4") * 0.01
        else:
            add = rng.standard_normal((S - prev.shape[0], D)).astype("<f4") * 0.01
            x = np.concatenate([prev, add], axis=0)
        prev = x
        payload = struct.pack("<iii", 1, S, D) + np.ascontiguousarray(x).tobytes()
        need = S * D * 4

        def one():
            p.stdin.write(payload); p.stdin.flush()
            buf = b""
            while len(buf) < need:
                c = p.stdout.read(need - len(buf))
                if not c:
                    raise RuntimeError("closed")
                buf += c
        t0 = time.time()
        for _ in range(reps):
            one()
        dt = (time.time() - t0) / reps * 1000.0
        out.append((S, dt))
    p.stdin.close()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/wt_fullisa")
    ap.add_argument("--D", type=int, default=1725)
    ap.add_argument("--full-cap-S", type=int, default=211,
                    help="cap S for the O(S^2) full-recompute (its tail is minutes)")
    ap.add_argument("--incr-max-S", type=int, default=1651,
                    help="max S for the incremental curve (quine-scale)")
    args = ap.parse_args()
    binp = os.path.join(args.out_dir, "blockstack.nblbin")
    full = os.path.join(args.out_dir, "prog")
    o2 = os.path.join(args.out_dir, "prog_incr_o2")
    simd = os.path.join(args.out_dir, "prog_incr_simd")
    for e in (binp, full, o2, simd):
        assert os.path.exists(e), f"missing {e}"

    # full-recompute: a handful of points up to the cap (each is O(S^2))
    full_S = [s for s in (31, 61, 91, 151, 211, 301) if s <= args.full_cap_S]
    # incremental: append-only stream to quine scale
    incr_S = list(range(31, args.incr_max_S + 1, 30))

    print(f"D={args.D}  full-recompute S up to {full_S[-1]}  "
          f"incremental S up to {incr_S[-1]}\n")

    print("=== FULL-RECOMPUTE  [O(S^2)] ===", flush=True)
    fr = _feed(full, binp, False, full_S, args.D, reps=1)
    for S, ms in fr:
        print(f"  S={S:5d}  {ms:10.1f} ms/step")
    if len(fr) >= 2:
        (s0, m0), (s1, m1) = fr[0], fr[-1]
        print(f"  -> {m0:.0f}ms@S{s0} to {m1:.0f}ms@S{s1}: "
              f"{m1/m0:.1f}x slower for {s1/s0:.1f}x S "
              f"(quadratic ~= {(s1/s0)**2:.1f}x)")

    print("\n=== INCREMENTAL windowed  [O(S)] -O2 ===", flush=True)
    ir = _feed(o2, binp, True, incr_S, args.D, reps=1)
    show = ir[::max(1, len(ir) // 10)]
    for S, ms in show:
        print(f"  S={S:5d}  {ms:10.1f} ms/step")
    med = np.median([m for _, m in ir])
    print(f"  -> median {med:.1f} ms/step; "
          f"first {ir[0][1]:.1f}ms@S{ir[0][0]} last {ir[-1][1]:.1f}ms@S{ir[-1][0]}  "
          f"(growth {ir[-1][1]/max(ir[0][1],1e-9):.1f}x for {ir[-1][0]/ir[0][0]:.0f}x S)")

    print("\n=== INCREMENTAL windowed + SIMD (AVX) ===", flush=True)
    isd = _feed(simd, binp, True, incr_S, args.D, reps=1)
    show = isd[::max(1, len(isd) // 10)]
    for S, ms in show:
        print(f"  S={S:5d}  {ms:10.1f} ms/step")
    med_s = np.median([m for _, m in isd])
    print(f"  -> median {med_s:.1f} ms/step  (SIMD speedup over -O2: "
          f"{med/med_s:.2f}x)")

    # windowing speedup at the S values the full-recompute reached
    print("\n=== WINDOWING SPEEDUP (incremental-simd vs full-recompute, same S) ===")
    ir_map = {S: ms for S, ms in isd}
    for S, ms_full in fr:
        # nearest incremental S
        best = min(ir_map, key=lambda s: abs(s - S))
        ms_incr = ir_map[best]
        print(f"  S~{S:4d}: full {ms_full:9.1f} ms  incr-simd {ms_incr:7.1f} ms  "
              f"-> {ms_full/max(ms_incr,1e-9):6.1f}x faster")
    # extrapolate quine (S=1651)
    if fr:
        (s1, m1) = fr[-1]
        quad = m1 * (1651.0 / s1) ** 2
        print(f"\n  quine S=1651: full-recompute EXTRAPOLATED ~{quad/1000:.0f}s/step "
              f"(O(S^2) from {m1:.0f}ms@S{s1}); incremental-simd "
              f"~{ir_map.get(1651, med_s):.0f}ms/step MEASURED")


if __name__ == "__main__":
    main()
