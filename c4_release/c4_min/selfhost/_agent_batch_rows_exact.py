#!/usr/bin/env python3
"""_agent_batch_rows_exact.py — MODEL-FREE proof that the C runtime's batched-K
forward exposes, at each frame-end row, the SAME block-stack hidden as the
single-step forward's row -1 when the stream is truncated there.

This validates the ONLY C change (materialise every tail row) + the windowing
invariant under a MULTI-row jump: a speculative round appends K*30 rows in ONE
forward; row (base + i*30 + 29) must equal the row -1 of a stream ended there.

We feed identical synthetic append-only residual frames two ways and compare the
decode rows bit-for-bit.  No model, no torch — just the C serve process.

Run:  python -m c4_min.selfhost._agent_batch_rows_exact /tmp/fullisa_sparse
"""
from __future__ import annotations
import os
import struct
import subprocess
import sys

import numpy as np

FRAME = 30


def serve(exe, binp):
    return subprocess.Popen(
        [exe, binp, "-", "--residual-in", "--serve", "--incremental"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def fwd(p, x):
    B, S, D = x.shape
    p.stdin.write(struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, "<f4").tobytes())
    p.stdin.flush()
    need = B * S * D * 4
    buf = bytearray()
    while len(buf) < need:
        c = p.stdout.read(need - len(buf))
        if not c:
            raise RuntimeError("closed")
        buf += c
    return np.frombuffer(bytes(buf), "<f4").reshape(B, S, D)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
    binp = os.path.join(out_dir, "blockstack.nblbin")
    exe = os.path.join(out_dir, "prog_incr_simd")
    if not os.path.exists(exe):
        exe = os.path.join(out_dir, "prog_incr_o2")
    D = 1725
    K = 4
    base_S = 31                          # BOS + init frame ~= 31 rows
    rng = np.random.default_rng(1)

    # a fixed append-only stream: base rows + K frames of 30 rows each
    base = (rng.standard_normal((base_S, D)) * 0.01).astype("<f4")
    tail = (rng.standard_normal((K * FRAME, D)) * 0.01).astype("<f4")
    full = np.concatenate([base, tail], axis=0)      # [base_S + K*30, D]

    # ---- (1) SINGLE-STEP: grow one 30-row frame at a time, keep each row -1 ----
    p = serve(exe, binp)
    single_decode = []      # decoded row -1 after each of the K frame appends
    S = base_S
    x = base.copy()
    fwd(p, x[None])         # prime the base
    for i in range(K):
        x = full[: base_S + (i + 1) * FRAME]
        h = fwd(p, x[None])
        single_decode.append(h[0, x.shape[0] - 1].copy())   # row -1
    p.stdin.close(); p.wait(timeout=10)

    # ---- (2) BATCHED: prime the base, then append ALL K*30 rows in ONE forward ----
    p = serve(exe, binp)
    fwd(p, base[None])                    # same prime
    hb = fwd(p, full[None])               # one batched forward over base + K*30
    batch_decode = []
    for i in range(K):
        row = base_S + i * FRAME + (FRAME - 1)     # frame i's last row
        batch_decode.append(hb[0, row].copy())
    p.stdin.close(); p.wait(timeout=10)

    # ---- compare ----
    print(f"K={K}  base_S={base_S}  D={D}")
    all_ok = True
    for i in range(K):
        d = np.abs(single_decode[i] - batch_decode[i]).max()
        ok = d == 0.0
        all_ok = all_ok and ok
        print(f"  frame {i}: row {base_S + i*FRAME + FRAME - 1:4d}  "
              f"max|batch - single| = {d:.3e}  {'EXACT' if ok else 'DIFF'}")
    print(f"\n=== {'BYTE-EXACT (batched decode rows == single-step row -1)' if all_ok else 'MISMATCH'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
