#!/usr/bin/env python3
"""_agent_verify_sparse_byteexact.py — verify the SPARSE whole-forward runtime is
BYTE-EXACT to the DENSE runtime (and to the torch reference) on the FULL-ISA
compact block-stack model, and measure per-step wall for both.

Uses the frames.npz dumped by _agent_build_fullisa_sparse (real pre-embedded VM
residual frames + torch block-stack reference outputs), so NO model rebuild.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)
DENSE_RT = os.path.join(C4MIN, "onnx_runtime_nibble.c")
SPARSE_RT = os.path.join(C4MIN, "onnx_runtime_nibble_sparse.c")


def _cc(csrc, exe):
    err = ""
    for flags in (["-O2", "-static", "-static-libgcc"], ["-O2", "-static"], ["-O2"]):
        r = subprocess.run(["gcc"] + flags + ["-o", exe, csrc, "-lm"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        err = r.stderr
    raise RuntimeError("gcc failed:\n" + err)


def _run_oneshot(exe, binp, x, stats=False):
    B, S, D = x.shape
    payload = struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, dtype="<f4").tobytes()
    with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as f:
        f.write(payload)
        fp = f.name
    try:
        args = [exe, binp, fp, "--residual-in"]
        if stats:
            args.append("--stats")
        r = subprocess.run(args, capture_output=True, check=True)
    finally:
        os.unlink(fp)
    out = np.frombuffer(r.stdout, dtype="<f4").reshape(B, S, D)
    return out, r.stderr.decode()


def _serve_time(exe, binp, x, reps=20):
    B, S, D = x.shape
    proc = subprocess.Popen([exe, binp, "-", "--residual-in", "--serve"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    payload = struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, "<f4").tobytes()
    need = B * S * D * 4

    def _one():
        proc.stdin.write(payload); proc.stdin.flush()
        buf = b""
        while len(buf) < need:
            chunk = proc.stdout.read(need - len(buf))
            if not chunk:
                raise RuntimeError("serve closed early")
            buf += chunk
    _one()  # warmup
    t0 = time.time()
    for _ in range(reps):
        _one()
    dt = (time.time() - t0) / reps
    proc.stdin.close(); proc.wait(timeout=10)
    return dt


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
    # optional: max #frames to check (dense forward is slow at large S), and max S
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    max_S = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    binp = os.path.join(out_dir, "blockstack.nblbin")
    npz = np.load(os.path.join(out_dir, "frames.npz"))
    n = int(npz["n"]); D = int(npz["D"])
    allframes = [npz[f"frame_{i}"] for i in range(n)]
    allrefs = [npz[f"ref_{i}"] for i in range(n)]
    frames, refs = [], []
    for f, r in zip(allframes, allrefs):
        if f.shape[1] <= max_S:
            frames.append(f); refs.append(r)
        if len(frames) >= max_frames:
            break
    n = len(frames)
    print(f"checking {n} frames (S<= {max_S}), D={D}, "
          f"S in {sorted(set(f.shape[1] for f in frames))}")

    dense_exe = os.path.join(out_dir, "rt_dense")
    sparse_exe = os.path.join(out_dir, "rt_sparse")
    print("compiling dense + sparse runtimes (static)...", flush=True)
    _cc(DENSE_RT, dense_exe)
    _cc(SPARSE_RT, sparse_exe)
    print(f"  dense:  {os.path.getsize(dense_exe):,} bytes")
    print(f"  sparse: {os.path.getsize(sparse_exe):,} bytes")

    max_ds = 0.0   # sparse vs dense
    max_dt = 0.0   # dense  vs torch-ref
    max_st = 0.0   # sparse vs torch-ref
    n_mismatch = 0
    stats_line = ""
    for fi, x in enumerate(frames):
        d_out, _ = _run_oneshot(dense_exe, binp, x)
        s_out, s_err = _run_oneshot(sparse_exe, binp, x, stats=(fi == 0))
        if fi == 0:
            for ln in s_err.splitlines():
                if "SPARSE_MATMUL_ITERS" in ln:
                    stats_line = ln.strip()
        ds = float(np.abs(d_out - s_out).max())
        dt = float(np.abs(d_out - refs[fi]).max())
        st = float(np.abs(s_out - refs[fi]).max())
        max_ds = max(max_ds, ds); max_dt = max(max_dt, dt); max_st = max(max_st, st)
        if ds != 0.0:
            n_mismatch += 1
            print(f"  frame {fi}: sparse!=dense max|d|={ds:.3e} shape={x.shape}")
    print(f"\nBYTE-EXACT over {n} frames:")
    print(f"  sparse vs dense  : max|delta| = {max_ds:.3e}  mismatches={n_mismatch}")
    print(f"  dense  vs torch  : max|delta| = {max_dt:.3e}  (fp accum-order noise)")
    print(f"  sparse vs torch  : max|delta| = {max_st:.3e}")
    print(f"  sparse-matmul iters: {stats_line}")

    x = frames[0]; B, S, _ = x.shape
    print(f"\nper-step wall (serve mode, B={B} S={S} D={D}, amortized load):")
    dt_d = _serve_time(dense_exe, binp, x)
    dt_s = _serve_time(sparse_exe, binp, x)
    print(f"  DENSE  : {dt_d*1000:9.3f} ms/step")
    print(f"  SPARSE : {dt_s*1000:9.3f} ms/step")
    print(f"  -> sparse/dense speedup = {dt_d/dt_s:.2f}x")
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
