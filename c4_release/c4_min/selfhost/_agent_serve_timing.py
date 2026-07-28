#!/usr/bin/env python3
"""_agent_serve_timing.py — measure per-step wall in SERVE mode (amortized graph
load) for dense vs sparse on a saved frame, plus the one-shot load time.
Args: out_dir frame_index reps"""
import os, struct, subprocess, sys, time
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
fi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
binp = os.path.join(out_dir, "blockstack.nblbin")
npz = np.load(os.path.join(out_dir, "frames.npz"))
x = npz[f"frame_{fi}"]; B, S, D = x.shape
print(f"frame {fi}: B={B} S={S} D={D}, reps={reps}", flush=True)
payload = struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, "<f4").tobytes()
need = B * S * D * 4


def serve_time(exe):
    t_load0 = time.time()
    proc = subprocess.Popen([os.path.join(out_dir, exe), binp, "-", "--residual-in", "--serve"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def one():
        proc.stdin.write(payload); proc.stdin.flush()
        buf = b""
        while len(buf) < need:
            c = proc.stdout.read(need - len(buf))
            if not c:
                raise RuntimeError("closed")
            buf += c
        return buf
    one()  # first call includes graph load lazily? no—load is at startup; warmup anyway
    load_wall = time.time() - t_load0
    t0 = time.time()
    for _ in range(reps):
        one()
    dt = (time.time() - t0) / reps
    proc.stdin.close(); proc.wait(timeout=10)
    return dt, load_wall


for exe in ("rt_sparse", "rt_dense"):
    dt, lw = serve_time(exe)
    print(f"  {exe:10s}: {dt*1000:10.2f} ms/step   (startup+first-step {lw:.2f}s)", flush=True)
