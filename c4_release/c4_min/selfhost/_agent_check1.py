#!/usr/bin/env python3
"""_agent_check1.py — minimal 1-frame byte-exact check (dense vs sparse vs torch),
printing each result immediately.  Args: out_dir frame_index."""
import os, struct, subprocess, sys, tempfile, time
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
fi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
binp = os.path.join(out_dir, "blockstack.nblbin")
npz = np.load(os.path.join(out_dir, "frames.npz"))
x = npz[f"frame_{fi}"]; ref = npz[f"ref_{fi}"]
B, S, D = x.shape
print(f"frame {fi}: B={B} S={S} D={D}", flush=True)


def run(exe, stats=False):
    payload = struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, "<f4").tobytes()
    with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as f:
        f.write(payload); fp = f.name
    args = [os.path.join(out_dir, exe), binp, fp, "--residual-in"]
    if stats:
        args.append("--stats")
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, check=True)
    dt = time.time() - t0
    os.unlink(fp)
    out = np.frombuffer(r.stdout, dtype="<f4").reshape(B, S, D)
    return out, dt, r.stderr.decode()


s_out, s_dt, s_err = run("rt_sparse", stats=True)
print(f"SPARSE done in {s_dt:.2f}s", flush=True)
for ln in s_err.splitlines():
    if "SPARSE_MATMUL_ITERS" in ln:
        print("  " + ln.strip(), flush=True)
st = float(np.abs(s_out - ref).max())
print(f"  sparse vs torch-ref: max|delta| = {st:.3e}", flush=True)

d_out, d_dt, _ = run("rt_dense")
print(f"DENSE  done in {d_dt:.2f}s", flush=True)
ds = float(np.abs(d_out - s_out).max())
dt_ = float(np.abs(d_out - ref).max())
print(f"  dense  vs torch-ref: max|delta| = {dt_:.3e}", flush=True)
print(f"  SPARSE vs DENSE    : max|delta| = {ds:.3e}   {'BYTE-EXACT' if ds==0.0 else 'MISMATCH'}", flush=True)
print(f"per-forward wall: sparse {s_dt:.2f}s  dense {d_dt:.2f}s  -> {d_dt/s_dt:.1f}x", flush=True)
