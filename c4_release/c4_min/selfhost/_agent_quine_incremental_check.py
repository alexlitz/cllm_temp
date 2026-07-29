#!/usr/bin/env python3
"""_agent_quine_incremental_check.py — run the QUINE (S~1651 from step 0) through the
incremental windowed C runtime for a BOUNDED number of steps, confirm the emitted
bytes are byte-exact to the quine's own source prefix, and separate the C-forward
wall from the Python driver (overlay/torch) overhead.

The quine is the S=1651-scale program that made the base full-recompute runtime
intractable (~46 min/step extrapolated).  This shows the incremental path runs it at
~seconds/C-step and BYTE-EXACT.  Args: out_dir [n_steps]
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from c4_min import compact_alloc as CA
from c4_min import nibble_pure_forward_complete as PFC
from c4_min import blogspec_vocab as V
from c4_min import quine_prtf as Q
from c4_min import isa


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wt_fullisa"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    binp = os.path.join(out_dir, "blockstack.nblbin")
    exe = os.path.join(out_dir, os.environ.get("RT_EXE", "prog_incr_mt"))
    print(f"runtime: {exe}  INCR_THREADS={os.environ.get('INCR_THREADS','(default)')}")

    t0 = time.time()
    model, L, _ = CA.build_compact_pure_forward_model(code_size=48)
    model.eval()
    embed = model.embed.detach().numpy().astype(np.float32)
    print(f"model built {time.time()-t0:.0f}s  D={L.D}")

    code, seed_mem, S_src = Q.build_quine()
    print(f"quine: {len(code)} instrs, seed_mem {len(seed_mem)}, source {len(S_src)} bytes")

    # serve process (incremental)
    proc = subprocess.Popen([exe, binp, "-", "--residual-in", "--serve", "--incremental"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    def c_forward(x):
        B, Sn, D = x.shape
        payload = struct.pack("<iii", B, Sn, D) + np.ascontiguousarray(x, "<f4").tobytes()
        need = B * Sn * D * 4
        t = time.time()
        proc.stdin.write(payload); proc.stdin.flush()
        buf = b""
        while len(buf) < need:
            c = proc.stdout.read(need - len(buf))
            if not c:
                raise RuntimeError("closed")
            buf += c
        dt = time.time() - t
        return np.frombuffer(buf, dtype="<f4").reshape(B, Sn, D), dt

    init_frame = PFC._build_frame(0, 0, PFC.SP_INIT, PFC.SP_INIT, 0)
    seed_frames, store_log = PFC._seed_frames(seed_mem)
    n_seed = len(store_log)
    stream = [V.BOS] + seed_frames + init_frame
    out = []
    cur_pc = 0
    cur_sp = cur_bp = PFC.SP_INIT
    cur_ax = 0
    frame_idx = n_seed
    c_ms = []
    py_ms = []
    print(f"\nrunning {n_steps} quine steps (S starts ~{len(stream)}):", flush=True)
    for step in range(n_steps):
        tp = time.time()
        overlay = PFC.make_overlay_complete(code, L, store_log=store_log)
        x = torch.from_numpy(embed[np.asarray(stream)]).unsqueeze(0).clone()
        overlay(x)
        py_pre = time.time() - tp
        hidden, dt = c_forward(x.numpy())
        c_ms.append(dt * 1000)
        state = torch.from_numpy(hidden[0, -1])
        pc = PFC._snap_lane(state[L.PC_VAL]); sp = PFC._snap_lane(state[L.SP_VAL])
        bp = PFC._snap_lane(state[L.BP_VAL]); stk = PFC._snap_lane(state[L.STK_VAL])
        ax = PFC._decode_reg_from_nibbles(state, L, L.AX)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        s_addr = s_val = 0; is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = PFC._mem_top(store_log, cur_sp); s_val = ax & 0xFF
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFF
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = PFC._build_frame(pc, ax, sp, bp, stk,
                                 mem_addr=(s_addr if is_store else 0),
                                 mem_val=(s_val if is_store else 0))
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        if op == isa.PRTF:
            out.append(ax & 0xFF)
        py_ms.append((py_pre + (time.time() - tp - py_pre - dt)) * 1000)
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        print(f"  step {step:3d}  S={x.shape[1]:5d}  "
              f"C-forward {dt*1000:7.0f} ms  emitted={len(out)}", flush=True)
        if cur_pc < 0 or cur_pc >= len(code):
            break
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()

    # byte-exactness: the emitted bytes must be a prefix of the quine's source
    n = len(out)
    exact = out == S_src[:n]
    print(f"\nemitted {n} bytes; byte-exact-prefix vs quine source: {exact}")
    print(f"  emitted : {bytes(out)!r}")
    print(f"  expected: {bytes(S_src[:n])!r}")
    print(f"\nC-forward per step: median {np.median(c_ms):.0f} ms  "
          f"(min {min(c_ms):.0f}, max {max(c_ms):.0f}) at S~{x.shape[1]}")
    print(f"quine-scale S~{x.shape[1]}: base full-recompute would be "
          f"~{(x.shape[1]/211)**2 * 45.3:.0f}s/step (O(S^2) extrapolated from "
          f"45.3s@S211) -> incremental is ~{(x.shape[1]/211)**2*45300/np.median(c_ms):.0f}x faster")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
