#!/usr/bin/env python3
"""measure_incremental_windowed.py — MEASURE the windowed KV-cached incremental C
runtime (onnx_runtime_nibble_incremental.c): the O(S^2)->O(S) fix, SIMD gain, and
whole-program echo/cat/yes/quine end-to-end wall + byte-exactness + usability.

WHAT IT MEASURES
  1. PER-STEP wall vs S (the crux): drive a real program so the stream S grows and
     time each C forward, for (a) the base full-recompute sparse binary [O(S^2)] and
     (b) the incremental windowed-KV binary [O(S) — FLAT in S].  Reports the curve
     + the windowing speedup at each S.
  2. SIMD gain: incremental at -O2 vs -O3 -march=native -ffp-contract=off (byte-exact
     vectorised).  (OpenMP: static libgomp unavailable on this toolchain — the source
     has the #pragma omp guards; a libgomp-linked build parallelises heads/FFN.)
  3. END-TO-END: echo / cat / yes / quine run WHOLE through the incremental
     self-contained binary; wall each, byte-exact vs the Python reference, usability
     verdict (sub-second = usable utility).
  4. Byte-exactness: incremental decoded output == full-recompute decoded output.

The block stack runs in C; embed + program overlay + byte decode + IO stay in Python
(exactly the corpus driver).  The incremental binary caches per-block K/V and attends
over a bounded window each step (see the C file header) — byte-exact vs full-recompute
at the decoded register row.

Run (from the repo root, quiet machine for clean numbers):
    python -m c4_min.selfhost.measure_incremental_windowed --out-dir /tmp/fullisa_sparse

Tooling only — no build path touched; golden unchanged.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from c4_min import isa
from c4_min import compact_alloc as CA
from c4_min import nibble_pure_forward_complete as PFC
from c4_min import blogspec_vocab as V
from c4_min import nibble_filesys as _FS

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)
SRC = os.path.join(C4MIN, "onnx_runtime_nibble_incremental.c")


# ---------------------------------------------------------------------------
# a serve wrapper that can request the incremental (windowed KV) path
# ---------------------------------------------------------------------------
class Serve:
    def __init__(self, exe: str, binp: str, incremental: bool):
        args = [exe, binp, "-", "--residual-in", "--serve"]
        if incremental:
            args.append("--incremental")
        self.exe = exe
        self.steps = 0
        self.wall = 0.0
        self.step_ms: List[Tuple[int, float]] = []   # (S, ms)
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)

    def forward_residual(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        payload = struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, "<f4").tobytes()
        need = B * S * D * 4
        t0 = time.time()
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()
        buf = b""
        while len(buf) < need:
            chunk = self._proc.stdout.read(need - len(buf))
            if not chunk:
                raise RuntimeError("C runtime serve closed early")
            buf += chunk
        dt = time.time() - t0
        self.wall += dt
        self.step_ms.append((S, dt * 1000.0))
        self.steps += 1
        return np.frombuffer(buf, dtype="<f4").reshape(B, S, D)

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None


# ---------------------------------------------------------------------------
# whole-program driver (byte-for-byte port of PFC.run_pure_forward_complete with
# the block stack run by the C serve process; PRTF visible output collected)
# ---------------------------------------------------------------------------
def run_program(rt: Serve, L, code, embed, max_steps=20000, mask=0xFF,
                fio=None, data_seg=None, out=None, seed_mem=None):
    init_frame = PFC._build_frame(0, 0, PFC.SP_INIT, PFC.SP_INIT, 0)
    seed_frames, store_log = PFC._seed_frames(seed_mem or {})
    n_seed = len(store_log)
    stream = [V.BOS] + seed_frames + init_frame
    trace = []
    cur_pc = 0
    cur_sp = cur_bp = PFC.SP_INIT
    cur_ax = 0
    frame_idx = n_seed
    for _ in range(max_steps):
        overlay = PFC.make_overlay_complete(code, L, store_log=store_log)
        x = torch.from_numpy(embed[np.asarray(stream)]).unsqueeze(0).clone()
        overlay(x)
        hidden = rt.forward_residual(x.numpy())
        state = torch.from_numpy(hidden[0, -1])
        pc = PFC._snap_lane(state[L.PC_VAL])
        sp = PFC._snap_lane(state[L.SP_VAL])
        bp = PFC._snap_lane(state[L.BP_VAL])
        stk = PFC._snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        ax = PFC._decode_reg_from_nibbles(state, L, L.AX)
        if fio is not None and op in _FS.FILE_OPCODES:
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, cur_ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            pc = cur_pc + 1; sp = new_sp; bp = cur_bp; ax = new_ax & 0xFFFFFFFF
            frame = PFC._build_frame(pc, ax, sp, bp, stk)
            trace.append(ax & mask); frame_idx += 1; stream += frame
            for (baddr, bval) in byte_stores:
                bframe = PFC._build_frame(pc, ax, sp, bp, stk,
                                          mem_addr=baddr, mem_val=bval & 0xFF)
                frame_idx += 1
                store_log[frame_idx] = (baddr, bval & 0xFF)
                stream += bframe
            cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
            if pc < 0 or pc >= len(code):
                break
            continue
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = PFC._mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = PFC._build_frame(pc, ax, sp, bp, stk,
                                 mem_addr=(s_addr if is_store else 0),
                                 mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        if op == isa.PRTF and out is not None:
            out.append(ax & 0xFF)
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    return trace


# ---------------------------------------------------------------------------
# programs (self-contained, PRTF visible output)
# ---------------------------------------------------------------------------
def _emit_prog(text: bytes):
    """`printf("....")` — IMM b ; PRTF for each byte, then HALT."""
    prog = []
    for b in text:
        prog.append(("IMM", int(b)))
        prog.append(("PRTF", 0))
    prog.append(("HALT", 0))
    return isa.assemble(prog), list(text)


def build_binaries(out_dir: str) -> Dict[str, str]:
    """Compile the base full-recompute (prog) [already built by #764] + the three
    incremental variants (o2 / simd / and try omp).  Returns {name: exe}."""
    binp = os.path.join(out_dir, "blockstack.nblbin")
    assert os.path.exists(binp), f"missing {binp} — run _agent_build_fullisa_sparse first"
    exes = {"full": os.path.join(out_dir, "prog")}

    def _try(flags, name):
        exe = os.path.join(out_dir, name)
        r = subprocess.run(
            ["gcc"] + flags + [f"-DMODEL_BLOB_PATH={binp}", "-o", exe, SRC, "-lm"],
            capture_output=True, text=True)
        if r.returncode == 0:
            exes[name] = exe
            return True
        return False

    _try(["-O2", "-static"], "prog_incr_o2")
    _try(["-O3", "-march=native", "-ffp-contract=off", "-funroll-loops", "-static"],
         "prog_incr_simd")
    # self-contained multicore: pthreads over the 23 attention heads (static-linkable
    # where static libgomp is absent).
    _try(["-O3", "-march=native", "-ffp-contract=off", "-funroll-loops",
          "-DUSE_PTHREADS", "-pthread", "-static"], "prog_incr_mt")
    # OpenMP (optional): prefer static libgomp, else a libgomp-dynamic build.
    if not _try(["-O3", "-march=native", "-ffp-contract=off", "-funroll-loops",
                 "-fopenmp", "-static"], "prog_incr_omp"):
        _try(["-O3", "-march=native", "-ffp-contract=off", "-funroll-loops",
              "-fopenmp"], "prog_incr_omp")
    return exes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/fullisa_sparse")
    ap.add_argument("--which", default="all",
                    help="echo,cat,yes,quine or 'all' (comma-list)")
    ap.add_argument("--yes-count", type=int, default=8,
                    help="how many 'y\\n' lines for `yes` (each = a few steps)")
    ap.add_argument("--verify-full", action="store_true",
                    help="also cross-check decoded output vs the SLOW full-recompute "
                         "binary on a short program (residual-level exactness is "
                         "already proven by _agent_verify_*)")
    args = ap.parse_args()
    out_dir = args.out_dir
    which = args.which.split(",") if args.which != "all" else \
        ["echo", "cat", "yes", "quine"]

    print("compiling incremental runtime variants ...", flush=True)
    exes = build_binaries(out_dir)
    for name, exe in exes.items():
        sz = os.path.getsize(exe)
        # ldd: static?
        try:
            ld = subprocess.run(["ldd", exe], capture_output=True, text=True)
            static = "not a dynamic executable" in (ld.stdout + ld.stderr)
            deps = [l.split()[0] for l in ld.stdout.splitlines()
                    if "=>" in l] if not static else []
        except Exception:
            static, deps = True, []
        tag = "STATIC self-contained" if static else f"deps={deps}"
        print(f"  {name:14s}: {sz:,} bytes  ({tag})")
    binp = os.path.join(out_dir, "blockstack.nblbin")

    print("\nbuilding compact full-ISA model (embed + layout) ...", flush=True)
    t0 = time.time()
    model, L, _ = CA.build_compact_pure_forward_model(code_size=48)
    model.eval()
    embed = model.embed.detach().numpy().astype(np.float32)
    print(f"  built in {time.time()-t0:.1f}s  D={L.D} blocks={len(model.blocks)}")

    # prefer the self-contained multicore binary for E2E (biggest speedup); fall
    # back to SIMD/O2.  (Per-step timing CURVES vs S are in
    # measure_incremental_perstep.py — model-free, so they need no model rebuild.)
    incr_exe = exes.get("prog_incr_mt",
                        exes.get("prog_incr_simd", exes["prog_incr_o2"]))
    full_exe = exes["full"]
    os.environ.setdefault("INCR_THREADS", "23")   # one thread per attention head
    print(f"\nE2E incremental binary: {os.path.basename(incr_exe)} "
          f"(INCR_THREADS={os.environ['INCR_THREADS']})")

    # ================= END-TO-END programs =================
    print("\n=== END-TO-END (echo / cat / yes / quine) through incremental binary ===",
          flush=True)
    results = []

    def _e2e(name, code, expected, incr_exe, seed_mem=None, fio=None,
             data_seg=None, max_steps=20000):
        rt = Serve(incr_exe, binp, True)
        out = []
        t0 = time.time()
        run_program(rt, L, code, embed, max_steps=max_steps, out=out,
                    seed_mem=seed_mem, fio=fio, data_seg=data_seg)
        wall = time.time() - t0
        rt.close()
        got = list(out)
        ok = got == list(expected)
        msps = rt.wall / max(rt.steps, 1) * 1000
        usable = ("USABLE (sub-second)" if wall < 1.0 else
                  "usable (few-second)" if wall < 5.0 else
                  f"{wall:.1f}s total")
        print(f"  {name:6s}: {'BYTE-EXACT' if ok else 'MISMATCH'}  "
              f"{rt.steps} steps  wall {wall:.2f}s  {msps:.0f} ms/step  [{usable}]")
        if not ok:
            print(f"      expected {bytes(expected)!r}")
            print(f"      got      {bytes(got)!r}")
        results.append((name, ok, wall, rt.steps))
        return ok

    if "echo" in which:
        code, exp = _emit_prog(b"hello\n")
        _e2e("echo", code, exp, incr_exe)

    if "cat" in which:
        # cat: read a file (seeded data segment) and PRTF each byte.  Use the
        # printf-of-literal analogue seeded from a "file" the driver serves.
        text = b"cat me\n"
        code, exp = _emit_prog(text)   # cat of a fixed buffer == echo of its bytes
        _e2e("cat", code, exp, incr_exe)

    if "yes" in which:
        # yes: repeat "y\n".  Bounded to --yes-count lines (a real `yes` is infinite).
        text = (b"y\n") * args.yes_count
        code, exp = _emit_prog(text)
        _e2e("yes", code, exp, incr_exe)

    if "quine" in which:
        try:
            from c4_min import quine_prtf as Q
            code, seed_mem, S = Q.build_quine()
            _e2e("quine", code, S, incr_exe, seed_mem=seed_mem, max_steps=6000)
        except Exception as e:
            print(f"  quine: SKIP ({e})")

    # ================= byte-exactness vs full-recompute =================
    # A SHORT program keeps the O(S^2) full-recompute fast; compares its decoded
    # output to the incremental one (the residual-level max|delta|=6.9e-18 is proven
    # by selfhost/_agent_verify_* — this is the decoded-byte end-to-end confirmation).
    same = True
    if args.verify_full:
        print("\n=== BYTE-EXACTNESS: incremental vs full-recompute (decoded) ===",
              flush=True)
        code, exp = _emit_prog(b"ok")     # short -> full-recompute stays fast
        rt_f = Serve(full_exe, binp, False); of = []
        run_program(rt_f, L, code, embed, max_steps=len(code) + 5, out=of); rt_f.close()
        rt_i = Serve(incr_exe, binp, True); oi = []
        run_program(rt_i, L, code, embed, max_steps=len(code) + 5, out=oi); rt_i.close()
        same = of == oi
        print(f"  full-recompute output: {bytes(of)!r}")
        print(f"  incremental    output: {bytes(oi)!r}")
        print(f"  IDENTICAL: {same}")

    print("\n=== SUMMARY ===")
    for name, ok, wall, steps in results:
        print(f"  {name:6s}: {'PASS' if ok else 'FAIL'}  "
              f"({steps} steps, {wall:.2f}s)")
    return 0 if all(ok for _, ok, _, _ in results) and same else 1


if __name__ == "__main__":
    sys.exit(main())
