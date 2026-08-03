"""Verify + measure the SELF-EMU direct-CAM (C4_SELFEMU_DIRECT_CAM) fix.

(1) BYTE-EXACT: the K-batch trace with direct-CAM ON == the K=1 sequential bounded
    reference (softmax path) == the direct-CAM OFF K-batch, on the battery + a deep
    nested loop, for K in {1,8,32}.
(2) MEASURE the store-row-growth attention: time forward_span at growing store-row
    counts (n_store) with direct-CAM OFF (softmax over stores, O(n_store)) vs ON
    (O(1) gather).  Report the per-step speedup + whether the O(n_store) softmax is
    collapsed.

Lean streaming build only.  CUDA_VISIBLE_DEVICES respected.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_POS_SPARSE"] = "1"

import torch

from . import isa
from .compact_alloc import build_compact_sparse_streaming
from .pf_kbatch import KBatchBoundedRunner
from . import bench_pf_kbatch as B


def _meminfo_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return 999.0


def build(device, code_size=64):
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    print("[built] n_blocks=%d dim=%d dev=%s build=%.1fs MemAvail=%.1fGB"
          % (len(model.blocks), model.embed.shape[1], device,
             time.time() - t0, _meminfo_gb()), flush=True)
    return model, L


def verify(model, L, device):
    print("\n=== BYTE-EXACT VERIFY (direct-CAM ON vs seq/softmax reference) ===",
          flush=True)
    B.runner_window = 64
    all_ok = True
    for K in (1, 8, 32):
        runner = KBatchBoundedRunner(model, L, window=64)
        n_prog = 0
        n_ok = 0
        for name, prog, seed in B._battery():
            code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
            # softmax reference (K=1 sequential bounded).  direct-CAM overlay in
            # drive_seq_bounded patches the decode band identically either way, so
            # this is the same reference the vanilla path uses.
            ref = B.drive_seq_bounded(model, L, code, seed_mem=seed)
            # direct-CAM ON
            os.environ["C4_SELFEMU_DIRECT_CAM"] = "1"
            kb_on, _ = B.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed)
            # direct-CAM OFF (the vanilla store-row softmax path)
            os.environ["C4_SELFEMU_DIRECT_CAM"] = "0"
            kb_off, _ = B.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed)
            n = min(len(ref), len(kb_on))
            ok = (kb_on[:n] == ref[:n] and len(kb_on) == len(ref)
                  and kb_on == kb_off)
            n_prog += 1
            n_ok += int(ok)
            all_ok = all_ok and ok
            if not ok:
                print("    [MISMATCH K=%d] %-10s" % (K, name), flush=True)
                print("      REF   =%s" % ref, flush=True)
                print("      DCAM  =%s" % kb_on, flush=True)
                print("      OFF   =%s" % kb_off, flush=True)
        # deep nested loop vs free draft
        deep = B._nested_prog(3, 4)
        draft = B.draft_pf_program(deep, max_steps=60, mask=0xFF)
        draft_ax = [f["ax"] & 0xFF for f in draft.frames]
        os.environ["C4_SELFEMU_DIRECT_CAM"] = "1"
        kb_on, _ = B.drive_kbatch(model, L, runner, deep, K=K, seed_mem={}, max_steps=60)
        n = min(len(kb_on), len(draft_ax))
        dok = kb_on[:n] == draft_ax[:n] and n > 0
        all_ok = all_ok and dok
        print("    K=%2d  battery %d/%d OK   nested_deep=%s (%d steps)"
              % (K, n_ok, n_prog, dok, n), flush=True)
    os.environ["C4_SELFEMU_DIRECT_CAM"] = "0"
    print("  OVERALL BYTE-EXACT: %s" % all_ok, flush=True)
    return all_ok


def _time_forward(runner, x, ops, q_idxs, cuda, n=20, warmup=5):
    for _ in range(warmup):
        runner.forward_span(x, ops, q_idxs)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        runner.forward_span(x, ops, q_idxs)
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def measure(model, L, device):
    """Time the K-batch forward at GROWING store-row counts, direct-CAM ON vs OFF.

    We drive a real store-heavy program (repeated SI to distinct addresses) so the
    store set GROWS, and time forward_span with the global heads doing softmax-over-
    stores (OFF) vs O(1) gather (ON)."""
    print("\n=== MEASURE: store-row-growth attention (softmax vs O(1) gather) ===",
          flush=True)
    cuda = device.startswith("cuda")
    from .selfemu_direct_cam import (build_direct_cam_table)
    from .nibble_pure_forward_complete import make_overlay_complete
    from .nibble_pure_forward import N_ROLES

    K = 32
    runner = KBatchBoundedRunner(model, L, window=64)
    B.runner_window = 64

    # Build a memory-write-heavy program: push a value and SI to a growing address,
    # repeated, so n_store grows.  We synthesise the stream directly (store mask +
    # role tags) at target S values -- the exact residual values do not affect the
    # attention TIME, only the store-row COUNT drives the softmax GEMM.
    dev = model.embed.device
    D = model.embed.shape[1]
    ops = [isa.ADD] * K

    print("  %-10s %12s %12s %10s %12s   %s" % ("n_store", "OFF ms/fwd", "ON ms/fwd",
                                           "speedup", "ON ms/step", "VRAM off/on GB"),
          flush=True)
    results = []
    # PACK many store rows into a bounded resident window: the real self-emu uses
    # eviction to keep the resident stream bounded, but the GLOBAL CAM softmax scores a
    # [Q, n_store] matrix that grows with the WORKING SET (number of distinct live
    # stores).  We hold S bounded (safe input) and pack up to n_store store rows into
    # it (1 store per 2 rows) so the softmax-over-stores cost grows while the input
    # tensor stays modest.
    for n_store in (32, 128, 512, 2048, 8192, 32768, 131072):
        pack = 2                                   # store rows packed every `pack` rows
        S = max(n_store * pack + 200, K * 30 + 200)
        if S > 700000:                             # ~700k rows x 1840 fp32 ~= 5GB input
            print("  (n_store=%d needs S=%d rows -> skipped, input too large)"
                  % (n_store, S), flush=True)
            break
        x = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)
        x[0, :, int(L.ONE)] = 1.0
        # pack n_store store rows near the tail (all <= the query rows).
        srows = list(range(1, min(S - 40, n_store * pack), pack))
        for r in srows:
            x[0, r, int(L.IS_STORE)] = 1.0
        q_idxs = [S - 1 - 30 * (K - 1 - j) for j in range(K)]
        q_idxs = [q for q in q_idxs if q >= 0]
        for q in q_idxs:
            for role in range(N_ROLES):
                x[0, q, L.ROLE + role] = 1.0

        # OFF: vanilla softmax over stores.  (May OOM at large n_store -> that IS the
        # O(n_store) VRAM ceiling; catch it and report OOM.)
        runner.arm_direct_cam(None)
        ms_off = vram_off = float("nan")
        oom_off = False
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            ms_off = _time_forward(runner, x, ops, q_idxs, cuda)
            vram_off = (torch.cuda.max_memory_allocated() / 1e9) if cuda else 0.0
        except torch.cuda.OutOfMemoryError:
            oom_off = True
            if cuda:
                torch.cuda.empty_cache()

        # ON: O(1) gather.  Build a resolved table that maps each query pos to a pop
        # read (so the pop head gathers) -- the TIMING is layout-driven, independent
        # of the actual value.
        tbl = build_direct_cam_table(
            {q: [_mk_resolved(q)] for q in q_idxs})
        runner.arm_direct_cam(tbl)
        ms_on = vram_on = float("nan")
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            ms_on = _time_forward(runner, x, ops, q_idxs, cuda)
            vram_on = (torch.cuda.max_memory_allocated() / 1e9) if cuda else 0.0
        except torch.cuda.OutOfMemoryError:
            if cuda:
                torch.cuda.empty_cache()
        runner.arm_direct_cam(None)

        spd = (ms_off / ms_on) if (ms_on and not oom_off and ms_off == ms_off) else float("nan")
        off_s = "OOM" if oom_off else ("%.3f" % ms_off)
        results.append((len(srows), ms_off, ms_on, spd))
        print("  %-10d %12s %12.3f %9s %12.4f   %5s / %5.2f"
              % (len(srows), off_s, ms_on,
                 ("%.2fx" % spd) if spd == spd else "-",
                 ms_on / max(len(q_idxs), 1),
                 ("OOM" if oom_off else "%.2f" % vram_off), vram_on),
              flush=True)
        del x
        if cuda:
            torch.cuda.empty_cache()
    return results


class _RR:
    __slots__ = ("head", "value")
    def __init__(self, head, value):
        self.head = head
        self.value = value


def _mk_resolved(q):
    # a "pop" read of some value -> exercises the pop global head's direct gather.
    return _RR("pop", (q * 7) & 0xFF)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--skip-measure", action="store_true")
    a = ap.parse_args(argv)
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[dcam] CUDA unavailable -> cpu", file=sys.stderr)
        device = "cpu"
    if _meminfo_gb() < 25:
        print("[dcam] MemAvailable < 25GB -> abort (memory discipline)", file=sys.stderr)
        return 2
    model, L = build(device, code_size=a.code_size)
    ok = verify(model, L, device)
    if not a.skip_measure:
        measure(model, L, device)
    print("\nRESULT byte_exact=%s" % ok, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
