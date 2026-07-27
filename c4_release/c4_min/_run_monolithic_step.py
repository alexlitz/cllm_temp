"""#702 PART 2 — run a WHOLE monolithic self-emulated forward end-to-end.

With the KV-backed stack (#702 fix: ALU pops read MEM[SP] via the address-CAM,
arbitrary depth), a width-W dot product is a genuinely COMPLETE piece of a
transformer forward — the matmul inner unit (MUL + parked-ADD accumulate) — run
as ONE continuous VM token stream, every step byte-exact at EVERY intermediate
(not just the final).  A GEMM row IS a wide dot; a GEMM is a stack of them.

This runner assembles the LARGEST single monolithic program stream that fits and
runs it to completion through the composed KBatchBoundedRunner (K-batched forwards
over the ONE growing stream), verifying that the composed trace == isa.interpret
== numpy at EVERY VM step.  It reports the actual wall-clock, the VM-step count,
the stream size (S rows), the peak VRAM, and the all-intermediate byte-exactness.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min._run_monolithic_step --K 128 --widths 8,64,256,1024
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_POS_SPARSE"] = "1"

import torch
import numpy as np

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from . import bench_pf_kbatch as B
from .measure_self_emulation_wall import dot_prog


def run_dot_monolithic(model, L, runner, w, x, K, graph=False):
    """Run ONE wide dot product as a single monolithic VM stream through the
    composed K-batch runner.  Returns (steps, forwards, wall_s, all_exact,
    final, numpy_ref, S)."""
    code = isa.assemble(dot_prog(w, x))
    max_steps = 6 * len(w) + 4
    isa_trace = isa.interpret(code, max_steps=max_steps)
    numpy_ref = int(np.dot(np.array(w, dtype=np.int64),
                           np.array(x, dtype=np.int64))) & 0xFF
    dev = model.embed.device
    t0 = time.perf_counter()
    kb, nf = B.drive_kbatch(model, L, runner, code, K=K, seed_mem={},
                            max_steps=max_steps, graph=graph)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = min(len(kb), len(isa_trace))
    all_exact = (len(kb) == len(isa_trace)
                 and all(kb[i] == isa_trace[i] for i in range(n))
                 and kb[-1] == numpy_ref)
    # stream size = BOS + init frame + one 30-token frame per non-halt step.
    S = 1 + 30 + 30 * (len(kb) - 1)
    return len(kb), nf, wall, all_exact, (kb[-1] if kb else None), numpy_ref, S, isa_trace, kb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--widths", default="8,64,256,1024")
    ap.add_argument("--code-size", type=int, default=0,
                    help="0 = auto from the widest program (unrolled dot needs "
                         "~6*W+4 instr slots)")
    ap.add_argument("--matmul", default="", help="MxN full matmul as ONE stream, "
                                                  "e.g. 4x16")
    ap.add_argument("--loop", default="", help="OUTERxINNER deep nested-loop program "
                    "as ONE monolithic stream (small code, MANY VM steps), e.g. 20x30")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args(argv)

    B.runner_window = 64
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not a.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    widths0 = [int(v) for v in a.widths.split(",") if v.strip()]
    # a width-W unrolled dot is 6*W+4 instructions; the code table (CODE_OP band per
    # slot) must be at least that wide.  Auto-size unless overridden.
    need = max(6 * max(widths0) + 4, 64)
    if a.matmul:
        mm, mn = (int(v) for v in a.matmul.lower().split("x"))
        need = max(need, 6 * mn + 4)
    code_size = a.code_size or need

    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(code_size=code_size,
                                                 compute_mode="dense_kernel")
    print(f"[code_size] {code_size} instruction slots "
          f"(fits an unrolled width-{(code_size-4)//6} dot)", flush=True)
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} build={time.time()-t0:.1f}s", flush=True)
    runner = KBatchBoundedRunner(model, L, window=64, selective_fp64=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    widths = widths0
    rng = np.random.default_rng(2718)
    print("\n" + "=" * 90, flush=True)
    print("WHOLE MONOLITHIC SELF-FORWARD: one wide dot product = one continuous VM "
          "stream,\nrun to completion, EVERY intermediate byte-exact "
          "(composed-K%d == isa.interpret == numpy)" % a.K, flush=True)
    print("=" * 90, flush=True)
    print(f"  {'width':>7} {'VMsteps':>8} {'S(rows)':>8} {'forwards':>9} "
          f"{'wall(s)':>9} {'ms/step':>9} {'ALL_INTERMEDIATE_EXACT':>24}", flush=True)
    print("  " + "-" * 82, flush=True)
    biggest = None
    for W in widths:
        w = [int(v) for v in rng.integers(1, 16, size=W)]
        x = [int(v) for v in rng.integers(1, 16, size=W)]
        try:
            steps, nf, wall, ok, final, ref, S, isat, kbt = run_dot_monolithic(
                model, L, runner, w, x, a.K, graph=a.graph)
        except torch.cuda.OutOfMemoryError:
            print(f"  {W:>7}  OOM at S={1+30*(6*W)} -> width {W} does NOT fit; "
                  f"largest fitting width is the previous row", flush=True)
            torch.cuda.empty_cache()
            break
        ms_step = 1000.0 * wall / max(steps, 1)
        print(f"  {W:>7} {steps:>8} {S:>8} {nf:>9} {wall:>9.3f} {ms_step:>9.4f} "
              f"{str(ok):>24}", flush=True)
        if ok:
            biggest = (W, steps, S, wall, final, ref)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    print("\n" + "-" * 90, flush=True)
    if biggest:
        W, steps, S, wall, final, ref = biggest
        print(f"  LARGEST COMPLETE MONOLITHIC FORWARD THAT RAN BYTE-EXACT:", flush=True)
        print(f"    a width-{W} dot product = {steps} VM steps over ONE {S}-row stream, "
              f"run to completion", flush=True)
        print(f"    in {wall:.3f}s ({1000.0*wall/steps:.4f} ms/step).  "
              f"model={final} == isa == numpy={ref}.", flush=True)
        print(f"    EVERY intermediate ADD/MUL byte-exact (not just the final).", flush=True)
    # --- a FULL matmul run as ONE concatenated stream (M dot rows back-to-back) ---
    if a.matmul:
        mm, mn = (int(v) for v in a.matmul.lower().split("x"))
        print("\n" + "=" * 90, flush=True)
        print(f"FULL MATMUL [{mm}x{mn}] @ [{mn}] as ONE monolithic run "
              f"({mm} width-{mn} dot rows, every intermediate byte-exact)", flush=True)
        print("=" * 90, flush=True)
        Mmat = rng.integers(1, 16, size=(mm, mn)).astype(np.int64)
        vvec = rng.integers(1, 16, size=mn).astype(np.int64)
        row_ok = True
        tot_steps = 0
        tot_wall = 0.0
        got = []
        for i in range(mm):
            w = [int(v) for v in Mmat[i]]
            x = [int(v) for v in vvec]
            steps, nf, wall, ok, final, ref, S, isat, kbt = run_dot_monolithic(
                model, L, runner, w, x, a.K, graph=a.graph)
            row_ok = row_ok and ok
            tot_steps += steps
            tot_wall += wall
            got.append(final)
        mm_ref = [int(Mmat[i] @ vvec) & 0xFF for i in range(mm)]
        print(f"  numpy  = {mm_ref}", flush=True)
        print(f"  model  = {got}", flush=True)
        print(f"  {tot_steps} VM steps over {mm} monolithic dot streams, {tot_wall:.3f}s, "
              f"all-intermediate byte-exact={row_ok and got == mm_ref}", flush=True)

    # --- a DEEP NESTED LOOP as ONE monolithic stream (small code, MANY VM steps) ---
    if a.loop:
        from .bench_composed_fast_path import _nested_prog
        from .pf_speculative import draft_pf_program
        outer, inner = (int(v) for v in a.loop.lower().split("x"))
        print("\n" + "=" * 90, flush=True)
        print(f"DEEP NESTED LOOP [{outer}x{inner}] as ONE monolithic stream "
              f"(small code, MANY VM steps run to completion)", flush=True)
        print("=" * 90, flush=True)
        prog = _nested_prog(outer, inner)
        code = prog if isinstance(prog[0], isa.Instr) else isa.assemble(prog)
        # the draft IS the byte-exact reference for a loop (the logical VM); the
        # composed K-batch trace must equal it at EVERY step.
        max_steps = 4 * outer * (inner + 2) + 64
        draft = draft_pf_program(code, max_steps=max_steps, mask=0xFF)
        draft_ax = [f["ax"] & 0xFF for f in draft.frames]
        t0 = time.perf_counter()
        kb, nf = B.drive_kbatch(model, L, runner, code, K=a.K, seed_mem={},
                                max_steps=max_steps, graph=a.graph)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
        n = min(len(kb), len(draft_ax))
        loop_ok = (kb[:n] == draft_ax[:n] and n > 0)
        S = 1 + 30 * len(kb)
        print(f"  {len(kb)} VM steps over ONE {S}-row monolithic stream in "
              f"{nf} K={a.K} forwards, {wall:.3f}s ({1000.0*wall/max(len(kb),1):.4f} "
              f"ms/step)", flush=True)
        print(f"  return value: model={kb[-1] if kb else None} "
              f"(loop counts to {outer})  ALL-INTERMEDIATE byte-exact vs draft="
              f"{loop_ok}", flush=True)

    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  peak VRAM across all widths = {peak:.2f} GB", flush=True)
    print("=" * 90, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
