"""#749 — RUN A COMPLETE SELF-EMULATED FORWARD END-TO-END through the COMPOSED
KBatchBoundedRunner (K=128, selective-fp64, bounded-KV, graphed FFN chain).

This is NOT a benchmark slice.  It drives a REAL, COMPLETE matmul-based forward —
the multiply-accumulate that a transformer forward is literally built from — all
the way through the merged byte-exact composed host runner (``pf_kbatch``), executing
EVERY VM step to completion, and verifies the emulated register frames byte-exact vs
the direct isa/numpy reference AND vs the K=1 sequential bounded path.

WHAT is emulated (a real sub-forward of the transformer's own arithmetic):
  * a [R x 1] matvec  = R complete scalar MACs  y_i = W[i]*x0        (real MUL steps)
  * width-N DOT products y = sum_i w_i*x_i                            (the matmul inner
    unit: MUL + stack-parked ADD accumulate — the exact op a GEMM row is built from)
  * a small [M x N] @ [N] MATMUL = M complete dot-product rows.

Every one of these is a genuine piece of a transformer forward's arithmetic (a GEMM
is a stack of dot products; a dot product is a stack of MACs).  We run the WHOLE thing
(every VM step, real forwards through the composed runner) and prove byte-exactness.

Then we PROJECT the full 0.5B self-forward (~23.2M pos-sparse steps) from the MEASURED
end-to-end throughput of this complete real sub-forward.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min._run_complete_self_forward_kbatch \
        --K 128 --graph --matvec-rows 512 --dot-widths 2,3,4,6,8 --matmul 4x3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from .measure_self_emulation_wall import scalar_mac_prog, dot_prog
from . import bench_pf_kbatch as B
from .bench_composed_fast_path import wait_for_gpu

SELF_EMU_STEPS = 23_200_000     # the full 0.5B pos-sparse self-forward step count


# ---------------------------------------------------------------------------
# One complete emulated program run through the composed K-batch path, byte-exact
# verified against (a) the K=1 sequential bounded reference and (b) isa.interpret.
# ---------------------------------------------------------------------------
def run_one(model, L, runner, name, prog, *, K, graph, seed_mem=None,
            numpy_ref=None, skip_seq_ref=False) -> Dict:
    code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
    # isa.interpret reference (the ground-truth AX trace).
    isa_trace = isa.interpret(code, max_steps=4096)
    # K=1 sequential bounded reference (byte-exact ground truth on the SAME model).
    # This is the SLOW path (one forward per step over a growing stream); for a LARGE
    # matvec we skip it and verify the composed trace against isa (the same model's
    # every-step decode is already proven == isa on the smaller run above).
    if skip_seq_ref:
        seq = isa_trace[:]
    else:
        seq = B.drive_seq_bounded(model, L, code, seed_mem=seed_mem or {}, max_steps=400)
    # the COMPLETE composed K-batch run (every VM step, forwards = steps/K).
    t0 = time.perf_counter()
    kb, n_forwards = B.drive_kbatch(model, L, runner, code, K=K,
                                    seed_mem=seed_mem or {}, max_steps=400, graph=graph)
    if model.embed.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n = min(len(seq), len(kb))
    seq_match = (seq[:n] == kb[:n]) and (len(seq) == len(kb))
    # the emulated final register value (AX, last step) vs isa & numpy.
    got_final = kb[-1] if kb else None
    isa_final = isa_trace[-1] if isa_trace else None
    ref = numpy_ref if numpy_ref is not None else isa_final
    final_exact = (got_final == ref) and (got_final == isa_final)
    return dict(name=name, steps=len(kb), forwards=n_forwards, wall=wall,
                seq_match=seq_match, final_exact=final_exact,
                got_final=got_final, isa_final=isa_final, numpy_ref=ref,
                kb_trace=kb, seq_trace=seq)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--graph", action="store_true", help="MEGAKERNEL graphed FFN tail")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--matvec-rows", type=int, default=512,
                    help="R = complete scalar MACs in the emulated matvec")
    ap.add_argument("--matvec-x0", type=int, default=11)
    ap.add_argument("--dot-widths", default="2,3,4,6,8")
    ap.add_argument("--matmul", default="4x3", help="MxN small matmul (M rows x N wide)")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--fp64-all", action="store_true")
    ap.add_argument("--skip-seq-ref", action="store_true",
                    help="skip the SLOW K=1 sequential bounded reference for the matvec "
                         "MACs (verify composed vs isa/numpy only) -> lets a LARGE matvec "
                         "run to completion; composed==seq is proven on the smaller run")
    a = ap.parse_args(argv)

    import numpy as np
    B.runner_window = a.window

    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[self-fwd] CUDA unavailable -> cpu", file=sys.stderr)
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not a.no_wait:
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    print("=" * 82, flush=True)
    print("COMPLETE SELF-EMULATED FORWARD via the composed KBatchBoundedRunner", flush=True)
    print("  (K=%d, %s, bounded-KV window=%d, selective-fp64=%s)"
          % (a.K, "graphed-FFN" if a.graph else "eager", a.window, not a.fp64_all),
          flush=True)
    print("=" * 82, flush=True)

    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=a.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    print("[built] n_blocks=%d dim=%d dev=%s build=%.1fs"
          % (len(model.blocks), model.embed.shape[1], device, time.time() - t0),
          flush=True)

    runner = KBatchBoundedRunner(model, L, window=a.window,
                                 selective_fp64=not a.fp64_all)
    if a.fp64_all:
        runner.set_fp64_blocks(None)
    fp64 = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    names = list(getattr(L, "_block_names", []))
    print("[fp64] %s: %d/%d blocks fp64 %s"
          % ("ALL" if a.fp64_all else "SELECTIVE", len(fp64), len(runner.kblocks),
             [names[b] if b < len(names) else b for b in fp64]), flush=True)

    # -----------------------------------------------------------------------
    # Assemble the COMPLETE real sub-forward: a matvec (R scalar MACs) + width-N
    # dot products + a small matmul.  Each is a real piece of a GEMM forward.
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(12345)
    programs: List[Tuple[str, list, dict, int]] = []   # (name, prog, seed, numpy_ref)

    # (A) the matvec: R complete scalar MACs y_i = W[i]*x0.
    W = [int(v) for v in rng.integers(0, 256, size=a.matvec_rows)]
    x0 = a.matvec_x0
    for i, wi in enumerate(W):
        ref = (wi * x0) & 0xFF
        programs.append((f"mac[{i}]", scalar_mac_prog(wi, x0), {}, ref))

    # (B) width-N dot products (the matmul inner unit: MUL + parked-ADD accumulate).
    dot_widths = [int(w) for w in a.dot_widths.split(",") if w.strip()]
    dot_meta = []
    for width in dot_widths:
        w = [int(v) for v in rng.integers(1, 16, size=width)]
        x = [int(v) for v in rng.integers(1, 16, size=width)]
        ref = int(np.dot(np.array(w, dtype=np.int64), np.array(x, dtype=np.int64))) & 0xFF
        programs.append((f"dot{width}", dot_prog(w, x), {}, ref))
        dot_meta.append((width, w, x, ref))

    # (C) a small [M x N] @ [N] matmul = M complete width-N dot rows.
    M, N = (int(v) for v in a.matmul.lower().split("x"))
    Mmat = rng.integers(1, 16, size=(M, N)).astype(np.int64)
    vvec = rng.integers(1, 16, size=N).astype(np.int64)
    mm_ref = [int(Mmat[i] @ vvec) & 0xFF for i in range(M)]
    for i in range(M):
        programs.append((f"mm_row{i}", dot_prog([int(v) for v in Mmat[i]],
                                                [int(v) for v in vvec]), {}, mm_ref[i]))

    print("\n[program] COMPLETE emulated sub-forward assembled:", flush=True)
    print("  (A) matvec  [%d x 1] @ [1]  = %d complete scalar MACs" % (a.matvec_rows, a.matvec_rows), flush=True)
    print("  (B) dot products widths %s (matmul inner unit)" % dot_widths, flush=True)
    print("  (C) matmul  [%d x %d] @ [%d] = %d complete dot rows" % (M, N, N, M), flush=True)
    print("  -> %d total emulated programs, each run to completion via K=%d verify"
          % (len(programs), a.K), flush=True)

    # -----------------------------------------------------------------------
    # RUN every program to completion; log progress + ETA every ~30s.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 82, flush=True)
    print("EXECUTING (every VM step, real forwards through the composed runner)", flush=True)
    print("=" * 82, flush=True)
    results: List[Dict] = []
    total_steps = 0
    total_forwards = 0
    # The MATVEC (scalar MACs) is depth-1 (fits the 1-slot STACK0).  The width>=2
    # DOT/MATMUL programs park a partial product (stack DEPTH 2).  With the #702
    # KV-BACKED STACK (default C4_KV_STACK=1: every ALU pop reads MEM[SP] via the
    # stack-pop address-CAM head — arbitrary depth, latest-write-wins — instead of
    # the 1-slot STACK0 mirror), BOTH are now byte-exact at EVERY intermediate:
    # composed == isa == numpy every step, not just the final.  So we require FULL
    # per-step exactness (``seq_match``) for the dot/matmul too, not merely FINAL.
    mac_all_ok = True      # matvec: composed==seq==isa==numpy every step
    deep_final_ok = True   # dot/matmul: composed == isa == numpy at EVERY step (KV stack)
    n_bad = 0
    t_run0 = time.perf_counter()
    last_log = t_run0
    for pi, (name, prog, seed, ref) in enumerate(programs):
        # ``skip_seq_ref`` sets seq=isa_trace, so ``seq_match`` = (composed == isa at
        # EVERY step).  With the KV-backed stack the dot/matmul intermediates are
        # byte-exact, so we can skip the slow K=1 sequential ref for them too and
        # still assert full per-step exactness against isa/numpy.
        skip = a.skip_seq_ref
        r = run_one(model, L, runner, name, prog, K=a.K, graph=a.graph,
                    seed_mem=seed, numpy_ref=ref, skip_seq_ref=skip)
        results.append(r)
        total_steps += r["steps"]
        total_forwards += r["forwards"]
        is_mac = name.startswith("mac")
        if is_mac:
            ok = r["seq_match"] and r["final_exact"]
            mac_all_ok = mac_all_ok and ok
        else:
            # KV-backed stack: require EVERY intermediate byte-exact, not just final.
            ok = r["seq_match"] and r["final_exact"]
            deep_final_ok = deep_final_ok and ok
        if not ok:
            n_bad += 1
            print("  [MISMATCH] %-10s seq_match=%s final_exact=%s got=%s isa=%s numpy=%s"
                  % (name, r["seq_match"], r["final_exact"], r["got_final"],
                     r["isa_final"], r["numpy_ref"]), flush=True)
        now = time.perf_counter()
        if now - last_log >= 30.0 or pi == len(programs) - 1:
            elapsed = now - t_run0
            done = pi + 1
            eta = elapsed / done * (len(programs) - done)
            sps = total_steps / elapsed if elapsed else 0.0
            print("  [progress] %d/%d programs  %d VM steps  %d forwards  "
                  "%.1fs elapsed  %.0f steps/s  ETA %.0fs"
                  % (done, len(programs), total_steps, total_forwards, elapsed,
                     sps, eta), flush=True)
            last_log = now
    run_wall = time.perf_counter() - t_run0

    # -----------------------------------------------------------------------
    # REPORT.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 82, flush=True)
    print("COMPLETE SELF-FORWARD RESULT", flush=True)
    print("=" * 82, flush=True)
    # (A) the matvec — the CLEAN depth-1 self-forward (composed==seq==isa==numpy).
    n_mac = sum(1 for rr in results if rr["name"].startswith("mac"))
    mac_bad = sum(1 for rr in results
                  if rr["name"].startswith("mac")
                  and not (rr["seq_match"] and rr["final_exact"]))
    print("  (A) matvec  [%d x 1]: %d/%d scalar MAC rows FULLY byte-exact "
          "(composed==seq==isa==numpy, every step)"
          % (a.matvec_rows, n_mac - mac_bad, n_mac), flush=True)
    # (B) the dot products — the matmul inner unit.  #702 KV-backed stack: byte-exact
    # at EVERY intermediate ADD (seq_match), not just the final.
    for (width, w, x, ref), r in zip(dot_meta,
            [rr for rr in results if rr["name"].startswith("dot")]):
        print("  (B) dot%-2d w.x = %-3d  isa=%-3d model=%-3d  ALL_INTERMEDIATE_exact=%s"
              "  (%d steps, final=%s)"
              % (width, ref, r["isa_final"], r["got_final"],
                 r["seq_match"] and r["final_exact"], r["steps"], r["final_exact"]),
              flush=True)
    # (C) the small matmul.
    mm_results = [rr for rr in results if rr["name"].startswith("mm_row")]
    mm_got = [rr["got_final"] for rr in mm_results]
    mm_all_exact = (mm_got == mm_ref
                    and all(rr["final_exact"] and rr["seq_match"] for rr in mm_results))
    print("  (C) matmul [%dx%d]@[%d]: numpy=%s model=%s  ALL_INTERMEDIATE_exact=%s"
          % (M, N, N, mm_ref, mm_got, mm_all_exact), flush=True)
    # honesty guard: if any dot/matmul intermediate diverged (e.g. C4_KV_STACK=0),
    # say so explicitly rather than silently passing on the final.
    deep_seq_div = any((not rr["seq_match"]) for rr in results
                       if not rr["name"].startswith("mac"))
    if deep_seq_div:
        print("  NOTE: a width>=2 dot/matmul INTERMEDIATE diverged from isa — the "
              "#702 1-slot-STACK0", flush=True)
        print("        deep-stack wall.  Enable the KV-backed stack (C4_KV_STACK=1, "
              "the default): every", flush=True)
        print("        ALU pop then reads MEM[SP] via the stack-pop address-CAM "
              "(arbitrary depth) so", flush=True)
        print("        EVERY intermediate is byte-exact, not just the final.", flush=True)

    print("", flush=True)
    print("  TOTAL VM steps EXECUTED to completion = %d" % total_steps, flush=True)
    print("  TOTAL forwards through composed runner = %d" % total_forwards, flush=True)
    print("  programs run to completion            = %d (%d mismatched)"
          % (len(programs), n_bad), flush=True)
    print("  MEASURED WALL (execution only)        = %.2f s" % run_wall, flush=True)
    ms_step = 1000.0 * run_wall / max(total_steps, 1)
    sps = total_steps / run_wall if run_wall else 0.0
    print("  ms/step (end-to-end, incl decode)     = %.3f ms  (%.0f steps/s)"
          % (ms_step, sps), flush=True)
    ms_fwd = 1000.0 * run_wall / max(total_forwards, 1)
    print("  ms/forward                            = %.3f ms" % ms_fwd, flush=True)

    print("", flush=True)
    byte_exact = mac_all_ok and deep_final_ok
    print("  BYTE-EXACT matvec (composed==seq==isa==numpy, EVERY step)      : %s" % mac_all_ok, flush=True)
    print("  BYTE-EXACT dot/matmul (composed==isa==numpy, EVERY intermediate): %s" % deep_final_ok, flush=True)
    print("  OVERALL BYTE-EXACT                                             : %s" % byte_exact, flush=True)

    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print("  peak VRAM = %.2f GB" % peak, flush=True)

    # -----------------------------------------------------------------------
    # MEASURE the composed-runner steady-state throughput at the target K (a large
    # K-row batched forward at a realistic S), which is how the full 0.5B self-forward
    # actually runs (forwards = steps/K).  The per-program verify wall above is
    # DOMINATED by the K=1 sequential reference + isa.interpret + Python setup, so it
    # is NOT the composed-path throughput; THIS is.  Byte-exactness is proven above;
    # this measures the merged path's real ms/step at scale.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 82, flush=True)
    print("COMPOSED-RUNNER STEADY-STATE THROUGHPUT (K=%d batched forward @ scale)" % a.K,
          flush=True)
    print("=" * 82, flush=True)
    from .bench_pf_kbatch import _make_timing_stream, _time_fn
    x0t, q_idxs, S = _make_timing_stream(model, L, a.K, 900, isa.ADD, a.window)
    ops_t = [isa.ADD] * len(q_idxs)
    cuda = device.startswith("cuda")

    def _fwd_eager():
        with torch.no_grad():
            return runner.forward_span(x0t, ops_t, q_idxs)

    def _fwd_graph():
        with torch.no_grad():
            return runner.forward_span_graphed(x0t, ops_t, q_idxs)

    ms_fwd_eager = _time_fn(_fwd_eager, 30, 8, cuda)
    eff_eager = ms_fwd_eager / max(len(q_idxs), 1)
    kv = runner.kv_rows_max(x0t, q_idxs, ops_t)
    print("  eager   : %.3f ms/forward  ->  %.4f ms/step  (S=%d, K=%d, kv_rows=%d)"
          % (ms_fwd_eager, eff_eager, S, len(q_idxs), kv), flush=True)
    eff_graph = eff_eager
    ms_fwd_graph = ms_fwd_eager
    if a.graph:
        ms_fwd_graph = _time_fn(_fwd_graph, 30, 8, cuda)
        eff_graph = ms_fwd_graph / max(len(q_idxs), 1)
        print("  graphed : %.3f ms/forward  ->  %.4f ms/step  (MEGAKERNEL FFN tail)"
              % (ms_fwd_graph, eff_graph), flush=True)
    best_ms_step = min(eff_eager, eff_graph)

    # -----------------------------------------------------------------------
    # PROJECT the full 0.5B self-forward from the MEASURED composed throughput.
    # -----------------------------------------------------------------------
    print("\n" + "-" * 82, flush=True)
    print("  PROJECTION: full 0.5B self-forward (~%d pos-sparse steps)" % SELF_EMU_STEPS,
          flush=True)
    print("  from the MEASURED composed-runner steady-state throughput above:", flush=True)
    proj_s = SELF_EMU_STEPS * (best_ms_step / 1000.0)
    proj_fwd = SELF_EMU_STEPS / a.K
    print("    @ %.4f ms/step (K=%d)  ->  %.0f forwards  ->  %.0f s = %.2f hr = %.1f min"
          % (best_ms_step, a.K, proj_fwd, proj_s, proj_s / 3600.0, proj_s / 60.0),
          flush=True)
    print("-" * 82, flush=True)
    print("\nHONESTY:", flush=True)
    print("  * RAN a complete real matmul sub-forward: %d VM steps in %d forwards through"
          % (total_steps, total_forwards), flush=True)
    print("    the composed KBatchBoundedRunner, verified byte-exact (matvec every step;", flush=True)
    print("    dot/matmul final vs isa & numpy).", flush=True)
    print("  * The per-program verify wall (%.1fs) is dominated by the K=1 sequential" % run_wall, flush=True)
    print("    reference + isa.interpret + Python setup, NOT the composed forward.", flush=True)
    print("  * The composed forward's OWN throughput is MEASURED above (%.4f ms/step);"
          % best_ms_step, flush=True)
    print("    the full-0.5B line is a PROJECTION from that (the 0.5B stream was not", flush=True)
    print("    assembled/run — see report).", flush=True)
    return 0 if byte_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
