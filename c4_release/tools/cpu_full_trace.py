#!/usr/bin/env python3
"""Truthful CPU full_trace verdict for specific programs (the framing self-check).

THIS is the tool a lane MUST use to self-check a framing / full_trace fix on
CPU. It runs the byte-identical CPU *autoregressive* decode
(:class:`neural_vm.verification.faithful_autoregressive.FaithfulAutoregressiveRunner`)
and reports the SAME per-program pass/fail verdict as
``tools/run_1096_canonical.py --criterion full_trace`` — but on CPU, in ~85s
per ~10-step program, with NO GPU.

Why NOT ``tools/interp_oracle_gate.py`` for framing fixes
---------------------------------------------------------
``interp_oracle_gate.py`` (and the FaithfulInterpreter single-forward it wraps)
RE-ANCHORS each step's register markers so it can attribute a wrong byte to its
owning rule. Re-anchoring deliberately HIDES the production decoder's
autoregressive framing desync: when a step emits 34 or 37 tokens instead of 35,
the production fixed-35-token slice (``batched_pure_neural`` ``_UNSAFE_OFFSETS``
+ ``_step_one``) misreads the NEXT step's PC and the run drifts. A re-anchored
single forward decodes every step independently, so it CANNOT see that
cumulative miscount — it reports PASS for programs the GPU fails. The canonical
false positive: ``func_identity`` "passes step 9" under the re-anchoring gate
but is 0/150 on GPU.

This runner closes that gap. It is a thin wrapper over
``FaithfulAutoregressiveRunner.run_batch_fail_fast`` whose verdict logic is the
UNMODIFIED production ``BatchedPureNeuralRunner`` decode (only the per-token
argmax source is swapped for the validated byte-identical CPU forward). It
feeds emitted tokens back as the next step's input — so the 34/37-token
miscount is reproduced EXACTLY when the neural model would miscount.

Trust envelope (validated 2026-06-18 vs GPU ground truth on current main,
golden ``b9d8861f`` flag-off, HINIB default-ON)
----------------------------------------------------------------------------
At ``--spec-k 0`` the CPU full_trace verdict is BIT-EXACT to the GPU
``run_1096_canonical --criterion full_trace`` verdict on the validation sample
(add / var / func / expr / if / edge_literal / gcd — pass AND fail cases),
INCLUDING the saturated-tie programs (``expr_paren`` / ``expr_mul_div``) that a
spec_k=32 CPU run reports as FAIL where the GPU passes. Running both at
``eff_spec_k=1`` (spec_k=0) removes the fused 32-ahead verify, so the CPU
one-step-per-forward argmax matches the GPU one-step decode at those
~1e22-logit, gap=0 positions. ``--spec-k 0`` is therefore the GPU-independent,
bit-exact full-corpus verdict; the legacy ``--spec-k 32`` default is kept only
for back-compat with the earlier validation runs.

GPU-INDEPENDENCE, not speed: the full corpus on ~16 CPU workers is ~1-2h vs
~30min on a dedicated GPU. The point is to make the full_trace corpus loop run
with NO GPU at all (the bake is ~15-40s one-time per worker; the per-token
forward is the real CPU ``model.forward`` over one row via
``ModelExactForward`` — bit-exact to the neural model, including the saturated
ties the old recovered-weight forward mis-decoded).

Pass criterion (identical to ``run_1096_canonical.py --criterion full_trace``)
------------------------------------------------------------------------------
PASS iff every completed VM step's decoded ``(PC, AX)`` matched the declarative
oracle through HALT (the production fail-fast verdict). A FAIL records the
divergence step + expected/got register state. ``error`` means the decode could
not produce a verdict (e.g. ran out of context room).

Usage
-----
    # Self-check a framing fix on a handful of programs (CPU; no GPU):
    python tools/cpu_full_trace.py --ids 550,275,800,250

    # GPU-bit-exact verdict (the saturated ties resolve at spec_k=0):
    python tools/cpu_full_trace.py --ids 825,850 --spec-k 0

    # Full-corpus CPU verdict across all cores (drop-in for the GPU gate):
    python tools/cpu_full_trace.py --all --spec-k 0 --workers 16 \
        --output /tmp/cpu_full_corpus.json

    # Ranges work too (like run_1096_canonical):
    python tools/cpu_full_trace.py --ids 1031-1035

Speed: ~0.47s/CPU forward, ~85s for a ~10-step diverging program (O(steps^2),
no KV cache). For the full corpus use ``--all --workers N`` (parallel across
cores); each worker bakes the model once then processes its chunk.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

# Force CPU before torch initialises a CUDA context. This is a CPU-only tool by
# construction (the faithful forward runs the recovered dense weights on CPU);
# pinning here also keeps the per-program speed measurement honest.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from tools.run_1096_canonical import (  # noqa: E402
    _cluster_breakdown,
    _print_cluster_table,
    _score_fail_fast_results,
    cluster_of,
)
from tools.run_1096_fast import (  # noqa: E402
    ProgramResult,
    _compile_and_oracle,
    _parse_ids,
)

# Default fail-fast speculation K.
#
# Historically pinned to 32 to match ``run_1096_canonical`` — whose chunk
# runner clamps ``spec_k=(spec_k if spec_k > 0 else 32)`` in fail-fast mode, so
# the GPU gate at ``--spec-k 0`` actually ran the fail-fast verify 32-ahead. We
# keep 32 as the DEFAULT for back-compat, but expose ``--spec-k`` so the CPU
# verdict can run at the SAME effective K as the GPU gate.
#
# ``--spec-k 0`` (or any value <= 0) flows to ``run_batch_fail_fast`` which
# clamps it internally to ``eff_spec_k=1`` (one VM step verified per forward —
# the raw, non-speculative decode). The per-step (PC, AX) VERDICT is
# K-independent by construction (the model is the arbiter at every slot, see
# the ``run_batch_fail_fast`` docstring), so the pass SET is identical for any
# K — EXCEPT at the documented saturated-tie positions (~1e22 logits, exact
# gap=0) where the argmax winner is fp-accumulation-order dependent: there the
# fused 32-ahead verify and the one-step-per-forward path can pick different
# ids. At ``eff_spec_k=1`` the CPU decode matches the GPU gate's saturated-tie
# verdicts (closing the residual CPU-vs-GPU disagreements), so ``--spec-k 0``
# is the GPU-bit-exact setting.
_DEFAULT_FULL_TRACE_SPEC_K = 32

# Skip (count as 'skipped', NEVER pass) any program whose declarative oracle
# step count exceeds this cap. Identical default + rationale to
# ``run_1096_canonical._DEFAULT_MAX_STEPS_CAP``: all 846 passable
# (non-diverging) short programs are <=39 steps, so the default 1000 drops NO
# pass; it folds the WHOLE deep diverging loop/gcd/rec band (233 programs at
# 41..8369 steps) into the run so each gets a real pass/fail verdict. The model
# forward has no position ceiling (ALiBi relative-distance; a 5000-token forward
# runs with no mask ceiling, validated 2026-07-03), so a deep program that does
# not hit an unfixed per-step root PASSES. On CPU (no KV cache) each deep member
# is MINUTES (O(steps^2)); use ``--max-steps-cap 40`` for the fast short-only
# run, or ``--max-steps-cap 0`` to disable the cap (the deepest rec_fib is hours
# on CPU). Programs above 1000 steps stay skipped so a full run never OOMs.
_DEFAULT_MAX_STEPS_CAP = 1000

# A "prepared" program entry (the tuple ``_compile_and_oracle`` yields):
#   (idx, suite_expected, description, decl_exit, decl_steps, bytecode, data)
PreparedEntry = Tuple[int, int, str, int, int, list, bytes]


def _select(all_tests, ids_spec: Optional[str], *, want_all: bool):
    """Select ``(idx, src, expected, desc)`` tuples for the requested ids.

    With ``want_all`` (``--all``) every program is selected, in id order. Else
    the comma/range ``ids_spec`` is parsed (deduped, request order preserved).
    """
    by_idx = {idx: (idx, src, exp, desc)
              for idx, (src, exp, desc) in enumerate(all_tests)}
    if want_all:
        return [by_idx[idx] for idx in sorted(by_idx)]
    wanted = list(dict.fromkeys(_parse_ids(ids_spec or "")))  # dedup, keep order
    out = []
    missing = []
    for idx in wanted:
        if idx in by_idx:
            out.append(by_idx[idx])
        else:
            missing.append(idx)
    if missing:
        print(f"[cpu-full-trace] WARNING: ids out of range (skipped): "
              f"{missing}", file=sys.stderr, flush=True)
    return out


def _split_over_cap(
    prepared: List[PreparedEntry],
    max_steps_cap: Optional[int],
) -> Tuple[List[PreparedEntry], List[ProgramResult]]:
    """Partition prepared programs into (runnable, skipped-over-cap).

    Mirrors ``run_1096_canonical._split_over_cap`` so the CPU verdict's skip
    set matches the GPU gate's exactly: a program is skipped iff its
    DECLARATIVE oracle step count exceeds the cap. Skipped programs become
    ``ProgramResult(status="skipped")`` — counted separately, NEVER as pass.
    """
    if not max_steps_cap or max_steps_cap <= 0:
        return list(prepared), []
    runnable: List[PreparedEntry] = []
    skipped: List[ProgramResult] = []
    for entry in prepared:
        idx, expected, description, decl_exit, decl_steps, _bc, _data = entry
        if decl_steps is not None and decl_steps > max_steps_cap:
            skipped.append(
                ProgramResult(
                    idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=decl_exit,
                    declarative_steps=decl_steps,
                    neural_exit=None,
                    status="skipped",
                    error=(
                        f"skipped: declarative steps {decl_steps} "
                        f"> --max-steps-cap {max_steps_cap}"
                    ),
                )
            )
        else:
            runnable.append(entry)
    return runnable, skipped


# ---------------------------------------------------------------------------
# Parallel CPU execution. Each worker process bakes the faithful runner ONCE
# (the ~15-40s cold bake), then decodes its whole chunk. The bake is the only
# cold cost; the per-program decode is the real per-row ``model.forward``
# (ModelExactForward, O(steps^2), no KV cache). We split the runnable programs into
# ``workers`` chunks and Pool-map ``_worker_run_chunk`` over them.
# ---------------------------------------------------------------------------

# Module-level per-worker runner cache. A Pool worker may be handed more than
# one chunk (if there are more chunks than workers); the cache makes the bake
# happen exactly once per OS process, not once per chunk.
_WORKER_RUNNER = None


def _get_worker_runner():
    global _WORKER_RUNNER
    if _WORKER_RUNNER is None:
        # CPU-only, single-thread per worker (we get parallelism from the
        # process pool, not from intra-op threads — oversubscription would
        # thrash). Pin BEFORE the heavy import inside the runner build.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:  # noqa: BLE001
            pass
        from neural_vm.verification.faithful_autoregressive import (
            FaithfulAutoregressiveRunner,
        )

        _WORKER_RUNNER = FaithfulAutoregressiveRunner()
    return _WORKER_RUNNER


def _worker_run_chunk(
    payload: Tuple[List[PreparedEntry], int, str],
) -> List[ProgramResult]:
    """Decode one chunk of prepared programs in a worker process.

    ``payload`` is ``(entries, spec_k, criterion)``. Returns the scored
    ``ProgramResult`` list. The runner is baked once per worker process (cached
    module-globally) and reused across every chunk the pool hands this worker.
    """
    entries, spec_k, criterion = payload
    runner = _get_worker_runner()
    out: List[ProgramResult] = []
    for entry in entries:
        _idx, _expected, _description, _decl_exit, decl_steps, bytecode, data = entry
        r = runner.run_batch_fail_fast(
            [bytecode], data_list=[data], max_steps=None,
            expected_steps_list=[decl_steps], spec_k=spec_k,
            criterion=criterion,
        )[0]
        out.append(_score_fail_fast_results([entry], [r], criterion=criterion)[0])
    return out


def _chunk_evenly(entries: List[PreparedEntry], n_chunks: int) -> List[List[PreparedEntry]]:
    """Split ``entries`` into ``n_chunks`` work-balanced chunks.

    Balance by total declarative steps (the CPU cost is ~O(steps^2) per
    program, but a steps-balanced greedy assignment keeps the slowest worker
    from hoarding all the deep programs). Sort by steps descending and
    round-robin onto the least-loaded chunk.
    """
    n_chunks = max(1, n_chunks)
    if n_chunks == 1 or len(entries) <= 1:
        return [list(entries)]
    ordered = sorted(entries, key=lambda e: (e[4] or 0) ** 2, reverse=True)
    buckets: List[List[PreparedEntry]] = [[] for _ in range(n_chunks)]
    loads = [0] * n_chunks
    for entry in ordered:
        j = min(range(n_chunks), key=lambda k: loads[k])
        buckets[j].append(entry)
        loads[j] += (entry[4] or 1) ** 2
    return [b for b in buckets if b]


def _run_serial(
    prepared: List[PreparedEntry], *, spec_k: int, criterion: str,
) -> Tuple[List[ProgramResult], int, float]:
    """Decode every prepared program in-process (1 worker). Returns
    ``(results, n_forwards, wall_seconds)``."""
    print("[cpu-full-trace] building CPU faithful autoregressive runner "
          "(cold model bake)...", file=sys.stderr, flush=True)
    from neural_vm.verification.faithful_autoregressive import (
        FaithfulAutoregressiveRunner,
    )

    t_build = time.monotonic()
    runner = FaithfulAutoregressiveRunner()
    print(f"[cpu-full-trace] runner ready ({time.monotonic() - t_build:.1f}s).",
          file=sys.stderr, flush=True)

    results: List[ProgramResult] = []
    n_fwd_total = 0
    t0 = time.monotonic()
    for n, entry in enumerate(prepared):
        idx, _expected, description, _decl_exit, decl_steps, bytecode, data = entry
        before = runner.forward_count
        pt0 = time.monotonic()
        r = runner.run_batch_fail_fast(
            [bytecode], data_list=[data], max_steps=None,
            expected_steps_list=[decl_steps], spec_k=spec_k,
            criterion=criterion,
        )[0]
        n_fwd = runner.forward_count - before
        n_fwd_total += n_fwd
        results.append(_score_fail_fast_results([entry], [r], criterion=criterion)[0])
        print(f"[cpu-full-trace]   {n + 1}/{len(prepared)} id={idx} "
              f"{cluster_of(description)} steps={decl_steps}: "
              f"{results[-1].status} ({n_fwd} fwd, {time.monotonic() - pt0:.1f}s)",
              file=sys.stderr, flush=True)
    return results, n_fwd_total, time.monotonic() - t0


def _run_parallel(
    prepared: List[PreparedEntry], *, workers: int, spec_k: int, criterion: str,
) -> Tuple[List[ProgramResult], float]:
    """Decode prepared programs across ``workers`` processes. Returns
    ``(results, wall_seconds)``.

    Each worker bakes the faithful runner ONCE then decodes its chunk. With
    more chunks than workers the pool reuses each worker's cached runner, so
    the bake still happens exactly ``workers`` times total.
    """
    chunks = _chunk_evenly(prepared, workers)
    payloads = [(chunk, spec_k, criterion) for chunk in chunks]

    # Warm the on-disk compile cache ONCE in the parent BEFORE spawning the
    # pool. Otherwise all ``workers`` workers call ``compile_full_vm_dynamic``
    # simultaneously, miss the cold cache, and bake the model in parallel —
    # each bake is multi-threaded, so N simultaneous bakes thrash the cores and
    # the time-to-first-result balloons (measured: 16 simultaneous bakes ~= 6
    # min). With the cache warm, each worker's build is a ~3 s ``torch.load``
    # instead of a ~40 s contended bake.
    print("[cpu-full-trace] warming the on-disk compile cache in the parent "
          "(so workers torch.load instead of all baking at once)...",
          file=sys.stderr, flush=True)
    t_warm = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from neural_vm.verification.faithful_autoregressive import build_cpu_model

    build_cpu_model(disk_cache=True)  # populates the disk cache; result dropped
    print(f"[cpu-full-trace] cache warm in {time.monotonic() - t_warm:.1f}s.",
          file=sys.stderr, flush=True)

    print(f"[cpu-full-trace] parallel decode: {len(prepared)} programs across "
          f"{len(chunks)} chunk(s) on {workers} worker(s) "
          f"(each worker loads the warmed model)...",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    results: List[ProgramResult] = []
    done = 0
    cum_pass = cum_fail = cum_err = 0
    # ``spawn`` avoids inheriting a fork-unsafe torch/CUDA state; the worker
    # builds its own CPU model. Each worker stays alive for multiple chunks
    # (maxtasksperchild=None) so the bake is once per worker.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for chunk_results in pool.imap_unordered(_worker_run_chunk, payloads):
            results.extend(chunk_results)
            done += len(chunk_results)
            for r in chunk_results:
                if r.status == "ok":
                    cum_pass += 1
                elif r.status == "error":
                    cum_err += 1
                else:
                    cum_fail += 1
            print(f"[cpu-full-trace] {done}/{len(prepared)} done | "
                  f"cum: pass={cum_pass} fail={cum_fail} err={cum_err} "
                  f"({time.monotonic() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    return results, time.monotonic() - t0


def _row(r: ProgramResult) -> str:
    """One-line per-program summary in run_1096_canonical's style."""
    if r.status == "ok":
        tag = "PASS"
    elif r.status == "error":
        tag = "ERR "
    else:
        tag = "FAIL"
    extra = ""
    if r.status == "fail" and r.divergence_step is not None:
        extra = (f"  div_step={r.divergence_step} "
                 f"got=(pc={r.got_pc},ax={r.got_ax}) "
                 f"oracle=(pc={r.expected_pc},ax={r.expected_ax})")
    return (f"[{tag}] id={r.idx:04d} "
            f"cluster={cluster_of(r.description):14s} "
            f"decl_exit={r.declarative_exit} "
            f"decoded_exit={r.neural_exit} "
            f"steps={r.declarative_steps}  {r.description}{extra}")


def _default_workers() -> int:
    """min(cpu-2, 16) — leave 2 cores for the OS / the parent, cap at 16."""
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 2, 16))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--ids",
                     help="comma-separated ids / ranges, e.g. 550,275,800 or "
                          "1031-1035 (same syntax as run_1096_canonical)")
    sel.add_argument("--all", action="store_true",
                     help="run the WHOLE corpus (every program; same set as "
                          "run_1096_canonical with no --ids).")
    ap.add_argument("--spec-k", type=int, default=_DEFAULT_FULL_TRACE_SPEC_K,
                    help="fail-fast speculation K. Default 32 (back-compat with "
                         "the earlier validation runs). Pass 0 for the "
                         "GPU-bit-exact verdict (eff_spec_k=1 inside "
                         "run_batch_fail_fast; resolves the saturated-tie "
                         "CPU-vs-GPU disagreements).")
    ap.add_argument("--workers", type=int, default=_default_workers(),
                    help=f"parallel CPU worker processes (default "
                         f"min(cpu-2,16)={_default_workers()}). Each worker "
                         f"bakes the model once then decodes its chunk. 1 = "
                         f"serial (in-process, with per-program progress).")
    ap.add_argument("--max-steps-cap", type=int, default=_DEFAULT_MAX_STEPS_CAP,
                    help=f"skip (count as 'skipped', NEVER pass) any program "
                         f"whose declarative oracle step count exceeds this cap. "
                         f"Default {_DEFAULT_MAX_STEPS_CAP} matches the GPU gate "
                         f"and drops NO pass (all 846 short programs are <=39 "
                         f"steps); it folds in the 233 deep loop/gcd/rec programs "
                         f"so each gets a real verdict. Pass 40 for the fast "
                         f"short-only run; 0 to disable (hours on CPU for deep rec).")
    ap.add_argument("--criterion", default="full_trace",
                    choices=["full_trace", "strict_trace"],
                    help="verdict criterion (default: full_trace)")
    ap.add_argument("--output", default=None,
                    help="optional JSON path for the per-program results")
    args = ap.parse_args(argv)

    workers = max(1, int(args.workers))
    spec_k = int(args.spec_k)

    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
    selected = _select(all_tests, args.ids, want_all=args.all)
    if not selected:
        print("[cpu-full-trace] no valid ids selected; nothing to do.",
              file=sys.stderr, flush=True)
        return 2

    print(f"[cpu-full-trace] selected={len(selected)} spec_k={spec_k} "
          f"workers={workers} max_steps_cap={args.max_steps_cap} "
          f"criterion={args.criterion} device=CPU",
          file=sys.stderr, flush=True)
    print(f"[cpu-full-trace] compiling + oracling {len(selected)} program(s)...",
          file=sys.stderr, flush=True)
    prepared, pre_errors = _compile_and_oracle(selected)

    # Partition out over-cap (too-deep) programs BEFORE any decode, so they
    # never enter a forward — matches the GPU gate's 'skipped' set exactly.
    runnable, skipped = _split_over_cap(prepared, args.max_steps_cap)
    if skipped:
        print(f"[cpu-full-trace] SKIPPING {len(skipped)} program(s) over "
              f"--max-steps-cap={args.max_steps_cap} (counted as 'skipped', "
              f"NOT pass).", file=sys.stderr, flush=True)

    # Decode (serial in-process or parallel across workers). One worker => the
    # serial path (per-program progress + a forward-count speed envelope);
    # >1 worker => the process pool (each worker bakes once).
    n_fwd_total = 0
    if workers <= 1 or len(runnable) <= 1:
        decoded, n_fwd_total, secs = _run_serial(
            runnable, spec_k=spec_k, criterion=args.criterion)
    else:
        decoded, secs = _run_parallel(
            runnable, workers=min(workers, len(runnable)),
            spec_k=spec_k, criterion=args.criterion)

    results: List[ProgramResult] = list(pre_errors) + skipped + decoded
    results.sort(key=lambda r: r.idx)

    print("\n" + "=" * 78)
    print("CPU FAITHFUL AUTOREGRESSIVE full_trace VERDICT")
    print(f"  criterion={args.criterion}  spec_k={spec_k}  workers={workers}  "
          f"device=CPU")
    print("=" * 78)
    n_pass = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_err = sum(1 for r in results if r.status == "error")
    n_skip = sum(1 for r in results if r.status == "skipped")
    # Per-program rows only for small selections (the full corpus is 1096
    # rows — the cluster table is the readable summary there).
    if len(results) <= 60:
        for r in results:
            print(_row(r))
        print("-" * 78)
    print(f"  total={len(results)}  pass={n_pass}  fail={n_fail}  "
          f"error={n_err}  skipped={n_skip}")
    if n_fwd_total:
        print(f"  CPU speed: {secs / max(1, len(decoded)):.1f}s/program, "
              f"{secs / max(1, n_fwd_total):.3f}s/forward, "
              f"{n_fwd_total} forwards total, wall={secs:.1f}s.")
    else:
        rate = secs / max(1, len(decoded))
        print(f"  CPU wall: {secs:.1f}s for {len(decoded)} program(s) "
              f"({rate:.1f}s/program effective across {workers} workers).")

    # Per-cluster breakdown (run_1096_canonical style) — the drop-in summary.
    table = _cluster_breakdown(results)
    _print_cluster_table(table, fh=sys.stdout)

    if args.output:
        payload = {
            "criterion": args.criterion,
            "spec_k": spec_k,
            "workers": workers,
            "max_steps_cap": args.max_steps_cap,
            "device": "cpu",
            "wall_seconds": secs,
            "summary": {"total": len(results), "pass": n_pass,
                        "fail": n_fail, "error": n_err, "skipped": n_skip},
            "clusters": {k: v for k, v in table.items()},
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[cpu-full-trace] wrote {args.output}", file=sys.stderr,
              flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
