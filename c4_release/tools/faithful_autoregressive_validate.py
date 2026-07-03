#!/usr/bin/env python3
"""Validate the CPU faithful autoregressive decode against neural ground-truth.

The headline question this tool answers: does the CPU faithful autoregressive
decoder (:class:`neural_vm.verification.faithful_autoregressive.FaithfulAutoregressiveRunner`)
reproduce the NEURAL full_trace verdict BYTE-FOR-BYTE for EVERY cluster —
including the autoregressive framing-drift bucket (``var_*`` / ``func_identity``
/ ``nested_*`` / ``if_var``) that ``tools/interp_oracle_gate.py`` currently
DEFERS to GPU because its re-anchored single-forward cannot see the 34/37-token
miscount?

It runs, on the SAME representative sample (drawn across ALL clusters, with the
framing-drift clusters explicitly over-sampled):

  1. The CPU faithful autoregressive decode (``FaithfulAutoregressiveRunner.
     run_batch_fail_fast``, ``CUDA_VISIBLE_DEVICES=""``), which produces a
     per-program ``{status, divergence_step, got_pc, got_ax, decoded_exit}``.

  2. (Optional, ``--neural``) The NEURAL ground-truth decode
     (``BatchedPureNeuralRunner.run_batch_fail_fast``, spec_k=0) — the EXACT
     path ``tools/run_1096_canonical.py --criterion full_trace`` uses.

It then compares, per program:

  * the VERDICT (``status`` pass/fail/error) — the trust-critical signal;
  * the first-divergence step (``divergence_step``) — a stronger check that
    the CPU decode diverges at the SAME step, not just lands on the same
    pass/fail bit;
  * the decoded exit code.

ANY disagreement is a faithfulness bug in the CPU decode (wrong slice offset,
argmax tie-break, STEP_END handling, position / ALiBi) and is printed with the
program's cluster + source so it can be found and fixed.

When ``--neural`` is NOT given (e.g. no GPU / GPU contended) the tool runs the
CPU decode only and reports the CPU verdict distribution + per-cluster
breakdown, plus the CPU-vs-declarative-EXIT agreement (a weaker self-check that
needs no neural run).

Resolve dims (when debugging a divergence) via the BUILT
``compile_full_vm_dynamic()[1].dim_positions`` — NOT the static registry.

Usage
-----
    # CPU-only self-check (no GPU needed):
    CUDA_VISIBLE_DEVICES="" python tools/faithful_autoregressive_validate.py \
        --sample 40 --framing-drift

    # Full CPU-vs-neural byte-for-byte verdict agreement (needs a GPU for the
    # neural ground-truth; the CPU decode still runs on CPU):
    python tools/faithful_autoregressive_validate.py --sample 40 \
        --framing-drift --neural --neural-device 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from tools.run_1096_canonical import cluster_of  # noqa: E402


# The clusters whose full_trace verdict is autoregressive framing-drift driven
# (a step emits 34/37 tokens not 35), the ~13% the gate defers to GPU.
_FRAMING_DRIFT_CLUSTERS = frozenset(
    {"var", "func", "nested", "if_var", "rec", "expr"}
)
# Coarse value vs ALU split (for the broken-out agreement report).
_ALU_CLUSTERS = frozenset({"mul", "div", "mod", "add", "sub"})


def _cluster_bucket(cluster: str) -> str:
    """Coarse bucket of a cluster for the broken-out agreement report."""
    c = cluster.lower()
    for fd in _FRAMING_DRIFT_CLUSTERS:
        if c == fd or c.startswith(fd):
            return "framing_drift"
    for alu in _ALU_CLUSTERS:
        if c == alu or c.startswith(alu):
            return "alu"
    return "value"


@dataclass
class _Prog:
    idx: int
    source: str
    description: str
    cluster: str
    bucket: str
    expected: int
    bytecode: list
    data: bytes
    decl_exit: int
    decl_steps: int


def _compile_sample(
    n: int, *, framing_drift_first: bool, max_decl_steps: int, seed_stride: int,
) -> List[_Prog]:
    """Compile a representative sample spanning ALL clusters.

    With ``framing_drift_first`` the framing-drift clusters (var/func/nested/
    if_var/rec/expr) are taken FIRST (so the deferred-to-GPU bucket is densely
    sampled), then the rest fill in round-robin across clusters. Deep diverging
    programs (> ``max_decl_steps`` oracle steps) are skipped: their per-token
    CPU forward over a 1000+-token tape is minutes each and they are not where
    the framing-drift faithfulness signal lives.
    """
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program

    tests = generate_test_programs()

    # Per-cluster cap: keep oracling bounded. We want enough per cluster to draw
    # the sample (round-robin), not the whole corpus. The framing-drift clusters
    # get a bigger cap so the deferred-to-GPU bucket is densely covered.
    per_cluster_cap = max(4, (n // 4) + 2)
    fd_cap = per_cluster_cap * 3

    # First-pass index by cluster (cheap; no compile) so we can stride within a
    # cluster to spread the sub-sample, and bound how many we actually oracle.
    cluster_indices: "OrderedDict[str, List[int]]" = OrderedDict()
    for idx, (_src, _exp, description) in enumerate(tests):
        cluster_indices.setdefault(cluster_of(description), []).append(idx)

    by_cluster: "OrderedDict[str, List[_Prog]]" = OrderedDict()
    for cluster, indices in cluster_indices.items():
        cap = fd_cap if _cluster_bucket(cluster) == "framing_drift" else per_cluster_cap
        # Stride across the cluster's members so we sample its full range, then
        # oracle only until we have ``cap`` in-budget candidates (bounds work).
        strided = indices[::max(1, seed_stride)] or indices[:1]
        kept: List[_Prog] = []
        for idx in strided:
            if len(kept) >= cap:
                break
            source, expected, description = tests[idx]
            try:
                bc, data = compile_c(source)
            except Exception:
                continue
            try:
                # Oracle bounded by max_decl_steps+slack so deep programs that
                # are out-of-budget bail fast instead of running a huge horizon.
                oracle = declarative_oracle_for_program(
                    bc, data, suite_expected=expected,
                    label=f"id={idx:04d}", max_steps=max_decl_steps + 2,
                )
            except Exception:
                continue
            if oracle.error is not None or oracle.steps is None:
                continue
            if oracle.steps > max_decl_steps:
                continue
            kept.append(_Prog(
                idx=idx, source=source, description=description,
                cluster=cluster, bucket=_cluster_bucket(cluster),
                expected=expected, bytecode=bc, data=data,
                decl_exit=int(oracle.exit_code), decl_steps=int(oracle.steps),
            ))
        if kept:
            by_cluster[cluster] = kept

    fd_clusters = [
        cl for cl in by_cluster if _cluster_bucket(cl) == "framing_drift"
    ]
    other_clusters = [
        cl for cl in by_cluster if _cluster_bucket(cl) != "framing_drift"
    ]
    order = (fd_clusters + other_clusters) if framing_drift_first else (
        list(by_cluster.keys())
    )

    # Round-robin draw across the ordered clusters until we have n.
    out: List[_Prog] = []
    cursors = {cl: 0 for cl in order}
    while len(out) < n:
        progressed = False
        for cl in order:
            progs = by_cluster[cl]
            c = cursors[cl]
            if c < len(progs):
                out.append(progs[c])
                cursors[cl] = c + 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def _run_faithful(
    progs: List[_Prog], criterion: str, spec_k: int = 32,
) -> Tuple[List[dict], float, int]:
    """Run the CPU faithful autoregressive decode; return (results, secs, n_fwd).

    Each program runs solo (the faithful forward is per-row, so batching gives
    no speedup and only complicates the speed measurement). ``spec_k`` MUST match
    the neural canonical run (which uses 32): the speculative path teacher-forces
    the unsafe MEM offsets from the DraftVM, and a mismatch there would make the
    CPU verdict disagree with neural even though the per-token argmax is identical.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from neural_vm.verification.faithful_autoregressive import (
        FaithfulAutoregressiveRunner,
    )

    runner = FaithfulAutoregressiveRunner()
    results: List[dict] = []
    t0 = time.monotonic()
    for n, p in enumerate(progs):
        before = runner.forward_count
        pt0 = time.monotonic()
        r = runner.run_batch_fail_fast(
            [p.bytecode], data_list=[p.data], max_steps=None,
            expected_steps_list=[p.decl_steps], spec_k=spec_k, criterion=criterion,
        )[0]
        r["_n_fwd"] = runner.forward_count - before
        results.append(r)
        print(f"[faithful-ar]   CPU {n + 1}/{len(progs)} id={p.idx} "
              f"{p.cluster} steps={p.decl_steps}: {r.get('status')} "
              f"({r['_n_fwd']} fwd, {time.monotonic() - pt0:.1f}s)",
              file=sys.stderr, flush=True)
    secs = time.monotonic() - t0
    return results, secs, runner.forward_count


def _run_neural(
    progs: List[_Prog], criterion: str, device: str, spec_k: int = 32,
) -> Tuple[List[dict], float]:
    """Run the NEURAL ground-truth decode (the run_1096_canonical path).

    ``spec_k`` matches ``run_1096_canonical`` (which passes 32 to
    ``run_batch_fail_fast`` in the fail-fast path), so this IS the canonical
    full_trace verdict.

    When ``device == ""`` this runs the REAL model on CPU — the apples-to-apples
    comparison that isolates the decoder's FAITHFULNESS (faithful CPU math vs
    fused CPU kernels) from the GPU-vs-CPU fp32 confound. When ``device`` is a
    GPU index this is the literal GPU ground-truth, which can DISAGREE with the
    CPU decode on programs whose drifted positions produce saturated-logit ties
    (~1e22, top-1/top-2 gap = 0): there the argmax winner is fp32-accumulation-
    order-dependent, so GPU (fused) and CPU (rule-by-rule) pick different ids.
    That divergence is a MODEL numerical-instability artifact, not a decoder bug
    (the real CPU model.forward decode AGREES with the faithful CPU decode there).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    runner = BatchedPureNeuralRunner(csr_inference=False)
    results: List[dict] = []
    t0 = time.monotonic()
    # Batch-of-1 per program to mirror the faithful path's per-program verdict
    # (the verdict is per-element and batch-independent by construction).
    for n, p in enumerate(progs):
        r = runner.run_batch_fail_fast(
            [p.bytecode], data_list=[p.data], max_steps=None,
            expected_steps_list=[p.decl_steps],
            spec_k=spec_k, criterion=criterion,
        )[0]
        results.append(r)
        print(f"[faithful-ar]   neural {n + 1}/{len(progs)} id={p.idx} "
              f"{p.cluster}: {r.get('status')}", file=sys.stderr, flush=True)
    secs = time.monotonic() - t0
    return results, secs


def _status(r: dict) -> str:
    return r.get("status", "?")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=30,
                    help="number of programs to sample across clusters")
    ap.add_argument("--framing-drift", action="store_true",
                    help="over-sample the framing-drift clusters first")
    ap.add_argument("--max-decl-steps", type=int, default=40,
                    help="skip programs whose oracle horizon exceeds this")
    ap.add_argument("--seed-stride", type=int, default=7,
                    help="stride for spreading the per-cluster sub-sample")
    ap.add_argument("--criterion", default="full_trace",
                    choices=["full_trace", "strict_trace"])
    ap.add_argument("--neural", action="store_true",
                    help="also run the NEURAL ground-truth and compare verdicts")
    ap.add_argument("--neural-device", default="0",
                    help="CUDA_VISIBLE_DEVICES for the neural ground-truth run")
    ap.add_argument("--cpu-neural", action="store_true",
                    help="compare against the REAL model on CPU (apples-to-apples "
                         "faithfulness check: isolates the decoder from the "
                         "GPU-vs-CPU fp32 saturated-tie confound). Implies --neural "
                         "with neural-device=''.")
    args = ap.parse_args(argv)
    if args.cpu_neural:
        args.neural = True
        args.neural_device = ""

    print(f"[faithful-ar] compiling sample (n={args.sample}, "
          f"framing_drift_first={args.framing_drift}, "
          f"max_decl_steps={args.max_decl_steps})...", file=sys.stderr, flush=True)
    progs = _compile_sample(
        args.sample, framing_drift_first=args.framing_drift,
        max_decl_steps=args.max_decl_steps, seed_stride=args.seed_stride,
    )
    bucket_counts: Dict[str, int] = {}
    for p in progs:
        bucket_counts[p.bucket] = bucket_counts.get(p.bucket, 0) + 1
    print(f"[faithful-ar] sampled {len(progs)} programs: "
          + ", ".join(f"{b}={c}" for b, c in sorted(bucket_counts.items())),
          file=sys.stderr, flush=True)

    print("[faithful-ar] running CPU faithful autoregressive decode...",
          file=sys.stderr, flush=True)
    cpu_results, cpu_secs, cpu_fwd = _run_faithful(progs, args.criterion)
    print(f"[faithful-ar] CPU decode done: {cpu_secs:.1f}s, "
          f"{cpu_fwd} faithful forwards "
          f"({cpu_secs / max(1, len(progs)):.2f}s/program, "
          f"{cpu_secs / max(1, cpu_fwd):.3f}s/forward)",
          file=sys.stderr, flush=True)

    neural_results: Optional[List[dict]] = None
    neural_secs = 0.0
    if args.neural:
        print("[faithful-ar] running NEURAL ground-truth decode...",
              file=sys.stderr, flush=True)
        neural_results, neural_secs = _run_neural(
            progs, args.criterion, args.neural_device
        )
        print(f"[faithful-ar] neural decode done: {neural_secs:.1f}s",
              file=sys.stderr, flush=True)

    # ---- report -------------------------------------------------------
    baseline = (
        "REAL-CPU (apples-to-apples)" if args.neural and args.neural_device == ""
        else f"GPU device {args.neural_device}" if args.neural else "none"
    )
    print("\n" + "=" * 78)
    print("FAITHFUL AUTOREGRESSIVE CPU DECODE — VALIDATION REPORT")
    print(f"  neural baseline: {baseline}")
    print("=" * 78)

    cpu_dist: Dict[str, int] = {}
    for r in cpu_results:
        cpu_dist[_status(r)] = cpu_dist.get(_status(r), 0) + 1
    print(f"\nCPU verdict distribution: "
          + ", ".join(f"{k}={v}" for k, v in sorted(cpu_dist.items())))

    if neural_results is None:
        # CPU-only self-check: CPU verdict vs declarative exit.
        agree = 0
        for p, r in zip(progs, cpu_results):
            cpu_pass = _status(r) == "pass"
            de = r.get("decoded_exit")
            decl_ok = (de is not None) and (
                int(de) & 0xFFFFFFFF == int(p.decl_exit) & 0xFFFFFFFF
            )
            # On a PASS the decoded exit must equal the declarative exit; on a
            # FAIL it generally won't (a self-consistency check, not ground
            # truth).
            if cpu_pass == decl_ok:
                agree += 1
        print(f"\nCPU-vs-declarative self-consistency: {agree}/{len(progs)} "
              f"(pass<=>exit-matches-declarative). NOTE: this is a weaker "
              f"self-check; run --neural for true byte-for-byte verdict trust.")
        _print_cluster_table(progs, cpu_results, None)
        print(f"\nCPU speed: {cpu_secs / max(1, len(progs)):.2f}s/program, "
              f"{cpu_secs / max(1, cpu_fwd):.3f}s/forward, "
              f"{cpu_fwd} forwards total.")
        return 0

    # CPU-vs-neural byte-for-byte verdict agreement.
    total = len(progs)
    verdict_match = 0
    divstep_match = 0
    exit_match = 0
    mismatches: List[Tuple[_Prog, dict, dict]] = []
    bucket_total: Dict[str, int] = {}
    bucket_match: Dict[str, int] = {}
    for p, cr, nr in zip(progs, cpu_results, neural_results):
        bucket_total[p.bucket] = bucket_total.get(p.bucket, 0) + 1
        v_ok = _status(cr) == _status(nr)
        d_ok = cr.get("divergence_step") == nr.get("divergence_step")
        e_ok = cr.get("decoded_exit") == nr.get("decoded_exit")
        if v_ok:
            verdict_match += 1
            bucket_match[p.bucket] = bucket_match.get(p.bucket, 0) + 1
        if d_ok:
            divstep_match += 1
        if e_ok:
            exit_match += 1
        if not (v_ok and d_ok):
            mismatches.append((p, cr, nr))

    pct = 100.0 * verdict_match / max(1, total)
    print(f"\nCPU-vs-NEURAL byte-for-byte verdict agreement:")
    print(f"  status (pass/fail/error)  : {verdict_match}/{total} ({pct:.1f}%)")
    print(f"  + first-divergence step    : {divstep_match}/{total} "
          f"({100.0 * divstep_match / max(1, total):.1f}%)")
    print(f"  + decoded exit code        : {exit_match}/{total} "
          f"({100.0 * exit_match / max(1, total):.1f}%)")

    print("\nVerdict agreement broken out by bucket:")
    for b in sorted(bucket_total):
        bt = bucket_total[b]
        bm = bucket_match.get(b, 0)
        print(f"  {b:16s}: {bm}/{bt} ({100.0 * bm / max(1, bt):.1f}%)")

    if mismatches:
        print(f"\n{len(mismatches)} DISAGREEMENT(S) — faithfulness bugs to fix:")
        for p, cr, nr in mismatches:
            print(f"  id={p.idx} cluster={p.cluster} ({p.bucket}) "
                  f"steps={p.decl_steps}")
            print(f"    src: {p.source.strip()}")
            print(f"    CPU   : status={_status(cr)} "
                  f"div_step={cr.get('divergence_step')} "
                  f"got=(pc={cr.get('got_pc')},ax={cr.get('got_ax')}) "
                  f"exit={cr.get('decoded_exit')}")
            print(f"    NEURAL: status={_status(nr)} "
                  f"div_step={nr.get('divergence_step')} "
                  f"got=(pc={nr.get('got_pc')},ax={nr.get('got_ax')}) "
                  f"exit={nr.get('decoded_exit')}")
    else:
        print("\nNO DISAGREEMENTS: the CPU autoregressive decode reproduces the "
              "neural full_trace verdict BYTE-FOR-BYTE on this sample.")

    _print_cluster_table(progs, cpu_results, neural_results)
    print(f"\nCPU speed: {cpu_secs / max(1, total):.2f}s/program, "
          f"{cpu_secs / max(1, cpu_fwd):.3f}s/forward, {cpu_fwd} forwards.")
    if args.neural:
        print(f"Neural speed: {neural_secs / max(1, total):.2f}s/program "
              f"(GPU device {args.neural_device}).")

    # Exit nonzero if any disagreement (so CI can gate on byte-for-byte trust).
    return 1 if mismatches else 0


def _print_cluster_table(
    progs: List[_Prog], cpu: List[dict], neural: Optional[List[dict]],
) -> None:
    print("\nper-cluster breakdown:")
    rows: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for i, p in enumerate(progs):
        row = rows.setdefault(p.cluster, {"n": 0, "cpu_pass": 0, "agree": 0})
        row["n"] += 1
        if _status(cpu[i]) == "pass":
            row["cpu_pass"] += 1
        if neural is not None and _status(cpu[i]) == _status(neural[i]):
            row["agree"] += 1
    hdr = f"  {'cluster':16s} {'n':>4s} {'cpu_pass':>9s}"
    if neural is not None:
        hdr += f" {'agree':>6s}"
    print(hdr)
    for cl, row in rows.items():
        line = f"  {cl:16s} {row['n']:4d} {row['cpu_pass']:9d}"
        if neural is not None:
            line += f" {row['agree']:6d}"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
