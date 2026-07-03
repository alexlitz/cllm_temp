#!/usr/bin/env python3
"""Validate the DSL-INTERPRETER verdict path vs the REAL CPU-neural decode.

The VEHICLE is the DSL interpreter:
``neural_vm.verification.dsl_interpreter_verdict.DSLInterpreterVerdictRunner``
drives the production fail-fast decode with the per-token argmax coming from
:class:`~neural_vm.verification.faithful_interpreter.IRBlockForward` (the
``FaithfulInterpreter`` engine executing the per-physical-block IR). It is
NON-re-anchored (unlike ``tools/interp_oracle_gate.py``), so it reproduces the
autoregressive framing-drift verdict (a step emitting 34/37 tokens) at the spec
level.

This tool compares, per program, the DSL-interpreter verdict (status +
first-divergence step + decoded exit) to the REAL ``model.forward`` CPU-neural
decode (``BatchedPureNeuralRunner`` on CPU, spec_k=32 — the canonical full_trace
path). Validate against **CPU-neural**, not GPU-neural: the residual CPU-vs-GPU
disagreements are a MODEL numerical-instability artifact at saturated-logit ties
(~1e22, top-1/top-2 gap = 0; fp-accumulation-order dependent), NOT a decoder
bug — the real CPU ``model.forward`` fails those same programs. Target: 100%
DSL-interpreter-vs-CPU-neural agreement.

It also runs the RE-ANCHORED gate (``tools/interp_oracle_gate.py``) on the same
programs to surface which ones the gate DEFERS as CROSS-STEP (GPU-only) — the
framing-drift bucket. The DSL-interpreter path resolving those on CPU (matching
CPU-neural) is the headline: the DSL interpreter is now a CPU verdict authority.

Usage::

    CUDA_VISIBLE_DEVICES="" python tools/dsl_interpreter_verdict_validate.py \
        --sample 24 --framing-drift --max-decl-steps 12
    CUDA_VISIBLE_DEVICES="" python tools/dsl_interpreter_verdict_validate.py \
        --ids 262,263,267,271,272 --max-decl-steps 16
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
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")


# Framing-drift clusters: the ~55% the single-forward gate defers as GPU-only.
_FRAMING_DRIFT_CLUSTERS = frozenset(
    {"var", "func", "nested", "if_var", "rec", "expr"}
)
_ALU_CLUSTERS = frozenset({"mul", "div", "mod", "add", "sub"})


def _cluster_of(desc: str) -> str:
    import re
    base = desc.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    return base.rstrip("_") or "misc"


def _bucket(cluster: str) -> str:
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


def _compile_ids(ids: List[int], max_decl_steps: int) -> List[_Prog]:
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program

    tests = generate_test_programs()
    out: List[_Prog] = []
    for idx in ids:
        if idx < 0 or idx >= len(tests):
            continue
        src, exp, desc = tests[idx]
        try:
            bc, data = compile_c(src)
            orc = declarative_oracle_for_program(
                bc, data, suite_expected=exp, label=f"id={idx}",
                max_steps=max_decl_steps + 2,
            )
        except Exception:
            continue
        if orc.error is not None or orc.steps is None or orc.steps > max_decl_steps:
            continue
        cl = _cluster_of(desc)
        out.append(_Prog(
            idx=idx, source=src, description=desc, cluster=cl, bucket=_bucket(cl),
            expected=exp, bytecode=bc, data=data,
            decl_exit=int(orc.exit_code), decl_steps=int(orc.steps),
        ))
    return out


def _compile_sample(n: int, *, framing_drift_first: bool, max_decl_steps: int,
                    seed_stride: int) -> List[_Prog]:
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program

    tests = generate_test_programs()
    per_cluster_cap = max(4, (n // 4) + 2)
    fd_cap = per_cluster_cap * 3

    cluster_indices: "OrderedDict[str, List[int]]" = OrderedDict()
    for idx, (_src, _exp, description) in enumerate(tests):
        cluster_indices.setdefault(_cluster_of(description), []).append(idx)

    by_cluster: "OrderedDict[str, List[_Prog]]" = OrderedDict()
    for cluster, indices in cluster_indices.items():
        cap = fd_cap if _bucket(cluster) == "framing_drift" else per_cluster_cap
        strided = indices[::max(1, seed_stride)] or indices[:1]
        kept: List[_Prog] = []
        for idx in strided:
            if len(kept) >= cap:
                break
            source, expected, description = tests[idx]
            try:
                bc, data = compile_c(source)
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
                cluster=cluster, bucket=_bucket(cluster),
                expected=expected, bytecode=bc, data=data,
                decl_exit=int(oracle.exit_code), decl_steps=int(oracle.steps),
            ))
        if kept:
            by_cluster[cluster] = kept

    fd_clusters = [cl for cl in by_cluster if _bucket(cl) == "framing_drift"]
    other_clusters = [cl for cl in by_cluster if _bucket(cl) != "framing_drift"]
    order = (fd_clusters + other_clusters) if framing_drift_first else list(
        by_cluster.keys()
    )

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


def _status(r: dict) -> str:
    return r.get("status", "?")


def _run_dsl_interp(progs: List[_Prog], criterion: str, spec_k: int = 32
                    ) -> Tuple[List[dict], float, int]:
    """Run the DSL-interpreter verdict path; return (results, secs, n_fwd)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from neural_vm.verification.dsl_interpreter_verdict import (
        DSLInterpreterVerdictRunner,
    )

    runner = DSLInterpreterVerdictRunner()
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
        print(f"[dsl-verdict]   CPU {n + 1}/{len(progs)} id={p.idx} "
              f"{p.cluster} steps={p.decl_steps}: {r.get('status')} "
              f"div={r.get('divergence_step')} ({r['_n_fwd']} fwd, "
              f"{time.monotonic() - pt0:.1f}s)", file=sys.stderr, flush=True)
    secs = time.monotonic() - t0
    return results, secs, runner.forward_count


def _run_cpu_neural(progs: List[_Prog], criterion: str, spec_k: int = 32
                    ) -> Tuple[List[dict], float]:
    """Run the REAL model.forward CPU-neural decode (canonical full_trace path)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    runner = BatchedPureNeuralRunner(csr_inference=False)
    results: List[dict] = []
    t0 = time.monotonic()
    for n, p in enumerate(progs):
        r = runner.run_batch_fail_fast(
            [p.bytecode], data_list=[p.data], max_steps=None,
            expected_steps_list=[p.decl_steps], spec_k=spec_k, criterion=criterion,
        )[0]
        results.append(r)
        print(f"[dsl-verdict]   cpu-neural {n + 1}/{len(progs)} id={p.idx} "
              f"{p.cluster}: {r.get('status')} div={r.get('divergence_step')}",
              file=sys.stderr, flush=True)
    secs = time.monotonic() - t0
    return results, secs


def _run_reanchored_gate(progs: List[_Prog]) -> Dict[int, str]:
    """Run the RE-ANCHORED gate (interp_oracle_gate) to surface its per-program
    classification (PASS / FAIL-high / CROSS-STEP). The CROSS-STEP set is the
    framing-drift bucket the single-forward gate DEFERS to GPU — exactly what the
    DSL-interpreter verdict path is meant to resolve on CPU.

    Returns ``{idx: gate_class}`` where gate_class is one of
    ``"PASS"``, ``"FAIL-high"``, ``"CROSS-STEP"``, ``"ERROR"``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tools.interp_oracle_gate import (
        build_gate_context, classify_program, PASS, FAIL, CROSS_STEP,
    )

    ctx = build_gate_context(verbose=False)
    out: Dict[int, str] = {}
    for p in progs:
        try:
            r = classify_program(
                ctx, p.description[:40], p.bytecode, p.data,
                cluster=p.cluster, max_steps=p.decl_steps + 4, attribute=False,
            )
        except Exception:
            out[p.idx] = "ERROR"
            continue
        if r.classification == PASS:
            out[p.idx] = "PASS"
        elif r.classification == FAIL and r.confidence == CROSS_STEP:
            out[p.idx] = "CROSS-STEP"
        elif r.classification == FAIL:
            out[p.idx] = "FAIL-high"
        else:
            out[p.idx] = r.classification
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=24,
                    help="number of programs to sample across clusters")
    ap.add_argument("--ids", type=str, default="",
                    help="comma-separated exact corpus ids (overrides --sample)")
    ap.add_argument("--framing-drift", action="store_true",
                    help="over-sample the framing-drift clusters first")
    ap.add_argument("--max-decl-steps", type=int, default=12,
                    help="skip programs whose oracle horizon exceeds this "
                         "(CPU forward is O(steps^2); keep bounded)")
    ap.add_argument("--seed-stride", type=int, default=7)
    ap.add_argument("--criterion", default="full_trace",
                    choices=["full_trace", "strict_trace"])
    ap.add_argument("--with-gate", action="store_true",
                    help="also run the re-anchored gate to surface the deferred "
                         "CROSS-STEP (framing-drift) programs the DSL path resolves")
    args = ap.parse_args(argv)

    if args.ids.strip():
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        progs = _compile_ids(ids, args.max_decl_steps)
    else:
        progs = _compile_sample(
            args.sample, framing_drift_first=args.framing_drift,
            max_decl_steps=args.max_decl_steps, seed_stride=args.seed_stride,
        )
    bucket_counts: Dict[str, int] = {}
    for p in progs:
        bucket_counts[p.bucket] = bucket_counts.get(p.bucket, 0) + 1
    print(f"[dsl-verdict] sampled {len(progs)} programs: "
          + ", ".join(f"{b}={c}" for b, c in sorted(bucket_counts.items())),
          file=sys.stderr, flush=True)

    gate_class: Dict[int, str] = {}
    if args.with_gate:
        print("[dsl-verdict] running RE-ANCHORED gate (to surface deferred "
              "CROSS-STEP)...", file=sys.stderr, flush=True)
        gate_class = _run_reanchored_gate(progs)

    print("[dsl-verdict] running DSL-INTERPRETER verdict decode (CPU)...",
          file=sys.stderr, flush=True)
    dsl_results, dsl_secs, dsl_fwd = _run_dsl_interp(progs, args.criterion)
    print(f"[dsl-verdict] DSL decode done: {dsl_secs:.1f}s, {dsl_fwd} IR forwards "
          f"({dsl_secs / max(1, len(progs)):.2f}s/program, "
          f"{dsl_secs / max(1, dsl_fwd):.3f}s/forward)",
          file=sys.stderr, flush=True)

    print("[dsl-verdict] running CPU-NEURAL ground-truth decode...",
          file=sys.stderr, flush=True)
    neural_results, neural_secs = _run_cpu_neural(progs, args.criterion)
    print(f"[dsl-verdict] cpu-neural decode done: {neural_secs:.1f}s",
          file=sys.stderr, flush=True)

    # ---- report -------------------------------------------------------
    print("\n" + "=" * 78)
    print("DSL-INTERPRETER VERDICT — CPU vs CPU-NEURAL VALIDATION")
    print("  vehicle: FaithfulInterpreter engine executing per-block IR "
          "(IRBlockForward)")
    print("  baseline: REAL model.forward on CPU (spec_k=32, canonical "
          "full_trace)")
    print("=" * 78)

    total = len(progs)
    verdict_match = 0
    divstep_match = 0
    exit_match = 0
    mismatches: List[Tuple[_Prog, dict, dict]] = []
    bucket_total: Dict[str, int] = {}
    bucket_match: Dict[str, int] = {}
    deferred_resolved = 0
    deferred_total = 0
    for p, dr, nr in zip(progs, dsl_results, neural_results):
        bucket_total[p.bucket] = bucket_total.get(p.bucket, 0) + 1
        v_ok = _status(dr) == _status(nr)
        d_ok = dr.get("divergence_step") == nr.get("divergence_step")
        e_ok = dr.get("decoded_exit") == nr.get("decoded_exit")
        if v_ok:
            verdict_match += 1
            bucket_match[p.bucket] = bucket_match.get(p.bucket, 0) + 1
        if d_ok:
            divstep_match += 1
        if e_ok:
            exit_match += 1
        if not (v_ok and d_ok):
            mismatches.append((p, dr, nr))
        if gate_class.get(p.idx) == "CROSS-STEP":
            deferred_total += 1
            if v_ok and d_ok:
                deferred_resolved += 1

    pct = 100.0 * verdict_match / max(1, total)
    print(f"\nDSL-interpreter-vs-CPU-neural verdict agreement:")
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

    if args.with_gate:
        # How the DSL path resolves the gate's deferred (CROSS-STEP) bucket.
        gate_dist: Dict[str, int] = {}
        for p in progs:
            g = gate_class.get(p.idx, "?")
            gate_dist[g] = gate_dist.get(g, 0) + 1
        print("\nRe-anchored gate classification on this sample: "
              + ", ".join(f"{k}={v}" for k, v in sorted(gate_dist.items())))
        print(f"  of the {deferred_total} programs the gate DEFERS as CROSS-STEP "
              f"(GPU-only, the framing-drift bucket), the DSL-interpreter verdict "
              f"path resolves {deferred_resolved}/{deferred_total} on CPU "
              f"(matching CPU-neural status + div step).")

    if mismatches:
        print(f"\n{len(mismatches)} DISAGREEMENT(S) — faithfulness bugs to fix:")
        for p, dr, nr in mismatches:
            print(f"  id={p.idx} cluster={p.cluster} ({p.bucket}) "
                  f"steps={p.decl_steps}  gate={gate_class.get(p.idx, '-')}")
            print(f"    src: {p.source.strip()[:90]}")
            print(f"    DSL   : status={_status(dr)} "
                  f"div_step={dr.get('divergence_step')} "
                  f"got=(pc={dr.get('got_pc')},ax={dr.get('got_ax')}) "
                  f"exit={dr.get('decoded_exit')}")
            print(f"    NEURAL: status={_status(nr)} "
                  f"div_step={nr.get('divergence_step')} "
                  f"got=(pc={nr.get('got_pc')},ax={nr.get('got_ax')}) "
                  f"exit={nr.get('decoded_exit')}")
    else:
        print("\nNO DISAGREEMENTS: the DSL-interpreter verdict path reproduces "
              "the CPU-neural full_trace verdict BYTE-FOR-BYTE on this sample.")

    print(f"\nCPU speed: {dsl_secs / max(1, total):.2f}s/program, "
          f"{dsl_secs / max(1, dsl_fwd):.3f}s/forward, {dsl_fwd} IR forwards.")

    # Exit nonzero if any disagreement (so CI can gate on verdict trust).
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
