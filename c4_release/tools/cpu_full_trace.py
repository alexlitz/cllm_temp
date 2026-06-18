#!/usr/bin/env python3
"""Truthful CPU full_trace verdict for specific programs (the framing self-check).

THIS is the tool a lane MUST use to self-check a framing / full_trace fix on
CPU. It runs the byte-identical CPU *autoregressive* decode
(:class:`neural_vm.unified_compiler.faithful_autoregressive.FaithfulAutoregressiveRunner`)
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

Trust envelope (validated 2026-06-17 vs GPU ground truth on current main,
golden ``b9d8861f`` flag-off, HINIB default-ON)
----------------------------------------------------------------------------
The CPU full_trace verdict matches the GPU full_trace verdict for the
framing / value / ALU sample (add / var / func / expr / if / edge_literal /
gcd). The only residual CPU-vs-GPU disagreements are a MODEL fp-instability
artifact at SATURATED-tie positions (drifted positions where the LM logits
saturate ~1e22 with an EXACT top-1/top-2 gap=0, so the argmax winner is fp32-
accumulation-order dependent and fused-GPU vs rule-by-rule-CPU pick different
ids). The REAL CPU ``model.forward`` decode fails those too — i.e. the CPU
decode is faithful to CPU neural; CPU neural just != GPU neural there. See
``docs/CPU_FULL_TRACE_TRUTHFUL_2026_06_17.md`` for the validation table.

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

    # Ranges work too (like run_1096_canonical):
    python tools/cpu_full_trace.py --ids 1031-1035

    # Emit a JSON sidecar for tooling:
    python tools/cpu_full_trace.py --ids 550 --output /tmp/cpu_ft.json

Speed: ~0.47s/CPU forward, ~85s for a ~10-step diverging program (O(steps^2),
no KV cache). Fine for debugging specific programs; use
``tools/run_1096_canonical.py`` on GPU for the whole corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional

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
    _score_fail_fast_results,
    cluster_of,
)
from tools.run_1096_fast import (  # noqa: E402
    ProgramResult,
    _compile_and_oracle,
    _parse_ids,
)

# The full_trace fail-fast path is ALWAYS run at spec_k=32 — that is exactly
# what ``run_1096_canonical._run_one_chunk_with_oom_retry`` does
# (``spec_k=(spec_k if spec_k > 0 else 32)``). The speculative path teacher-
# forces the unsafe MEM offsets from the DraftVM; a mismatch there would make
# the CPU verdict disagree with neural even though the per-token argmax is
# identical. So we pin 32 to be byte-identical to the canonical verdict.
_FULL_TRACE_SPEC_K = 32


def _select(all_tests, ids_spec: str):
    """Select ``(idx, src, expected, desc)`` tuples for the requested ids."""
    wanted = list(dict.fromkeys(_parse_ids(ids_spec)))  # dedup, keep order
    by_idx = {idx: (idx, src, exp, desc)
              for idx, (src, exp, desc) in enumerate(all_tests)}
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True,
                    help="comma-separated ids / ranges, e.g. 550,275,800 or "
                         "1031-1035 (same syntax as run_1096_canonical)")
    ap.add_argument("--criterion", default="full_trace",
                    choices=["full_trace", "strict_trace"],
                    help="verdict criterion (default: full_trace)")
    ap.add_argument("--output", default=None,
                    help="optional JSON path for the per-program results")
    args = ap.parse_args(argv)

    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
    selected = _select(all_tests, args.ids)
    if not selected:
        print("[cpu-full-trace] no valid ids selected; nothing to do.",
              file=sys.stderr, flush=True)
        return 2

    print(f"[cpu-full-trace] compiling + oracling {len(selected)} program(s)...",
          file=sys.stderr, flush=True)
    prepared, pre_errors = _compile_and_oracle(selected)
    # Sort prepared by the requested id order for stable, readable output.
    order = {idx: i for i, (idx, *_rest) in enumerate(selected)}
    prepared.sort(key=lambda e: order.get(e[0], 1 << 30))

    # Build the CPU faithful runner ONCE (the model bake is the cold cost).
    print("[cpu-full-trace] building CPU faithful autoregressive runner "
          "(cold model bake)...", file=sys.stderr, flush=True)
    from neural_vm.unified_compiler.faithful_autoregressive import (
        FaithfulAutoregressiveRunner,
    )

    t_build = time.monotonic()
    runner = FaithfulAutoregressiveRunner()
    print(f"[cpu-full-trace] runner ready ({time.monotonic() - t_build:.1f}s).",
          file=sys.stderr, flush=True)

    results: List[ProgramResult] = list(pre_errors)
    n_fwd_total = 0
    t0 = time.monotonic()
    for n, entry in enumerate(prepared):
        idx, expected, description, decl_exit, decl_steps, bytecode, data = entry
        before = runner.forward_count
        pt0 = time.monotonic()
        r = runner.run_batch_fail_fast(
            [bytecode], data_list=[data], max_steps=None,
            expected_steps_list=[decl_steps], spec_k=_FULL_TRACE_SPEC_K,
            criterion=args.criterion,
        )[0]
        n_fwd = runner.forward_count - before
        n_fwd_total += n_fwd
        pr = _score_fail_fast_results([entry], [r], criterion=args.criterion)[0]
        results.append(pr)
        print(f"[cpu-full-trace]   {n + 1}/{len(prepared)} id={idx} "
              f"{cluster_of(description)} steps={decl_steps}: "
              f"{pr.status} ({n_fwd} fwd, {time.monotonic() - pt0:.1f}s)",
              file=sys.stderr, flush=True)
    secs = time.monotonic() - t0

    # Keep the printed order stable (requested id order, errors interleaved).
    results.sort(key=lambda r: order.get(r.idx, 1 << 30))

    print("\n" + "=" * 78)
    print("CPU FAITHFUL AUTOREGRESSIVE full_trace VERDICT")
    print(f"  criterion={args.criterion}  spec_k={_FULL_TRACE_SPEC_K}  device=CPU")
    print("=" * 78)
    n_pass = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_err = sum(1 for r in results if r.status == "error")
    for r in results:
        print(_row(r))
    print("-" * 78)
    print(f"  total={len(results)}  pass={n_pass}  fail={n_fail}  error={n_err}")
    print(f"  CPU speed: {secs / max(1, len(prepared)):.1f}s/program, "
          f"{secs / max(1, n_fwd_total):.3f}s/forward, "
          f"{n_fwd_total} forwards total.")

    if args.output:
        payload = {
            "criterion": args.criterion,
            "spec_k": _FULL_TRACE_SPEC_K,
            "device": "cpu",
            "summary": {"total": len(results), "pass": n_pass,
                        "fail": n_fail, "error": n_err},
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[cpu-full-trace] wrote {args.output}", file=sys.stderr,
              flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
