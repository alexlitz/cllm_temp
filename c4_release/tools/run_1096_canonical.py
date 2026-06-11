#!/usr/bin/env python3
"""Canonical fast runner for the 1096 corpus.

This is the FAST (~10-15 min) equivalent of the pytest suite
``tests/test_suite_1096_pure_neural_pytest.py``. It reports the SAME
per-program pass/fail criterion as that suite, so the count it prints is the
canonical 1096 score (the real ~62/1096 number) rather than a ``--limit 128``
slice artifact.

Why a separate entry point from ``run_1096_fast.py``
---------------------------------------------------
``run_1096_fast.py`` already applies the same pass *criterion* (neural
exit-code == declarative oracle exit-code, masked to 32 bits). The "~1%"
numbers attributed to that tool historically came from running only a small
leading slice (``--limit 128`` = the add/sub/mul band, whose declarative
results the model gets wrong on the high-byte carry path), NOT from a
criterion mismatch. See ``docs/1096_CANONICAL_RUNNER_2026_06_11.md`` for the
full reconciliation.

This wrapper:

  * applies the SUITE's exact pass criterion (no KV cache, declarative halt
    horizon as ``max_steps``, neural-vs-declarative exit-code compare);
  * defaults to ``spec_k=0`` (raw one-token-per-forward) instead of the
    suite's literal ``spec_k=adaptive``. DraftVM is verifier-arbitrated and
    byte-identity with spec_k=0, so the pass SET is identical, but spec_k=0 is
    FASTER on the heavily-rejecting clusters (var/if/func/loop/rec) where
    adaptive K wastes a draft-and-rollback per rejected step — that adaptive
    waste is the main reason the full pytest run takes >5h. Pass ``--spec-k -1``
    to reproduce the suite's literal path (same passes, slower);
  * runs the FULL corpus by default (all 1096 programs);
  * emits a per-cluster breakdown (var / func / if / loop / rec / expr / ...)
    so we can see which clusters move as fixes land.

Pass criterion (identical to the suite's ``test_program``)
----------------------------------------------------------
A program counts as PASS iff ALL hold:

  1. ``compile_c`` succeeds, AND
  2. the declarative oracle halts and its exit code equals the suite's
     ``expected`` value masked to 32 bits
     (``decl_exit == suite_expected & 0xFFFFFFFF``), AND
  3. the pure-neural batched decode halts with the same exit code as the
     declarative oracle (``neural_exit & 0xFFFFFFFF == decl_exit & 0xFFFFFFFF``).

Anything else is FAIL. (In the pytest suite, 2 and 3 are two ``assert``
statements and 1 surfaces as ``err``; xfail(strict=False) means XPASS=pass,
xfail=fail, so PASS here == XPASS there.)

This is the PURE-NEURAL path (the vanilla thesis), exactly like the suite. It
is NOT the production handler path.

Usage
-----

    # Full canonical 1096 score + per-cluster breakdown (GPU 1):
    CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py \
        --output /tmp/canonical_1096.json

    # A validation sample (compare these exact ids against the pytest suite):
    CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py \
        --ids 0,49,250,550,975 --output /tmp/canonical_sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, List, Optional

# Make ``import neural_vm`` / ``import tools`` work from any cwd / worktree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tools.run_1096_fast import (  # noqa: E402
    ProgramResult,
    _build_neural_runner,
    _compile_and_oracle,
    _parse_ids,
    _run_chunks,
)

# Speculation default for the CANONICAL fast runner.
#
# The pytest suite's literal default is C4_SPEC_K="adaptive" -> -1 (per-element
# adaptive DraftVM starting at K=32). DraftVM is verifier-arbitrated and
# byte-identity with spec_k=0 (the model is the final arbiter), so the pass SET
# is identical for any spec_k. We therefore default this fast runner to
# spec_k=0 (raw one-token-per-forward batched decode) NOT to -1, because:
#
#   * Correctness is identical (byte-identity, documented in
#     batched_pure_neural.run_batch / project_probe_path_spec_k_not_hooks).
#   * spec_k=0 is FASTER on the heavily-rejecting clusters (var / if / func /
#     loop / rec). Adaptive K=32 wastes a full draft-and-rollback cycle on
#     every rejected step; on those clusters (~600 of 1096) that turns each
#     step into many wasted forwards. The suite's adaptive default is the main
#     reason the full pytest run takes >5h — it is throughput-optimal only on
#     clean single-step programs.
#
# spec_k=0 is also the smoke-gate ground-truth path
# (project_probe_path_spec_k_not_hooks). Pass --spec-k -1 to reproduce the
# suite's literal adaptive path (same pass set, slower).
_SUITE_SPEC_K = 0


def cluster_of(description: str) -> str:
    """Derive a stable cluster key from a program's description.

    Descriptions look like ``add_0: 654 + 114`` / ``var_simple_3: x = 7`` /
    ``rec_fib_12: fib(9)`` / ``edge_literal_4`` / ``gcd_17: ...``. The cluster
    is the alphabetic prefix with the trailing per-case index stripped. This
    reproduces the cluster families in docs/1096_TRIAGE_2026_06_11.md:
    add, sub, mul, div, mod, var_*, if_*, loop_*, func_*, rec_*, expr_*,
    gcd, nested_*, edge*, absdiff, bool_and.
    """
    base = description.split(":", 1)[0].strip()
    # Strip a trailing _<digits> (per-case index), then any trailing digits.
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


def _select(all_tests, ids_spec: Optional[str], offset: int, limit: Optional[int]):
    enumerated = list(enumerate(all_tests))
    if ids_spec:
        wanted = set(_parse_ids(ids_spec))
        return [
            (idx, src, exp, desc)
            for idx, (src, exp, desc) in enumerated
            if idx in wanted
        ]
    windowed = enumerated[offset:]
    if limit is not None:
        windowed = windowed[:limit]
    return [(idx, src, exp, desc) for idx, (src, exp, desc) in windowed]


def _cluster_breakdown(results: List[ProgramResult]) -> "OrderedDict[str, Dict[str, int]]":
    """Aggregate pass/fail/error per cluster, in cluster-first-seen order."""
    table: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for r in results:
        key = cluster_of(r.description)
        row = table.setdefault(key, {"n": 0, "pass": 0, "fail": 0, "error": 0})
        row["n"] += 1
        if r.status == "ok":
            row["pass"] += 1
        elif r.status == "error":
            row["error"] += 1
        else:
            row["fail"] += 1
    return table


def _print_cluster_table(table, fh=sys.stderr) -> None:
    print("\n[1096-canonical] per-cluster breakdown", file=fh, flush=True)
    print(
        f"  {'cluster':22s} {'n':>5s} {'pass':>5s} {'fail':>5s} {'err':>5s} {'pass%':>6s}",
        file=fh,
    )
    for key, row in table.items():
        n = row["n"]
        pct = (100.0 * row["pass"] / n) if n else 0.0
        print(
            f"  {key:22s} {n:5d} {row['pass']:5d} {row['fail']:5d} "
            f"{row['error']:5d} {pct:6.1f}",
            file=fh,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical fast runner for the 1096 corpus. Reports the SAME "
            "per-program pass/fail as test_suite_1096_pure_neural_pytest.py."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N tests (default: all 1096).",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first M tests before applying --limit.",
    )
    parser.add_argument(
        "--ids", type=str, default=None,
        help="Comma-separated ids/ranges (e.g. '0,5-9,42'). Overrides offset/limit.",
    )
    parser.add_argument(
        "--chunk", type=int,
        default=int(os.environ.get("C4_BATCH_CHUNK", "32")),
        help="Programs per neural batch (default: env C4_BATCH_CHUNK or 32, "
             "matching the suite fixture's default).",
    )
    parser.add_argument(
        "--spec-k", type=int, default=_SUITE_SPEC_K,
        help="Speculative-decode K (default: 0 = raw one-token-per-forward). "
             "DraftVM is byte-identity with spec_k=0 so this does NOT change "
             "the pass set vs the suite's adaptive default; 0 is chosen for "
             "speed on the rejecting clusters. Pass -1 to reproduce the "
             "suite's literal adaptive path (same passes, slower).",
    )
    parser.add_argument(
        "--max-context-window", type=int,
        default=int(os.environ.get("C4_BATCH_CONTEXT_WINDOW", "512")),
        help="Tail context window (default: 512, matching the suite).",
    )
    parser.add_argument(
        "--model-max-seq-len", type=int,
        default=int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096")),
        help="Model max seq len for the runner (default: 4096).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to dump full per-program results + cluster breakdown as JSON.",
    )
    parser.add_argument(
        "--print-failures", action="store_true",
        help="Stream each non-passing program row to stdout.",
    )
    args = parser.parse_args(argv)

    # Match run_1096_fast: only set a default device if the caller hasn't. The
    # task mandates GPU 1; callers pass CUDA_VISIBLE_DEVICES=1.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    overall_t0 = time.monotonic()

    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
    selected = _select(all_tests, args.ids, args.offset, args.limit)

    print(
        f"[1096-canonical] selected={len(selected)} (of {len(all_tests)}) "
        f"chunk={args.chunk} spec_k={args.spec_k} "
        f"(suite-exact criterion; pure-neural path) "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        file=sys.stderr,
        flush=True,
    )

    # Phase 1: compile + declarative oracle (CPU). Oracle is called with
    # suite_expected so an oracle-vs-suite disagreement becomes an oracle
    # error == suite assert #2 failing == FAIL. Identical to the suite.
    prepared, oracle_errors = _compile_and_oracle(selected)

    # Phase 2: bake the pure-neural model once.
    neural_runner = _build_neural_runner(model_max_seq_len=args.model_max_seq_len)

    # Phase 3: batched pure-neural decode in chunks; compare neural exit code
    # to the declarative exit code (suite assert #3). Identical to the suite.
    neural_results = _run_chunks(
        neural_runner,
        prepared,
        chunk_size=max(1, int(args.chunk)),
        spec_k=int(args.spec_k),
        max_context_window=int(args.max_context_window),
    )

    all_results: List[ProgramResult] = oracle_errors + neural_results
    all_results.sort(key=lambda r: r.idx)

    total_elapsed = time.monotonic() - overall_t0

    pass_n = sum(1 for r in all_results if r.status == "ok")
    fail_n = sum(1 for r in all_results if r.status == "fail")
    err_n = sum(1 for r in all_results if r.status == "error")
    total_n = len(all_results)

    # The suite counts FAIL and ERROR identically (both are "not a pass" /
    # xfail). The canonical headline number is PASS / total.
    print(
        f"\n[1096-canonical] CANONICAL SCORE: {pass_n}/{total_n} PASS "
        f"({100.0 * pass_n / total_n:.2f}%)  "
        f"[fail={fail_n} error={err_n}]  wall={total_elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    table = _cluster_breakdown(all_results)
    _print_cluster_table(table)

    if args.print_failures:
        for r in all_results:
            if r.status != "ok":
                print(r.to_row(), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "wall_seconds": total_elapsed,
                    "criterion": (
                        "suite-exact: compile_c ok AND "
                        "decl_exit==(suite_expected & 0xFFFFFFFF) AND "
                        "neural_exit==decl_exit (both masked 0xFFFFFFFF); "
                        "pure-neural batched path"
                    ),
                    "spec_k": args.spec_k,
                    "chunk": args.chunk,
                    "summary": {
                        "total": total_n,
                        "pass": pass_n,
                        "fail": fail_n,
                        "error": err_n,
                    },
                    "clusters": {
                        k: v for k, v in table.items()
                    },
                    "results": [asdict(r) for r in all_results],
                },
                fh,
                indent=2,
            )
        print(f"[1096-canonical] wrote {args.output}", file=sys.stderr, flush=True)

    # Exit 0 always: this is a measurement tool, not a gate. (The suite's
    # gate semantics are xfail; a non-zero pass count is success here.)
    return 0


if __name__ == "__main__":
    sys.exit(main())
