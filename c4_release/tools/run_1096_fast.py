#!/usr/bin/env python3
"""Fast batched runner for the 1096 corpus.

Loads the pure-neural VM model once, compiles all 1096 programs once, then
streams them through ``BatchedPureNeuralRunner.run_batch`` in fixed-size
chunks (default 32 per inner batch, tuned to match the pytest-fixture path
under ``C4_BATCH_CHUNK=32`` + ``C4_SPEC_K=4``).

Per-chunk progress is printed to stderr so the run is observable. The final
summary tallies pass/fail/timeout/error counts and reports total wall time.

Output parity vs the pytest harness
-----------------------------------
Each program's neural exit code is compared against the declarative oracle's
exit code (which is also what the suite checks against in the
``test_1096_neural_declarative_diagnostic_slice`` path). A program is
counted as PASS iff:

  * compile_c + declarative oracle succeed (matching suite_expected), AND
  * the neural exit code matches the declarative exit code.

Anything else is FAIL (or ERROR if a compile / oracle exception is raised
before the neural run even starts).

Usage
-----

    python c4_release/tools/run_1096_fast.py [--limit N] [--chunk K] [--spec-k S]
                                              [--ids LIST] [--output FILE]

By default the script targets GPU 0 (cuda:0). Override with
``CUDA_VISIBLE_DEVICES`` if you need to redirect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# Make ``import neural_vm`` work whether the tool is run from c4_release/tools
# or from the repo root or from a worktree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


@dataclass
class ProgramResult:
    idx: int
    description: str
    suite_expected: int
    declarative_exit: Optional[int]
    declarative_steps: Optional[int]
    neural_exit: Optional[int]
    status: str  # "ok", "fail", "error"
    error: Optional[str] = None

    # Full-trace (fail-fast) divergence point, populated only by the
    # ``--fail-fast`` path in run_1096_canonical.py. ``divergence_step`` is the
    # 0-based VM step where the model's decoded (PC, AX) first diverged from the
    # declarative oracle; the expected/got pairs are that step's register state.
    divergence_step: Optional[int] = None
    expected_pc: Optional[int] = None
    expected_ax: Optional[int] = None
    got_pc: Optional[int] = None
    got_ax: Optional[int] = None

    # strict_trace (token-identity) divergence detail. Populated only by the
    # ``--criterion strict_trace`` path in run_1096_canonical.py: the OFFSET
    # (0..34) inside the diverging 35-token VM step, its human-readable field
    # name (e.g. ``AX[1]``), and the expected/got token ids at that offset.
    divergence_offset: Optional[int] = None
    divergence_offset_name: Optional[str] = None
    expected_tok: Optional[int] = None
    got_tok: Optional[int] = None

    def to_row(self) -> str:
        if self.status == "ok":
            tag = "OK  "
        elif self.status == "error":
            tag = "ERR "
        else:
            tag = "FAIL"
        return (
            f"[{tag}] id={self.idx:04d} "
            f"expected={self.suite_expected} "
            f"decl={self.declarative_exit} "
            f"neural={self.neural_exit} "
            f"steps={self.declarative_steps} "
            f"desc={self.description}"
        )


def _parse_ids(spec: str) -> List[int]:
    """Parse a comma-separated list of ids / ranges like ``0,5-9,42``."""
    out: List[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(piece))
    return out


def _build_neural_runner(*, model_max_seq_len: int):
    """Instantiate ``BatchedPureNeuralRunner`` and time the cold model bake."""
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    t0 = time.monotonic()
    runner = BatchedPureNeuralRunner(max_seq_len=model_max_seq_len)
    elapsed = time.monotonic() - t0
    print(
        f"[1096-fast] model bake complete in {elapsed:.1f}s "
        f"(device={runner._device})",
        file=sys.stderr,
        flush=True,
    )
    return runner


def _compile_and_oracle(
    tests: List[Tuple[int, str, int, str]],
) -> Tuple[
    List[Tuple[int, int, str, int, int, list, bytes]],
    List[ProgramResult],
]:
    """Compile + run the symbolic declarative oracle for every selected test.

    Returns ``(prepared, errors)``:

      * ``prepared`` is the list of ``(idx, suite_expected, description,
        decl_exit, decl_steps, bytecode, data)`` tuples for tests that
        passed compile + oracle.
      * ``errors`` is the list of ``ProgramResult`` entries for tests that
        failed before the neural run (compile error, oracle disagreement,
        non-halt, etc.). These are counted as FAIL in the summary.
    """
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program

    prepared: List[Tuple[int, int, str, int, int, list, bytes]] = []
    errors: List[ProgramResult] = []

    t0 = time.monotonic()
    for idx, source, expected, description in tests:
        try:
            bytecode, data = compile_c(source)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                ProgramResult(
                    idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=None,
                    declarative_steps=None,
                    neural_exit=None,
                    status="error",
                    error=f"compile error: {exc!r}",
                )
            )
            continue
        try:
            oracle = declarative_oracle_for_program(
                bytecode,
                data,
                suite_expected=expected,
                label=f"id={idx:04d}",
                max_steps=None,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                ProgramResult(
                    idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=None,
                    declarative_steps=None,
                    neural_exit=None,
                    status="error",
                    error=f"oracle exception: {exc!r}",
                )
            )
            continue
        if oracle.error is not None or oracle.steps is None:
            errors.append(
                ProgramResult(
                    idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=oracle.exit_code,
                    declarative_steps=oracle.steps,
                    neural_exit=None,
                    status="error",
                    error=oracle.error or "declarative did not halt",
                )
            )
            continue
        prepared.append(
            (
                idx,
                expected,
                description,
                int(oracle.exit_code),
                int(oracle.steps),
                bytecode,
                data,
            )
        )
    elapsed = time.monotonic() - t0
    print(
        f"[1096-fast] compile+oracle done in {elapsed:.1f}s "
        f"(prepared={len(prepared)}, oracle-errors={len(errors)})",
        file=sys.stderr,
        flush=True,
    )
    return prepared, errors


def _run_chunks(
    neural_runner,
    prepared: List[Tuple[int, int, str, int, int, list, bytes]],
    *,
    chunk_size: int,
    spec_k: int,
    max_context_window: int,
) -> List[ProgramResult]:
    """Run prepared programs through the neural runner in fixed chunks.

    Each chunk is a single ``run_batch`` call. Progress (cumulative
    pass/fail) is printed after every chunk.
    """
    results: List[ProgramResult] = []
    total = len(prepared)
    if total == 0:
        return results

    cum_pass = 0
    cum_fail = 0
    cum_error = 0

    for start in range(0, total, chunk_size):
        chunk = prepared[start : start + chunk_size]
        bytecodes = [entry[5] for entry in chunk]
        data_list = [entry[6] for entry in chunk]
        expected_steps = [entry[4] for entry in chunk]
        chunk_ids = [entry[0] for entry in chunk]

        chunk_t0 = time.monotonic()
        try:
            neural_results = neural_runner.run_batch(
                bytecodes,
                data_list=data_list,
                max_steps=None,
                expected_steps_list=expected_steps,
                max_context_window=max_context_window,
                spec_k=spec_k,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=2)
            for entry in chunk:
                idx, expected, description, decl_exit, decl_steps, _, _ = entry
                results.append(
                    ProgramResult(
                        idx=idx,
                        description=description,
                        suite_expected=expected,
                        declarative_exit=decl_exit,
                        declarative_steps=decl_steps,
                        neural_exit=None,
                        status="error",
                        error=f"neural batch error: {exc!r}",
                    )
                )
                cum_error += 1
            print(
                f"[1096-fast] chunk {start//chunk_size + 1} CRASHED: {exc!r}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
            continue

        chunk_elapsed = time.monotonic() - chunk_t0

        for entry, (_neural_output, neural_exit) in zip(chunk, neural_results):
            idx, expected, description, decl_exit, decl_steps, _, _ = entry
            if neural_exit is None:
                status = "error"
                cum_error += 1
                err = "neural exit is None (no halt within horizon)"
            else:
                ne = int(neural_exit) & 0xFFFFFFFF
                de = int(decl_exit) & 0xFFFFFFFF
                if ne == de:
                    status = "ok"
                    cum_pass += 1
                    err = None
                else:
                    status = "fail"
                    cum_fail += 1
                    err = None
            results.append(
                ProgramResult(
                    idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=decl_exit,
                    declarative_steps=decl_steps,
                    neural_exit=(
                        None if neural_exit is None else int(neural_exit) & 0xFFFFFFFF
                    ),
                    status=status,
                    error=err,
                )
            )

        done = start + len(chunk)
        print(
            f"[1096-fast] {done}/{total} done "
            f"ids={min(chunk_ids):04d}-{max(chunk_ids):04d} "
            f"chunk_wall={chunk_elapsed:.1f}s "
            f"max_steps={max(expected_steps)} | "
            f"cum: pass={cum_pass} fail={cum_fail} err={cum_error}",
            file=sys.stderr,
            flush=True,
        )

    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fast batched runner for the 1096 corpus.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N tests (default: all 1098).",
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
        help="Programs per neural batch (default: env C4_BATCH_CHUNK or 32).",
    )
    parser.add_argument(
        "--spec-k", type=int,
        default=int(os.environ.get("C4_SPEC_K", "4")),
        help="Speculative-decode K (default: env C4_SPEC_K or 4).",
    )
    parser.add_argument(
        "--max-context-window", type=int,
        default=int(os.environ.get("C4_BATCH_CONTEXT_WINDOW", "512")),
        help="Tail context window (default: 512).",
    )
    parser.add_argument(
        "--model-max-seq-len", type=int,
        default=int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096")),
        help="Model max seq len for the runner (default: 4096).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to dump full per-program results as JSON.",
    )
    parser.add_argument(
        "--print-failures", action="store_true",
        help="Stream per-failure row to stdout (in addition to summary).",
    )
    args = parser.parse_args(argv)

    # Honor the goal of running on GPU 0 by default. The harness only sets
    # CUDA_VISIBLE_DEVICES when nothing is already set, so the caller can
    # still override on the command line.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    overall_t0 = time.monotonic()

    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
    enumerated = list(enumerate(all_tests))

    if args.ids:
        wanted = set(_parse_ids(args.ids))
        selected = [
            (idx, src, exp, desc)
            for idx, (src, exp, desc) in enumerated
            if idx in wanted
        ]
    else:
        windowed = enumerated[args.offset:]
        if args.limit is not None:
            windowed = windowed[: args.limit]
        selected = [(idx, src, exp, desc) for idx, (src, exp, desc) in windowed]

    print(
        f"[1096-fast] selected={len(selected)} (of {len(all_tests)}) "
        f"chunk={args.chunk} spec_k={args.spec_k} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        file=sys.stderr,
        flush=True,
    )

    # Phase 1: compile + oracle (CPU-only).
    prepared, oracle_errors = _compile_and_oracle(selected)

    # Phase 2: build the neural runner (model bake).
    neural_runner = _build_neural_runner(model_max_seq_len=args.model_max_seq_len)

    # Phase 3: run prepared programs in chunks.
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

    print(
        f"\n[1096-fast] SUMMARY "
        f"total={len(all_results)} "
        f"pass={pass_n} fail={fail_n} error={err_n} "
        f"wall={total_elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    if args.print_failures:
        for r in all_results:
            if r.status != "ok":
                print(r.to_row(), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "wall_seconds": total_elapsed,
                    "summary": {
                        "total": len(all_results),
                        "pass": pass_n,
                        "fail": fail_n,
                        "error": err_n,
                    },
                    "chunk": args.chunk,
                    "spec_k": args.spec_k,
                    "results": [asdict(r) for r in all_results],
                },
                fh,
                indent=2,
            )
        print(f"[1096-fast] wrote {args.output}", file=sys.stderr, flush=True)

    return 0 if fail_n == 0 and err_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
