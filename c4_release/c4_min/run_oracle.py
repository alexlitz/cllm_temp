#!/usr/bin/env python3
"""c4_min per-op ORACLE RUNNER — compile a green-field program, decode the
model's autoregressive output, check it == expected, print a per-op PASS/FAIL
table (mirroring ``tools/run_per_op_oracle.py`` for the reference model).

Pipeline per program (see ``c4_min/DESIGN.md``):

  1. EXPECTED (ground truth) via ``c4_min.oracle.expected_for_program`` —
     the reference ISA semantics (``neural_vm.verification.symbolic_program``),
     the same source the reference per-op oracle trusts. Expected side ONLY.
  2. ``state_dict = c4_min.compile.compile_program(prog)``  (green-field).
  3. ``decoded  = c4_min.model.run(state_dict, prog)``       (green-field).
  4. verdict = ``oracle.compare(prog, expected, decoded)``.

The green-field ``compile`` / ``model`` modules are being built in parallel; if
they are not importable yet, the runner falls back to the ``--self-test`` mock
harness (a hand-mocked correct model AND an incorrect one) which PROVES the
harness detects both PASS and FAIL. When the real compiler lands it plugs
straight into ``_get_model_backend()`` with no other change.

Usage
-----
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_oracle.py --self-test
    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/run_oracle.py           # real backend
    python c4_min/run_oracle.py --op-classes ADD,SUB,LI
    python c4_min/run_oracle.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Protocol, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from c4_min import oracle  # noqa: E402
from c4_min.oracle import (  # noqa: E402
    Decoded, Expected, OpClassVerdict, Program, ProgramVerdict, compare,
)


# ---------------------------------------------------------------------------
# The model backend contract (see DESIGN.md "Model interface").
# ---------------------------------------------------------------------------


class ModelBackend(Protocol):
    """The green-field compile+run pair the harness drives."""

    def compile_program(self, prog: Program) -> dict: ...

    def run(self, state_dict: dict, prog: Program, *, max_steps: int = 64) -> Decoded: ...


class _GreenfieldBackend:
    """The real backend: ``c4_min.compile`` + ``c4_min.model`` (when they land).

    Kept behind a lazy import so the harness (and its self-test) work today,
    before the compiler exists. When ``c4_min/compile.py`` and ``c4_min/model.py``
    are present with the DESIGN.md signatures, this backend drives them directly.
    """

    def __init__(self) -> None:
        from c4_min import compile as _compile  # type: ignore
        from c4_min import model as _model      # type: ignore

        self._compile = _compile
        self._model = _model

    def compile_program(self, prog: Program) -> dict:
        return self._compile.compile_program(prog)

    def run(self, state_dict: dict, prog: Program, *, max_steps: int = 64) -> Decoded:
        out = self._model.run(state_dict, prog, max_steps=max_steps)
        # Accept either a Decoded or a raw byte stream (adapter path).
        if isinstance(out, Decoded):
            return out
        return oracle.decoded_from_bytes(out)


def _get_model_backend() -> Optional[ModelBackend]:
    """Return the real green-field backend, or None if not yet importable."""
    try:
        return _GreenfieldBackend()
    except Exception:  # noqa: BLE001 — compiler not built yet
        return None


# ---------------------------------------------------------------------------
# Mock backends for the SELF-TEST (prove the harness detects pass AND fail).
# ---------------------------------------------------------------------------


class MockCorrectBackend:
    """A hand-mocked CORRECT model: decodes exactly the reference behaviour.

    It "compiles" to a trivial state_dict and, on ``run``, returns the reference
    ISA VM's own (exit_code, steps, trace) — i.e. a perfectly faithful model.
    The harness MUST report every op-class PASS against this backend. (This
    proves the harness does not spuriously FAIL a correct model.)
    """

    def compile_program(self, prog: Program) -> dict:
        return {"_mock": "correct", "n_instr": len(prog.bytecode)}

    def run(self, state_dict: dict, prog: Program, *, max_steps: int = 64) -> Decoded:
        exp = oracle.expected_for_program(prog, max_steps=max_steps)
        return Decoded(exit_code=exp.exit_code, steps=exp.steps,
                       halted=exp.halted, trace=exp.trace)


class MockBuggyBackend:
    """A hand-mocked INCORRECT model, buggy in TWO structural ways:

      * exit-code bug: every exit code is off by one (``+1``).
      * trace bug: the AX of the last decoded step is corrupted.

    The harness MUST report FAIL for (almost) every op-class against this
    backend. (This proves the harness actually detects a wrong model.)
    """

    def compile_program(self, prog: Program) -> dict:
        return {"_mock": "buggy"}

    def run(self, state_dict: dict, prog: Program, *, max_steps: int = 64) -> Decoded:
        exp = oracle.expected_for_program(prog, max_steps=max_steps)
        if exp.exit_code is None:
            return Decoded(exit_code=None, steps=exp.steps, halted=exp.halted)
        bad_exit = (int(exp.exit_code) + 1) & 0xFFFFFFFF
        bad_trace: Tuple[Tuple[int, int], ...] = exp.trace
        if bad_trace:
            pc, ax = bad_trace[-1]
            bad_trace = bad_trace[:-1] + ((pc, (ax + 1) & 0xFFFFFFFF),)
        return Decoded(exit_code=bad_exit, steps=exp.steps,
                       halted=exp.halted, trace=bad_trace)


# ---------------------------------------------------------------------------
# The verdict engine over a backend.
# ---------------------------------------------------------------------------


def run_op_class(op: str, backend: ModelBackend, *, max_steps: int = 64) -> OpClassVerdict:
    """Decode-check every representative program of one op-class via ``backend``."""
    verdict = OpClassVerdict(op=op)
    for prog in oracle._programs_for(op):
        expected = oracle.expected_for_program(prog, max_steps=max_steps)
        try:
            state_dict = backend.compile_program(prog)
            decoded = backend.run(state_dict, prog, max_steps=max_steps)
        except Exception as exc:  # noqa: BLE001
            verdict.programs.append(ProgramVerdict(
                op=op, label=prog.label,
                expected_exit=expected.exit_code, got_exit=None,
                expected_steps=expected.steps, got_steps=None,
                status="model_error", detail=f"backend raised: {exc!r}",
            ))
            continue
        verdict.programs.append(compare(prog, expected, decoded))
    return verdict


def run_all(
    backend: ModelBackend,
    ops: Optional[List[str]] = None,
    *,
    max_steps: int = 64,
) -> List[OpClassVerdict]:
    ops = ops or list(oracle.ALL_OP_CLASSES)
    return [run_op_class(op, backend, max_steps=max_steps) for op in ops]


def print_table(results: List[OpClassVerdict], *, title: str) -> Tuple[int, int]:
    """Print a per-op PASS/FAIL table; return (n_ok, n_total)."""
    print("=" * 78)
    print(title)
    print("=" * 78)
    n_ok = 0
    for v in results:
        n_ok += 1 if v.ok else 0
        badge = "PASS" if v.ok else "FAIL"
        fails = [p for p in v.programs if not p.ok]
        extra = ""
        if fails:
            extra = "  <-- " + "; ".join(
                f"{p.label}[{p.status}: {p.detail}]" for p in fails
            )
        print(f"  [{badge}] {v.op:5s} {v.n_pass}/{len(v.programs)} progs pass{extra}")
    print("-" * 78)
    print(f"OP-CLASS PASS: {n_ok}/{len(results)}")
    return n_ok, len(results)


# ---------------------------------------------------------------------------
# Self-test: prove the harness detects a correct AND an incorrect model.
# ---------------------------------------------------------------------------


def self_test(*, verbose: bool = True) -> int:
    """Run the harness against a mocked CORRECT and a mocked BUGGY model.

    Asserts:
      * the CORRECT model -> every op-class PASS (harness doesn't false-fail);
      * the BUGGY model    -> every op-class FAIL (harness detects the bug).

    Returns 0 iff both hold. This is the harness's own regression gate: it runs
    with NO green-field compiler and NO GPU, purely on the reference semantics.
    """
    ops = list(oracle.ALL_OP_CLASSES)

    correct = run_all(MockCorrectBackend(), ops)
    if verbose:
        print_table(correct, title="SELF-TEST 1/2 — mocked CORRECT model "
                                   "(expect ALL PASS)")
        print()
    correct_ok = all(v.ok for v in correct)

    buggy = run_all(MockBuggyBackend(), ops)
    if verbose:
        print_table(buggy, title="SELF-TEST 2/2 — mocked BUGGY model "
                                 "(expect ALL FAIL)")
        print()
    buggy_all_fail = all(not v.ok for v in buggy)

    # Report the two structural claims explicitly.
    ok = True
    print("SELF-TEST RESULT")
    print("-" * 40)
    if correct_ok:
        print(f"  [OK]   correct model -> {len(ops)}/{len(ops)} op-classes PASS "
              f"(harness does not false-fail a correct model)")
    else:
        bad = [v.op for v in correct if not v.ok]
        print(f"  [BAD]  correct model FAILED op-classes: {bad}")
        ok = False
    if buggy_all_fail:
        print(f"  [OK]   buggy model   -> {len(ops)}/{len(ops)} op-classes FAIL "
              f"(harness detects a wrong model)")
    else:
        good = [v.op for v in buggy if v.ok]
        print(f"  [BAD]  buggy model PASSED op-classes (undetected bug!): {good}")
        ok = False
    print("-" * 40)
    print("SELF-TEST", "PASSED — harness detects BOTH pass and fail." if ok
          else "FAILED.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _list_programs() -> None:
    for op, progs in oracle.programs_by_op().items():
        print(f"  {op:5s}: {[p.label for p in progs]}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run the mocked correct+buggy harness self-test "
                         "(no green-field compiler needed).")
    ap.add_argument("--op-classes", default=None,
                    help="comma-separated op-classes (default: all 30).")
    ap.add_argument("--list", action="store_true",
                    help="print the op-class -> representative programs and exit.")
    ap.add_argument("--max-steps", type=int, default=64)
    args = ap.parse_args(argv)

    if args.list:
        _list_programs()
        return 0

    if args.self_test:
        return self_test()

    backend = _get_model_backend()
    if backend is None:
        print("[c4_min] green-field backend (c4_min.compile / c4_min.model) is "
              "not importable yet — running the harness SELF-TEST instead.\n"
              "         (the real backend plugs in automatically once it lands; "
              "see c4_min/DESIGN.md).\n", file=sys.stderr)
        return self_test()

    ops = (
        [o.strip().upper() for o in args.op_classes.split(",") if o.strip()]
        if args.op_classes else list(oracle.ALL_OP_CLASSES)
    )
    unknown = [o for o in ops if o not in oracle.ALL_OP_CLASSES]
    if unknown:
        print(f"unknown op-classes: {unknown}", file=sys.stderr)
        return 2

    results = run_all(backend, ops, max_steps=args.max_steps)
    n_ok, n_total = print_table(
        results,
        title="c4_min PER-OP DECODE ORACLE  (green-field model vs reference ISA)",
    )
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
