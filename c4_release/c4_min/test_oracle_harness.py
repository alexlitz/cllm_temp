"""Pytest gate for the c4_min per-op oracle harness (no green-field compiler,
no GPU). Proves the harness itself is sound: it PASSes a correct model and
FAILs a buggy one, and every generated program has a well-defined reference
expected output.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) pytest -q c4_min/test_oracle_harness.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from c4_min import oracle                       # noqa: E402
from c4_min.oracle import Decoded, compare      # noqa: E402
from c4_min.run_oracle import (                 # noqa: E402
    MockBuggyBackend, MockCorrectBackend, run_all,
)


def test_every_generated_program_has_a_reference_expected_output():
    """The reference ISA VM must halt with an exit code for every program."""
    for op, pairs in oracle.expected_by_op().items():
        assert pairs, f"{op}: no representative programs"
        for prog, exp in pairs:
            assert exp.error is None, f"{op}/{prog.label}: {exp.error}"
            assert exp.halted, f"{op}/{prog.label}: did not halt"
            assert exp.exit_code is not None
            assert exp.steps and exp.steps > 0


def test_harness_passes_a_correct_model():
    """A perfectly faithful model -> every op-class PASS (no false-fail)."""
    results = run_all(MockCorrectBackend())
    bad = [v.op for v in results if not v.ok]
    assert not bad, f"harness FALSE-FAILED a correct model on: {bad}"


def test_harness_detects_a_buggy_model():
    """A wrong model -> every op-class FAIL (bug is detected)."""
    results = run_all(MockBuggyBackend())
    undetected = [v.op for v in results if v.ok]
    assert not undetected, f"harness MISSED the bug on: {undetected}"


def test_trace_divergence_detected_even_with_correct_exit_code():
    """The full-trace comparison catches a wrong step even if exit matches."""
    prog = oracle._programs_for("ADD")[0]
    exp = oracle.expected_for_program(prog)
    bad_trace = list(exp.trace)
    assert len(bad_trace) >= 2
    pc, ax = bad_trace[1]
    bad_trace[1] = (pc, (ax + 7) & 0xFFFFFFFF)   # corrupt a mid-step AX
    decoded = Decoded(exit_code=exp.exit_code, steps=exp.steps,
                      halted=True, trace=tuple(bad_trace))
    v = compare(prog, exp, decoded)
    assert v.status == "fail" and "trace diverged" in v.detail


def test_exit_only_conformer_is_accepted():
    """A model that reports only an exit code (no trace) still PASSes."""
    prog = oracle._programs_for("MUL")[0]
    exp = oracle.expected_for_program(prog)
    decoded = Decoded(exit_code=exp.exit_code, steps=exp.steps, halted=True)
    assert compare(prog, exp, decoded).status == "pass"


def test_self_test_entry_point_returns_zero():
    from c4_min.run_oracle import self_test

    assert self_test(verbose=False) == 0
