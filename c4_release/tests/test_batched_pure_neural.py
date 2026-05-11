"""Equivalence + benchmark tests for `BatchedPureNeuralRunner`.

The batched runner is an opt-in alternative to the serial pure-neural runner
(`AutoregressiveVMRunner(pure_neural=True)`). For correctness it must produce
exactly the same `(output, exit_code)` as the serial runner on every batch
element.

Tests in this file:

* `test_batched_matches_serial_phase1_small`: build a small batch of Phase-1
  style programs (IMM N; EXIT) and verify element-by-element equality with the
  serial runner.

* `test_batched_speedup_phase1`: benchmark a clean Phase-1 batch through both
  runners and assert the batched runner is at least 1.5x faster. Skipped if
  no GPU is available, since CPU forward time is dominated by Python-side work
  and doesn't benefit from batching.

The fixture is session-scoped so the compiled model is reused across tests in
this file.
"""

from __future__ import annotations

import time
import pytest

from neural_vm.embedding import Opcode


def _encode(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


# Programs deliberately chosen to halt cleanly in pure_neural mode (Phase 1).
# IMM with byte values that DON'T hit the 254/255 sign-extension bug.
_CLEAN_PHASE1_PROGRAMS = [
    [(Opcode.IMM, v), Opcode.EXIT] for v in [0, 1, 2, 5, 7, 11, 17, 23, 42, 100, 200]
]


def test_batched_matches_serial_phase1_small(
    pure_neural_runner, batched_pure_neural_runner
):
    """Element-by-element equivalence on a small batch."""
    programs = _CLEAN_PHASE1_PROGRAMS[:4]
    bcs = [_encode(p) for p in programs]

    # Serial reference.
    serial_results = []
    for bc in bcs:
        pure_neural_runner._memory = {}
        pure_neural_runner._mem_history = {}
        pure_neural_runner._mem_access_order = []
        out, code = pure_neural_runner.run(bc, b"", max_steps=10)
        serial_results.append((out, code))

    # Batched.
    batched_results = batched_pure_neural_runner.run_batch(bcs, max_steps=10)

    assert len(batched_results) == len(serial_results)
    for i, (b, s) in enumerate(zip(batched_results, serial_results)):
        assert b == s, f"batch element {i}: batched={b!r} != serial={s!r}"


@pytest.mark.slow
def test_batched_speedup_phase1(
    pure_neural_runner, batched_pure_neural_runner
):
    """Benchmark serial vs batched on Phase-1-style programs.

    Asserts a minimum speedup of 1.5x. On GPU the typical speedup is 3-4x
    for clean Phase-1 batches (11+ short programs that all halt quickly).
    """
    bcs = [_encode(p) for p in _CLEAN_PHASE1_PROGRAMS]
    N = len(bcs)

    # Serial.
    t0 = time.time()
    serial_results = []
    for bc in bcs:
        pure_neural_runner._memory = {}
        pure_neural_runner._mem_history = {}
        pure_neural_runner._mem_access_order = []
        serial_results.append(pure_neural_runner.run(bc, b"", max_steps=10))
    serial_total = time.time() - t0

    # Batched.
    t0 = time.time()
    batched_results = batched_pure_neural_runner.run_batch(bcs, max_steps=10)
    batched_total = time.time() - t0

    # Equivalence check (defensive; the small test above also covers this).
    for i, (b, s) in enumerate(zip(batched_results, serial_results)):
        assert b == s, (
            f"speedup-test element {i}: batched={b!r} != serial={s!r}"
        )

    speedup = serial_total / batched_total
    print(
        f"\n[batched bench] N={N}  serial={serial_total:.2f}s  "
        f"batched={batched_total:.2f}s  speedup={speedup:.2f}x"
    )
    assert speedup >= 1.5, (
        f"Expected batched runner to be >= 1.5x faster on {N} clean Phase-1 "
        f"programs, got {speedup:.2f}x (serial={serial_total:.2f}s, "
        f"batched={batched_total:.2f}s)"
    )
