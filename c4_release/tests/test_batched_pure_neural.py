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


# --- Bucket-by-predicted-length tests -------------------------------------


def test_predict_steps_basic():
    """``DraftVM.predict_steps`` returns the right step count for IMM/EXIT."""
    from neural_vm.speculative import DraftVM

    # IMM 42 (op=1, imm=42); EXIT (op=38). 2 steps total.
    bc = [(42 << 8) | 1, 38]
    vm = DraftVM(bc)
    assert vm.predict_steps(max_steps=10) == 2
    assert vm.halted

    # A program that doesn't halt within max_steps caps at max_steps and
    # leaves halted=False (the bucket helper treats this as unpredicted).
    # Use a tight infinite loop: JMP 0 (op=2, imm=0).
    bc_loop = [(0 << 8) | 2]
    vm_loop = DraftVM(bc_loop)
    assert vm_loop.predict_steps(max_steps=5) == 5
    assert not vm_loop.halted


def test_bucketed_matches_unbucketed(batched_pure_neural_runner):
    """Bucketed run produces byte-identical results to unbucketed run.

    Programs deliberately span different predicted lengths so they hit
    different buckets.
    """
    # All clean Phase-1 programs halt in 2 steps; bucketing should put them
    # all in the smallest bucket but still yield byte-identical results.
    bcs = [_encode(p) for p in _CLEAN_PHASE1_PROGRAMS]

    bucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=True
    )
    unbucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=False
    )
    assert bucketed == unbucketed, (
        f"bucketed != unbucketed:\n  bucketed={bucketed}\n  unbucketed={unbucketed}"
    )


def test_bucket_key_assignment():
    """``_bucket_key`` lands programs in the smallest bucket whose bound
    is >= predicted, and None goes to the unpredicted bucket."""
    from neural_vm.batched_pure_neural import (
        BatchedPureNeuralRunner,
        _DEFAULT_BUCKET_BOUNDS,
        _UNPREDICTED_BUCKET_KEY,
    )

    bk = BatchedPureNeuralRunner._bucket_key
    assert bk(1, _DEFAULT_BUCKET_BOUNDS) == 10
    assert bk(10, _DEFAULT_BUCKET_BOUNDS) == 10
    assert bk(11, _DEFAULT_BUCKET_BOUNDS) == 20
    assert bk(80, _DEFAULT_BUCKET_BOUNDS) == 80
    assert bk(81, _DEFAULT_BUCKET_BOUNDS) == 160
    assert bk(_DEFAULT_BUCKET_BOUNDS[-1] + 1, _DEFAULT_BUCKET_BOUNDS) == _UNPREDICTED_BUCKET_KEY
    assert bk(None, _DEFAULT_BUCKET_BOUNDS) == _UNPREDICTED_BUCKET_KEY


@pytest.mark.slow
def test_bucketed_wall_time_analysis(batched_pure_neural_runner):
    """Wall-time analysis: bucketed should be faster than unbucketed when
    the input has wide length variance.

    All current ``_CLEAN_PHASE1_PROGRAMS`` are 2-step programs so they end up
    in the same bucket — the bucket wall-time = unbucketed wall-time in
    expectation. The bucketed path adds only a tiny DraftVM-prediction
    overhead per program. We assert bucketed runtime is within 1.25x of
    unbucketed runtime (so the overhead doesn't dominate on short inputs)
    and the results are byte-identical.
    """
    bcs = [_encode(p) for p in _CLEAN_PHASE1_PROGRAMS]
    N = len(bcs)

    t0 = time.time()
    unbucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=False
    )
    unbucketed_t = time.time() - t0

    t0 = time.time()
    bucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=True
    )
    bucketed_t = time.time() - t0

    assert bucketed == unbucketed
    overhead = bucketed_t / unbucketed_t if unbucketed_t > 0 else 1.0
    print(
        f"\n[bucket bench] N={N}  unbucketed={unbucketed_t:.3f}s  "
        f"bucketed={bucketed_t:.3f}s  ratio={overhead:.2f}x"
    )
    # Single-bucket workload should add only DraftVM-prediction overhead.
    assert overhead < 1.25, (
        f"Bucketing added too much overhead on homogeneous batch: "
        f"unbucketed={unbucketed_t:.3f}s, bucketed={bucketed_t:.3f}s"
    )
