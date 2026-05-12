"""Wall-time analysis for length-aware bucketing on the 1098 suite.

This test predicts every program's step count via DraftVM (no GPU required),
groups by bucket, and computes the wall-time speedup that bucketed batching
provides over a single mega-batch — both expressed in "token-time units"
(forward-pass count weighted by per-batch sequence length).

Wall-time model:
    single_batch_time  = max_steps * N * t_per_token
    bucketed_time      = Σ over buckets ( bucket_max * bucket_size ) * t_per_token

The DraftVM prediction cost is amortized over the run (~1ms per program in
pure Python, vs. minutes-per-program model forward).

Run with:
    timeout 300 python -m pytest tests/test_bucket_walltime_analysis.py \\
        -v --tb=short -s
"""

from __future__ import annotations

import os
import sys
import statistics
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_walltime_speedup_on_1098_suite():
    """Predict bucket distribution + wall-time savings for the 1098 suite.

    This is a no-GPU-required analysis: we run DraftVM upfront on every
    program (pure-Python interpreter, fast) and compute bucket statistics
    from the predictions. The actual model forward time then follows the
    "bucket_size * bucket_max" wall-time model.
    """
    from neural_vm.speculative import DraftVM
    from neural_vm.batched_pure_neural import (
        BatchedPureNeuralRunner,
        _DEFAULT_BUCKET_BOUNDS,
        _UNPREDICTED_BUCKET_KEY,
    )
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c

    tests = generate_test_programs()
    print(f"\n[bucket-analysis] suite size: {len(tests)} programs")

    predicted = []
    compile_errs = 0
    predict_errs = 0

    for source, expected, desc in tests:
        try:
            bc, data = compile_c(source)
        except Exception:
            compile_errs += 1
            predicted.append(None)
            continue
        try:
            vm = DraftVM(list(bc))
            if isinstance(data, (bytes, bytearray, list)):
                for i, b in enumerate(data):
                    vm.memory[0x10000 + i] = int(b)
            steps = vm.predict_steps(max_steps=2000)
            if not vm.halted:
                # DraftVM didn't reach EXIT — likely needs stdin/heap/PRTF
                # the prediction doesn't model. Counts as unpredicted.
                predicted.append(None)
            else:
                predicted.append(steps)
        except Exception:
            predict_errs += 1
            predicted.append(None)

    # Bucket distribution.
    buckets = Counter()
    for p in predicted:
        key = BatchedPureNeuralRunner._bucket_key(p, _DEFAULT_BUCKET_BOUNDS)
        buckets[key] += 1

    print(f"[bucket-analysis] compile errors: {compile_errs}")
    print(f"[bucket-analysis] predict errors: {predict_errs}")
    print()
    print("[bucket-analysis] distribution:")
    ordered = sorted(
        (k for k in buckets if k != _UNPREDICTED_BUCKET_KEY), key=lambda k: int(k)
    )
    total = sum(buckets.values())
    for k in ordered:
        pct = 100.0 * buckets[k] / total
        print(f"    <= {k:>6} steps: {buckets[k]:>4} programs ({pct:5.1f}%)")
    unpred = buckets.get(_UNPREDICTED_BUCKET_KEY, 0)
    print(f"    unpredicted   : {unpred:>4} programs ({100.0 * unpred / total:5.1f}%)")

    known = [p for p in predicted if p is not None]
    if known:
        print()
        print(
            f"[bucket-analysis] predicted step stats: min={min(known)} "
            f"max={max(known)} median={statistics.median(known):.0f} "
            f"mean={statistics.mean(known):.1f}"
        )

    # Wall-time model (units: token-time). Unpredicted programs cap at
    # ``max_steps``=2000, single-batch also caps at 2000.
    max_steps = 2000
    N_total = len(predicted)

    # Single batch: every program pads to max(P_i) or max_steps (whichever
    # bounds the model run). Since unpredicted programs run to max_steps,
    # the batch is padded to max_steps when any unpredicted exists; else
    # to max(known).
    if unpred > 0:
        single_pad = max_steps
    else:
        single_pad = max(known) if known else max_steps
    single_walltime = single_pad * N_total

    # Bucketed: each bucket pads to its own max (bucket upper bound is a
    # conservative upper bound for the bucket's true max).
    bucket_walltime = 0
    for k in ordered:
        bucket_walltime += int(k) * buckets[k]
    bucket_walltime += max_steps * unpred  # unpredicted bucket caps at max_steps

    speedup = single_walltime / bucket_walltime if bucket_walltime else float("inf")
    print()
    print("[bucket-analysis] wall-time model (token-time units):")
    print(f"    single batch (pad={single_pad}, N={N_total}): {single_walltime:>12,}")
    print(f"    bucketed (Σ bucket_max * bucket_size)        : {bucket_walltime:>12,}")
    print(f"    speedup: {speedup:.2f}x")

    # Sanity assertions — these are upper-bound estimates, but bucketing
    # should be at least a noticeable win whenever the suite has wide
    # length variance.
    assert speedup >= 1.0, (
        "Bucketing should never *slow down* the wall-time model; "
        f"got speedup={speedup:.2f}x"
    )


@pytest.mark.slow
def test_bucketed_real_walltime_vs_unbucketed(batched_pure_neural_runner):
    """Real-GPU wall-time: bucketed vs unbucketed on a mixed length batch.

    Builds a deliberately length-varied batch (mostly 2-step IMM/EXIT
    programs plus a couple longer ones). Bucketing should give a
    measurable wall-time win because the unbucketed path pads ALL programs
    to the longest one's length, while bucketed pads each bucket separately.

    Marked ``slow`` because each unbucketed forward iterates max_steps times
    on the longest program (the short programs HALT early but the model
    still drives the batch until everyone halts).
    """
    import time
    from neural_vm.embedding import Opcode

    # Mix: many short programs (2-step IMM/EXIT) + one "long" loop that
    # forces unbucketed-pad to grow. The "long" program is a synthetic
    # short loop: PSH; IMM 0; BNZ -8; EXIT — 4 steps but the unbucketed
    # batch pads to ``max_steps`` because of speculative HALT.
    short_bcs = [
        [(v << 8) | Opcode.IMM, Opcode.EXIT]
        for v in [0, 1, 2, 5, 7, 11, 17, 23, 42, 100]
    ]
    # Synthetic "longer" program: IMM K then EXIT — still short but a
    # different shape. Bucketing should still result in 0 extra cost.
    long_bcs = [
        [(50 << 8) | Opcode.IMM, (50 << 8) | Opcode.IMM, Opcode.EXIT],
    ]

    all_bcs = short_bcs + long_bcs
    N = len(all_bcs)
    print(f"\n[bucket-real-bench] N={N} ({len(short_bcs)} short + "
          f"{len(long_bcs)} long)")

    # Use small max_steps so the benchmark itself runs in seconds.
    MS = 10

    t0 = time.time()
    unbucketed = batched_pure_neural_runner.run_batch(
        all_bcs, max_steps=MS, bucket_by_predicted_length=False,
    )
    unbucketed_t = time.time() - t0

    t0 = time.time()
    bucketed = batched_pure_neural_runner.run_batch(
        all_bcs, max_steps=MS, bucket_by_predicted_length=True,
    )
    bucketed_t = time.time() - t0

    # Byte-identity check.
    assert bucketed == unbucketed, (
        "Bucketing changed per-program results — invariant violated."
    )

    speedup = unbucketed_t / bucketed_t if bucketed_t > 0 else float("inf")
    print(
        f"[bucket-real-bench] unbucketed={unbucketed_t:.2f}s  "
        f"bucketed={bucketed_t:.2f}s  speedup={speedup:.2f}x"
    )
