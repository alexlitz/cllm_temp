#!/usr/bin/env python3
"""Pure-neural smoke ratchet (micro) — fast sanity gate.

Companion to ``test_pure_neural_smoke_ratchet.py``. The full ratchet
spawns a fresh ``pytest`` against the whole 51-test smoke suite in
pure-neural mode and takes 5-15 min, so it lives behind ``@slow``.

This micro version runs only a small representative subset (one
``TestSmokeBasic`` class through the existing per-class batched
fixture) so it completes in <30s on a warm GPU. It is meant as a
fast pre-push sanity check — if the micro count drops, the full
ratchet is almost certainly regressed too. If the micro count is
fine but the full ratchet fails, the regression is in a class the
micro does not cover.

Like the full ratchet, the assertion is on the no-cheats pass count
under ``C4_DISABLE_BATCHED_ALU_RECOVERY=1``.

Run with:
    pytest tests/test_pure_neural_smoke_ratchet_micro.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Representative micro-subset: the 6 ``TestSmokeBasic`` tests
# (IMM/ADD/SUB/MUL/DIV/MOD on small immediates). Covers the smallest
# ALU + EXIT happy path. This subset's no-cheats baseline is recorded
# below; it ratchets the same way as the full baseline.
#
# As of inventory df50a8c9 the basic subset reports 1/6 honestly
# (IMM-only — see ``test_smoke_pure_neural.py`` docstring), matching
# the "Phase 8 starting point" the Phase 1-7 closure work targets.
_PURE_NEURAL_MICRO_BASELINE = 1
_PURE_NEURAL_MICRO_TOTAL = 6


# Hard wall-clock cap so a hung subprocess doesn't blow past the
# "fast CI" budget. The basic class on a warm GPU should clear in
# well under 30s; the timeout is set higher to absorb cold-bake.
_MICRO_TIMEOUT_SECONDS = 180


def _parse_pytest_summary(stdout: str, stderr: str) -> dict:
    """Parse ``X passed, Y failed, ...`` counts from pytest's summary line.

    Mirrors the parser in ``test_pure_neural_smoke_ratchet.py`` (kept
    inline so the micro file stays self-contained).
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    combined = (stdout + "\n" + stderr).splitlines()
    for line in reversed(combined):
        line = line.strip()
        if not line:
            continue
        if "passed" not in line and "failed" not in line and "error" not in line:
            continue
        for key, token in (
            ("passed", "passed"),
            ("failed", "failed"),
            ("errors", "error"),
            ("skipped", "skipped"),
        ):
            m = re.search(rf"(\d+)\s+{token}", line)
            if m:
                counts[key] = int(m.group(1))
        if counts["passed"] or counts["failed"] or counts["errors"]:
            return counts
    return counts


def test_pure_neural_smoke_micro_baseline_non_decreasing():
    """Run only ``TestSmokeBasic`` under the no-cheats env var.

    Asserts ``passed >= _PURE_NEURAL_MICRO_BASELINE``. Prints the actual
    count to help future agents bump the baseline as fixes land.
    """
    repo_tests = Path(__file__).resolve().parent
    smoke_path = repo_tests / "test_smoke.py"
    assert smoke_path.exists(), f"missing smoke target: {smoke_path}"

    env = {**os.environ, "C4_DISABLE_BATCHED_ALU_RECOVERY": "1"}
    # ``-k`` selects only the basic class. ``test_smoke.py`` honors the
    # pytest selection through its ``_selected_group_tests`` helper so
    # only the basic 6 actually run through the batched fixture.
    cmd = [
        sys.executable, "-m", "pytest", str(smoke_path),
        "-k", "TestSmokeBasic",
        "--tb=no", "-q", "-p", "no:cacheprovider",
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=False,
            timeout=_MICRO_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"micro ratchet exceeded {_MICRO_TIMEOUT_SECONDS}s wall budget "
            f"— pure-neural runner likely hung. Partial stdout:\n"
            f"{(exc.stdout or b'')[-2000:]!r}"
        )
    elapsed = time.time() - t0

    counts = _parse_pytest_summary(result.stdout, result.stderr)
    passed = counts["passed"]
    failed = counts["failed"]
    errors = counts["errors"]

    print(
        f"\n[ratchet:micro] pure-neural basic: {passed}/{_PURE_NEURAL_MICRO_TOTAL} "
        f"passed, {failed} failed, {errors} errors "
        f"(baseline {_PURE_NEURAL_MICRO_BASELINE}, wall {elapsed:.1f}s)",
        flush=True,
    )

    if passed == 0 and failed == 0 and errors == 0:
        pytest.fail(
            "Could not parse pytest summary; subprocess likely failed before "
            "tests ran. Tail of stdout:\n"
            f"{result.stdout[-2000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
        )

    assert passed >= _PURE_NEURAL_MICRO_BASELINE, (
        f"pure-neural micro smoke regressed: {passed} passed < baseline "
        f"{_PURE_NEURAL_MICRO_BASELINE}. The full ratchet "
        f"(test_pure_neural_smoke_ratchet.py) is almost certainly also "
        f"regressed; run it (under --runslow) to confirm and read the "
        f"per-class breakdown."
    )

    if passed > _PURE_NEURAL_MICRO_BASELINE:
        print(
            f"[ratchet:micro] HEADROOM: now {passed}, baseline "
            f"{_PURE_NEURAL_MICRO_BASELINE}. Bump _PURE_NEURAL_MICRO_BASELINE "
            f"in the same commit as the fix.",
            flush=True,
        )
