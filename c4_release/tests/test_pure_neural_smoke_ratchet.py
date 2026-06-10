#!/usr/bin/env python3
"""Pure-neural smoke ratchet — CI gate on the no-cheats baseline.

The Python ALU recovery overrides in ``neural_vm/batched_pure_neural.py``
(see ``test_batched_alu_recovery_toggle.py`` for the framework spec) mask
the vanilla-transformer thesis: the headline smoke pass count depends on
Python cheats, not neural compute. With
``C4_DISABLE_BATCHED_ALU_RECOVERY=1`` the cheats are off and the smoke
suite reports the honest neural-only pass count.

Per inventory ``df50a8c9``, that honest baseline is **28/51** as of the
current main. As upstream fixes land (each closing one of the per-op
neural blockers tracked in
``docs/SERIAL_MODE_DIVERGENCE_ATTRIBUTION_2026_06_09.md``) the pure-neural
count should monotone-increase toward 46/51 (the cheats-on number).

This test is the ratchet: it re-runs ``test_smoke.py`` with the global
disable env var set, parses the pytest summary, and asserts
``passed >= _PURE_NEURAL_BASELINE``. Regressions fail the gate; honest
improvements get ratcheted in by bumping ``_PURE_NEURAL_BASELINE``.

Marked ``@pytest.mark.slow`` because it forks a fresh ``pytest`` process
that rebuilds the pure-neural runner (the cheats-on session-scoped
fixture is unavailable to a subprocess) — wall time runs ~5-15 min.
Run with ``pytest --runslow tests/test_pure_neural_smoke_ratchet.py``
or ``make pure-neural-ratchet``.

For a fast (<30s) sanity gate on a representative subset see
``test_pure_neural_smoke_ratchet_micro.py``.

Ratcheting protocol
-------------------
1. After landing a fix that improves pure-neural smoke, re-run this
   gate locally and read the actual count from the printed line
   ``[ratchet] pure-neural smoke: X passed (baseline Y)``.
2. Bump ``_PURE_NEURAL_BASELINE`` to the new (higher) ``X`` in the same
   commit as the fix. Never decrement the baseline.
3. If you cannot raise the baseline because the fix is partial,
   document in the commit message why the count did not move and what
   blocker remains.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


# Honest pure-neural smoke pass count with C4_DISABLE_BATCHED_ALU_RECOVERY=1
# (no Python ALU recovery cheats). Source: inventory df50a8c9 — current
# baseline as of the commit landing this test. RATCHET ONLY UPWARD.
_PURE_NEURAL_BASELINE = 28


# Reporting-only constants: target = pass count with cheats ON, total =
# tests collected from test_smoke.py (excluding @slow). The assertion is
# on ``passed`` alone so a fix that flips a fail to skip does not silently
# improve the count.
_PURE_NEURAL_TARGET = 46
_PURE_NEURAL_TOTAL = 51


def _parse_pytest_summary(stdout: str, stderr: str) -> dict:
    """Pull ``X passed, Y failed, ...`` counts out of the pytest summary line.

    Pytest prints the final tally on the last non-empty line in formats like
    ``"28 passed, 23 failed in 412.34s"`` or ``"51 passed in 30s"``.
    We scan from the bottom and accept the first line that mentions
    ``passed`` or ``failed`` to be robust to interleaved warnings.
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    combined = (stdout + "\n" + stderr).splitlines()
    for line in reversed(combined):
        line = line.strip()
        if not line:
            continue
        if "passed" not in line and "failed" not in line and "error" not in line:
            continue
        # Match "N passed", "N failed", "N error[s]", "N skipped".
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


def _run_smoke_pure_neural(target: Path, *, extra_args: list[str] | None = None,
                           timeout: int = 1800) -> tuple[dict, subprocess.CompletedProcess]:
    """Spawn pytest against ``target`` with the ALU-cheat global disable.

    Returns ``(counts, completed_process)``. The caller owns assertion
    and reporting.
    """
    env = {**os.environ, "C4_DISABLE_BATCHED_ALU_RECOVERY": "1"}
    cmd = [
        sys.executable, "-m", "pytest", str(target),
        "--tb=no", "-q", "-p", "no:cacheprovider",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=False,
        timeout=timeout,
    )
    counts = _parse_pytest_summary(result.stdout, result.stderr)
    return counts, result


@pytest.mark.slow
def test_pure_neural_smoke_baseline_non_decreasing():
    """Re-run test_smoke.py with ALU recovery disabled; gate on the count.

    Fails if the honest pass count drops below ``_PURE_NEURAL_BASELINE``.
    Prints the actual count so a follow-up commit can ratchet the baseline
    upward when a fix lands.
    """
    repo_tests = Path(__file__).resolve().parent
    smoke_path = repo_tests / "test_smoke.py"
    assert smoke_path.exists(), f"missing smoke target: {smoke_path}"

    counts, result = _run_smoke_pure_neural(smoke_path)
    passed = counts["passed"]
    failed = counts["failed"]
    errors = counts["errors"]

    # Always print so follow-up agents can see the delta even when we pass.
    print(
        f"\n[ratchet] pure-neural smoke: {passed} passed, "
        f"{failed} failed, {errors} errors "
        f"(baseline {_PURE_NEURAL_BASELINE}, target {_PURE_NEURAL_TARGET}, "
        f"total {_PURE_NEURAL_TOTAL})",
        flush=True,
    )

    if passed == 0 and failed == 0 and errors == 0:
        # Collection failure or pytest startup error — surface stdout/stderr.
        pytest.fail(
            "Could not parse pytest summary; the subprocess likely failed "
            "before any test ran. Tail of stdout:\n"
            f"{result.stdout[-2000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
        )

    assert passed >= _PURE_NEURAL_BASELINE, (
        f"pure-neural smoke regressed: {passed} passed < baseline "
        f"{_PURE_NEURAL_BASELINE}. Someone either (a) re-enabled a Python "
        f"cheat the framework was meant to keep off, or (b) broke a neural "
        f"op that previously worked under "
        f"C4_DISABLE_BATCHED_ALU_RECOVERY=1. See "
        f"docs/SERIAL_MODE_DIVERGENCE_ATTRIBUTION_2026_06_09.md §7."
    )

    if passed > _PURE_NEURAL_BASELINE:
        print(
            f"[ratchet] HEADROOM: pure-neural smoke is now {passed}, "
            f"exceeding baseline {_PURE_NEURAL_BASELINE}. Bump "
            f"_PURE_NEURAL_BASELINE in this file (same commit as the fix) "
            f"so the gate ratchets upward.",
            flush=True,
        )
