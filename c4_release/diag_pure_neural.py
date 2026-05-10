#!/usr/bin/env python3
"""Diagnostic runner for the pure-neural test suite (Phases 1-7).

Iterates the phase test files. Some live on sibling branches rather than the
current branch; for those we materialise the test file from
``git show <branch>:<path>`` into a temp file and point pytest at it.

Outputs a one-line-per-phase summary plus a grand total.

Usage examples
--------------

    python c4_release/diag_pure_neural.py
    python c4_release/diag_pure_neural.py --phases 1,2
    python c4_release/diag_pure_neural.py --quick
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------

# (phase -> (test path relative to c4_release/, branch (None == use working tree)))
PHASE_TESTS: dict[int, tuple[str, str | None]] = {
    1: ("tests/test_pure_neural_pc.py", None),                                          # on main
    2: ("tests/test_pure_neural_psh_add.py", None),                                     # on main
    3: ("tests/test_pure_neural_multibyte.py", "phase3-multibyte-pure-neural-tests"),
    4: ("tests/test_pure_neural_jmp_bz.py", "worktree-agent-af8a628b11c3ef028"),
    5: ("tests/test_pure_neural_jsr_ent_lev.py", "phase5-pure-neural-jsr-ent-lev"),
    6: ("tests/test_pure_neural_io.py", "phase6-io-pure-neural-tests"),
    7: ("tests/test_pure_neural_heap_div.py", "worktree-agent-a6893efcef9837dd3"),
}

# Phases that historically run > 120s (typically every full ALU sweep).
# --quick will skip these.
SLOW_PHASES: set[int] = {3, 7}

PER_TEST_TIMEOUT_SEC = 60
WALL_TIMEOUT_SEC_PER_PHASE = 900  # 15 min ceiling for a phase invocation


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Script lives at <worktree>/c4_release/diag_pure_neural.py.  The "repo top
# level" the user invokes from is the worktree root (one directory up).
HERE = Path(__file__).resolve().parent                 # .../c4_release
REPO_ROOT = HERE.parent                                # worktree root


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    phase: int
    test_path: str
    status: str  # 'ok', 'missing', 'collect-error', 'timeout', 'error', 'skipped'
    passed: int = 0
    failed: int = 0
    xfailed: int = 0
    xpassed: int = 0
    errored: int = 0
    skipped: int = 0
    total: int = 0
    failure_names: list[str] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.failure_names is None:
            self.failure_names = []

    @property
    def attempted(self) -> int:
        # tests that actually ran and produced a pass/fail signal
        return self.passed + self.failed + self.xfailed + self.xpassed + self.errored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_show(branch: str, repo_relative_path: str) -> str | None:
    """Return the contents of ``branch:c4_release/<repo_relative_path>`` or None.

    ``repo_relative_path`` is relative to the inner ``c4_release/`` package
    (matches the way PHASE_TESTS records paths).
    """
    git_path = f"c4_release/{repo_relative_path}"
    try:
        out = subprocess.run(
            ["git", "show", f"{branch}:{git_path}"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _materialise_test_file(phase: int, rel_path: str, branch: str, scratch_dir: Path) -> Path | None:
    """Fetch a test file from a sibling branch into a scratch location.

    The scratch layout mirrors ``c4_release/tests/<file>`` so pytest's
    ``conftest.py`` discovery still works (we copy conftest alongside it).
    """
    contents = _git_show(branch, rel_path)
    if contents is None:
        return None

    target_dir = scratch_dir / f"phase{phase}" / "tests"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy the project's conftest so fixtures resolve.  Symlinking would
    # also work but copy is portable across filesystems.
    src_conftest = HERE / "tests" / "conftest.py"
    if src_conftest.exists():
        shutil.copy2(src_conftest, target_dir / "conftest.py")

    target = target_dir / Path(rel_path).name
    target.write_text(contents)
    return target


def _resolve_test_file(phase: int, rel_path: str, branch: str | None,
                       scratch_dir: Path) -> tuple[Path | None, str]:
    """Return (path-on-disk, source-description) or (None, reason)."""
    in_tree = HERE / rel_path
    if in_tree.exists():
        return in_tree, "working tree"

    if branch is None:
        return None, "expected on current branch but file missing"

    materialised = _materialise_test_file(phase, rel_path, branch, scratch_dir)
    if materialised is None:
        return None, f"could not git show {branch}:c4_release/{rel_path}"
    return materialised, f"branch {branch}"


def _parse_junit(xml_path: Path) -> tuple[int, int, int, int, int, int, list[str]]:
    """Parse a JUnit XML report into counts.

    Returns (passed, failed, xfailed, xpassed, errored, skipped, failure_names).

    pytest emits xfail/xpass either as <skipped> or <failure> depending on
    the run flags; we use the message text and the ``--runxfail`` semantics
    to classify them.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return (0, 0, 0, 0, 0, 0, [])

    root = tree.getroot()
    # JUnit XML may have a top-level <testsuites> or just a single <testsuite>.
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    passed = failed = xfailed = xpassed = errored = skipped = 0
    failure_names: list[str] = []

    for suite in suites:
        for case in suite.findall("testcase"):
            name = case.get("name", "?")
            classname = case.get("classname", "")
            short_id = name
            # Use just the test method/parametrise id; full classname is noisy.
            failure_label = short_id

            failure_el = case.find("failure")
            error_el = case.find("error")
            skipped_el = case.find("skipped")

            if error_el is not None:
                errored += 1
                failure_names.append(failure_label)
                continue

            if failure_el is not None:
                # With --runxfail, expected-failures that still fail come through
                # as failures (xfail strict).  We treat everything here as a real
                # failure for the user-visible summary.
                msg = (failure_el.get("message") or "").lower()
                if "xfail" in msg or "expected failure" in msg:
                    xfailed += 1
                else:
                    failed += 1
                    failure_names.append(failure_label)
                continue

            if skipped_el is not None:
                msg = (skipped_el.get("message") or "").lower()
                stype = (skipped_el.get("type") or "").lower()
                if "xfail" in msg or "xfail" in stype or "expected failure" in msg:
                    xfailed += 1
                elif "xpass" in msg or "xpass" in stype:
                    xpassed += 1
                else:
                    skipped += 1
                continue

            passed += 1

    return passed, failed, xfailed, xpassed, errored, skipped, failure_names


def _format_failure_list(names: list[str], cap: int = 4) -> str:
    if not names:
        return ""
    if len(names) <= cap:
        return ", ".join(names)
    return ", ".join(names[:cap]) + f", +{len(names) - cap} more"


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------


def run_phase(phase: int, rel_path: str, branch: str | None,
              scratch_dir: Path) -> PhaseResult:
    test_file, source = _resolve_test_file(phase, rel_path, branch, scratch_dir)
    if test_file is None:
        return PhaseResult(
            phase=phase,
            test_path=rel_path,
            status="missing",
            detail=source,
        )

    junit_path = scratch_dir / f"phase{phase}.junit.xml"

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "--runxfail",
        f"--timeout={PER_TEST_TIMEOUT_SEC}",
        "-q", "--no-header", "--tb=no",
        "-p", "no:cacheprovider",
        f"--junit-xml={junit_path}",
    ]

    env = os.environ.copy()
    # Make sure the inner package is importable regardless of CWD.
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(HERE),
            capture_output=True,
            text=True,
            timeout=WALL_TIMEOUT_SEC_PER_PHASE,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return PhaseResult(
            phase=phase,
            test_path=rel_path,
            status="timeout",
            detail=f"wall timeout after {WALL_TIMEOUT_SEC_PER_PHASE}s",
        )
    except FileNotFoundError as exc:
        return PhaseResult(
            phase=phase,
            test_path=rel_path,
            status="error",
            detail=str(exc),
        )

    # rc=0 all pass; rc=1 some failed; rc=2 collection error; rc=5 no tests
    if proc.returncode == 2:
        # Collection error -- usually missing fixture / import.
        first_err = ""
        for line in proc.stdout.splitlines():
            if line.startswith("E ") or "ERROR" in line:
                first_err = line.strip()
                break
        return PhaseResult(
            phase=phase,
            test_path=rel_path,
            status="collect-error",
            detail=first_err or proc.stdout.splitlines()[-1] if proc.stdout else "(no output)",
        )

    if not junit_path.exists():
        # pytest crashed before writing the report
        tail = "\n".join((proc.stdout or proc.stderr).splitlines()[-3:])
        return PhaseResult(
            phase=phase,
            test_path=rel_path,
            status="error",
            detail=f"no junit output (rc={proc.returncode}): {tail}",
        )

    passed, failed, xfailed, xpassed, errored, skipped, failure_names = _parse_junit(junit_path)
    total = passed + failed + xfailed + xpassed + errored + skipped

    return PhaseResult(
        phase=phase,
        test_path=rel_path,
        status="ok",
        passed=passed,
        failed=failed,
        xfailed=xfailed,
        xpassed=xpassed,
        errored=errored,
        skipped=skipped,
        total=total,
        failure_names=failure_names,
        detail=source,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_phase_line(r: PhaseResult) -> str:
    if r.status == "missing":
        return f"Phase {r.phase}: MISSING ({r.detail})"
    if r.status == "collect-error":
        return f"Phase {r.phase}: COLLECT ERROR ({r.detail})"
    if r.status == "timeout":
        return f"Phase {r.phase}: TIMEOUT ({r.detail})"
    if r.status == "error":
        return f"Phase {r.phase}: ERROR ({r.detail})"
    if r.status == "skipped":
        return f"Phase {r.phase}: SKIPPED ({r.detail})"

    parts: list[str] = [f"{r.passed}/{r.total} PASS"]
    if r.failed:
        names = _format_failure_list(r.failure_names)
        parts.append(f"{r.failed} fail" + (f" ({names})" if names else ""))
    if r.errored:
        parts.append(f"{r.errored} error")
    if r.xfailed:
        parts.append(f"{r.xfailed} xfail")
    if r.xpassed:
        parts.append(f"{r.xpassed} xpass")
    if r.skipped:
        parts.append(f"{r.skipped} skipped")

    return f"Phase {r.phase}: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_phases(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--phases",
        type=str,
        default=None,
        help="Comma-separated phase numbers (e.g. '1,2,4' or '1-3'). Default: all.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=f"Skip slow phases (currently: {sorted(SLOW_PHASES)}).",
    )
    args = p.parse_args(argv)

    if args.phases:
        wanted = parse_phases(args.phases)
    else:
        wanted = sorted(PHASE_TESTS.keys())

    if args.quick:
        wanted = [ph for ph in wanted if ph not in SLOW_PHASES]

    unknown = [ph for ph in wanted if ph not in PHASE_TESTS]
    if unknown:
        print(f"unknown phase(s): {unknown}", file=sys.stderr)
        return 2

    print("=" * 64)
    print("pure-neural diagnostic")
    print(f"  worktree:   {REPO_ROOT}")
    print(f"  package:    {HERE}")
    print(f"  phases:     {wanted}")
    print(f"  per-test timeout: {PER_TEST_TIMEOUT_SEC}s")
    print("=" * 64)

    results: list[PhaseResult] = []
    with tempfile.TemporaryDirectory(prefix="diag_pure_neural_") as scratch:
        scratch_dir = Path(scratch)
        for phase in wanted:
            rel_path, branch = PHASE_TESTS[phase]
            print(f"-- running phase {phase} ({rel_path}) ...", flush=True)
            result = run_phase(phase, rel_path, branch, scratch_dir)
            results.append(result)
            print("   " + format_phase_line(result), flush=True)

    print()
    print("=" * 64)
    print("summary")
    print("=" * 64)
    total_pass = total_attempted = 0
    for r in results:
        print(format_phase_line(r))
        if r.status == "ok":
            total_pass += r.passed
            total_attempted += r.attempted

    skipped_phases = [
        ph for ph in PHASE_TESTS
        if ph not in wanted and (not args.phases or args.quick)
    ]
    if skipped_phases:
        print(f"(phases not run: {sorted(skipped_phases)})")

    print(f"Total: {total_pass}/{total_attempted} passing across {len(results)} phase(s)")

    # Non-zero exit if any phase had a hard failure or runtime error so
    # that CI shells notice.  xfails alone are not exit-worthy.
    bad = any(r.status not in {"ok", "skipped"} or r.failed or r.errored
              for r in results)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
