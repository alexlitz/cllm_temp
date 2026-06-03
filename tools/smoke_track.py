#!/usr/bin/env python3
"""Smoke pass-count tracker.

Runs ``pytest c4_release/tests/test_smoke.py -q --tb=no`` and emits a JSON
manifest describing pass/fail counts per test class and the per-test outcome
for the current commit. Snapshots accumulate in a non-gitignored directory so
historical regressions become a simple JSON diff (see ``tools/smoke_diff.py``).

Output schema (JSON):

    {
      "schema_version": 1,
      "commit": "<full 40-char SHA>",
      "commit_short": "<short SHA>",
      "branch": "<current branch or 'detached'>",
      "timestamp_utc": "YYYY-mm-ddTHH:MM:SSZ",
      "pytest_args": ["c4_release/tests/test_smoke.py", "-q", "--tb=no", ...],
      "duration_seconds": <float>,
      "returncode": <int>,
      "totals": {
        "passed": <int>,
        "failed": <int>,
        "errors": <int>,
        "skipped": <int>,
        "xfailed": <int>,
        "xpassed": <int>
      },
      "by_class": {
        "TestSmokeBasic": {"passed": N, "failed": N, "errors": N,
                            "skipped": N, "xfailed": N, "xpassed": N},
        ...
      },
      "tests": [
        {"class": "TestSmokeBasic", "name": "test_add_basic",
         "nodeid": "c4_release/tests/test_smoke.py::TestSmokeBasic::test_add_basic",
         "outcome": "passed|failed|error|skipped|xfailed|xpassed"},
        ...
      ]
    }

Default output path:
    c4_release/.smoke-snapshots/smoke_<short_sha>_<utc_timestamp>.json

The directory is NOT in any ``.gitignore`` so snapshots can be committed if
desired (the tracker itself does not commit them).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# pytest -q line format:
#     c4_release/tests/test_smoke.py::TestSmokeBasic::test_add_basic PASSED
# (or FAILED / ERROR / SKIPPED / XFAIL / XPASS). With ``--tb=no`` failures
# still show one line in this form. We parse those lines directly.
_OUTCOME_RE = re.compile(
    r"^(?P<nodeid>\S+::(?P<cls>Test\w+)::(?P<name>test_\w+))"
    r"(?:\[[^\]]*\])?\s+"
    r"(?P<outcome>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
)

# pytest also emits short-form lines in the summary section, e.g.:
#     PASSED c4_release/tests/test_smoke.py::TestSmokeBasic::test_add_basic
_OUTCOME_RE_SHORT = re.compile(
    r"^(?P<outcome>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+"
    r"(?P<nodeid>\S+::(?P<cls>Test\w+)::(?P<name>test_\w+))"
)

_OUTCOME_MAP = {
    "PASSED": "passed",
    "FAILED": "failed",
    "ERROR": "error",
    "SKIPPED": "skipped",
    "XFAIL": "xfailed",
    "XPASS": "xpassed",
}


def _repo_root() -> Path:
    """Find the c4_release repo root (directory containing this tools/ dir)."""
    here = Path(__file__).resolve().parent
    return here.parent


def _git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def parse_pytest_output(text: str) -> list[dict]:
    """Extract per-test outcomes from pytest -q output.

    Returns a list of ``{"class", "name", "nodeid", "outcome"}`` dicts in
    the order they appear in the output. Duplicate node ids are dropped
    (first occurrence wins).
    """
    seen: set[str] = set()
    out: list[dict] = []
    for line in text.splitlines():
        line = line.rstrip()
        m = _OUTCOME_RE.match(line) or _OUTCOME_RE_SHORT.match(line)
        if not m:
            continue
        nodeid = m.group("nodeid")
        if nodeid in seen:
            continue
        seen.add(nodeid)
        out.append(
            {
                "class": m.group("cls"),
                "name": m.group("name"),
                "nodeid": nodeid,
                "outcome": _OUTCOME_MAP[m.group("outcome")],
            }
        )
    return out


def build_manifest(
    tests: list[dict],
    *,
    commit: str,
    commit_short: str,
    branch: str,
    pytest_args: list[str],
    duration: float,
    returncode: int,
) -> dict:
    counts_template = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    totals = dict(counts_template)
    by_class: dict[str, dict[str, int]] = {}
    for entry in tests:
        cls = entry["class"]
        if cls not in by_class:
            by_class[cls] = dict(counts_template)
        # totals key is plural for errors only
        key = "errors" if entry["outcome"] == "error" else entry["outcome"]
        totals[key] += 1
        by_class[cls][key] += 1
    return {
        "schema_version": 1,
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pytest_args": pytest_args,
        "duration_seconds": round(duration, 2),
        "returncode": returncode,
        "totals": totals,
        "by_class": dict(sorted(by_class.items())),
        "tests": tests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--smoke-path",
        default="c4_release/tests/test_smoke.py",
        help="pytest path to the smoke test file (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory to write the JSON snapshot into "
            "(default: c4_release/.smoke-snapshots, NOT gitignored)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Explicit output JSON path (overrides --out-dir naming).",
    )
    parser.add_argument(
        "--from-log",
        default=None,
        help=(
            "Skip pytest; parse outcomes from an existing log file. Useful for "
            "post-hoc snapshot generation. Returncode/duration recorded as 0."
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra pytest arg (repeatable). E.g. --extra -k --extra TestSmokeBasic.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pytest stdout/stderr stream (still saved into snapshot).",
    )
    args = parser.parse_args(argv)

    repo = _repo_root()
    commit = _git(["rev-parse", "HEAD"], repo) or "unknown"
    commit_short = (_git(["rev-parse", "--short", "HEAD"], repo) or commit[:8])[:12]
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo) or "detached"

    # ``-rA`` reports every test outcome (PASSED/FAILED/ERROR/SKIPPED/XFAIL/XPASS)
    # by name in the summary, which is what the parser relies on. Plain ``-q``
    # uses single-char status (``...EEE.FF``) and only lists failures/errors
    # by name in the summary, so PASSED counts would silently drop to 0.
    pytest_args = [args.smoke_path, "-q", "--tb=no", "-rA", *args.extra]

    if args.from_log:
        with open(args.from_log, "r", encoding="utf-8") as f:
            output_text = f.read()
        duration = 0.0
        returncode = 0
    else:
        cmd = [sys.executable, "-m", "pytest", *pytest_args]
        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        duration = time.time() - start
        output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        returncode = proc.returncode
        if not args.quiet:
            sys.stdout.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")

    tests = parse_pytest_output(output_text)

    manifest = build_manifest(
        tests,
        commit=commit,
        commit_short=commit_short,
        branch=branch,
        pytest_args=pytest_args,
        duration=duration,
        returncode=returncode,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else (repo / "c4_release" / ".smoke-snapshots")
        # If invoked from inside c4_release dir layout, the default uses repo/c4_release;
        # otherwise the worktree puts smoke snapshots at <repo>/c4_release/.smoke-snapshots.
        # Fall back to <repo>/.smoke-snapshots if c4_release is not a subdir.
        if not (repo / "c4_release").is_dir():
            out_dir = Path(args.out_dir) if args.out_dir else (repo / ".smoke-snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"smoke_{commit_short}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    totals = manifest["totals"]
    sys.stdout.write(
        f"\n[smoke_track] commit={commit_short} branch={branch} "
        f"passed={totals['passed']} failed={totals['failed']} "
        f"errors={totals['errors']} skipped={totals['skipped']} "
        f"xfail={totals['xfailed']} xpass={totals['xpassed']} "
        f"duration={manifest['duration_seconds']}s\n"
        f"[smoke_track] snapshot: {out_path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
