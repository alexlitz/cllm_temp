#!/usr/bin/env python3
"""Diff two smoke-track snapshots produced by ``tools/smoke_track.py``.

Two modes:

  smoke_diff.py <before.json> <after.json>
      Diff two explicit snapshot files.

  smoke_diff.py <after.json> --vs HEAD~N
      Pick the most recent snapshot (by mtime) in the same directory whose
      ``commit`` field matches ``git rev-parse HEAD~N`` and use it as
      ``before``. Errors out if no matching snapshot exists.

The diff shows:
  * Total pass/fail/error delta.
  * Per-class delta where any count changed.
  * Per-test transitions (passed->failed, failed->passed, etc.).
  * Tests added/removed between snapshots.

A non-zero exit status indicates ``passed`` regressed (i.e. ``after.passed
< before.passed``); otherwise the exit status is 0 even if individual
tests flipped (a net-neutral churn is still a clean status).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _git_rev_parse(rev: str, cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _find_snapshot_for_commit(directory: Path, commit_sha: str) -> Path | None:
    """Return the newest snapshot whose ``commit`` matches ``commit_sha``."""
    candidates: list[tuple[float, Path]] = []
    for p in directory.glob("smoke_*.json"):
        try:
            data = _load(p)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("commit") == commit_sha:
            candidates.append((p.stat().st_mtime, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def diff_manifests(before: dict, after: dict) -> dict:
    """Return a dict describing the diff (totals, by_class, transitions)."""
    totals_delta = {
        k: after["totals"].get(k, 0) - before["totals"].get(k, 0)
        for k in sorted(set(before["totals"]) | set(after["totals"]))
    }
    classes = sorted(set(before["by_class"]) | set(after["by_class"]))
    class_delta: dict[str, dict[str, int]] = {}
    empty = {
        "passed": 0, "failed": 0, "errors": 0,
        "skipped": 0, "xfailed": 0, "xpassed": 0,
    }
    for cls in classes:
        b = before["by_class"].get(cls, empty)
        a = after["by_class"].get(cls, empty)
        delta = {k: a.get(k, 0) - b.get(k, 0) for k in empty}
        if any(delta.values()):
            class_delta[cls] = delta

    before_by_nodeid = {t["nodeid"]: t for t in before.get("tests", [])}
    after_by_nodeid = {t["nodeid"]: t for t in after.get("tests", [])}

    transitions: list[dict] = []
    for nodeid, a in sorted(after_by_nodeid.items()):
        b = before_by_nodeid.get(nodeid)
        if b is None:
            transitions.append(
                {"nodeid": nodeid, "from": "<absent>", "to": a["outcome"]}
            )
        elif b["outcome"] != a["outcome"]:
            transitions.append(
                {"nodeid": nodeid, "from": b["outcome"], "to": a["outcome"]}
            )
    removed = [
        {"nodeid": nodeid, "from": b["outcome"], "to": "<absent>"}
        for nodeid, b in sorted(before_by_nodeid.items())
        if nodeid not in after_by_nodeid
    ]

    return {
        "before": {
            "commit": before.get("commit_short"),
            "snapshot_timestamp_utc": before.get("timestamp_utc"),
        },
        "after": {
            "commit": after.get("commit_short"),
            "snapshot_timestamp_utc": after.get("timestamp_utc"),
        },
        "totals_delta": totals_delta,
        "totals_before": before["totals"],
        "totals_after": after["totals"],
        "by_class_delta": class_delta,
        "transitions": transitions,
        "removed": removed,
    }


def _format_signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def render_diff(diff: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"smoke diff: {diff['before']['commit']} -> {diff['after']['commit']}"
    )
    lines.append("")
    lines.append("Totals (before -> after [delta]):")
    for key in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed"):
        b = diff["totals_before"].get(key, 0)
        a = diff["totals_after"].get(key, 0)
        d = diff["totals_delta"].get(key, 0)
        marker = ""
        if key == "passed" and d > 0:
            marker = "  <- improved"
        elif key == "passed" and d < 0:
            marker = "  <- REGRESSED"
        elif key in ("failed", "errors") and d > 0:
            marker = "  <- REGRESSED"
        elif key in ("failed", "errors") and d < 0:
            marker = "  <- improved"
        lines.append(
            f"  {key:8s} {b:4d} -> {a:4d}  [{_format_signed(d)}]{marker}"
        )
    lines.append("")

    if diff["by_class_delta"]:
        lines.append("Per-class changes:")
        for cls, delta in diff["by_class_delta"].items():
            parts = []
            for key, val in delta.items():
                if val:
                    parts.append(f"{key}={_format_signed(val)}")
            lines.append(f"  {cls}: {', '.join(parts)}")
        lines.append("")
    else:
        lines.append("Per-class changes: none.")
        lines.append("")

    if diff["transitions"]:
        lines.append("Test transitions:")
        for t in diff["transitions"]:
            lines.append(f"  [{t['from']} -> {t['to']}] {t['nodeid']}")
        lines.append("")

    if diff["removed"]:
        lines.append("Tests removed in 'after':")
        for t in diff["removed"]:
            lines.append(f"  [{t['from']} -> <absent>] {t['nodeid']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "after",
        help="Path to the 'after' snapshot JSON.",
    )
    parser.add_argument(
        "before",
        nargs="?",
        default=None,
        help="Path to the 'before' snapshot JSON (omit when using --vs).",
    )
    parser.add_argument(
        "--vs",
        default=None,
        help=(
            "Git rev (e.g. HEAD~1) to resolve as 'before' by looking up "
            "the matching snapshot in the same directory as 'after'."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the rendered text diff.",
    )
    args = parser.parse_args(argv)

    after_path = Path(args.after)
    after = _load(after_path)

    if args.before:
        before_path = Path(args.before)
        before = _load(before_path)
    elif args.vs:
        sha = _git_rev_parse(args.vs, after_path.parent)
        if not sha:
            sys.stderr.write(f"smoke_diff: cannot resolve git rev {args.vs!r}\n")
            return 2
        before_path = _find_snapshot_for_commit(after_path.parent, sha)
        if before_path is None:
            sys.stderr.write(
                f"smoke_diff: no snapshot found in {after_path.parent} "
                f"for commit {sha[:12]} ({args.vs}).\n"
            )
            return 2
        before = _load(before_path)
    else:
        sys.stderr.write(
            "smoke_diff: must supply either <before> or --vs <gitrev>.\n"
        )
        return 2

    diff = diff_manifests(before, after)
    if args.json:
        json.dump(diff, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_diff(diff))

    # exit non-zero only when passed count regressed
    return 1 if diff["totals_delta"].get("passed", 0) < 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
