#!/usr/bin/env python3
"""Raw ``FFNRule`` constructor linter (Wave V9, DSL migration).

Scans ``c4_release/neural_vm/**/*.py`` for raw
``FFNRule.constant_write(...)`` / ``FFNRule.gated_write(...)`` calls
made OUTSIDE the declarative-IR / building-blocks DSL modules.

Background — Waves V1-V7 migrated the per-layer FFN bakes onto a
small set of building-block constructors in
``neural_vm/unified_compiler/building_blocks_dsl.py``
(``step_function_rule``, ``multi_way_and_rule``,
``band_range_check_rules``, ``cancel_residual_rule``,
``lookup_table_rules``, ``multi_way_or_rules``, etc.). New op code
should call one of those helpers rather than build ``FFNRule``s
directly so the BLOG_SPEC.md §504-568 building blocks are the source
of truth, not raw rule dicts.

Allowed call sites (no warning):
  * ``c4_release/neural_vm/unified_compiler/building_blocks_dsl.py``
    — every constructor lowers to ``FFNRule.constant_write`` /
    ``gated_write`` here on purpose.
  * ``c4_release/neural_vm/unified_compiler/wide_alu_dsl.py``
    — wide-ALU helpers (Waves W1-W7 follow-on).
  * ``c4_release/neural_vm/unified_compiler/ir.py``
    — the definitions of ``constant_write`` / ``gated_write`` live here.
  * ``c4_release/tests/**`` — tests may use the raw constructors for
    byte-identity comparisons.

Anywhere else (especially ``c4_release/neural_vm/unified_compiler/ops/*.py``)
is a migration target: rewrite via building-blocks DSL helpers, then
remove the entry from ``_BASELINE`` below.

The baseline (``_BASELINE``) is a per-file count of pre-existing raw
constructor calls captured at Wave V9 commit time. CI fails when:
  * A NEW non-allow-listed file appears, OR
  * An EXISTING baselined file's raw-constructor count GROWS.

Migrations shrink the count: update the baseline downward (or to 0)
in the same commit. This is a ratchet: counts only go down.

Usage::

    python c4_release/tools/lint_raw_ffn_rule.py             # repo root or c4_release
    python c4_release/tools/lint_raw_ffn_rule.py --json      # machine-readable
    python c4_release/tools/lint_raw_ffn_rule.py --list      # print every hit
    python c4_release/tools/lint_raw_ffn_rule.py --path PATH # lint a single file/dir

Exit codes:
  0  no regression vs baseline
  1  regression: new files or growing counts
  2  invocation / IO error
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Modules where raw ``FFNRule.constant_write`` / ``gated_write`` calls
# are expected (the DSL definitions themselves, the wide-ALU DSL, and
# the building-blocks DSL). Paths are relative to the repo root that
# contains ``c4_release/``.
_ALLOWED_FILES: frozenset = frozenset(
    {
        "c4_release/neural_vm/unified_compiler/building_blocks_dsl.py",
        "c4_release/neural_vm/unified_compiler/wide_alu_dsl.py",
        "c4_release/neural_vm/unified_compiler/ir.py",
    }
)


# Directories scanned. Restricted to ``neural_vm`` per the migration
# scope; tests are allowed to use raw constructors.
_SCAN_ROOTS: Tuple[str, ...] = ("c4_release/neural_vm",)


_SKIP_DIR_SEGMENTS = {
    "__pycache__",
    ".git",
    "archive",
    "deprecated",
    "old",
    "tests",
    "test_archive",
}


# Per-file count baseline captured at Wave V9 (commit cef9d046 +
# follow-ups). Files listed here had N raw-constructor calls at
# baseline; the lint allows up to N. New files or growing counts are
# regressions.
#
# After each per-file migration through the building-blocks DSL,
# decrement (or delete) the entry in the SAME commit so the ratchet
# only walks downward.
_BASELINE: Dict[str, int] = {
    "c4_release/neural_vm/unified_compiler/ops/l0_ops.py": 2,
    "c4_release/neural_vm/unified_compiler/ops/l11_ops.py": 1,
    "c4_release/neural_vm/unified_compiler/ops/l12_ops.py": 1,
    "c4_release/neural_vm/unified_compiler/ops/l1_ops.py": 2,
    "c4_release/neural_vm/unified_compiler/ops/l6_ops.py": 26,
    "c4_release/neural_vm/unified_compiler/ops/l9_ops.py": 5,
}


def _walk_python_files(root: Path) -> Iterable[Path]:
    """Yield every ``.py`` under ``root``, skipping caches / archives / tests."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_SEGMENTS]
        parts = set(Path(dirpath).parts)
        if parts & _SKIP_DIR_SEGMENTS:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _is_raw_ffn_rule_call(node: ast.AST) -> bool:
    """True when ``node`` is ``FFNRule.constant_write(...)`` or ``.gated_write(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in ("constant_write", "gated_write"):
        return False
    value = func.value
    if isinstance(value, ast.Name) and value.id == "FFNRule":
        return True
    return False


def _scan_file(path: Path) -> List[Tuple[int, str]]:
    """Return ``[(lineno, method)]`` for every raw FFNRule call in ``path``."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if _is_raw_ffn_rule_call(node):
            assert isinstance(node, ast.Call)
            attr = node.func.attr  # type: ignore[union-attr]
            hits.append((node.lineno, attr))
    return hits


def _is_allowed(rel_path: str) -> bool:
    """True when ``rel_path`` is a DSL module exempted from the lint."""
    for allowed in _ALLOWED_FILES:
        if rel_path == allowed or rel_path.endswith(
            allowed[len("c4_release/") :]
        ):
            return True
    return False


def scan(repo_root: Path, scan_roots: Iterable[str] = _SCAN_ROOTS) -> Dict[str, List[Tuple[int, str]]]:
    """Return ``{relative_path: [(lineno, method), ...]}`` for every non-allow-listed file."""
    by_file: Dict[str, List[Tuple[int, str]]] = {}
    for scan_rel in scan_roots:
        root = repo_root / scan_rel
        if not root.exists():
            # Maybe we were given the c4_release dir directly.
            alt = repo_root / scan_rel.replace("c4_release/", "")
            if alt.exists():
                root = alt
            else:
                continue
        for path in _walk_python_files(root):
            try:
                rel = str(path.relative_to(repo_root))
            except ValueError:
                rel = str(path)
            # Normalize to the c4_release/... shape so the baseline keys
            # match whether the caller passed the repo root or the
            # ``c4_release/`` directory.
            if not rel.startswith("c4_release/"):
                rel = "c4_release/" + rel
            if _is_allowed(rel):
                continue
            hits = _scan_file(path)
            if hits:
                by_file[rel] = hits
    return by_file


def diff_against_baseline(
    hits_by_file: Dict[str, List[Tuple[int, str]]],
    baseline: Dict[str, int] = _BASELINE,
) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """Compare current hits against baseline.

    Returns ``(regressions, new_files)`` where:
      * ``regressions`` is ``[(file, baseline_count, current_count)]`` for files
        that grew beyond their baseline.
      * ``new_files`` is the list of files NOT in baseline that have raw calls.
    """
    regressions: List[Tuple[str, int, int]] = []
    new_files: List[str] = []
    for rel, hits in sorted(hits_by_file.items()):
        n = len(hits)
        if rel not in baseline:
            new_files.append(rel)
            continue
        if n > baseline[rel]:
            regressions.append((rel, baseline[rel], n))
    return regressions, new_files


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument(
        "--list",
        action="store_true",
        help="print every hit (not just regressions vs baseline)",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="repo root (defaults to auto-detect from cwd)",
    )
    ap.add_argument(
        "--path",
        default=None,
        help=(
            "lint a single .py file or directory instead of the default "
            "scan roots (used by the unit test for planted violations)"
        ),
    )
    args = ap.parse_args(argv)

    if args.path is not None:
        target = Path(args.path).resolve()
        if not target.exists():
            print(f"error: --path target {target} not found", file=sys.stderr)
            return 2
        if target.is_file():
            hits = _scan_file(target)
            if args.json:
                print(
                    json.dumps(
                        [{"file": str(target), "line": ln, "method": m} for ln, m in hits],
                        indent=2,
                    )
                )
            elif hits:
                print(
                    f"lint_raw_ffn_rule: {len(hits)} raw FFNRule constructor(s) in {target}:"
                )
                for ln, m in hits:
                    print(
                        f"  {target}:{ln}: raw FFNRule constructor; use "
                        f"building_blocks_dsl helpers (step_function_rule, "
                        f"multi_way_and_rule, etc.)  [{m}]"
                    )
            else:
                print(f"lint_raw_ffn_rule: 0 raw FFNRule constructors in {target}")
            return 1 if hits else 0
        # Directory: walk it like a scan root.
        repo_root = target.parent
        hits_by_file: Dict[str, List[Tuple[int, str]]] = {}
        for p in _walk_python_files(target):
            file_hits = _scan_file(p)
            if file_hits:
                hits_by_file[str(p)] = file_hits
        total = sum(len(v) for v in hits_by_file.values())
        if args.json:
            print(
                json.dumps(
                    {
                        rel: [{"line": ln, "method": m} for ln, m in hs]
                        for rel, hs in hits_by_file.items()
                    },
                    indent=2,
                )
            )
        elif total:
            print(
                f"lint_raw_ffn_rule: {total} raw FFNRule constructor(s) "
                f"across {len(hits_by_file)} file(s) under {target}:"
            )
            for rel, hs in sorted(hits_by_file.items()):
                for ln, m in hs:
                    print(
                        f"  {rel}:{ln}: raw FFNRule constructor; use "
                        f"building_blocks_dsl helpers (step_function_rule, "
                        f"multi_way_and_rule, etc.)  [{m}]"
                    )
        else:
            print(f"lint_raw_ffn_rule: 0 raw FFNRule constructors under {target}")
        return 1 if total else 0

    if args.root:
        root = Path(args.root).resolve()
    else:
        cwd = Path.cwd().resolve()
        root = cwd
        # Walk up until we find a directory that contains ``c4_release/``
        # or whose name IS ``c4_release``.
        for _ in range(5):
            if (root / "c4_release").is_dir():
                break
            if (root / "neural_vm").is_dir() and root.name == "c4_release":
                root = root.parent
                break
            parent = root.parent
            if parent == root:
                break
            root = parent
        if not (root / "c4_release").is_dir():
            print(
                f"error: could not find c4_release/ from {cwd}",
                file=sys.stderr,
            )
            return 2

    hits_by_file = scan(root)
    total_hits = sum(len(v) for v in hits_by_file.values())
    regressions, new_files = diff_against_baseline(hits_by_file)

    if args.json:
        print(
            json.dumps(
                {
                    "total_raw_calls": total_hits,
                    "files": {
                        rel: [{"line": ln, "method": m} for ln, m in hs]
                        for rel, hs in hits_by_file.items()
                    },
                    "regressions": [
                        {"file": rel, "baseline": b, "current": c}
                        for rel, b, c in regressions
                    ],
                    "new_files": new_files,
                },
                indent=2,
            )
        )
        return 1 if (regressions or new_files) else 0

    if args.list:
        if not hits_by_file:
            print(
                "lint_raw_ffn_rule: 0 raw FFNRule constructors outside the "
                "allow-listed DSL modules."
            )
        else:
            print(
                f"lint_raw_ffn_rule: {total_hits} raw FFNRule constructor(s) "
                f"across {len(hits_by_file)} file(s):"
            )
            for rel, hs in sorted(hits_by_file.items()):
                base = _BASELINE.get(rel, 0)
                marker = " [BASELINED]" if rel in _BASELINE else " [NEW]"
                print(f"  {rel}: {len(hs)} hit(s), baseline {base}{marker}")
                for ln, m in hs:
                    print(
                        f"    {rel}:{ln}: raw FFNRule constructor; use "
                        f"building_blocks_dsl helpers (step_function_rule, "
                        f"multi_way_and_rule, etc.)  [{m}]"
                    )

    if not regressions and not new_files:
        print(
            f"lint_raw_ffn_rule: OK — {total_hits} raw FFNRule call(s) "
            f"across {len(hits_by_file)} file(s), all within baseline. "
            f"({len(_BASELINE)} files tracked.)"
        )
        return 0

    print(
        f"lint_raw_ffn_rule: REGRESSION — "
        f"{len(regressions)} growing file(s), {len(new_files)} new file(s)."
    )
    if regressions:
        print("\n  Files that grew beyond baseline:")
        for rel, baseline, current in regressions:
            print(f"    {rel}: baseline={baseline}, current={current} (+{current - baseline})")
    if new_files:
        print("\n  Files NOT in baseline that contain raw FFNRule calls:")
        for rel in new_files:
            for ln, m in hits_by_file[rel]:
                print(
                    f"    {rel}:{ln}: raw FFNRule constructor; use "
                    f"building_blocks_dsl helpers (step_function_rule, "
                    f"multi_way_and_rule, etc.)  [{m}]"
                )
    print(
        "\nSee c4_release/docs/BUILDING_BLOCKS_DSL.md for the canonical "
        "helpers. Migration shrinks counts; update _BASELINE in this "
        "tool in the same commit."
    )
    summary_total = sum(c - b for _, b, c in regressions) + sum(
        len(hits_by_file[f]) for f in new_files
    )
    print(
        f"\nSummary: {summary_total} violation(s) across "
        f"{len(regressions) + len(new_files)} file(s)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
