#!/usr/bin/env python3
"""Raw IR-constructor linter (Wave V9 + AP/AO expansion, DSL migration).

Scans ``c4_release/neural_vm/**/*.py`` for two classes of raw IR-constructor
calls outside the declarative-IR / building-blocks DSL modules:

  1. ``FFNRule.constant_write(...)`` / ``FFNRule.gated_write(...)`` —
     the original Wave V9 lint. These are the FFN-rule constructors;
     new op code should call a ``building_blocks_dsl`` helper instead.
  2. ``AP(...)`` / ``AO(...)`` — the attention-projection / output
     constructors from ``primitives.py``. New op code should compose
     attention specs through a building-block / allocator helper
     rather than hand-rolling per-slot ``AP``/``AO`` lists.

Background — Waves V1-V7 migrated the per-layer FFN bakes onto a
small set of building-block constructors in
``neural_vm/unified_compiler/building_blocks_dsl.py``
(``step_function_rule``, ``multi_way_and_rule``,
``band_range_check_rules``, ``cancel_residual_rule``,
``lookup_table_rules``, ``multi_way_or_rules``, etc.). The same
ratcheting discipline now applies at the attention layer: ops
should not hand-build ``AP(slot, BD.X, w)`` / ``AO(BD.X, slot, w)``
lists; instead they should compose through the building-block
attention helpers in ``building_blocks_dsl.py`` (or call into the
``attention_head_allocator``).

Allowed call sites (no warning):
  * ``c4_release/neural_vm/unified_compiler/building_blocks_dsl.py``
    — every constructor lowers to ``FFNRule.constant_write`` /
    ``gated_write`` / ``AP`` / ``AO`` here on purpose.
  * ``c4_release/neural_vm/unified_compiler/wide_alu_dsl.py``
    — wide-ALU helpers (Waves W1-W7 follow-on).
  * ``c4_release/neural_vm/unified_compiler/ir.py``
    — the definitions of ``constant_write`` / ``gated_write`` live here.
  * ``c4_release/neural_vm/unified_compiler/primitives.py``
    — the definitions of ``AP`` / ``AO`` live here.
  * ``c4_release/neural_vm/attention_head_allocator.py``
    — the head allocator owns raw attention-spec composition.
  * ``c4_release/tests/**`` — tests may use the raw constructors for
    byte-identity comparisons.

Anywhere else (especially ``c4_release/neural_vm/unified_compiler/ops/*.py``)
is a migration target: rewrite via building-blocks DSL helpers, then
remove the entry from the appropriate baseline below.

Two baselines (``_BASELINE`` for FFN, ``_AP_AO_BASELINE`` for attention)
are per-file count snapshots captured at lint-extension time. CI fails
when:
  * A NEW non-allow-listed file appears in the FFN scan, OR
  * An EXISTING baselined file's FFN raw-constructor count GROWS.

The AP/AO scan is **advisory only** for now: it prints a warning
summary and a per-file breakdown, but does not flip the exit code
unless ``--strict-ap-ao`` is passed. The baseline still ratchets — if
an AP/AO file grows beyond its baseline count under ``--strict-ap-ao``
the lint exits 1. Default is non-blocking so the existing CI gate
remains the FFN ratchet only.

Migrations shrink the count: update the baseline downward (or to 0)
in the same commit. This is a ratchet: counts only go down.

Usage::

    python c4_release/tools/lint_raw_ffn_rule.py             # repo root or c4_release
    python c4_release/tools/lint_raw_ffn_rule.py --json      # machine-readable
    python c4_release/tools/lint_raw_ffn_rule.py --list      # print every hit
    python c4_release/tools/lint_raw_ffn_rule.py --path PATH # lint a single file/dir
    python c4_release/tools/lint_raw_ffn_rule.py --strict-ap-ao
                                                              # AP/AO ratchet also exits 1

Exit codes:
  0  no regression vs baseline
  1  regression: new files or growing counts (FFN; AP/AO only with --strict-ap-ao)
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


# Modules where raw ``AP(...)`` / ``AO(...)`` calls are expected: the
# primitives module that defines them, the building-blocks/wide-ALU
# DSLs that compose declarative attention specs, the IR module, and
# the attention-head allocator (which legitimately threads raw
# AP/AO writes through generic per-head plumbing).
_ALLOWED_FILES_AP_AO: frozenset = frozenset(
    {
        "c4_release/neural_vm/unified_compiler/building_blocks_dsl.py",
        "c4_release/neural_vm/unified_compiler/wide_alu_dsl.py",
        "c4_release/neural_vm/unified_compiler/ir.py",
        "c4_release/neural_vm/unified_compiler/primitives.py",
        "c4_release/neural_vm/attention_head_allocator.py",
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
    # All previously-baselined files migrated to 0 raw constructor
    # calls as of 2026-06-04 (commits bc8d999a + ce56aab2 + 608b3c65 +
    # the model_ops/flag_gated_ops/band_guarantees migration series).
    # Ratchet held at empty — any future raw constructor in any
    # non-allowlisted file under ``c4_release/neural_vm/`` fails CI.
}


# Per-file ``AP(...)`` / ``AO(...)`` baseline captured 2026-06-05 at the
# lint-extension commit. The count is intentionally generous so the
# ratchet sets the ceiling at "today's number" and only walks downward
# as ops migrate to building-block attention helpers.
#
# Migration target files (top of the list = most leverage). Decrement
# entries in the SAME commit that migrates a file through the DSL.
_AP_AO_BASELINE: Dict[str, int] = {
    "c4_release/neural_vm/unified_compiler/ops/l10_ops.py": 311,
    "c4_release/neural_vm/unified_compiler/ops/l14_ops.py": 161,
    "c4_release/neural_vm/unified_compiler/ops/l6_ops.py": 147,
    "c4_release/neural_vm/unified_compiler/ops/l7_ops.py": 108,
    "c4_release/neural_vm/unified_compiler/ops/l8_ops.py": 61,
    "c4_release/neural_vm/unified_compiler/ops/model_ops.py": 57,
    "c4_release/neural_vm/unified_compiler/ops/l3_ops.py": 56,
    "c4_release/neural_vm/unified_compiler/ops/l15_ops.py": 49,
    "c4_release/neural_vm/unified_compiler/ops/l5_ops.py": 47,
    "c4_release/neural_vm/unified_compiler/ops/l9_ops.py": 33,
    "c4_release/neural_vm/unified_compiler/ops/l4_ops.py": 25,
    "c4_release/neural_vm/unified_compiler/ops/flag_gated_ops.py": 23,
    "c4_release/neural_vm/unified_compiler/ops/l13_ops.py": 21,
    "c4_release/neural_vm/unified_compiler/ops/l1_ops.py": 20,
    "c4_release/neural_vm/unified_compiler/ops/control_flow_heads.py": 15,
    "c4_release/neural_vm/unified_compiler/ops/l2_ops.py": 8,
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


def _is_raw_ap_ao_call(node: ast.AST) -> bool:
    """True when ``node`` is ``AP(...)`` or ``AO(...)`` — a bare-name call.

    We intentionally only match bare-name calls (``AP(...)`` and
    ``AO(...)``) — both ``primitives.AP(...)`` and
    ``Primitives.something_AP(...)`` are out of scope: the ops files
    universally ``from ..primitives import AO, AP``, so the bare-name
    form covers every real call site.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Name):
        return False
    return func.id in ("AP", "AO")


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


def _scan_file_ap_ao(path: Path) -> List[Tuple[int, str]]:
    """Return ``[(lineno, name)]`` for every raw ``AP(...)``/``AO(...)`` call."""
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
        if _is_raw_ap_ao_call(node):
            assert isinstance(node, ast.Call)
            name = node.func.id  # type: ignore[union-attr]
            hits.append((node.lineno, name))
    return hits


def _is_allowed(rel_path: str) -> bool:
    """True when ``rel_path`` is a DSL module exempted from the FFN lint."""
    for allowed in _ALLOWED_FILES:
        if rel_path == allowed or rel_path.endswith(
            allowed[len("c4_release/") :]
        ):
            return True
    return False


def _is_allowed_ap_ao(rel_path: str) -> bool:
    """True when ``rel_path`` is exempted from the AP/AO lint."""
    for allowed in _ALLOWED_FILES_AP_AO:
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


def scan_ap_ao(
    repo_root: Path, scan_roots: Iterable[str] = _SCAN_ROOTS
) -> Dict[str, List[Tuple[int, str]]]:
    """Return ``{relative_path: [(lineno, name), ...]}`` for AP/AO outside allowlist."""
    by_file: Dict[str, List[Tuple[int, str]]] = {}
    for scan_rel in scan_roots:
        root = repo_root / scan_rel
        if not root.exists():
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
            if not rel.startswith("c4_release/"):
                rel = "c4_release/" + rel
            if _is_allowed_ap_ao(rel):
                continue
            hits = _scan_file_ap_ao(path)
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
    ap.add_argument(
        "--strict-ap-ao",
        action="store_true",
        help=(
            "treat AP/AO ratchet regressions as failures (default is "
            "advisory-only: warning printed, exit code unaffected)"
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
            ap_ao_hits = _scan_file_ap_ao(target)
            if args.json:
                print(
                    json.dumps(
                        {
                            "ffn": [
                                {"file": str(target), "line": ln, "method": m}
                                for ln, m in hits
                            ],
                            "ap_ao": [
                                {"file": str(target), "line": ln, "name": n}
                                for ln, n in ap_ao_hits
                            ],
                        },
                        indent=2,
                    )
                )
            else:
                if hits:
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
                if ap_ao_hits:
                    print(
                        f"lint_raw_ffn_rule: {len(ap_ao_hits)} raw AP/AO call(s) in {target}:"
                    )
                    for ln, n in ap_ao_hits:
                        print(
                            f"  {target}:{ln}: raw {n}(...) call; compose via "
                            f"building_blocks_dsl attention helpers or "
                            f"attention_head_allocator instead.  [{n}]"
                        )
            # FFN hits flip exit code; AP/AO hits in --path mode are
            # advisory unless --strict-ap-ao is set.
            if hits:
                return 1
            if args.strict_ap_ao and ap_ao_hits:
                return 1
            return 0
        # Directory: walk it like a scan root.
        repo_root = target.parent
        hits_by_file: Dict[str, List[Tuple[int, str]]] = {}
        ap_ao_by_file: Dict[str, List[Tuple[int, str]]] = {}
        for p in _walk_python_files(target):
            file_hits = _scan_file(p)
            if file_hits:
                hits_by_file[str(p)] = file_hits
            file_ap_ao = _scan_file_ap_ao(p)
            if file_ap_ao:
                ap_ao_by_file[str(p)] = file_ap_ao
        total = sum(len(v) for v in hits_by_file.values())
        total_ap_ao = sum(len(v) for v in ap_ao_by_file.values())
        if args.json:
            print(
                json.dumps(
                    {
                        "ffn": {
                            rel: [{"line": ln, "method": m} for ln, m in hs]
                            for rel, hs in hits_by_file.items()
                        },
                        "ap_ao": {
                            rel: [{"line": ln, "name": n} for ln, n in hs]
                            for rel, hs in ap_ao_by_file.items()
                        },
                    },
                    indent=2,
                )
            )
        else:
            if total:
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
            if total_ap_ao:
                print(
                    f"lint_raw_ffn_rule: {total_ap_ao} raw AP/AO call(s) "
                    f"across {len(ap_ao_by_file)} file(s) under {target}."
                )
        if total:
            return 1
        if args.strict_ap_ao and total_ap_ao:
            return 1
        return 0

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

    ap_ao_by_file = scan_ap_ao(root)
    total_ap_ao = sum(len(v) for v in ap_ao_by_file.values())
    ap_ao_regressions, ap_ao_new_files = diff_against_baseline(
        ap_ao_by_file, baseline=_AP_AO_BASELINE
    )

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
                    "ap_ao": {
                        "total_raw_calls": total_ap_ao,
                        "files": {
                            rel: [{"line": ln, "name": n} for ln, n in hs]
                            for rel, hs in ap_ao_by_file.items()
                        },
                        "regressions": [
                            {"file": rel, "baseline": b, "current": c}
                            for rel, b, c in ap_ao_regressions
                        ],
                        "new_files": ap_ao_new_files,
                    },
                },
                indent=2,
            )
        )
        # Exit code: FFN regressions are always fatal; AP/AO only if
        # --strict-ap-ao is set.
        if regressions or new_files:
            return 1
        if args.strict_ap_ao and (ap_ao_regressions or ap_ao_new_files):
            return 1
        return 0

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
        # Per-file AP/AO summary in --list mode (advisory).
        if ap_ao_by_file:
            print(
                f"\nlint_raw_ffn_rule: {total_ap_ao} raw AP/AO call(s) "
                f"across {len(ap_ao_by_file)} file(s) (ADVISORY):"
            )
            for rel, hs in sorted(
                ap_ao_by_file.items(), key=lambda kv: -len(kv[1])
            ):
                base = _AP_AO_BASELINE.get(rel, 0)
                marker = " [BASELINED]" if rel in _AP_AO_BASELINE else " [NEW]"
                print(
                    f"  {rel}: {len(hs)} hit(s), baseline {base}{marker}"
                )

    # FFN block (the blocking ratchet).
    ffn_ok = not regressions and not new_files
    if ffn_ok:
        print(
            f"lint_raw_ffn_rule: OK — {total_hits} raw FFNRule call(s) "
            f"across {len(hits_by_file)} file(s), all within baseline. "
            f"({len(_BASELINE)} files tracked.)"
        )
    else:
        print(
            f"lint_raw_ffn_rule: REGRESSION — "
            f"{len(regressions)} growing file(s), {len(new_files)} new file(s)."
        )
        if regressions:
            print("\n  Files that grew beyond baseline:")
            for rel, baseline, current in regressions:
                print(
                    f"    {rel}: baseline={baseline}, current={current} "
                    f"(+{current - baseline})"
                )
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

    # AP/AO block (advisory, with optional strict mode).
    ap_ao_label = (
        "STRICT" if args.strict_ap_ao else "ADVISORY"
    )
    ap_ao_ok = not ap_ao_regressions and not ap_ao_new_files
    if not ap_ao_by_file:
        print(
            f"\nlint_raw_ffn_rule [AP/AO {ap_ao_label}]: 0 raw AP/AO call(s) "
            "outside the allow-listed DSL modules."
        )
    elif ap_ao_ok:
        print(
            f"\nlint_raw_ffn_rule [AP/AO {ap_ao_label}]: OK — {total_ap_ao} "
            f"raw AP/AO call(s) across {len(ap_ao_by_file)} file(s), all "
            f"within baseline. ({len(_AP_AO_BASELINE)} files tracked.)"
        )
    else:
        print(
            f"\nlint_raw_ffn_rule [AP/AO {ap_ao_label}]: REGRESSION — "
            f"{len(ap_ao_regressions)} growing file(s), "
            f"{len(ap_ao_new_files)} new file(s)."
        )
        if ap_ao_regressions:
            print("\n  AP/AO files that grew beyond baseline:")
            for rel, baseline, current in ap_ao_regressions:
                print(
                    f"    {rel}: baseline={baseline}, current={current} "
                    f"(+{current - baseline})"
                )
        if ap_ao_new_files:
            print("\n  AP/AO files NOT in baseline:")
            for rel in ap_ao_new_files:
                print(f"    {rel}: {len(ap_ao_by_file[rel])} raw AP/AO call(s)")
        print(
            "\nMigration target: compose attention specs through "
            "building_blocks_dsl helpers or attention_head_allocator. "
            "See c4_release/docs/ATTENTION_HEAD_IR_MIGRATION_PATTERN.md."
        )

    if not ffn_ok:
        return 1
    if args.strict_ap_ao and not ap_ao_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
