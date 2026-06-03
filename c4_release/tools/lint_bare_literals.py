#!/usr/bin/env python3
"""Bare-literal shape-value linter (Step 3, audit 2026-06-03).

Scans ``c4_release/neural_vm/**/*.py`` for bare literal shape values
(``d_model=512``, ``n_heads=8``, ``ffn_hidden=4096``, ``head_dim=64``)
that act as silent fallbacks in active bake / runner paths.

Background — see ``c4_release/docs/LITERAL_FALLBACK_AUDIT.md`` for the
five HIGH-risk sites that motivated this lint. The L10
``d_model=512`` fallback (``l10_ops.py:6764``) detonated under a
dim-allocator shift; this lint exists so a future migration that
forgets one runner does not re-introduce the same failure shape.

Allowed callers must source the value from one of:
  * the ``DEFAULT_D_MODEL`` / ``DEFAULT_N_HEADS`` / ``DEFAULT_FFN_HIDDEN``
    constants in ``neural_vm.vm_step``;
  * a derivation chain off the live target (``attn.W_q.shape[0] //
    attn.num_heads``, ``block.attn.dim``, ``compiled_model.d_model``,
    ``ModelArchitectureSpec``).

Usage::

    python c4_release/tools/lint_bare_literals.py            # repo root or c4_release
    python c4_release/tools/lint_bare_literals.py --json     # machine-readable

Exit codes:
  0  no HIGH-risk hits
  1  HIGH-risk hits found
  2  invocation / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# Patterns to flag. Each entry: (label, compiled regex). The regex
# matches a ``<name>=<literal>`` kwarg in a function signature or call,
# WITH WORD BOUNDARIES so ``d_model=5120`` and ``n_heads=80`` do not
# trip. ``head_dim`` is included even though it has no migrated
# constant (its derivation chain in layer_compiler now raises).
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("d_model=512", re.compile(r"\bd_model\s*=\s*512\b")),
    ("n_heads=8", re.compile(r"\bn_heads\s*=\s*8\b")),
    ("num_heads=8", re.compile(r"\bnum_heads\s*=\s*8\b")),
    ("ffn_hidden=4096", re.compile(r"\bffn_hidden\s*=\s*4096\b")),
    ("head_dim=64", re.compile(r"\bhead_dim\s*=\s*64\b")),
    # Hardcoded ``// 8`` head-count divider as a defense-in-depth check
    # (compiler.py:3488 was the L15 instance).
    ("d_model // 8", re.compile(r"\bd_model\s*//\s*8\b")),
    # ``or 4096`` / ``else 4096`` end-of-chain literal fallbacks.
    ("or 4096", re.compile(r"\bor\s+4096\b")),
    ("else 4096", re.compile(r"\belse\s+4096\b")),
    ("else 512", re.compile(r"\belse\s+512\b")),
    ("else 64", re.compile(r"\belse\s+64\b")),
]


# Files / directories scanned. Restricted to ``neural_vm`` per the
# audit's scope statement.
_SCAN_ROOTS = ("c4_release/neural_vm",)


# Allowlist: (relative_path_glob, allowed_line_substring). A hit on
# (file, line) is suppressed when the line CONTAINS the substring.
# Used to whitelist:
#   * the DEFAULT_X constants themselves (single source of truth);
#   * docstring / comment occurrences;
#   * MED-risk post-op ``__init__`` defaults that the audit explicitly
#     classifies as documented (vm_step.py 676, 776, 842, 1017, 1152,
#     1286 — overridden at every production call site);
#   * the L10 derivation-chain tail (``l10_ops.py`` lines 6659, 6801)
#     which is the last leg of a multi-step derivation, NOT a bare
#     fallback.
_ALLOW: List[Tuple[str, str]] = [
    # Single source of truth.
    ("c4_release/neural_vm/vm_step.py", "DEFAULT_D_MODEL = 512"),
    ("c4_release/neural_vm/vm_step.py", "DEFAULT_N_HEADS = 8"),
    ("c4_release/neural_vm/vm_step.py", "DEFAULT_FFN_HIDDEN = 4096"),
    # MED-risk post-op __init__ defaults (audit-documented).
    ("c4_release/neural_vm/vm_step.py", "def __init__(self, d_model=512, S=100.0, dim_positions=None)"),
    ("c4_release/neural_vm/vm_step.py", "def __init__(self, d_model=512, S=100.0, byte_idx=0"),
    ("c4_release/neural_vm/vm_step.py", "def __init__(self, d_model=512, S=100.0, mode="),
    # L10 end-of-chain literal — hardened by derivation steps above.
    ("c4_release/neural_vm/unified_compiler/ops/l10_ops.py", "d_model = 512"),
    # LOW-risk ``__main__`` demo drivers (audit classifies as LOW).
    ("c4_release/neural_vm/nibble_bytecode_executor.py", "AutoregressiveVM(d_model=1280"),
    ("c4_release/neural_vm/fully_neural_vm.py", "AutoregressiveVM(d_model=1280"),
    ("c4_release/neural_vm/autoregressive_nibble_vm.py", "AutoregressiveVM(d_model=1280"),
    # The linter's own error strings reference the historical literals.
    ("c4_release/neural_vm/unified_compiler/compiler.py", "(d_model // 8) removed by Step 3"),
    ("c4_release/neural_vm/unified_compiler/layer_compiler.py", "Bare-literal fallback (head_dim=64)"),
]


_SKIP_DIR_SEGMENTS = {
    "__pycache__",
    ".git",
    "archive",
    "deprecated",
    "old",
    "tests",
    "test_archive",
}


def _walk_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip caches, archives, tests (LOW-risk fixtures per audit).
        # Prune in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_SEGMENTS]
        # Also defensive: skip if a parent directory matched.
        parts = set(Path(dirpath).parts)
        if parts & _SKIP_DIR_SEGMENTS:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _is_allowed(rel_path: str, line: str) -> bool:
    for pat_path, needle in _ALLOW:
        if rel_path.endswith(pat_path.replace("c4_release/", "")) or rel_path == pat_path:
            if needle in line:
                return True
    return False


def _is_comment_or_docstring(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return True
    # Bare-line docstring (heuristic: line is inside triple-quoted block).
    # Cheap detection: line is entirely a string literal.
    if stripped.startswith(('"""', "'''", '"', "'")) and stripped.count('"""') + stripped.count("'''") + stripped.count('"') + stripped.count("'") > 0:
        # Probably inside a docstring. Conservative: only skip if line
        # has NO assignment after the literal (e.g. ``"d_model=512"``).
        if "=" not in stripped[:stripped.find('"')] if '"' in stripped else "=" not in stripped[:stripped.find("'")] if "'" in stripped else True:
            return True
    return False


def scan(repo_root: Path) -> List[Tuple[str, int, str, str]]:
    """Return a list of (relative_path, lineno, label, line) hits."""
    hits: List[Tuple[str, int, str, str]] = []
    seen: set = set()
    for scan_rel in _SCAN_ROOTS:
        root = repo_root / scan_rel
        if not root.exists():
            continue
        for path in _walk_python_files(root):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            rel = str(path.relative_to(repo_root))
            in_docstring = False
            for lineno, line in enumerate(lines, start=1):
                # Cheap docstring tracker: toggle on triple-quote.
                if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                    in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                # Skip pure-comment lines.
                if line.lstrip().startswith("#"):
                    continue
                for label, pat in _PATTERNS:
                    if pat.search(line):
                        if _is_allowed(rel, line):
                            continue
                        key = (rel, lineno, label)
                        if key in seen:
                            continue
                        seen.add(key)
                        hits.append((rel, lineno, label, line.rstrip()))
    return hits


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument(
        "--root",
        default=None,
        help="repo root (defaults to auto-detect from cwd)",
    )
    args = ap.parse_args(argv)

    if args.root:
        root = Path(args.root).resolve()
    else:
        cwd = Path.cwd().resolve()
        # Look for c4_release/ from cwd upward.
        root = cwd
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
            print(f"error: could not find c4_release/ from {cwd}", file=sys.stderr)
            return 2

    hits = scan(root)

    if args.json:
        print(json.dumps(
            [{"file": h[0], "line": h[1], "pattern": h[2], "text": h[3]} for h in hits],
            indent=2,
        ))
    else:
        if not hits:
            print(f"lint_bare_literals: 0 HIGH-risk hits in {root}/c4_release/neural_vm")
        else:
            print(
                f"lint_bare_literals: {len(hits)} HIGH-risk bare-literal hit(s):\n"
            )
            for rel, lineno, label, text in hits:
                print(f"  {rel}:{lineno}  [{label}]")
                print(f"      {text}")
            print(
                "\nSee c4_release/docs/LITERAL_FALLBACK_AUDIT.md for the "
                "five HIGH-risk patterns and migration guidance."
            )

    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
