#!/usr/bin/env python3
"""Atomic component-name codemod for the semantic-naming initiative.

Renames an op / FFN-rule / factory identifier AND every one of its
reference sites (``target_op_name=`` / ``requires=`` op-name strings /
``owner=`` / qualified ``<op>.<rule>`` keys / plain function-name uses)
in lock-step across the repo, so byte-identity is preserved.

Byte-identity rule (audit, task #277): the golden param hash is driven
by BLOCK INDEX (insertion-order topo-sort + DIM-derived dep edges), NOT
by op names. Names matter only as string-match keys. A rename is safe
IFF the name AND all its reference sites are renamed atomically. A
MISSED reference -> silent dep-edge drop -> op reassigned to a different
block -> byte-identity BREAK.

SAFETY MODEL
------------
This tool is "safe-set restricted". Each rename pair is APPLIED only if
EVERY whole-word occurrence of ``old`` in the repo lives in a file
inside ``--safe-files`` (the cold-file allow-list). If any occurrence is
in a file outside that set, the pair is REFUSED (printed and skipped) --
because applying it would require editing a file outside the safe set
(a deferred / hot-file rename). This is exactly the cold-file boundary
guard from the brief, and the same tool will later run the hot files by
widening ``--safe-files``.

A rename is also refused if ``new`` already occurs as a whole word
anywhere outside the rename sites (collision guard).

Usage::

    python tools/rename_component_names.py \
        --safe-files neural_vm/unified_compiler/ops/l1_ops.py \
        --rename l1_stack0_byte0=stack0_byte0 \
        [--apply]            # default is dry-run

The pairs may also be supplied via ``--map FILE`` (one ``old=new`` per
line, ``#`` comments allowed).

Each occurrence is matched on a word boundary (``\bold\b``) so substrings
of longer identifiers are never touched (e.g. renaming ``l1_byte_index``
will NOT touch ``l1_byte_index_0`` -- callers must enumerate the full
family or the f-string prefix explicitly).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File extensions we scan for occurrences. Docs/.md are scanned for the
# "all occurrences confined" check but, by default, NOT rewritten (they
# do not affect byte-identity and are handled separately); pass
# --rewrite-nonpy to also rewrite them.
CODE_EXTS = (".py",)
TEXT_EXTS = (".py", ".md", ".txt", ".rst")


def _iter_files(exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    skip_dirs = {".git", "__pycache__", ".claude", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if fn.endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return out


def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT)


def _word_re(name: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")


def _is_safe(path: str, safe_files: List[str]) -> bool:
    rp = _rel(path)
    base = os.path.basename(path)
    for sf in safe_files:
        if rp == sf or rp.endswith("/" + sf) or base == sf:
            return True
    return False


def find_occurrences(name: str, exts: Tuple[str, ...]) -> Dict[str, int]:
    pat = _word_re(name)
    hits: Dict[str, int] = {}
    for path in _iter_files(exts):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        n = len(pat.findall(text))
        if n:
            hits[path] = n
    return hits


def plan(
    pairs: List[Tuple[str, str]],
    safe_files: List[str],
    rewrite_exts: Tuple[str, ...],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Return (accepted_pairs, refused[(old,new,reason)])."""
    accepted: List[Tuple[str, str]] = []
    refused: List[Tuple[str, str, str]] = []
    for old, new in pairs:
        occ = find_occurrences(old, TEXT_EXTS)
        if not occ:
            refused.append((old, new, "no occurrences found"))
            continue
        outside = [p for p in occ if not _is_safe(p, safe_files)]
        if outside:
            refused.append(
                (old, new, "refs outside safe set: "
                 + ", ".join(sorted(_rel(p) for p in outside)))
            )
            continue
        # collision guard: ``new`` must not already exist as a word
        # anywhere except where ``old`` already is (same files are fine
        # since we substitute there).
        new_occ = find_occurrences(new, rewrite_exts)
        new_outside = [p for p in new_occ if p not in occ]
        if new_outside:
            refused.append(
                (old, new, "target name already exists in: "
                 + ", ".join(sorted(_rel(p) for p in new_outside)))
            )
            continue
        accepted.append((old, new))
    return accepted, refused


def apply_pairs(
    pairs: List[Tuple[str, str]],
    rewrite_exts: Tuple[str, ...],
    dry_run: bool,
) -> Dict[str, int]:
    """Rewrite ``old`` -> ``new`` whole-word in every rewrite-ext file.
    Returns per-file change counts."""
    counts: Dict[str, int] = {}
    compiled = [(_word_re(old), new, old) for old, new in pairs]
    for path in _iter_files(rewrite_exts):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        new_text = text
        total = 0
        for pat, new, _old in compiled:
            new_text, n = pat.subn(new, new_text)
            total += n
        if total and new_text != text:
            counts[_rel(path)] = total
            if not dry_run:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new_text)
    return counts


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safe-files", nargs="+", required=True,
                    help="allow-list of files (relative paths or basenames) "
                         "renames must be confined to")
    ap.add_argument("--rename", action="append", default=[],
                    metavar="OLD=NEW", help="a rename pair; repeatable")
    ap.add_argument("--map", default=None,
                    help="file with one OLD=NEW per line (# comments ok)")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run)")
    ap.add_argument("--rewrite-nonpy", action="store_true",
                    help="also rewrite .md/.txt/.rst (default: .py only)")
    args = ap.parse_args(argv)

    pairs: List[Tuple[str, str]] = []
    for r in args.rename:
        if "=" not in r:
            ap.error(f"bad --rename {r!r}; expected OLD=NEW")
        old, new = r.split("=", 1)
        pairs.append((old.strip(), new.strip()))
    if args.map:
        with open(args.map, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if "=" not in line:
                    ap.error(f"bad map line {line!r}; expected OLD=NEW")
                old, new = line.split("=", 1)
                pairs.append((old.strip(), new.strip()))
    if not pairs:
        ap.error("no rename pairs supplied (use --rename or --map)")

    rewrite_exts = TEXT_EXTS if args.rewrite_nonpy else CODE_EXTS

    accepted, refused = plan(pairs, args.safe_files, rewrite_exts)

    print(f"safe-files: {', '.join(args.safe_files)}")
    print(f"pairs: {len(pairs)}  accepted: {len(accepted)}  "
          f"refused: {len(refused)}")
    for old, new, reason in refused:
        print(f"  REFUSED {old} -> {new}: {reason}")
    if not accepted:
        print("nothing to apply.")
        return 1 if refused else 0

    counts = apply_pairs(accepted, rewrite_exts, dry_run=not args.apply)
    verb = "WOULD CHANGE" if not args.apply else "CHANGED"
    for old, new in accepted:
        print(f"  ACCEPT  {old} -> {new}")
    for path, n in sorted(counts.items()):
        print(f"  {verb} {path}: {n} occurrence(s)")
    if not args.apply:
        print("dry-run; re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
