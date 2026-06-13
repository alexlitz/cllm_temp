#!/usr/bin/env python3
"""Raw imperative-attention-write linter (DSL migration ratchet).

The sibling of ``tools/lint_raw_ffn_rule.py``. Where that tool ratchets
raw ``FFNRule.constant_write`` / ``gated_write`` (and ``AP`` / ``AO``)
constructor calls, THIS tool ratchets the *imperative tensor writes* into
attention projection / ALiBi-slope buffers:

  1. ``attn.W_q[...] = X`` / ``attn.W_k`` / ``attn.W_v`` / ``attn.W_o``
     (with or without ``.data``) — raw Q/K/V/O projection writes that
     should flow through ``DeclarativeAttentionHeadSpec`` +
     ``Primitives.generate_attention_head`` instead.
  2. ``attn.alibi_slopes[...] = X`` (with or without ``.data``,
     including ``alibi_slopes.fill_(...)`` — see note below) — raw
     ALiBi-slope writes that should be carried by the spec's
     ``alibi_slope`` field so a slope-ownership pass can detect
     collisions at compile time.

WHY THIS EXISTS
---------------
``DeclarativeAttentionHeadSpec`` carries an ``alibi_slope`` field, but
``alibi_slope=None`` means "the op writes ``attn.alibi_slopes[...]``
itself imperatively." Nothing checks whether two ops write the SAME
``(physical block, head)`` slope — and when they do, the later write
silently clobbers the earlier one. That is the operand-relay
transmission bug (``make_layer10_residual_alibi_slopes_op`` overwriting
``alibi_slopes[3]/[4]`` on a shared physical attention block). See
``tools/alibi_slope_collision_map.py`` and
``docs/ALIBI_SLOPE_COLLISION_MAP_2026_06_11.md`` for the per-head
collision diagnostic, and
``docs/ATTENTION_DSL_MIGRATION_PLAN_2026_06_11.md`` for the migration
waves this ratchet enforces.

Before this lint there was NO ratchet on imperative attention writes —
only the FFN side (``lint_raw_ffn_rule.py``) was guarded. This closes
the attention half of that gap.

ALLOWED CALL SITES (no warning)
-------------------------------
  * ``unified_compiler/primitives.py`` — defines
    ``generate_attention_head`` (the one legitimate site that writes
    ``attn.W_*`` / ``attn.alibi_slopes`` from a spec).
  * ``unified_compiler/building_blocks_dsl.py`` /
    ``unified_compiler/wide_alu_dsl.py`` / ``unified_compiler/ir.py`` —
    the declarative DSL + lowering modules.
  * ``attention_head_allocator.py`` — owns raw per-head plumbing.
  * ``tests/**`` — tests may use raw writes for byte-identity checks.

Everywhere else (the legacy imperative core — ``vm_step.py``,
``compiler.py``, ``setup_helpers_*.py``, ``weight_modules/*`` — and the
per-layer ``ops/*.py`` files) is a migration target.

BASELINE
--------
``_BASELINE`` is a per-file count snapshot captured 2026-06-11. CI fails
when a NEW non-allow-listed file appears, or an existing baselined
file's count GROWS. Migrations shrink counts: decrement (or delete) the
entry in the SAME commit. This is a ratchet — counts only go down.

NOTE on ``fill_`` and genuinely shape-dependent heads
-----------------------------------------------------
``alibi_slopes.fill_(d)`` is counted (it's an imperative whole-buffer
slope write). ``RuntimeAttentionFragment`` heads (e.g. L15
``memory_lookup`` branching on ``attn.num_heads``) write ``attn.W_*``
imperatively *by design* because their shape isn't known until bake
time; those are still counted so the ratchet stays honest, but the
migration plan marks them ``shape-dependent (keep imperative behind a
RuntimeAttentionFragment, but lift the slope into the spec)``.

USAGE
-----
    python c4_release/tools/lint_raw_attn.py             # repo root or c4_release
    python c4_release/tools/lint_raw_attn.py --json      # machine-readable
    python c4_release/tools/lint_raw_attn.py --list      # print every hit
    python c4_release/tools/lint_raw_attn.py --path PATH # lint a single file/dir

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


# Attribute names that identify an attention buffer write target.
_PROJ_NAMES = ("W_q", "W_k", "W_v", "W_o")
_SLOPE_NAME = "alibi_slopes"


# Modules where raw imperative attention writes are expected: the spec
# lowering primitive, the declarative DSL modules + IR, and the head
# allocator. (Tests are skipped by directory, below.)
_ALLOWED_FILES: frozenset = frozenset(
    {
        "c4_release/neural_vm/unified_compiler/primitives.py",
        "c4_release/neural_vm/unified_compiler/building_blocks_dsl.py",
        "c4_release/neural_vm/unified_compiler/wide_alu_dsl.py",
        "c4_release/neural_vm/unified_compiler/ir.py",
        "c4_release/neural_vm/attention_head_allocator.py",
    }
)


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


# Per-file count baseline captured 2026-06-11. The value is the TOTAL
# imperative attention writes (W_q/W_k/W_v/W_o subscript-assign +
# alibi_slopes subscript-assign + alibi_slopes.fill_) in the file. The
# inline comment splits projection-vs-slope counts for migration
# bookkeeping. Decrement entries in the SAME commit that migrates a file.
#
# Two "fronts":
#   * Legacy imperative core (vm_step / compiler / setup_helpers_* /
#     weight_modules) — the bulk, migrated by Phase 6/7 cuts.
#   * Per-layer ops/*.py — the active DSL-migration front (these are the
#     files the smoke agents touch; the migration plan orders waves to
#     avoid the currently-hot l7/l8/l9/l10 until those land).
_BASELINE: Dict[str, int] = {
    # ---- legacy imperative core (Phase 6/7 cut targets) ----
    "c4_release/neural_vm/unified_compiler/compiler.py": 749,        # W_*=735 alibi=6 fill=8
    "c4_release/neural_vm/vm_step.py": 537,                          # W_*=537 alibi=0 fill=0
    "c4_release/neural_vm/setup_helpers_l10.py": 159,               # W_*=158 alibi=1 fill=0
    "c4_release/neural_vm/setup_helpers_l5.py": 81,                 # W_*=81  alibi=0 fill=0
    "c4_release/neural_vm/weight_modules/function_calls.py": 51,    # W_*=51  alibi=0 fill=0
    "c4_release/neural_vm/setup_helpers_l6.py": 38,                 # W_*=38  alibi=0 fill=0
    "c4_release/neural_vm/setup_helpers_l13.py": 25,                # W_*=25  alibi=0 fill=0
    "c4_release/neural_vm/setup_helpers_l9.py": 24,                 # W_*=24  alibi=0 fill=0
    "c4_release/neural_vm/setup_helpers_l3.py": 9,                  # W_*=9   alibi=0 fill=0
    "c4_release/neural_vm/setup_helpers_l4.py": 7,                  # W_*=7   alibi=0 fill=0
    # ---- per-layer ops/*.py (active DSL-migration front) ----
    "c4_release/neural_vm/unified_compiler/ops/l15_ops.py": 234,    # W_*=231 alibi=3 fill=0
    "c4_release/neural_vm/unified_compiler/ops/l8_ops.py": 81,      # W_*=79  alibi=2 fill=0
    "c4_release/neural_vm/unified_compiler/ops/l14_ops.py": 80,     # W_*=78  alibi=2 fill=0
    "c4_release/neural_vm/unified_compiler/ops/l9_ops.py": 22,      # W_*=14  alibi=7 fill=1  (+2: Wave B Phase 2 SE-relay slope re-assert)
    "c4_release/neural_vm/unified_compiler/ops/model_ops.py": 17,   # W_*=0   alibi=13 fill=4
    "c4_release/neural_vm/unified_compiler/ops/l4_ops.py": 14,      # W_*=13  alibi=0 fill=1
    "c4_release/neural_vm/unified_compiler/ops/l10_ops.py": 11,     # W_*=0   alibi=11 fill=0
    "c4_release/neural_vm/unified_compiler/ops/alu_ops.py": 5,      # W_*=0   alibi=5 fill=0
    "c4_release/neural_vm/unified_compiler/ops/l7_ops.py": 5,       # W_*=0   alibi=4 fill=1
    "c4_release/neural_vm/unified_compiler/ops/l1_ops.py": 4,       # W_*=0   alibi=3 fill=1
    "c4_release/neural_vm/unified_compiler/ops/flag_gated_ops.py": 4,  # W_*=0 alibi=4 fill=0
    "c4_release/neural_vm/unified_compiler/ops/l2_ops.py": 2,       # W_*=0   alibi=1 fill=1
    "c4_release/neural_vm/unified_compiler/ops/l5_ops.py": 1,       # W_*=0   alibi=0 fill=1
    "c4_release/neural_vm/unified_compiler/ops/l3_ops.py": 1,       # W_*=0   alibi=0 fill=1
    "c4_release/neural_vm/unified_compiler/ops/l13_ops.py": 5,      # alibi_slopes[idx]= for relay heads 3/4/5/6 (known ALiBi-slope DSL gap)
    "c4_release/neural_vm/unified_compiler/ops/l0_ops.py": 1,       # W_*=0   alibi=0 fill=1
}


def _walk_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_SEGMENTS]
        if set(Path(dirpath).parts) & _SKIP_DIR_SEGMENTS:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _attr_chain(node: ast.AST) -> List[str]:
    """Return the dotted attribute chain for an ``Attribute``/``Name`` value."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _classify_target(target: ast.AST) -> str:
    """Return 'proj' / 'slope' / '' for an assignment target subscript.

    Matches ``<...>.W_q[...]`` / ``.W_k`` / ``.W_v`` / ``.W_o`` (proj) and
    ``<...>.alibi_slopes[...]`` (slope), with or without a trailing
    ``.data`` (the chain still contains the buffer attr).
    """
    if not isinstance(target, ast.Subscript):
        return ""
    chain = _attr_chain(target.value)
    if _SLOPE_NAME in chain:
        return "slope"
    if any(name in chain for name in _PROJ_NAMES):
        return "proj"
    return ""


def _is_slope_fill(node: ast.AST) -> bool:
    """True when ``node`` is ``<...>.alibi_slopes.fill_(...)``.

    ``fill_`` is an in-place whole-buffer slope write — semantically an
    imperative slope assignment, so it counts toward the ratchet.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "fill_":
        return False
    return _SLOPE_NAME in _attr_chain(func.value)


def _scan_file(path: Path) -> List[Tuple[int, str]]:
    """Return ``[(lineno, kind)]`` for every imperative attention write.

    ``kind`` is one of ``"W_q[]"``, ``"W_k[]"``, ``"W_v[]"``, ``"W_o[]"``,
    ``"alibi_slopes[]"`` (subscript-assign) or ``"alibi_slopes.fill_"``.
    """
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
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                kind = _classify_target(tgt)
                if kind:
                    hits.append((node.lineno, _kind_label(tgt, kind)))
        elif isinstance(node, ast.AugAssign):
            kind = _classify_target(node.target)
            if kind:
                hits.append((node.lineno, _kind_label(node.target, kind)))
        elif _is_slope_fill(node):
            hits.append((node.lineno, "alibi_slopes.fill_"))
    return hits


def _kind_label(target: ast.AST, kind: str) -> str:
    if kind == "slope":
        return "alibi_slopes[]"
    # proj — find which W_* it was for a more useful label.
    chain = _attr_chain(target.value) if isinstance(target, ast.Subscript) else []
    for name in _PROJ_NAMES:
        if name in chain:
            return f"{name}[]"
    return "W_*[]"


def _is_allowed(rel_path: str) -> bool:
    for allowed in _ALLOWED_FILES:
        if rel_path == allowed or rel_path.endswith(allowed[len("c4_release/"):]):
            return True
    return False


def scan(
    repo_root: Path, scan_roots: Iterable[str] = _SCAN_ROOTS
) -> Dict[str, List[Tuple[int, str]]]:
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


def _find_repo_root(args_root) -> Path:
    if args_root:
        return Path(args_root).resolve()
    cwd = Path.cwd().resolve()
    root = cwd
    for _ in range(5):
        if (root / "c4_release").is_dir():
            return root
        if (root / "neural_vm").is_dir() and root.name == "c4_release":
            return root.parent
        parent = root.parent
        if parent == root:
            break
        root = parent
    if (root / "c4_release").is_dir():
        return root
    raise FileNotFoundError(f"could not find c4_release/ from {cwd}")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument(
        "--list", action="store_true", help="print every hit (not just regressions)"
    )
    ap.add_argument("--root", default=None, help="repo root (defaults to auto-detect)")
    ap.add_argument(
        "--path",
        default=None,
        help="lint a single .py file or directory instead of the scan roots",
    )
    args = ap.parse_args(argv)

    # --path mode (used by the unit test for planted violations).
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
                        {
                            "file": str(target),
                            "hits": [
                                {"line": ln, "kind": k} for ln, k in hits
                            ],
                        },
                        indent=2,
                    )
                )
            else:
                if hits:
                    print(
                        f"lint_raw_attn: {len(hits)} raw attention write(s) "
                        f"in {target}:"
                    )
                    for ln, k in hits:
                        print(
                            f"  {target}:{ln}: raw imperative {k}; lower via "
                            f"DeclarativeAttentionHeadSpec + "
                            f"Primitives.generate_attention_head (carry the "
                            f"slope in spec.alibi_slope).  [{k}]"
                        )
                else:
                    print(f"lint_raw_attn: 0 raw attention writes in {target}")
            return 1 if hits else 0
        # Directory.
        hits_by_file: Dict[str, List[Tuple[int, str]]] = {}
        for p in _walk_python_files(target):
            fh = _scan_file(p)
            if fh:
                hits_by_file[str(p)] = fh
        total = sum(len(v) for v in hits_by_file.values())
        if args.json:
            print(
                json.dumps(
                    {
                        rel: [{"line": ln, "kind": k} for ln, k in hs]
                        for rel, hs in hits_by_file.items()
                    },
                    indent=2,
                )
            )
        else:
            print(
                f"lint_raw_attn: {total} raw attention write(s) across "
                f"{len(hits_by_file)} file(s) under {target}."
            )
        return 1 if total else 0

    try:
        root = _find_repo_root(args.root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    hits_by_file = scan(root)
    total_hits = sum(len(v) for v in hits_by_file.values())
    regressions, new_files = diff_against_baseline(hits_by_file)

    if args.json:
        print(
            json.dumps(
                {
                    "total_raw_writes": total_hits,
                    "files": {
                        rel: [{"line": ln, "kind": k} for ln, k in hs]
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
        print(
            f"lint_raw_attn: {total_hits} raw attention write(s) across "
            f"{len(hits_by_file)} file(s):"
        )
        for rel, hs in sorted(hits_by_file.items(), key=lambda kv: -len(kv[1])):
            base = _BASELINE.get(rel, 0)
            marker = " [BASELINED]" if rel in _BASELINE else " [NEW]"
            # Per-kind breakdown.
            kinds: Dict[str, int] = {}
            for _, k in hs:
                kinds[k] = kinds.get(k, 0) + 1
            kind_str = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            print(
                f"  {rel}: {len(hs)} hit(s), baseline {base}{marker}  "
                f"({kind_str})"
            )

    ok = not regressions and not new_files
    if ok:
        print(
            f"lint_raw_attn: OK — {total_hits} raw imperative attention "
            f"write(s) across {len(hits_by_file)} file(s), all within "
            f"baseline. ({len(_BASELINE)} files tracked.)"
        )
        return 0

    print(
        f"lint_raw_attn: REGRESSION — {len(regressions)} growing file(s), "
        f"{len(new_files)} new file(s)."
    )
    if regressions:
        print("\n  Files that grew beyond baseline:")
        for rel, baseline, current in regressions:
            print(
                f"    {rel}: baseline={baseline}, current={current} "
                f"(+{current - baseline})"
            )
    if new_files:
        print("\n  Files NOT in baseline that contain raw attention writes:")
        for rel in new_files:
            for ln, k in hits_by_file[rel]:
                print(
                    f"    {rel}:{ln}: raw imperative {k}; lower via "
                    f"DeclarativeAttentionHeadSpec + "
                    f"Primitives.generate_attention_head.  [{k}]"
                )
    print(
        "\nSee c4_release/docs/ATTENTION_DSL_MIGRATION_PLAN_2026_06_11.md "
        "for the wave plan and c4_release/docs/"
        "ALIBI_SLOPE_COLLISION_MAP_2026_06_11.md for the slope-collision "
        "diagnostic. Migration shrinks counts; update _BASELINE in this "
        "tool in the same commit."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
