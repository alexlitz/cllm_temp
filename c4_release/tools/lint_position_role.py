#!/usr/bin/env python3
"""Position-role lint — architectural principle ratchet.

Enforces the architectural directive that **all compute fires at
``MARK_SE`` (step-end), token-emit rules fire at ``MARK_X`` / per-byte
positions, and memory-read rules fire at ``MARK_AX`` only inside LI/LC
sub-steps.**

Background
==========

Historically the per-layer FFN bakes inferred their position-role
implicitly. ALU/CMP/bitwise units gated on ``MARK_AX`` (the same
position that emits AX bytes) and shared residual bandwidth with
token-emit logic. That conflation made it possible for an ALU rule to
fire at a position that was structurally meant for byte emission —
silently corrupting a sibling op's output. Sweeps periodically catch
the resulting drift but never the original sin: a compute rule
authored at the wrong position.

This lint reads each ``L*_ops.py`` file under
``c4_release/neural_vm/unified_compiler/ops/``, classifies every
declarative rule constructor call by its ``name=`` string (intent), and
checks the ``conditions=`` / ``scope=`` / ``gate=`` against the
position role required for that intent.

Three intents (inferred from the rule name)
===========================================

* ``COMPUTE``    — names matching ``*_cmp_*`` / ``*_alu_*`` /
                   ``*_bitwise_*`` / ``*_combine_*`` /
                   ``*_carry_*`` / ``*_mul_*`` / ``*_shift_*`` /
                   ``*_div_*`` / ``*_mod_*``. Must gate on
                   ``MARK_SE`` (step-end). Exception: explicit
                   ``MARK_AX`` gating is allowed only when a sibling
                   ``MARK_SE_ONLY`` or ``MARK_SE`` condition is
                   present (the migration target / documented
                   exception is allowed).
* ``TOKEN_EMIT`` — names matching ``*_byte*`` / ``*_emit*`` /
                   ``*_passthrough*`` / ``*_marker_*``. Must gate on
                   ``MARK_X`` (``MARK_AX``, ``MARK_PC``, ``MARK_SP``,
                   ``MARK_BP``, ``MARK_MEM``, ``MARK_STACK0``,
                   ``MARK_CS``) or a ``BYTE_INDEX_*`` flag.
* ``MEMORY_READ``— names matching ``*_lookup*`` / ``*_memory_*`` /
                   ``*_fetch*``. Allowed at ``MARK_AX`` during LI / LC
                   sub-steps only (a condition / gate mentioning
                   ``OP_LI`` / ``OP_LC`` / ``MARK_MEM`` /
                   ``MEM_STORE`` satisfies the gate).

Names that do not match any pattern are *un-classified*: the lint
reports them in advisory mode but does not raise a violation.

Baseline + ratchet
==================

A per-file count of *current* violations is captured in
``_BASELINE`` below. The lint exits 0 when every file's violation
count is ``<=`` its baseline, and exits 1 when a baselined file's count
grows OR a non-baselined file gains a violation. After fixing a
violation, decrement (or delete) the entry in the same commit so the
ratchet only walks downward.

Usage
=====

::

    python c4_release/tools/lint_position_role.py
    python c4_release/tools/lint_position_role.py --json
    python c4_release/tools/lint_position_role.py --list
    python c4_release/tools/lint_position_role.py --path <file_or_dir>

Exit codes
==========

* 0 — at or below baseline (no regression)
* 1 — regression: new file with violations OR baselined file grew
* 2 — invocation / IO error
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCAN_ROOTS: Tuple[str, ...] = (
    "c4_release/neural_vm/unified_compiler/ops",
)

_SKIP_DIR_SEGMENTS = {
    "__pycache__",
    ".git",
    "archive",
    "deprecated",
    "old",
    "tests",
    "test_archive",
}

# Rule constructors we inspect. Each is a positional callable that takes
# ``name=`` / ``conditions=`` / ``scope=`` / ``gate=`` kwargs.
_RULE_CONSTRUCTORS = {
    "multi_way_and_rule",
    "step_function_rule",
    "one_hot_indicator_rule",
    "band_range_check_rules",
    "cancel_residual_rule",
    "lookup_table_rules",
    "multi_way_or_rules",
    "multi_way_or_rule",
    "constant_write",
    "gated_write",
}

# Recognised position-gate dim names.
_STEP_END_GATES = {"MARK_SE", "MARK_SE_ONLY", "HAS_SE"}
_TOKEN_EMIT_GATES = {
    "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
    "MARK_STACK0", "MARK_CS",
}
_MEMORY_OP_HINTS = {
    "OP_LI", "OP_LC", "OP_LEA", "MARK_MEM", "MEM_STORE", "ADDR_KEY",
}

# Pattern matchers for intent classification. Substring matches against
# the rule name (case-insensitive).
_COMPUTE_PATTERNS = (
    "_cmp_", "_alu_", "_bitwise_", "_combine_",
    "_carry_", "_mul_", "_shift_",
    "_div_", "_mod_", "_add_", "_sub_",
    "_shl_", "_shr_", "_xor_", "_and_", "_or_",
)
_TOKEN_EMIT_PATTERNS = (
    "_byte", "_emit", "_passthrough", "_marker_",
    "_marker.", "_broadcast",
)
_MEMORY_READ_PATTERNS = (
    "_lookup", "_memory_", "_fetch",
)


# Per-file baseline of current violations. Captured 2026-06-10 at the
# lint-introduction commit. Migrations DECREMENT entries in the SAME
# commit; the ratchet only walks downward.
_BASELINE: Dict[str, int] = {
    "c4_release/neural_vm/unified_compiler/ops/l10_ops.py": 0,  # Wave B Cluster 1 (2026-06-10)
    "c4_release/neural_vm/unified_compiler/ops/l11_ops.py": 0,  # Wave B Cluster 4 (2026-06-10)
    "c4_release/neural_vm/unified_compiler/ops/l12_ops.py": 0,  # Wave B Cluster 4 (2026-06-10)
    "c4_release/neural_vm/unified_compiler/ops/l3_ops.py": 4,
    "c4_release/neural_vm/unified_compiler/ops/l6_ops.py": 2,
    "c4_release/neural_vm/unified_compiler/ops/l8_ops.py": 14,
    "c4_release/neural_vm/unified_compiler/ops/l9_ops.py": 0,
}


# Static mapping of ``dim_ref(category, role)`` -> canonical dim name.
# The lint can't evaluate the live ``DimRegistry`` so it bakes in a
# subset large enough to recognise position-marker references that use
# the semantic form. Lifted from
# ``c4_release/neural_vm/dim_registry.py``'s ``register_category``
# bindings list (the ``marker`` / ``byte_index`` / ``opcode_flag``
# blocks are the position-role cases we care about).
_DIM_REF_CANONICAL: Dict[Tuple[str, str], str] = {
    # marker
    ("marker", "PC"): "MARK_PC",
    ("marker", "AX"): "MARK_AX",
    ("marker", "SP"): "MARK_SP",
    ("marker", "BP"): "MARK_BP",
    ("marker", "MEM"): "MARK_MEM",
    ("marker", "SE"): "MARK_SE",
    ("marker", "CS"): "MARK_CS",
    ("marker", "SE_ONLY"): "MARK_SE_ONLY",
    ("marker", "STACK0"): "MARK_STACK0",
    # byte_index
    ("byte_index", "0"): "BYTE_INDEX_0",
    ("byte_index", "1"): "BYTE_INDEX_1",
    ("byte_index", "2"): "BYTE_INDEX_2",
    ("byte_index", "3"): "BYTE_INDEX_3",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _walk_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_SEGMENTS]
        parts = set(Path(dirpath).parts)
        if parts & _SKIP_DIR_SEGMENTS:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _call_func_name(call: ast.Call) -> Optional[str]:
    """Return the bare callable name of ``call`` (``foo(...)`` or ``X.foo(...)``)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kwarg(call: ast.Call, key: str) -> Optional[ast.AST]:
    for kw in call.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _string_value(node: ast.AST) -> Optional[str]:
    """Return the string value of ``node`` for plain strings + f-strings.

    F-strings are concatenated with their formatted-value parts replaced
    by ``"*"`` placeholders so substring matching against the rule's
    static name skeleton still works (e.g. ``f"l10_cmp_{op_name}_default"``
    -> ``"l10_cmp_*_default"``).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("*")
        return "".join(parts)
    return None


def _resolve_dim_ref_call(call: ast.Call) -> Optional[str]:
    """Resolve a ``dim_ref("cat", "role")`` call to its canonical dim name.

    Returns the dim base-name (without ``+offset`` suffix) when both
    ``cat`` and ``role`` are string literals AND the pair is registered
    in ``_DIM_REF_CANONICAL``. Returns ``None`` otherwise.

    Recognises bare ``dim_ref(...)`` and ``something.dim_ref(...)``.
    """
    func = call.func
    if isinstance(func, ast.Name):
        if func.id != "dim_ref":
            return None
    elif isinstance(func, ast.Attribute):
        if func.attr != "dim_ref":
            return None
    else:
        return None
    args = call.args
    if len(args) < 2:
        return None
    cat = _string_value(args[0])
    role = _string_value(args[1])
    if cat is None or role is None:
        return None
    return _DIM_REF_CANONICAL.get((cat, role))


def _collect_dim_names(node: ast.AST) -> List[str]:
    """Pull every string literal that looks like a dim name out of ``node``.

    Walks ``ast`` so it works on tuples, lists, dicts, set literals,
    generator expressions, and binary concatenations. Returns the dim
    "base" name (the part before any ``+offset`` suffix), so
    ``"OUTPUT_LO+4"`` -> ``"OUTPUT_LO"``.

    Also resolves ``dim_ref("category", "role")`` calls against the
    static ``_DIM_REF_CANONICAL`` mapping so rules that use the
    semantic-ref form for position markers still match.
    """
    out: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            ref = _resolve_dim_ref_call(sub)
            if ref is not None:
                out.append(ref)
                continue
        s = _string_value(sub)
        if s is None:
            continue
        # Strip ``+<digits>`` / ``+{...}`` style offsets.
        base = s.split("+", 1)[0].strip()
        if not base:
            continue
        # Heuristic: real dim names are UPPER_SNAKE_CASE with at least
        # 2 chars. This filters out doc strings and other prose.
        if re.fullmatch(r"[A-Z][A-Z0-9_]+", base):
            out.append(base)
    return out


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


def classify_intent(name: str) -> Optional[str]:
    """Return ``"COMPUTE"`` / ``"TOKEN_EMIT"`` / ``"MEMORY_READ"`` / ``None``."""
    if not name:
        return None
    lo = name.lower()
    # MEMORY_READ takes precedence over COMPUTE: a name like
    # ``l15_alu_lookup_*`` reads memory, even though it has ``_alu_``.
    for p in _MEMORY_READ_PATTERNS:
        if p in lo:
            return "MEMORY_READ"
    # TOKEN_EMIT names like ``l10_byte_passthrough`` should classify as
    # TOKEN_EMIT, not COMPUTE.
    for p in _TOKEN_EMIT_PATTERNS:
        if p in lo:
            return "TOKEN_EMIT"
    for p in _COMPUTE_PATTERNS:
        if p in lo:
            return "COMPUTE"
    return None


# ---------------------------------------------------------------------------
# Position-gate extraction
# ---------------------------------------------------------------------------


def _extract_position_dims(call: ast.Call) -> set:
    """Return the set of dim base-names referenced in conditions/scope/gate.

    This is the "what positions does this rule actually look at" answer,
    used to evaluate the position-role contract.
    """
    dims: set = set()
    for key in ("conditions", "scope", "gate", "dominates_at", "writes"):
        node = _kwarg(call, key)
        if node is None:
            continue
        if key == "scope":
            # ``scope=`` is a free-text predicate string like
            # ``"MARK_AX and OP_EQ"``. Tokenise on identifiers.
            s = _string_value(node)
            if s:
                for tok in re.findall(r"[A-Z][A-Z0-9_]+", s):
                    dims.add(tok)
            continue
        for d in _collect_dim_names(node):
            dims.add(d)
    return dims


def _is_position_role_violation(
    intent: str, position_dims: set
) -> Optional[str]:
    """Return a violation reason string or ``None`` if the rule is OK.

    Rules:
      * COMPUTE       -> require some STEP_END marker. ``MARK_AX`` /
                         ``MARK_PC`` / etc. without ``MARK_SE`` /
                         ``MARK_SE_ONLY`` / ``HAS_SE`` is a violation.
      * TOKEN_EMIT    -> require some token-position marker or
                         ``BYTE_INDEX_*``.
      * MEMORY_READ   -> allowed at ``MARK_AX`` only with an LI/LC hint.
    """
    has_step_end = bool(position_dims & _STEP_END_GATES)
    has_token_emit = bool(position_dims & _TOKEN_EMIT_GATES)
    has_byte_index = any(d.startswith("BYTE_INDEX") for d in position_dims)
    has_memory_op = bool(position_dims & _MEMORY_OP_HINTS)

    if intent == "COMPUTE":
        if has_step_end:
            return None
        if has_token_emit:
            return (
                "COMPUTE rule gates on token-emit position "
                f"({sorted(position_dims & _TOKEN_EMIT_GATES)}) without a "
                "MARK_SE / MARK_SE_ONLY / HAS_SE step-end guard"
            )
        # No recognised position gate at all — pass-through (lint can't
        # prove it's a violation).
        return None

    if intent == "TOKEN_EMIT":
        if has_token_emit or has_byte_index:
            return None
        if has_step_end:
            return (
                "TOKEN_EMIT rule gates on MARK_SE / MARK_SE_ONLY / HAS_SE "
                "without a token-position marker or BYTE_INDEX_*"
            )
        return None

    if intent == "MEMORY_READ":
        # Allowed at MARK_AX during LI/LC ops; disallowed elsewhere.
        if "MARK_AX" in position_dims and not has_memory_op:
            return (
                "MEMORY_READ rule fires at MARK_AX without an "
                "OP_LI / OP_LC / MARK_MEM / MEM_STORE / ADDR_KEY guard"
            )
        return None

    return None


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class _Hit:
    __slots__ = ("lineno", "name", "intent", "reason", "position_dims")

    def __init__(
        self,
        lineno: int,
        name: str,
        intent: str,
        reason: str,
        position_dims: set,
    ) -> None:
        self.lineno = lineno
        self.name = name
        self.intent = intent
        self.reason = reason
        self.position_dims = position_dims

    def as_dict(self) -> Dict[str, object]:
        return {
            "line": self.lineno,
            "name": self.name,
            "intent": self.intent,
            "reason": self.reason,
            "position_dims": sorted(self.position_dims),
        }


def _scan_file(path: Path) -> List[_Hit]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits: List[_Hit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        cname = _call_func_name(node)
        if cname not in _RULE_CONSTRUCTORS:
            continue
        # Skip the FFNRule.<attr> form unless it's a known method on FFNRule.
        # ``constant_write`` / ``gated_write`` can also appear on other
        # objects; require either ast.Name (bare call) or
        # ``FFNRule.constant_write`` / ``FFNRule.gated_write``.
        func = node.func
        if isinstance(func, ast.Attribute):
            if cname in ("constant_write", "gated_write"):
                receiver = func.value
                if not (
                    isinstance(receiver, ast.Name) and receiver.id == "FFNRule"
                ):
                    continue
            else:
                # Other building-block helpers should be bare calls.
                continue
        name_node = _kwarg(node, "name")
        if name_node is None:
            continue
        name = _string_value(name_node)
        if not name:
            continue
        intent = classify_intent(name)
        if intent is None:
            continue
        position_dims = _extract_position_dims(node)
        reason = _is_position_role_violation(intent, position_dims)
        if reason is None:
            continue
        hits.append(_Hit(node.lineno, name, intent, reason, position_dims))
    return hits


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def scan(
    repo_root: Path, scan_roots: Iterable[str] = _SCAN_ROOTS
) -> Dict[str, List[_Hit]]:
    by_file: Dict[str, List[_Hit]] = {}
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
            hits = _scan_file(path)
            if hits:
                by_file[rel] = hits
    return by_file


def diff_against_baseline(
    hits_by_file: Dict[str, List[_Hit]],
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


def _find_repo_root() -> Optional[Path]:
    cwd = Path.cwd().resolve()
    root = cwd
    for _ in range(6):
        if (root / "c4_release").is_dir():
            return root
        if (root / "neural_vm").is_dir() and root.name == "c4_release":
            return root.parent
        parent = root.parent
        if parent == root:
            break
        root = parent
    return None


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
                        {
                            "file": str(target),
                            "violations": [h.as_dict() for h in hits],
                        },
                        indent=2,
                    )
                )
            else:
                if hits:
                    print(
                        f"lint_position_role: {len(hits)} position-role "
                        f"violation(s) in {target}:"
                    )
                    for h in hits:
                        print(
                            f"  {target}:{h.lineno}: [{h.intent}] "
                            f"{h.name!r}: {h.reason}"
                        )
                else:
                    print(
                        f"lint_position_role: 0 position-role violations in "
                        f"{target}"
                    )
            return 1 if hits else 0
        # Directory: walk it like a scan root.
        hits_by_file: Dict[str, List[_Hit]] = {}
        for p in _walk_python_files(target):
            file_hits = _scan_file(p)
            if file_hits:
                hits_by_file[str(p)] = file_hits
        total = sum(len(v) for v in hits_by_file.values())
        if args.json:
            print(
                json.dumps(
                    {
                        rel: [h.as_dict() for h in hs]
                        for rel, hs in hits_by_file.items()
                    },
                    indent=2,
                )
            )
        else:
            if total:
                print(
                    f"lint_position_role: {total} position-role violation(s) "
                    f"across {len(hits_by_file)} file(s) under {target}:"
                )
                for rel, hs in sorted(hits_by_file.items()):
                    for h in hs:
                        print(
                            f"  {rel}:{h.lineno}: [{h.intent}] "
                            f"{h.name!r}: {h.reason}"
                        )
            else:
                print(
                    f"lint_position_role: 0 position-role violations under "
                    f"{target}"
                )
        return 1 if total else 0

    if args.root:
        root = Path(args.root).resolve()
    else:
        detected = _find_repo_root()
        if detected is None:
            print(
                f"error: could not find c4_release/ from {Path.cwd()}",
                file=sys.stderr,
            )
            return 2
        root = detected

    hits_by_file = scan(root)
    total_hits = sum(len(v) for v in hits_by_file.values())
    regressions, new_files = diff_against_baseline(hits_by_file)

    if args.json:
        print(
            json.dumps(
                {
                    "total_violations": total_hits,
                    "files": {
                        rel: [h.as_dict() for h in hs]
                        for rel, hs in hits_by_file.items()
                    },
                    "regressions": [
                        {"file": rel, "baseline": b, "current": c}
                        for rel, b, c in regressions
                    ],
                    "new_files": new_files,
                    "baseline_total": sum(_BASELINE.values()),
                },
                indent=2,
            )
        )
        return 1 if (regressions or new_files) else 0

    if args.list:
        if not hits_by_file:
            print("lint_position_role: 0 position-role violations.")
        else:
            print(
                f"lint_position_role: {total_hits} position-role violation(s) "
                f"across {len(hits_by_file)} file(s):"
            )
            for rel, hs in sorted(hits_by_file.items()):
                base = _BASELINE.get(rel, 0)
                marker = " [BASELINED]" if rel in _BASELINE else " [NEW]"
                print(
                    f"  {rel}: {len(hs)} hit(s), baseline {base}{marker}"
                )
                for h in hs:
                    print(
                        f"    {rel}:{h.lineno}: [{h.intent}] "
                        f"{h.name!r}: {h.reason}"
                    )

    ok = not regressions and not new_files
    if ok:
        print(
            f"lint_position_role: OK — {total_hits} violation(s) across "
            f"{len(hits_by_file)} file(s), all within baseline. "
            f"({len(_BASELINE)} files tracked.)"
        )
        return 0

    print(
        f"lint_position_role: REGRESSION — "
        f"{len(regressions)} growing file(s), {len(new_files)} new file(s)."
    )
    if regressions:
        print("\n  Files that grew beyond baseline:")
        for rel, baseline, current in regressions:
            print(
                f"    {rel}: baseline={baseline}, current={current} "
                f"(+{current - baseline})"
            )
            for h in hits_by_file[rel]:
                print(
                    f"      {rel}:{h.lineno}: [{h.intent}] "
                    f"{h.name!r}: {h.reason}"
                )
    if new_files:
        print("\n  Files NOT in baseline that contain violations:")
        for rel in new_files:
            print(f"    {rel}: {len(hits_by_file[rel])} violation(s)")
            for h in hits_by_file[rel]:
                print(
                    f"      {rel}:{h.lineno}: [{h.intent}] "
                    f"{h.name!r}: {h.reason}"
                )
    print(
        "\nFix:\n"
        "  * COMPUTE rules must gate on MARK_SE / MARK_SE_ONLY / HAS_SE.\n"
        "  * TOKEN_EMIT rules must gate on MARK_X or BYTE_INDEX_*.\n"
        "  * MEMORY_READ rules at MARK_AX need an OP_LI / OP_LC / MARK_MEM "
        "/ MEM_STORE guard.\n"
        "After a real migration, decrement the entry in `_BASELINE` "
        "in the SAME commit."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
