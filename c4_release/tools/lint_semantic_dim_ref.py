#!/usr/bin/env python3
"""Ratchet lint: flag RAW role-meaningful base-slot dim refs in ops FFNRule
authoring (Phase 7.E semantic-dim-ref migration).

WHY (companion to ``tools/lint_dim_resolution.py``): ``lint_dim_resolution``
guards the *resolution* trap (reading a dim POSITION through the static
registry, which the widen-repack moves). THIS lint guards the *authoring*
trap: an ``FFNRule`` condition / gate / write that hard-codes a
role-meaningful base slot NAME (``("MARK_AX", 1.0)``, ``("OUTPUT_LO+3",
2.0)``, ``("AX_CARRY_LO+0", -1.0)`` ...) instead of authoring it from its
semantic family via :func:`neural_vm.dim_registry.dim_ref` —
``dim_ref("marker", "AX")`` / ``dim_ref("output_lo", "nibble", 3)`` /
``dim_ref("ax_carry_lo", "AX", 0)``. Both produce the byte-identical
``"NAME+offset"`` string, but the ``dim_ref`` form survives a slot RENAME /
family REPACK: the ref follows the ``(category, role)`` binding, not a
frozen NAME. The Phase 7.E migration ported l0-l16's FFN authoring to
``dim_ref``; this ratchet freezes the current raw-ref count so NEW ops must
author role-meaningful slots via ``dim_ref``.

WHAT IS FLAGGED — a raw ref is a 2-tuple ``(NAME_literal, weight_expr)``
whose first element is a STRING LITERAL equal to (or an ``NAME+offset`` of) a
**category-registered** base slot NAME (the 78 names tagged by
``dim_registry._register_default_categories`` — markers, byte_index,
output/alu/ax_carry/carry/cmp/memory families, opcode_flags, the tagged
over-width STACK0 bands). That shape is exactly an ``FFNRule`` condition /
gate-term / write pair.

WHAT IS *NOT* FLAGGED (matches the l16/l15 "LEFT RAW" policy):
  * ``dim_ref(...)`` calls — an ``ast.Call``, not a string literal.
  * ``reads=`` / ``writes=`` op-metadata **sets** (``{...}``) — those are
    data-flow documentation, not weight authoring, so a bare NAME there is
    correct (never a ``(NAME, weight)`` tuple).
  * ``dp["MARK_AX"]`` attention name->BD-attribute resolver subscripts —
    the ``AP`` / ``AO`` Q/K/V/O writers resolve positions through ``BD``,
    not a family; those stay raw by design.
  * unbound / non-family flags (``PSH_AT_SP``, ``IS_BYTE``, ``HAS_SE``,
    ``H1+i``, ``EMBED_LO+k``, ``OUTPUT_HI_THIS_STEP+k`` ...): they carry NO
    ``(category, role)`` tag, so they are not category-registered and are
    never flagged.
  * genuinely-structural ``+N`` offsets stay raw *inside* the ref: the lint
    only requires the BASE NAME be family-authored; ``dim_ref("output_lo",
    "nibble", 3)`` keeps the ``3`` as a raw index.

RATCHET SEMANTICS — WARN by default (exit 0) and print the current-offender
count so a reviewer can see NEW code must not add to it. ``--max N`` (or
``--strict`` = ``--max <current baseline>``) makes it FAIL (exit 1) when the
count exceeds ``N``, for CI. Per-line opt-out: append
``# sem-dim-ref-lint: allow`` (for a legitimately-raw authoring ref).

Run:
    python tools/lint_semantic_dim_ref.py                  # scan ops/, WARN + count
    python tools/lint_semantic_dim_ref.py --max 1883       # FAIL if > 1883 offenders
    python tools/lint_semantic_dim_ref.py path/to/lN_ops.py
    python tools/lint_semantic_dim_ref.py --demo           # prove it discriminates
    python tools/lint_semantic_dim_ref.py -v               # per-ref listing
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import FrozenSet, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release (the package dir)

# Default scan set = the per-layer op modules (Phase 7.E authoring surface).
_OPS_DIR = os.path.join(_PKG, "neural_vm", "unified_compiler", "ops")

# Line-level opt-out marker.
_ALLOW_MARKER = "sem-dim-ref-lint: allow"


@lru_cache(maxsize=1)
def category_registered_names() -> FrozenSet[str]:
    """The base slot NAMEs that carry a ``(category, role)`` tag.

    These are exactly the names ``dim_ref`` can author (the map lives in
    ``dim_registry._register_default_categories`` + the over-width band
    tags). A raw FFN-authoring ref to one of these is a ratchet offender;
    any other NAME (unbound flag, threshold-head bank slot, embed nibble)
    is NOT family-authorable and is left raw by policy.

    Built once via ``build_default_registry`` (the same source ``dim_ref``
    resolves against), with deprecation warnings suppressed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Import lazily so ``--help`` / ``--demo`` don't pay the build cost
        # unless a real scan needs the names.
        from neural_vm.dim_registry import build_default_registry

        reg = build_default_registry()
    return frozenset(
        name
        for name, slot in reg.slots.items()
        if getattr(slot, "category", None) is not None
    )


def _base_name(ref: str) -> str:
    """Base slot NAME of a ``"NAME+offset"`` ref (drops the ``+offset``)."""
    return ref.split("+", 1)[0]


def is_raw_family_ref(ref: str, names: FrozenSet[str]) -> bool:
    """True iff ``ref`` is a raw role-meaningful family ref that ``dim_ref``
    should author.

    ``ref`` is a string literal like ``"MARK_AX"`` or ``"OUTPUT_LO+3"``. It
    is an offender iff its BASE NAME is category-registered.
    """
    return _base_name(ref) in names


@dataclass
class Finding:
    path: str
    line: int
    ref: str
    snippet: str


@dataclass
class FileReport:
    path: str
    findings: List[Finding] = field(default_factory=list)
    allow_lines: FrozenSet[int] = field(default_factory=frozenset)

    @property
    def offenders(self) -> List[Finding]:
        return [f for f in self.findings if f.line not in self.allow_lines]


class _Scanner(ast.NodeVisitor):
    """Collect raw ``(NAME_literal, weight)`` FFN-authoring tuples.

    A condition / gate-term / write is a 2-tuple whose first element is a
    string-literal family ref. ``reads=``/``writes=`` metadata are SETS
    (never 2-tuples), ``dp[...]`` resolvers are subscripts, and ``dim_ref``
    refs are Calls — none match this shape, so they are structurally
    excluded (no context heuristics needed).
    """

    def __init__(self, path: str, names: FrozenSet[str], src_lines: List[str]):
        self.path = path
        self.names = names
        self.src_lines = src_lines
        self.findings: List[Finding] = []

    def visit_Tuple(self, node: ast.Tuple) -> None:
        if len(node.elts) == 2:
            head = node.elts[0]
            if (
                isinstance(head, ast.Constant)
                and isinstance(head.value, str)
                and is_raw_family_ref(head.value, self.names)
            ):
                snippet = ""
                if 1 <= node.lineno <= len(self.src_lines):
                    snippet = self.src_lines[node.lineno - 1].strip()[:100]
                self.findings.append(
                    Finding(self.path, node.lineno, head.value, snippet)
                )
        self.generic_visit(node)


def scan_file(path: str, names: FrozenSet[str]) -> Optional[FileReport]:
    rel = os.path.relpath(path, _PKG)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        return None
    lines = src.splitlines()
    allow = frozenset(
        i for i, ln in enumerate(lines, start=1) if _ALLOW_MARKER in ln
    )
    scanner = _Scanner(rel, names, lines)
    scanner.visit(tree)
    return FileReport(path=rel, findings=scanner.findings, allow_lines=allow)


def _default_targets() -> List[str]:
    targets: List[str] = []
    for root, _dirs, files in os.walk(_OPS_DIR):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                targets.append(os.path.join(root, f))
    return sorted(targets)


def run(
    paths: List[str],
    *,
    max_offenders: Optional[int] = None,
    verbose: bool = False,
) -> int:
    names = category_registered_names()
    reports = [r for r in (scan_file(p, names) for p in paths) if r is not None]

    per_file = [(r, r.offenders) for r in reports]
    per_file = [(r, offs) for r, offs in per_file if offs]
    total = sum(len(offs) for _r, offs in per_file)

    if verbose:
        for r, offs in sorted(per_file, key=lambda x: x[0].path):
            print(f"\n  {r.path}  ({len(offs)} raw family refs)")
            for f in offs:
                print(f"    L{f.line}: {f.ref!r}  |  {f.snippet}")
    else:
        for r, offs in sorted(per_file, key=lambda x: -len(x[1])):
            print(f"  {len(offs):5d}  {r.path}")

    over = max_offenders is not None and total > max_offenders

    print()
    if max_offenders is None:
        print(
            f"lint_semantic_dim_ref: WARN — {total} raw role-meaningful "
            f"family refs in ops FFNRule authoring across {len(reports)} "
            f"file(s). New ops MUST author these via dim_ref(category, role, "
            f"offset). (Pass --max {total} to freeze this as the ratchet "
            f"baseline; --strict to fail if it grows.)"
        )
        return 0

    if over:
        print(
            f"lint_semantic_dim_ref: FAIL — {total} raw family refs exceed "
            f"the ratchet ceiling of {max_offenders}. Author new "
            f"role-meaningful slots via dim_ref(category, role, offset) "
            f"(see dim_registry._register_default_categories for the map), "
            f"or append '# {_ALLOW_MARKER}' to a legitimately-raw line."
        )
        return 1

    print(
        f"lint_semantic_dim_ref: PASS — {total} raw family refs "
        f"<= ceiling {max_offenders} (ratchet held)."
    )
    return 0


# ----------------------------------------------------------------------
# --demo: prove the lint flags a raw ref and passes a dim_ref one
# ----------------------------------------------------------------------
_DEMO_RAW = '''\
from neural_vm.unified_compiler.building_blocks_dsl import multi_way_and_rule
rule = multi_way_and_rule(
    name="demo",
    conditions=(("MARK_AX", 1.0), ("OUTPUT_LO+3", -1.0)),  # RAW family refs
    threshold=0.5,
    writes=(("AX_CARRY_LO+0", 2.0),),                        # RAW family ref
)
'''

_DEMO_CLEAN = '''\
from neural_vm.dim_registry import dim_ref
from neural_vm.unified_compiler.building_blocks_dsl import multi_way_and_rule
rule = multi_way_and_rule(
    name="demo",
    conditions=((dim_ref("marker", "AX"), 1.0),
                (dim_ref("output_lo", "nibble", 3), -1.0)),
    threshold=0.5,
    writes=((dim_ref("ax_carry_lo", "AX", 0), 2.0),),
    reads={"MARK_AX", "OUTPUT_LO"},        # metadata SET -> not flagged
)
gate_pos = dp["MARK_AX"]                    # dp resolver -> not flagged
'''


def _demo() -> int:
    import tempfile

    names = category_registered_names()

    print("=" * 70)
    print("DEMO — lint_semantic_dim_ref flags RAW family refs, passes dim_ref")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "demo_raw_refs.py")
        clean = os.path.join(td, "demo_dim_ref.py")
        with open(raw, "w") as fh:
            fh.write(_DEMO_RAW)
        with open(clean, "w") as fh:
            fh.write(_DEMO_CLEAN)

        rr = scan_file(raw, names)
        rc = scan_file(clean, names)

        raw_flagged = bool(rr and rr.offenders)
        clean_ok = bool(rc and not rc.offenders)

        print("\n[RAW]   hand-coded family NAMEs in an FFNRule:")
        for f in (rr.offenders if rr else []):
            print(f"   FLAGGED L{f.line}: {f.ref!r}  |  {f.snippet}")
        print(f"   -> {'FLAGGED (correct)' if raw_flagged else 'MISSED (BUG)'}")

        print("\n[CLEAN] dim_ref authoring + reads-set + dp[] resolver:")
        print(f"   offenders={len(rc.offenders) if rc else '?'}  "
              f"(expected 0: dim_ref=Call, reads=set, dp[]=subscript)")
        print(f"   -> {'CLEAN (correct)' if clean_ok else 'FLAGGED (BUG)'}")

    ok = raw_flagged and clean_ok
    print("\nRESULT:", "PASS" if ok else "FAIL",
          f"(raw-flagged={raw_flagged}, clean-ok={clean_ok})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                    help="files to scan (default: the ops/ modules)")
    ap.add_argument("--demo", action="store_true",
                    help="prove the lint flags a raw ref, passes a dim_ref one")
    ap.add_argument("--max", type=int, default=None,
                    help="ratchet ceiling: FAIL (exit 1) if offenders exceed it")
    ap.add_argument("--strict", action="store_true",
                    help="freeze at the CURRENT count: fail if it grows")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every offending ref, not just per-file counts")
    args = ap.parse_args()

    if args.demo:
        return _demo()

    paths = args.paths or _default_targets()
    resolved = []
    for p in paths:
        if os.path.isabs(p) and os.path.exists(p):
            resolved.append(p)
        elif os.path.exists(p):
            resolved.append(os.path.abspath(p))
        elif os.path.exists(os.path.join(_PKG, p)):
            resolved.append(os.path.join(_PKG, p))
        else:
            print(f"warning: path not found: {p}", file=sys.stderr)

    max_offenders = args.max
    if args.strict and max_offenders is None:
        names = category_registered_names()
        reports = [r for r in (scan_file(p, names) for p in resolved)
                   if r is not None]
        max_offenders = sum(len(r.offenders) for r in reports)

    return run(resolved, max_offenders=max_offenders, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
