#!/usr/bin/env python3
"""Ratchet lint: flag static-registry dim RESOLUTION in tools / ops.

THE TRAP (memory ``feedback_probe_dims_use_built_layout_not_static_registry``):
the widen-repack MOVES ~93% of residual dims relative to the static registry
(``build_default_registry_dynamic`` / ``build_default_registry``). Resolving a
dim's POSITION through the static registry — ``reg.slots[name].start``,
``reg.resolve_dim(cat, role)``, or a ``+N`` offset off a static start — reads
the WRONG cell of a BUILT model, producing false "signal is dead/constant"
walls. The ONE blessed resolution path is
``neural_vm.unified_compiler.dim_resolver.DimResolver`` over
``layout.dim_positions`` (the BUILT layout).

This lint scans Python files (default: ``tools/`` + the ops modules) and, for
each file, flags the trap when BOTH are present:

  (A) it references the static registry — a call to ``build_default_registry``
      / ``build_default_registry_dynamic``, OR an ``import`` of one; AND
  (B) it POSITIONALLY interprets that registry — attribute ``.start`` on a
      slot, ``.resolve_dim(...)``, or ``.slots[...]`` subscript.

A file that only uses the static registry to LIST names (never touches a
position) is fine. A file that goes through ``DimResolver`` is fine. Files can
opt out with a line comment ``# dim-resolution-lint: allow`` (for the handful
of legitimate registry-position uses — e.g. this repo's own trap-demo, or the
allocator byte-identity tests).

Exit code is non-zero when any un-allowlisted trap is found (ratchet).

Run:
    python tools/lint_dim_resolution.py                 # scan tools/ + ops/
    python tools/lint_dim_resolution.py path/to/file.py # scan specific files
    python tools/lint_dim_resolution.py --demo          # prove it discriminates
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release (the package dir)

# Static-registry builder names — referencing OR importing one of these is
# signal (A).
_STATIC_BUILDERS = {"build_default_registry", "build_default_registry_dynamic"}

# Line-level opt-out marker.
_ALLOW_MARKER = "dim-resolution-lint: allow"

# Files that are ALLOWED to touch the static registry positionally, project
# path relative to the package dir. These are the tools whose JOB is the
# static registry (byte-identity vs static, or the trap-demo itself).
_FILE_ALLOWLIST = {
    "tools/dim_resolver_demo.py",       # demonstrates the trap on purpose
    "tools/lint_dim_resolution.py",     # this file (mentions the names)
    "neural_vm/dim_registry_dynamic.py",
    "neural_vm/dim_registry.py",
}


@dataclass
class Finding:
    path: str
    line: int
    kind: str          # "slot_start" | "resolve_dim" | "slots_subscript"
    snippet: str


@dataclass
class FileReport:
    path: str
    references_static: bool = False
    positional_uses: List[Finding] = field(default_factory=list)
    allow_lines: Set[int] = field(default_factory=set)
    file_allowlisted: bool = False

    @property
    def traps(self) -> List[Finding]:
        if not self.references_static or self.file_allowlisted:
            return []
        return [f for f in self.positional_uses if f.line not in self.allow_lines]


class _Scanner(ast.NodeVisitor):
    """Collect static-registry references + positional uses in one AST pass."""

    def __init__(self, report: FileReport):
        self.r = report

    # (A) references: a call to a static builder, or an import of one.
    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        name = None
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if name in _STATIC_BUILDERS:
            self.r.references_static = True
        # (B) resolve_dim(...) positional resolution
        if isinstance(fn, ast.Attribute) and fn.attr == "resolve_dim":
            self.r.positional_uses.append(
                Finding(self.r.path, node.lineno, "resolve_dim",
                        _seg(node)))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            if a.name in _STATIC_BUILDERS:
                self.r.references_static = True
        self.generic_visit(node)

    # (B) `.start` attribute access (slot.start) — positional resolution.
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "start":
            self.r.positional_uses.append(
                Finding(self.r.path, node.lineno, "slot_start", _seg(node)))
        self.generic_visit(node)

    # (B) `.slots[...]` subscript — positional resolution.
    def visit_Subscript(self, node: ast.Subscript) -> None:
        val = node.value
        if isinstance(val, ast.Attribute) and val.attr == "slots":
            self.r.positional_uses.append(
                Finding(self.r.path, node.lineno, "slots_subscript",
                        _seg(node)))
        self.generic_visit(node)


def _seg(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # py3.9+
    except Exception:  # noqa: BLE001
        return f"<node@{getattr(node, 'lineno', '?')}>"


def scan_file(path: str) -> Optional[FileReport]:
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
    report = FileReport(path=rel)
    report.file_allowlisted = rel in _FILE_ALLOWLIST
    for i, line in enumerate(src.splitlines(), start=1):
        if _ALLOW_MARKER in line:
            report.allow_lines.add(i)
    _Scanner(report).visit(tree)
    return report


def _default_targets() -> List[str]:
    """Default scan set = ``tools/`` (where the documented trap bites).

    The ``ops/`` compile-time modules also read the static registry, but they
    run INSIDE the compile with ``dim_positions`` in scope, so a static
    ``.start`` there is a legacy-code smell, not the probe-reads-a-built-model
    trap this ratchet guards. Scan them explicitly (pass the path) when doing
    the Phase 7.E semantic-dim-ref migration.
    """
    targets: List[str] = []
    base = os.path.join(_PKG, "tools")
    for root, _dirs, files in os.walk(base):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                targets.append(os.path.join(root, f))
    return sorted(targets)


def run(paths: List[str], *, verbose: bool = False) -> int:
    reports = []
    for p in paths:
        r = scan_file(p)
        if r is not None:
            reports.append(r)

    trapped = [r for r in reports if r.traps]
    n_ref = sum(1 for r in reports if r.references_static)

    if verbose:
        for r in reports:
            if r.references_static and not r.traps and r.positional_uses:
                print(f"  ok (allowlisted/name-only): {r.path}")

    if not trapped:
        print(
            f"lint_dim_resolution: PASS — scanned {len(reports)} files, "
            f"{n_ref} reference the static registry, 0 resolve a dim POSITION "
            f"through it (all clean or allowlisted)."
        )
        return 0

    print(
        f"lint_dim_resolution: {len(trapped)} file(s) resolve a dim POSITION "
        f"via the STATIC registry (the widen-repack trap). Route through "
        f"DimResolver over layout.dim_positions instead."
    )
    for r in sorted(trapped, key=lambda x: x.path):
        print(f"\n  {r.path}")
        for f in r.traps:
            print(f"    L{f.line}: [{f.kind}] {f.snippet}")
    print(
        "\nFix: `from neural_vm.unified_compiler.dim_resolver import "
        "DimResolver` then `DimResolver.from_layout(layout).resolve(NAME)`. "
        "For a legitimate static-registry position use, append "
        f"`# {_ALLOW_MARKER}` to the line."
    )
    return 1


# ----------------------------------------------------------------------
# --demo: prove the lint discriminates bad (static) vs good (DimResolver)
# ----------------------------------------------------------------------
_DEMO_BAD = '''\
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
reg = build_default_registry_dynamic()
def col(name):
    return reg.slots[name].start   # STATIC position -> WRONG cell on a built model
op_psh = col("OP_PSH")             # resolves to 275 (static), built is 197
'''

_DEMO_GOOD = '''\
from neural_vm.unified_compiler.dim_resolver import DimResolver
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
_model, layout = compile_full_vm_dynamic(disk_cache=True)
r = DimResolver.from_layout(layout)
op_psh = r.resolve("OP_PSH")       # resolves to the BUILT column (197)
'''


def _demo() -> int:
    import tempfile

    print("=" * 70)
    print("DEMO — lint_dim_resolution discriminates static-registry resolution")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "demo_bad_static.py")
        good = os.path.join(td, "demo_good_resolver.py")
        with open(bad, "w") as fh:
            fh.write(_DEMO_BAD)
        with open(good, "w") as fh:
            fh.write(_DEMO_GOOD)

        rb = scan_file(bad)
        rg = scan_file(good)

        bad_flagged = bool(rb and rb.traps)
        good_clean = bool(rg and not rg.traps)

        print("\n[BAD]  static registry + .slots[...].start:")
        for f in (rb.traps if rb else []):
            print(f"   FLAGGED L{f.line}: [{f.kind}] {f.snippet}")
        print(f"   -> {'FLAGGED (correct)' if bad_flagged else 'MISSED (BUG)'}")

        print("\n[GOOD] DimResolver over layout.dim_positions:")
        print(f"   references_static={rg.references_static if rg else '?'}, "
              f"positional_uses={len(rg.positional_uses) if rg else '?'}")
        print(f"   -> {'CLEAN (correct)' if good_clean else 'FLAGGED (BUG)'}")

    ok = bad_flagged and good_clean
    print("\nRESULT:", "PASS" if ok else "FAIL",
          f"(bad-flagged={bad_flagged}, good-clean={good_clean})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*",
                    help="files to scan (default: tools/ + ops/)")
    ap.add_argument("--demo", action="store_true",
                    help="prove the lint discriminates bad vs good resolution")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.demo:
        return _demo()

    paths = args.paths or _default_targets()
    # Resolve relative CLI paths against cwd, then package dir.
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
    return run(resolved, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
