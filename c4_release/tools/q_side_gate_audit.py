"""Q-side single-condition gate audit for unified_compiler ops.

Scans ``c4_release/neural_vm/unified_compiler/ops/*_ops.py`` for the
anti-pattern documented in
``docs/DSL_ATTENTION_GATE_FINDING_2026_06_07.md`` (commit ``bf11d69d``):

    # Q side
    q = (..., AP(slot, BD.MARK_*, L), ...)
    # K side
    k = (..., AP(slot, BD.CONST, ...), ...)  # only CONST at the slot

After softmax the per-Q-row ``L * MARK_*`` offset cancels — the
"gate" does not filter K positions.

The audit handles the two flavours of weight-author code present on
``main`` (HEAD ``bf11d69d`` at audit time):

  * **Declarative DSL** — ``AP(slot, dim_expr, weight)`` calls inside
    ``q=`` / ``k=`` tuples of a ``DeclarativeAttentionHeadSpec``.  The
    main path.
  * **Imperative writes** — ``attn.W_q[slot_expr, BD.<DIM>] = ...`` /
    ``attn.W_k[...] = ...``.  Legacy / not-yet-migrated code.

For both forms it groups Q and K writes by ``(scope, slot_expr)``,
classifies each Q-side condition write as one of:

  * **safe** — K-side at the slot writes at least one non-``CONST``
    dim (a real K-side discriminator).
  * **no-op gate** — K-side at the slot has no writes, or only
    ``CONST`` writes.

Output is a Markdown summary plus per-file detail.

Run from the repo root:

    python3 c4_release/tools/q_side_gate_audit.py
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path


OPS_DIR = (
    Path(__file__).resolve().parent.parent
    / "neural_vm"
    / "unified_compiler"
    / "ops"
)

CONDITION_PREFIXES = ("MARK_", "OP_", "HAS_", "IS_")
UNIFORM_DIMS = {"CONST"}

# Match BD.<DIM> or proxy.<DIM>; the leading attribute may itself be
# a local alias (e.g. ``spec.CONST`` from the carry-forward primitive),
# so accept any dotted prefix.
DIM_RE = re.compile(
    r"(?:[A-Za-z_][\w]*\.)?(?P<dim>[A-Z][A-Z0-9_]+)"
)


def is_condition_dim(dim: str) -> bool:
    return any(dim.startswith(p) for p in CONDITION_PREFIXES)


def _normalize_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def _extract_dim(expr: str) -> str | None:
    """Best-effort extract the residual-dim symbol from a dim expression.

    Examples: ``BD.MARK_AX`` → ``MARK_AX``, ``BD.L1H1 + l1h1_idx`` →
    ``L1H1``, ``proxy.CONST`` → ``CONST``.
    """
    expr = expr.strip()
    # Strip outer parens.
    while expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1].strip()
    # Take the first BD/proxy.<DIM> token.
    m = DIM_RE.search(expr)
    if not m:
        return None
    return m.group("dim")


class _ApAoCallVisitor(ast.NodeVisitor):
    """Collect ``AP(slot, dim, weight)`` and ``AO(...)`` calls inside
    ``q=`` / ``k=`` / ``v=`` / ``o=`` keyword arguments of declarative
    head specs, plus inside assignments such as ``q = [...]``.
    """

    def __init__(self, source: str, tree: ast.Module | None = None):
        self.source = source.splitlines()
        # (scope, slot_expr) → {"q": [...], "k": [...]}
        self.entries = defaultdict(
            lambda: {"q": [], "k": [], "v": [], "o": []}
        )
        self.scope_stack: list[str] = ["<module>"]
        # Index of module-level helper functions for resolving calls
        # that build AP tuples. Maps name → FunctionDef node.
        self.helpers: dict[str, ast.FunctionDef] = {}
        if tree is not None:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.helpers[node.name] = node
        self._helper_depth = 0

    # Track scopes so cross-side matches are confined to one bake.
    def visit_FunctionDef(self, node):
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _scope(self) -> str:
        return self.scope_stack[-1]

    def _slot_expr(self, slot_node: ast.AST) -> str:
        return _normalize_expr(ast.unparse(slot_node))

    def _record_ap(self, side: str, args: list[ast.AST], lineno: int):
        if len(args) < 2:
            return
        slot_expr = self._slot_expr(args[0])
        dim_expr = ast.unparse(args[1])
        dim = _extract_dim(dim_expr)
        if dim is None:
            return
        raw = self.source[lineno - 1] if 0 <= lineno - 1 < len(self.source) else ""
        self.entries[(self._scope(), slot_expr)][side].append(
            {
                "lineno": lineno,
                "dim": dim,
                "dim_expr": dim_expr,
                "raw": raw.rstrip(),
            }
        )

    def _walk_list_for_side(self, side: str, container: ast.AST):
        """Walk a list/tuple/set literal containing AP(...) calls.

        Also handles list comprehensions and starred unpacks; for
        non-literal cases (e.g. ``q = build_q()``) we just skip — the
        regex fallback in :func:`parse_imperative` covers the rare
        cases where this matters.
        """
        if isinstance(container, (ast.List, ast.Tuple, ast.Set)):
            for elt in container.elts:
                self._walk_list_for_side(side, elt)
            return
        if isinstance(container, ast.GeneratorExp):
            # Walk the element expression.
            self._walk_list_for_side(side, container.elt)
            return
        if isinstance(container, ast.Starred):
            self._walk_list_for_side(side, container.value)
            return
        if isinstance(container, ast.BinOp) and isinstance(container.op, ast.Add):
            self._walk_list_for_side(side, container.left)
            self._walk_list_for_side(side, container.right)
            return
        if isinstance(container, ast.IfExp):
            self._walk_list_for_side(side, container.body)
            self._walk_list_for_side(side, container.orelse)
            return
        if isinstance(container, ast.Call):
            fn = container.func
            fn_name = (
                fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name)
                else None
            )
            if fn_name in ("AP", "AO"):
                self._record_ap(side, container.args, container.lineno)
                return
            elif fn_name in ("tuple", "list"):
                if container.args:
                    self._walk_list_for_side(side, container.args[0])
                return
            # Resolve module-local helper functions returning AP tuples
            # (e.g. ``_addr_key_match_writes``, ``_band_projection_writes``,
            # ``pc_gate(slot)`` — anything defined in this module that
            # returns a tuple/list of AP calls).
            self._resolve_helper(side, container)

    def visit_Assign(self, node):
        """Track ``q = [...]`` and ``q.append(AP(...))`` style writes."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("q", "k", "v", "o"):
                self._walk_list_for_side(name, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in ("q", "k", "v", "o"):
            self._walk_list_for_side(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Expr(self, node):
        # Catch ``q.append(AP(...))``.
        v = node.value
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr == "append"
            and isinstance(v.func.value, ast.Name)
            and v.func.value.id in ("q", "k", "v", "o")
            and v.args
        ):
            self._walk_list_for_side(v.func.value.id, v.args[0])
        self.generic_visit(node)

    def _resolve_helper(self, side: str, call: ast.Call):
        """Resolve module-level helper calls that return AP tuples by
        walking the helper's body for return statements containing
        AP(...) / AO(...) constructions.
        """
        if self._helper_depth > 4:
            return
        fn = call.func
        name = None
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if name is None or name not in self.helpers:
            return
        body = self.helpers[name]
        self._helper_depth += 1
        try:
            for sub in ast.walk(body):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    self._walk_list_for_side(side, sub.value)
                elif isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name) and tgt.id in (
                            "q", "k", "v", "o"
                        ) and tgt.id == side:
                            self._walk_list_for_side(side, sub.value)
        finally:
            self._helper_depth -= 1

    def visit_Call(self, node):
        """Catch ``DeclarativeAttentionHeadSpec(q=(...), k=(...))`` and
        any function-call form that passes ``q=`` / ``k=`` kwargs.
        """
        for kw in node.keywords:
            if kw.arg in ("q", "k", "v", "o"):
                self._walk_list_for_side(kw.arg, kw.value)
        self.generic_visit(node)


def parse_declarative(path: Path):
    """Parse declarative DSL writes via AST."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    v = _ApAoCallVisitor(text, tree)
    v.visit(tree)
    return v.entries


# ---------------------------------------------------------------------------
# Imperative path (legacy)
# ---------------------------------------------------------------------------

IMP_RE = re.compile(
    r"""
    (?:[\w\.]+\.)? W_(?P<side>[qkvo])
    \s*\[\s*
      (?P<slot_expr>[^,\]]+?)
      \s*,\s*
      (?P<dim_expr>[^\]]+?)
    \s*\]
    \s*=\s*
    (?P<rhs>[^\n#]+)
    """,
    re.VERBOSE,
)


def parse_imperative(path: Path):
    text = path.read_text(encoding="utf-8")
    entries = defaultdict(
        lambda: {"q": [], "k": [], "v": [], "o": []}
    )
    # AST for scope lookup.
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    scopes = []
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno + 200)
                scopes.append((node.lineno, end, node.name))
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in IMP_RE.finditer(line):
            slot_expr = _normalize_expr(m.group("slot_expr"))
            dim_expr = m.group("dim_expr").strip()
            dim = _extract_dim(dim_expr)
            if dim is None:
                continue
            scope_name = "<module>"
            best_start = -1
            for start, end, name in scopes:
                if start <= lineno <= end and start > best_start:
                    scope_name = name
                    best_start = start
            entries[(scope_name, slot_expr)][m.group("side")].append(
                {
                    "lineno": lineno,
                    "dim": dim,
                    "dim_expr": dim_expr,
                    "raw": line.rstrip(),
                }
            )
    return entries


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_file(path: Path):
    decl = parse_declarative(path)
    imp = parse_imperative(path)
    # Merge with a flavour tag.
    merged = defaultdict(
        lambda: {"q": [], "k": [], "v": [], "o": []}
    )
    for key, sides in decl.items():
        for s in "qkvo":
            for w in sides[s]:
                w["flavour"] = "decl"
                merged[key][s].append(w)
    for key, sides in imp.items():
        for s in "qkvo":
            for w in sides[s]:
                w["flavour"] = "imp"
                merged[key][s].append(w)

    findings = {
        "total_q_condition": 0,
        "safe": [],
        "noop": [],
    }
    for (scope_name, slot), sides in merged.items():
        q_writes = sides["q"]
        k_writes = sides["k"]
        for qw in q_writes:
            if not is_condition_dim(qw["dim"]):
                continue
            findings["total_q_condition"] += 1
            if not k_writes:
                findings["noop"].append(
                    {
                        "scope": scope_name,
                        "slot": slot,
                        "q_dim": qw["dim"],
                        "lineno": qw["lineno"],
                        "raw": qw["raw"],
                        "flavour": qw["flavour"],
                        "reason": "no K-side write at this slot",
                    }
                )
                continue
            k_dims = [kw["dim"] for kw in k_writes]
            non_uniform_k = [d for d in k_dims if d not in UNIFORM_DIMS]
            if not non_uniform_k:
                findings["noop"].append(
                    {
                        "scope": scope_name,
                        "slot": slot,
                        "q_dim": qw["dim"],
                        "lineno": qw["lineno"],
                        "raw": qw["raw"],
                        "flavour": qw["flavour"],
                        "reason": "K-side only writes CONST at this slot",
                    }
                )
                continue
            findings["safe"].append(
                {
                    "scope": scope_name,
                    "slot": slot,
                    "q_dim": qw["dim"],
                    "k_dims": sorted(set(k_dims)),
                    "lineno": qw["lineno"],
                    "raw": qw["raw"],
                    "flavour": qw["flavour"],
                }
            )
    return findings


def main():
    files = sorted(OPS_DIR.glob("*.py"))
    files = [f for f in files if f.name != "__init__.py"]
    print(f"Auditing {len(files)} files under {OPS_DIR}\n")

    totals = {"total_q": 0, "safe": 0, "noop": 0}
    per_file = {}
    for f in files:
        findings = audit_file(f)
        per_file[f.name] = findings
        totals["total_q"] += findings["total_q_condition"]
        totals["safe"] += len(findings["safe"])
        totals["noop"] += len(findings["noop"])

    print("## Per-file summary\n")
    print("| file | Q-cond writes | safe | no-op gate |")
    print("| --- | ---: | ---: | ---: |")
    for name in sorted(per_file):
        f = per_file[name]
        if f["total_q_condition"] == 0:
            continue
        print(
            f"| {name} | {f['total_q_condition']} | "
            f"{len(f['safe'])} | {len(f['noop'])} |"
        )
    print(
        f"| **total** | **{totals['total_q']}** | "
        f"**{totals['safe']}** | **{totals['noop']}** |"
    )

    print("\n## No-op gate candidates\n")
    for name in sorted(per_file):
        f = per_file[name]
        if not f["noop"]:
            continue
        print(f"\n### {name}\n")
        for n in f["noop"]:
            print(
                f"- L{n['lineno']} in `{n['scope']}` ({n['flavour']}) — "
                f"slot=`{n['slot']}` Q-dim=`{n['q_dim']}` ({n['reason']})"
            )
            print(f"  `{n['raw'].strip()}`")


if __name__ == "__main__":
    main()
