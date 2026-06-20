#!/usr/bin/env python3
"""Positional-invariant audit — the STEP_TOKENS shift-risk catalog.

The #1 campaign bug class
=========================

The ``C4_NO_STACK0_EMIT`` campaign collapses the per-step token block from
``STEP_TOKENS=35`` to ``30`` by dropping the 5-token STACK0 register block.
Every FFN rule / attention head that encodes POSITIONAL / DISTANCE logic at
a *fixed token offset* — ``BYTE_INDEX_*``, ``STACK0_BYTE0/1/2/3``, the
``MEM_VAL_B*`` ``d=N-from-MEM`` slot predictors, and the ``H*`` / ``L1H*`` /
``L2H0`` / ``L1H4`` marker-*distance* threshold heads — assumes the 35-token
frame. The IR does **not** record "this rule assumes ``STEP_TOKENS=35``", so
when the frame shifts 5 tokens EVERY such marker lands on the wrong physical
row and the rule silently mis-fires.

This class is confirmed, this session, behind:

* **div/mod** — ``efficient_alu_neural.py`` recovers operand-A byte 1 via a
  ``cummax`` over ``STACK0_BYTE1`` rows; when the STACK0 block is dropped the
  anchor row shifts and the dividend high byte is gathered from the wrong
  position.
* **var-multi-local operand-CAM** — ``make_layer8_mem_to_alu_op`` content-
  addresses the ``mem[SP]`` value-byte-0 row with the ``L2H0[MEM]=1 AND
  H1[MEM]=0`` (``d=6-from-MEM``) distance predicate; the ``H``/``L2H0``
  distance markers move when the frame shrinks.

NOTHING catches this statically today. This tool is that static audit: it
enumerates every rule / head / imperative weight-write that references a
positional-frame dim and reports whether it is campaign-guarded (consults
``no_stack0_emit_enabled()``) or **UNGUARDED** — the unguarded set is the
campaign's remaining shift-risk surface (the *next* bugs).

How it works
============

A pure-static AST walk (no model build — tooling-only, byte-identical
golden). For each scanned ``.py`` file it:

1. Derives the **positional-frame dim set** from ``dim_registry.py`` itself
   (every ``_pin(NAME, ..., "<description with a distance/byte-index/position
   phrase>")``) plus a hand-seeded core set, so the audit stays in sync with
   the registry rather than hard-coding a stale list.
2. Finds every reference to a positional-frame dim in three forms:
   * **string** literals (``"STACK0_BYTE1"``, ``"H1+4"``, ``"L2H0+4"``) used
     in rule ``conditions`` / ``scope`` / transition tables;
   * **attribute** access (``BD.L2H0``, ``proxy.STACK0_BYTE1``, ``bd.H1``)
     used by imperative attention / ALU forward code;
   * **dim_ref("category", "role")** semantic calls (``dim_ref("byte_index",
     "0")``, ``dim_ref("memory_lo", "val_b1")``).
3. Attributes each reference to its enclosing ``def`` (and the ``make_*`` /
   ``_layerN_*`` op factory that owns it), records ``file:line``, the dim,
   and whether that enclosing function — or its module — consults
   ``no_stack0_emit_enabled()`` (the campaign guard).

Catalog ranking
===============

Each hit is ranked by shift-risk:

* ``UNGUARDED``     — references a positional-frame dim, the enclosing op does
                      NOT consult ``no_stack0_emit_enabled()``. WILL silently
                      shift/break when ``STEP_TOKENS`` drops. **Highest risk.**
* ``CAMPAIGN_AWARE``— references a positional-frame dim AND the enclosing op
                      consults ``no_stack0_emit_enabled()``. Built for /
                      neutralized under the campaign, but may still hand-compute
                      a raw distance offset (``+MEM_I``) that the compiler does
                      not auto-shift — review, lower priority.

A reference that also carries a raw ``+<offset>`` (distance-from-marker, e.g.
``BD.L2H0 + MEM_I``, ``"H1+4"``) is additionally flagged ``DISTANCE_OFFSET``
because a bare positional-flag read is frame-robust-ish but an *offset into*
a distance bank is doubly frame-dependent.

Usage
=====

::

    python c4_release/tools/lint_positional_invariants.py            # ranked catalog
    python c4_release/tools/lint_positional_invariants.py --json
    python c4_release/tools/lint_positional_invariants.py --unguarded-only
    python c4_release/tools/lint_positional_invariants.py --dim STACK0_BYTE1
    python c4_release/tools/lint_positional_invariants.py --prove   # self-check: div/mod + operand-CAM

Exit codes
==========

* 0 — audit ran (always, when not in ``--prove`` mode; this is a catalog,
      not a ratchet — it does not fail CI on count growth)
* 1 — ``--prove`` mode and the known roots were NOT both caught
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
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Files to scan for positional-frame references. The per-layer op modules
# hold the declarative rules; ``efficient_alu_neural.py`` + ``vm_step.py``
# hold the surviving imperative compute (the div/mod cummax + the L14 borrow
# cascade); the L8 mem-to-ALU CAM lives in ``l8_ops.py``.
_SCAN_RELS: Tuple[str, ...] = (
    "neural_vm/unified_compiler/ops",
    "neural_vm/efficient_alu_neural.py",
    "neural_vm/vm_step.py",
    "neural_vm/setup_helpers.py",
)

_SKIP_DIR_SEGMENTS = {
    "__pycache__", ".git", "archive", "deprecated", "old",
    "tests", "test_archive",
}

# The campaign guard. A function (or module) that calls this is campaign-aware.
_CAMPAIGN_GUARD = "no_stack0_emit_enabled"

# The positional-invariant mechanism (``neural_vm/unified_compiler/
# positional_invariant.py``). A function that resolves its anchors through any
# of these helpers has DECLARED its frame assumption to the compiler — the
# STACK0 drop is a no-op for it BY CONSTRUCTION (``marker_bank_index`` for
# Class-1 marker-relative anchors, ``invariant_threshold`` /
# ``frame_byte_is_emitted`` for Class-2 absolute-slot anchors). Such refs are
# reclassified out of UNGUARDED into DECLARED_INVARIANT — a stronger guarantee
# than the per-op ``no_stack0_emit_enabled()`` branch (CAMPAIGN_AWARE).
_POSINV_HELPERS = frozenset(
    {"marker_bank_index", "invariant_threshold", "frame_byte_is_emitted"}
)

# ---------------------------------------------------------------------------
# Positional-frame dim set
# ---------------------------------------------------------------------------

# Hand-seeded CORE positional-frame dims (the known shift-risk set from the
# brief). Augmented at runtime from ``dim_registry.py`` descriptions so the
# list cannot silently drift from the registry.
_CORE_POSITIONAL_DIMS: Set[str] = {
    # byte-index-within-register flags
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    # STACK0 byte-position flags (the dropped block)
    "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
    "STACK0_BYTE1_PIN", "STACK0_BYTE2_PIN", "STACK0_BYTE3_PIN",
    # MEM value-byte slot predictors (d=N-from-MEM)
    "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
    # marker-distance threshold heads (L0/L1/L2)
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
    "L1H0", "L1H1", "L1H2", "L1H3", "L1H4",
    "L2H0",
}

# (category, role) -> canonical positional dim for dim_ref() resolution.
_DIM_REF_POSITIONAL: Dict[Tuple[str, str], str] = {
    ("byte_index", "0"): "BYTE_INDEX_0",
    ("byte_index", "1"): "BYTE_INDEX_1",
    ("byte_index", "2"): "BYTE_INDEX_2",
    ("byte_index", "3"): "BYTE_INDEX_3",
    ("memory_lo", "val_b0"): "MEM_VAL_B0",
    ("memory_lo", "val_b1"): "MEM_VAL_B1",
    ("memory_lo", "val_b2"): "MEM_VAL_B2",
    ("memory_lo", "val_b3"): "MEM_VAL_B3",
}

# A registry description matching ANY of these (case-insensitive) marks the
# pinned dim as positional-frame (distance / byte-index / slot-from-marker).
_POSITIONAL_DESC_PATTERNS = (
    re.compile(r"within dist", re.I),
    re.compile(r"byte index", re.I),
    re.compile(r"byte\s*\d\s*position", re.I),
    re.compile(r"d=\d+\s*from", re.I),
    re.compile(r"from nearest is_mark", re.I),
    re.compile(r"threshold\s*[\d.]+\s*from", re.I),
)


def derive_positional_dims_from_registry(repo_root: Path) -> Set[str]:
    """Augment the core set with every ``_pin(...)`` whose registry
    description is a positional / distance / byte-index phrase.

    Pure source scan of ``dim_registry.py`` (no import / no model build).
    Keeps the audit in lockstep with the registry: a newly added
    distance-bank dim is picked up automatically.
    """
    dims: Set[str] = set(_CORE_POSITIONAL_DIMS)
    reg = repo_root / "neural_vm" / "dim_registry.py"
    if not reg.exists():
        return dims
    try:
        src = reg.read_text(encoding="utf-8")
    except OSError:
        return dims
    # Match: _pin("NAME", <int>, <int>, "<description>" ...
    pin_re = re.compile(
        r'_pin\(\s*"([A-Z][A-Z0-9_]+)"\s*,\s*\d+\s*,\s*\d+\s*,\s*'
        r'"((?:[^"\\]|\\.)*)"'
    )
    for m in pin_re.finditer(src):
        name, desc = m.group(1), m.group(2)
        if any(p.search(desc) for p in _POSITIONAL_DESC_PATTERNS):
            dims.add(name)
    return dims


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _walk_python_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_SEGMENTS]
        if set(Path(dirpath).parts) & _SKIP_DIR_SEGMENTS:
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _string_value(node: ast.AST) -> Optional[str]:
    """Return the string of a plain str / f-string (formatted parts -> '*')."""
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


def _dim_base_and_offset(text: str) -> Tuple[Optional[str], bool]:
    """Split ``"L2H0+4"`` -> ("L2H0", True); ``"STACK0_BYTE1"`` -> (.., False).

    Returns (base_name_or_None, has_offset). ``base_name`` is None when the
    string is not an UPPER_SNAKE dim name skeleton.
    """
    base = text.split("+", 1)[0].strip()
    has_offset = "+" in text
    if not base or not re.fullmatch(r"[A-Z][A-Z0-9_]+", base):
        return None, has_offset
    return base, has_offset


def _resolve_dim_ref_call(call: ast.Call) -> Optional[str]:
    """Resolve a ``dim_ref("cat", "role")`` call to a positional dim, else None."""
    func = call.func
    if isinstance(func, ast.Name):
        if func.id != "dim_ref":
            return None
    elif isinstance(func, ast.Attribute):
        if func.attr != "dim_ref":
            return None
    else:
        return None
    if len(call.args) < 2:
        return None
    cat = _string_value(call.args[0])
    role = _string_value(call.args[1])
    if cat is None or role is None:
        return None
    return _DIM_REF_POSITIONAL.get((cat, role))


# Receivers we treat as the dim-namespace for ``<recv>.<DIM>`` attribute reads.
_DIM_RECEIVERS = {"BD", "bd", "proxy", "self", "_bd", "BD_", "BDc"}


# ---------------------------------------------------------------------------
# Reference record
# ---------------------------------------------------------------------------


class _Ref:
    __slots__ = ("lineno", "dim", "has_offset", "form", "func", "op",
                 "guarded", "declared", "snippet")

    def __init__(self, lineno: int, dim: str, has_offset: bool, form: str,
                 func: str, op: Optional[str], guarded: bool, declared: bool,
                 snippet: str) -> None:
        self.lineno = lineno
        self.dim = dim
        self.has_offset = has_offset
        self.form = form  # "string" | "attr" | "dim_ref"
        self.func = func
        self.op = op
        self.guarded = guarded
        self.declared = declared
        self.snippet = snippet

    @property
    def risk(self) -> str:
        if self.declared:
            return "DECLARED_INVARIANT"
        return "CAMPAIGN_AWARE" if self.guarded else "UNGUARDED"

    def as_dict(self, rel: str) -> Dict[str, object]:
        return {
            "file": rel,
            "line": self.lineno,
            "dim": self.dim,
            "form": self.form,
            "distance_offset": self.has_offset,
            "func": self.func,
            "op": self.op,
            "risk": self.risk,
            "campaign_guarded": self.guarded,
            "posinv_declared": self.declared,
            "snippet": self.snippet,
        }


# ---------------------------------------------------------------------------
# Function-scope index (line -> enclosing def / op + guard awareness)
# ---------------------------------------------------------------------------


class _FuncIndex:
    """Maps a source line to its enclosing ``def`` and the owning ``make_*``
    op factory, and records which functions consult the campaign guard."""

    def __init__(self, tree: ast.AST, module_guarded: bool) -> None:
        # (start, end, name, is_op_factory) sorted by widest-last so the
        # narrowest enclosing def wins.
        self._spans: List[Tuple[int, int, str, bool]] = []
        # function name -> calls no_stack0_emit_enabled() anywhere in its body
        self._func_guarded: Dict[str, bool] = {}
        # function name -> resolves anchors through a positional-invariant
        # helper (marker_bank_index / invariant_threshold / frame_byte_is_emitted)
        self._func_declared: Dict[str, bool] = {}
        self._module_guarded = module_guarded
        self._index(tree)

    @staticmethod
    def _is_op_factory(name: str) -> bool:
        return name.startswith("make_") or name.startswith("_layer")

    def _index(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                self._spans.append(
                    (node.lineno, end, node.name, self._is_op_factory(node.name))
                )
                # Does this function (directly in its own body, incl nested
                # bake closures) call the campaign guard, or resolve its anchors
                # through a positional-invariant helper?
                guarded = False
                declared = False
                for sub in ast.walk(node):
                    if not isinstance(sub, ast.Call):
                        continue
                    cn = self._call_name(sub)
                    if cn == _CAMPAIGN_GUARD:
                        guarded = True
                    elif cn in _POSINV_HELPERS:
                        declared = True
                self._func_guarded[node.name] = guarded
                self._func_declared[node.name] = declared
        # Narrowest span first when we search.
        self._spans.sort(key=lambda s: (s[1] - s[0]))

    @staticmethod
    def _call_name(call: ast.Call) -> Optional[str]:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    def enclosing(self, lineno: int) -> Tuple[str, Optional[str]]:
        """Return (innermost_func_name, owning_op_factory_or_None)."""
        inner = "<module>"
        op: Optional[str] = None
        # Spans sorted narrowest-first: first containing span is innermost.
        best_inner: Optional[Tuple[int, int, str, bool]] = None
        for s, e, name, is_op in self._spans:
            if s <= lineno <= e:
                if best_inner is None or (e - s) < (best_inner[1] - best_inner[0]):
                    best_inner = (s, e, name, is_op)
        if best_inner is not None:
            inner = best_inner[2]
        # Owning op factory = widest enclosing make_*/_layer* span.
        widest_op: Optional[Tuple[int, int, str, bool]] = None
        for s, e, name, is_op in self._spans:
            if is_op and s <= lineno <= e:
                if widest_op is None or (e - s) > (widest_op[1] - widest_op[0]):
                    widest_op = (s, e, name, is_op)
        if widest_op is not None:
            op = widest_op[2]
        return inner, op

    def guarded(self, lineno: int) -> bool:
        """Campaign-aware iff the innermost func, owning op, or module
        consults ``no_stack0_emit_enabled()``."""
        if self._module_guarded:
            return True
        inner, op = self.enclosing(lineno)
        if self._func_guarded.get(inner):
            return True
        if op is not None and self._func_guarded.get(op):
            return True
        return False

    def declared(self, lineno: int) -> bool:
        """Declared frame-invariant iff the innermost func or owning op
        resolves its anchors through a positional-invariant helper
        (``marker_bank_index`` / ``invariant_threshold`` /
        ``frame_byte_is_emitted``). The STACK0 drop is a no-op for such a ref
        by construction — a stronger guarantee than the campaign guard."""
        inner, op = self.enclosing(lineno)
        if self._func_declared.get(inner):
            return True
        if op is not None and self._func_declared.get(op):
            return True
        return False


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _source_line(lines: List[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def scan_file(path: Path, positional_dims: Set[str]) -> List[_Ref]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()

    # Module-level guard consultation: a module that imports + calls the guard
    # at module scope (rare) — be conservative, only flag the function-level.
    module_guarded = False

    findex = _FuncIndex(tree, module_guarded)
    refs: List[_Ref] = []
    seen: Set[Tuple[int, str, str]] = set()  # (line, dim, form) de-dup

    for node in ast.walk(tree):
        dim: Optional[str] = None
        has_offset = False
        form = ""
        lineno = getattr(node, "lineno", 0)

        if isinstance(node, ast.Call):
            ref = _resolve_dim_ref_call(node)
            if ref is not None and ref in positional_dims:
                dim, form = ref, "dim_ref"
        elif isinstance(node, ast.Attribute):
            # <recv>.<DIM>
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in _DIM_RECEIVERS
                and isinstance(node.attr, str)
            ):
                base, _ = _dim_base_and_offset(node.attr)
                if base is not None and base in positional_dims:
                    dim, form = base, "attr"
                    # Offset detection for attr form (BD.L2H0 + MEM_I) is done
                    # via the source line below.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            base, off = _dim_base_and_offset(node.value)
            if base is not None and base in positional_dims:
                dim, has_offset, form = base, off, "string"
        elif isinstance(node, ast.JoinedStr):
            s = _string_value(node)
            if s is not None:
                base, off = _dim_base_and_offset(s)
                if base is not None and base in positional_dims:
                    dim, has_offset, form = base, off, "string"

        if dim is None:
            continue

        snippet = _source_line(lines, lineno)
        # Attr form: detect a trailing ``+ <something>`` distance offset on the
        # source line (BD.L2H0 + MEM_I).
        if form == "attr":
            m = re.search(re.escape(f".{dim}") + r"\s*\+", snippet)
            if m:
                has_offset = True

        key = (lineno, dim, form)
        if key in seen:
            continue
        seen.add(key)

        inner, op = findex.enclosing(lineno)
        guarded = findex.guarded(lineno)
        declared = findex.declared(lineno)
        refs.append(_Ref(lineno, dim, has_offset, form, inner, op, guarded,
                         declared, snippet))
    refs.sort(key=lambda r: r.lineno)
    return refs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _find_repo_root() -> Optional[Path]:
    root = Path(__file__).resolve().parent
    for _ in range(6):
        if (root / "neural_vm").is_dir() and (root / "tools").is_dir():
            return root
        if (root / "c4_release" / "neural_vm").is_dir():
            return root / "c4_release"
        parent = root.parent
        if parent == root:
            break
        root = parent
    return None


def collect(
    repo_root: Path, positional_dims: Set[str]
) -> Dict[str, List[_Ref]]:
    by_file: Dict[str, List[_Ref]] = {}
    for rel in _SCAN_RELS:
        target = repo_root / rel
        if not target.exists():
            continue
        for path in _walk_python_files(target):
            try:
                relpath = str(path.relative_to(repo_root))
            except ValueError:
                relpath = str(path)
            file_refs = scan_file(path, positional_dims)
            if file_refs:
                by_file[relpath] = file_refs
    return by_file


def _rank_key(ref: _Ref) -> Tuple[int, int]:
    # Lower sorts first. UNGUARDED (the shift-risk surface) before
    # CAMPAIGN_AWARE before DECLARED_INVARIANT (resolved through the
    # positional-invariant mechanism); within each, distance-offset (doubly
    # frame-dependent) before bare flag.
    if ref.declared:
        tier = 2
    elif ref.guarded:
        tier = 1
    else:
        tier = 0
    return (tier, 0 if ref.has_offset else 1)


def _flatten(by_file: Dict[str, List[_Ref]]) -> List[Tuple[str, _Ref]]:
    flat: List[Tuple[str, _Ref]] = []
    for rel, refs in by_file.items():
        for r in refs:
            flat.append((rel, r))
    flat.sort(key=lambda t: (_rank_key(t[1]), t[0], t[1].lineno))
    return flat


def _print_catalog(
    by_file: Dict[str, List[_Ref]],
    positional_dims: Set[str],
    *,
    unguarded_only: bool,
    dim_filter: Optional[str],
) -> None:
    flat = _flatten(by_file)
    if dim_filter:
        flat = [(rel, r) for rel, r in flat if r.dim == dim_filter]
    if unguarded_only:
        flat = [(rel, r) for rel, r in flat
                if not r.guarded and not r.declared]

    n_unguarded = sum(1 for _, r in flat if not r.guarded and not r.declared)
    n_aware = sum(1 for _, r in flat if r.guarded and not r.declared)
    n_declared = sum(1 for _, r in flat if r.declared)
    n_unguarded_offset = sum(
        1 for _, r in flat if not r.guarded and not r.declared and r.has_offset
    )

    print("=" * 78)
    print("POSITIONAL-INVARIANT AUDIT — STEP_TOKENS shift-risk catalog")
    print("=" * 78)
    print(
        f"positional-frame dims tracked: {len(positional_dims)} "
        f"(seeded {len(_CORE_POSITIONAL_DIMS)} + registry-derived "
        f"{len(positional_dims) - len(_CORE_POSITIONAL_DIMS)})"
    )
    print(
        f"references: {len(flat)} total | "
        f"{n_unguarded} UNGUARDED ({n_unguarded_offset} with a distance "
        f"offset) | {n_aware} CAMPAIGN_AWARE | "
        f"{n_declared} DECLARED_INVARIANT"
    )
    print(
        "\nUNGUARDED = references a positional-frame dim AND the enclosing op "
        "does NOT\nconsult no_stack0_emit_enabled() -> WILL silently shift "
        "when STEP_TOKENS 35->30.\nThis is the next fix wave: re-anchor each "
        "one (record positional_invariant=\nSTEP_TOKENS so the compiler "
        "auto-shifts).\n"
    )

    # Group by op for the ranked catalog.
    cur_risk: Optional[str] = None
    for rel, r in flat:
        risk = r.risk
        if risk != cur_risk:
            print("\n" + "-" * 78)
            print(f"### {risk}")
            print("-" * 78)
            cur_risk = risk
        tags = [r.dim]
        if r.has_offset:
            tags.append("DISTANCE_OFFSET")
        tag_s = " ".join(tags)
        op_s = f" [{r.op}]" if r.op else ""
        print(f"  {rel}:{r.lineno}: {tag_s}  ({r.form}){op_s}")
        print(f"      in {r.func}()  |  {r.snippet}")


def _prove(by_file: Dict[str, List[_Ref]]) -> int:
    """Self-check: the catalog MUST include the two confirmed roots and flag
    each as shift-risk. Returns process exit code (0 ok, 1 fail)."""
    flat = _flatten(by_file)

    def _hits(pred) -> List[Tuple[str, _Ref]]:
        return [(rel, r) for rel, r in flat if pred(rel, r)]

    print("=" * 78)
    print("PROOF — catalog catches the two confirmed campaign roots")
    print("=" * 78)

    ok = True

    # Root 1: div/mod STACK0_BYTE1 cummax anchor in efficient_alu_neural.py.
    divmod_hits = _hits(
        lambda rel, r: rel.endswith("efficient_alu_neural.py")
        and r.dim == "STACK0_BYTE1"
    )
    print("\n[1] div/mod STACK0_BYTE1 dividend-anchor (efficient_alu_neural.py)")
    if divmod_hits:
        for rel, r in divmod_hits:
            print(
                f"    CAUGHT  {rel}:{r.lineno}  dim={r.dim}  risk={r.risk}  "
                f"in {r.func}()"
            )
        if not any(not r.guarded for _, r in divmod_hits):
            print("    FAIL: caught but NOT flagged UNGUARDED")
            ok = False
        else:
            print("    -> flagged UNGUARDED (would have been audit-time "
                  "shift-risk)")
    else:
        print("    FAIL: NOT in catalog")
        ok = False

    # Root 2: operand-CAM MEM_VAL_B* / L2H0 / H1 distance markers in the L8
    # mem-to-ALU op (make_layer8_mem_to_alu_op).
    cam_hits = _hits(
        lambda rel, r: rel.endswith("l8_ops.py")
        and (
            (r.op or "").startswith("make_layer8_mem_to_alu")
        )
        and r.dim in {"L2H0", "H1", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2"}
    )
    print("\n[2] operand-CAM L2H0/H1/MEM_VAL_B* distance markers "
          "(make_layer8_mem_to_alu_op, l8_ops.py)")
    if cam_hits:
        # Show a representative few.
        shown = 0
        for rel, r in cam_hits:
            if shown < 8:
                print(
                    f"    CAUGHT  {rel}:{r.lineno}  dim={r.dim}"
                    f"{'  +OFFSET' if r.has_offset else ''}  risk={r.risk}"
                )
                shown += 1
        n_off = sum(1 for _, r in cam_hits if r.has_offset)
        print(
            f"    ({len(cam_hits)} refs total, {n_off} with a DISTANCE_OFFSET "
            f"into the H/L2H0 distance bank)"
        )
        if not any(r.has_offset for _, r in cam_hits):
            print("    FAIL: caught but NO distance-offset flagged "
                  "(the +MEM_I shift surface)")
            ok = False
        else:
            print("    -> distance-offset refs flagged (the d=N-from-MEM "
                  "rows that shift)")
    else:
        print("    FAIL: NOT in catalog")
        ok = False

    print("\n" + "=" * 78)
    if ok:
        print("PROOF PASSED — both confirmed roots are in the catalog and "
              "flagged as shift-risk.")
        print("The audit would have flagged div/mod + operand-CAM at "
              "authoring time.")
    else:
        print("PROOF FAILED — a confirmed root was missed.")
    print("=" * 78)
    return 0 if ok else 1


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON catalog")
    ap.add_argument(
        "--unguarded-only", action="store_true",
        help="only show UNGUARDED references (the shift-risk surface)",
    )
    ap.add_argument(
        "--dim", default=None,
        help="filter to a single positional dim (e.g. STACK0_BYTE1)",
    )
    ap.add_argument(
        "--prove", action="store_true",
        help="self-check that the catalog catches div/mod + operand-CAM",
    )
    ap.add_argument("--root", default=None, help="repo root (auto-detected)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else _find_repo_root()
    if root is None or not (root / "neural_vm").is_dir():
        print("error: could not locate c4_release repo root", file=sys.stderr)
        return 2

    positional_dims = derive_positional_dims_from_registry(root)
    by_file = collect(root, positional_dims)

    if args.prove:
        return _prove(by_file)

    if args.json:
        flat = _flatten(by_file)
        if args.dim:
            flat = [(rel, r) for rel, r in flat if r.dim == args.dim]
        if args.unguarded_only:
            flat = [(rel, r) for rel, r in flat
                    if not r.guarded and not r.declared]
        payload = {
            "positional_dims": sorted(positional_dims),
            "n_refs": len(flat),
            "n_unguarded": sum(
                1 for _, r in flat if not r.guarded and not r.declared
            ),
            "n_campaign_aware": sum(
                1 for _, r in flat if r.guarded and not r.declared
            ),
            "n_declared_invariant": sum(1 for _, r in flat if r.declared),
            "references": [r.as_dict(rel) for rel, r in flat],
        }
        print(json.dumps(payload, indent=2))
        return 0

    _print_catalog(
        by_file, positional_dims,
        unguarded_only=args.unguarded_only,
        dim_filter=args.dim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
