#!/usr/bin/env python3
"""Phase 11.B audit: classify every FFNRule by condition-gate kind.

Walks every ``make_*`` factory in ``c4_release/neural_vm/unified_compiler/ops/``,
instantiates the op (tolerating failures), pulls FFNRules via the existing
``_collect_ffn_rules_from_op`` helper, and classifies each rule's
``conditions=`` set:

  opcode-gated  -- contains at least one ``OP_*`` dim
  token-gated   -- contains at least one ``MARK_*`` dim, no ``OP_*``
  mixed         -- both ``OP_*`` and ``MARK_*``
  dim-free      -- neither (rare)

Prints per-bucket counts plus a per-op breakdown for the token-gated bucket
to help pick easy migration candidates.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import pkgutil
from collections import defaultdict
from typing import Dict, List

# Make `neural_vm.*` importable.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c4_release"))
sys.path.insert(0, REPO)

from neural_vm.unified_compiler.decl_verifier import _collect_ffn_rules_from_op  # noqa: E402
from neural_vm.unified_compiler.ir import FFNRule  # noqa: E402


def _is_op_dim(name: str) -> bool:
    return (name.startswith("OP_")
            or name.startswith("ACTIVE_OPCODE_")
            or name.startswith("IO_IS_"))


def _is_mark_dim(name: str) -> bool:
    return name.startswith("MARK_")


def classify_rule(rule: FFNRule) -> str:
    has_op = False
    has_mark = False
    # conditions
    for term in rule.conditions:
        if _is_op_dim(term.dim.name):
            has_op = True
        elif _is_mark_dim(term.dim.name):
            has_mark = True
    # gate dim
    if rule.gate is not None:
        if _is_op_dim(rule.gate.name):
            has_op = True
        elif _is_mark_dim(rule.gate.name):
            has_mark = True
    # gate_terms
    for term in rule.gate_terms:
        if _is_op_dim(term.dim.name):
            has_op = True
        elif _is_mark_dim(term.dim.name):
            has_mark = True
    if has_op and has_mark:
        return "mixed"
    if has_op:
        return "opcode-gated"
    if has_mark:
        return "token-gated"
    return "dim-free"


def iter_op_modules():
    pkg_name = "neural_vm.unified_compiler.ops"
    pkg = importlib.import_module(pkg_name)
    for info in pkgutil.iter_modules(pkg.__path__, prefix=pkg_name + "."):
        # Skip dunder/private modules.
        if info.name.endswith(".shared"):
            yield importlib.import_module(info.name)
            continue
        try:
            yield importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001
            print(f"[skip-import] {info.name}: {e}", file=sys.stderr)


def main() -> int:
    counts: Dict[str, int] = defaultdict(int)
    per_op_token: Dict[str, int] = defaultdict(int)
    per_op_all: Dict[str, int] = defaultdict(int)
    sample_token_rules: Dict[str, list] = defaultdict(list)
    factories_seen = 0
    factories_ok = 0
    factories_failed = 0

    for mod in iter_op_modules():
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("make_"):
                continue
            if fn.__module__ != mod.__name__:
                continue
            sig = inspect.signature(fn)
            # Only call factories where every parameter has a default
            # (most authored ops are no-arg).
            required = [p for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                           inspect.Parameter.VAR_KEYWORD)]
            factories_seen += 1
            if required:
                # Try with dummy values for known patterns.
                continue
            try:
                op = fn()
            except Exception as e:  # noqa: BLE001
                factories_failed += 1
                continue
            factories_ok += 1
            try:
                rules = _collect_ffn_rules_from_op(op)
            except Exception as e:  # noqa: BLE001
                continue
            op_name = getattr(op, "name", name)
            for r in rules:
                cat = classify_rule(r)
                counts[cat] += 1
                per_op_all[op_name] += 1
                if cat == "token-gated":
                    per_op_token[op_name] += 1
                    if len(sample_token_rules[op_name]) < 3:
                        sample_token_rules[op_name].append(r)

    print(f"factories_seen={factories_seen} ok={factories_ok} "
          f"failed={factories_failed}", file=sys.stderr)
    total = sum(counts.values())
    print(f"\n=== Phase 11.B Audit ===")
    print(f"Total IR rules walked: {total}")
    for cat in ("opcode-gated", "token-gated", "mixed", "dim-free"):
        n = counts[cat]
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {cat:14s}: {n:6d}  ({pct:5.1f}%)")

    print("\n=== token-gated rules per op (top 30) ===")
    for op_name, n in sorted(per_op_token.items(), key=lambda x: -x[1])[:30]:
        print(f"  {n:5d}  {op_name}  (of {per_op_all[op_name]} total)")

    print("\n=== ops with FEW token-gated rules (1-5) — easy migration candidates ===")
    easy = sorted([(n, op) for op, n in per_op_token.items() if 1 <= n <= 5],
                  key=lambda x: x[0])
    for n, op_name in easy[:30]:
        print(f"  {n:3d}  {op_name}")
        for r in sample_token_rules[op_name]:
            cond_str = ", ".join(f"{t.dim.name}+{t.dim.offset}" for t in r.conditions)
            print(f"        rule={r.name!r} conds=({cond_str})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
