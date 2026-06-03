#!/usr/bin/env python3
"""Find easiest token→opcode migration candidates.

A candidate is "easy" if:
  - It's token-gated (per the audit classifier).
  - Its scope= or dominates_at= already names an OP_* (so the rule's
    intended discriminator IS opcode — just not in conditions).
  - The OP_* mentioned isn't already in conditions/gate (would be a no-op).
"""
from __future__ import annotations
import importlib, inspect, os, sys, pkgutil, re
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c4_release"))
sys.path.insert(0, REPO)

from neural_vm.unified_compiler.decl_verifier import _collect_ffn_rules_from_op  # noqa
from neural_vm.unified_compiler.ir import FFNRule  # noqa


def _is_op_dim(name): return name.startswith("OP_") or name.startswith("ACTIVE_OPCODE_") or name.startswith("IO_IS_")
def _is_mark_dim(name): return name.startswith("MARK_")


def classify(rule):
    has_op = has_mark = False
    for t in rule.conditions:
        if _is_op_dim(t.dim.name): has_op = True
        elif _is_mark_dim(t.dim.name): has_mark = True
    if rule.gate is not None:
        if _is_op_dim(rule.gate.name): has_op = True
        elif _is_mark_dim(rule.gate.name): has_mark = True
    for t in rule.gate_terms:
        if _is_op_dim(t.dim.name): has_op = True
        elif _is_mark_dim(t.dim.name): has_mark = True
    if has_op and has_mark: return "mixed"
    if has_op: return "opcode-gated"
    if has_mark: return "token-gated"
    return "dim-free"


OP_RE = re.compile(r"\bOP_[A-Z]+\b")


def main():
    pkg_name = "neural_vm.unified_compiler.ops"
    pkg = importlib.import_module(pkg_name)
    cand_by_op = defaultdict(list)
    for info in pkgutil.iter_modules(pkg.__path__, prefix=pkg_name + "."):
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("make_") or fn.__module__ != mod.__name__:
                continue
            sig = inspect.signature(fn)
            req = [p for p in sig.parameters.values()
                   if p.default is inspect.Parameter.empty
                   and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            if req:
                continue
            try:
                op = fn()
            except Exception:
                continue
            op_name = getattr(op, "name", name)
            try:
                rules = _collect_ffn_rules_from_op(op)
            except Exception:
                continue
            for r in rules:
                if classify(r) != "token-gated":
                    continue
                # Look for OP_* mentioned in scope or dominates_at.
                op_in_scope = set()
                if r.scope:
                    op_in_scope.update(OP_RE.findall(r.scope))
                if r.dominates_at:
                    for v in r.dominates_at.values():
                        if v:
                            op_in_scope.update(OP_RE.findall(v))
                if not op_in_scope:
                    continue
                # Verify the OP_* isn't already in conditions/gate.
                existing = set()
                for t in r.conditions:
                    if _is_op_dim(t.dim.name):
                        existing.add(t.dim.name)
                if r.gate is not None and _is_op_dim(r.gate.name):
                    existing.add(r.gate.name)
                for t in r.gate_terms:
                    if _is_op_dim(t.dim.name):
                        existing.add(t.dim.name)
                missing = op_in_scope - existing
                if missing:
                    cand_by_op[op_name].append((r.name, list(missing), r.scope))

    print(f"\n=== Easy migration candidates: token-gated rules whose scope= names OP_* ===")
    total = 0
    for op_name, lst in sorted(cand_by_op.items(), key=lambda x: -len(x[1])):
        print(f"\n  {op_name}  ({len(lst)} candidates)")
        for rname, missing, scope in lst[:10]:
            print(f"    {rname}  -- add: {missing}  scope={scope!r}")
        total += len(lst)
    print(f"\nTotal: {total}")


if __name__ == "__main__":
    main()
