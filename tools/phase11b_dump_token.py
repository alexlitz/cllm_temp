#!/usr/bin/env python3
"""Dump all token-gated rules for inspection, grouped by op."""
from __future__ import annotations
import importlib, inspect, os, sys, pkgutil
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


def rule_repr(r):
    conds = ", ".join(f"{t.dim.name}+{t.dim.offset}={t.weight:g}" for t in r.conditions)
    gate = f"gate={r.gate.name}" if r.gate else "gate=None"
    gt = ", ".join(f"{t.dim.name}+{t.dim.offset}={t.weight:g}" for t in r.gate_terms)
    writes = ", ".join(f"{w.dim.name}+{w.dim.offset}={w.weight:g}" for w in r.writes)
    return f"  rule={r.name!r}\n    conds=({conds})\n    {gate} gt=({gt}) thr={r.threshold}\n    writes=({writes})\n    scope={r.scope!r}"


def main():
    pkg_name = "neural_vm.unified_compiler.ops"
    pkg = importlib.import_module(pkg_name)
    target_ops = sys.argv[1:] if len(sys.argv) > 1 else None
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
            if target_ops and op_name not in target_ops:
                continue
            try:
                rules = _collect_ffn_rules_from_op(op)
            except Exception:
                continue
            tg = [r for r in rules if classify(r) == "token-gated"]
            if not tg:
                continue
            print(f"\n=== op: {op_name}  ({len(tg)} token-gated of {len(rules)}) ===")
            for r in tg:
                print(rule_repr(r))


if __name__ == "__main__":
    main()
