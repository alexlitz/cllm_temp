#!/usr/bin/env python3
"""Audit ``produces`` / ``consumes_fresh`` coverage across the op corpus.

Walks every ``make_*_op`` factory in
``c4_release.neural_vm.unified_compiler.ops.*`` and classifies the op as:

* ``both`` — declares non-empty ``produces`` AND non-empty ``consumes_fresh``.
* ``produces_only`` — declares ``produces`` but not ``consumes_fresh``.
* ``consumes_only`` — declares ``consumes_fresh`` but not ``produces``.
* ``none`` — neither.

Also reports how many of the ``none`` ops have a non-empty ``compiler_ir``
(i.e. would yield a non-empty derived set via
:mod:`tools.derive_produces_consumes`) vs. how many are imperative-only
(IR-less, needing manual migration).
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import Dict, List

REPO_ROOT = __file__.rsplit("/tools/", 1)[0]
sys.path.insert(0, os.path.join(REPO_ROOT, "c4_release"))
sys.path.insert(0, REPO_ROOT)

OPS_PACKAGE = "c4_release.neural_vm.unified_compiler.ops"
OPS_DIR = os.path.join(
    REPO_ROOT, "c4_release", "neural_vm", "unified_compiler", "ops"
)


def _iter_op_factories():
    for fname in sorted(os.listdir(OPS_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        mod_name = f"{OPS_PACKAGE}.{fname[:-3]}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            print(f"  (skip {mod_name}: {exc!r})", file=sys.stderr)
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("make_"):
                continue
            # Only top-level factories defined in this module
            if getattr(obj, "__module__", None) != mod_name:
                continue
            sig = inspect.signature(obj)
            # Skip factories with required positional params we can't fill;
            # accept factories whose params all have defaults.
            ok = True
            for p in sig.parameters.values():
                if p.default is inspect.Parameter.empty and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    ok = False
                    break
            if not ok:
                continue
            yield mod_name, name, obj


def main() -> int:
    buckets = {
        "both": 0,
        "produces_only": 0,
        "consumes_only": 0,
        "none": 0,
    }
    none_with_ir: List[str] = []
    none_without_ir: List[str] = []
    total = 0
    skipped = 0
    by_module: Dict[str, Dict[str, int]] = {}

    for mod_name, fname, factory in _iter_op_factories():
        total += 1
        try:
            op = factory()
        except Exception as exc:
            skipped += 1
            print(f"  (skip {mod_name}:{fname}: {exc!r})", file=sys.stderr)
            continue
        # Some factories return Operation, others return a tuple/list.
        ops = op if isinstance(op, (list, tuple)) else (op,)
        for o in ops:
            if not hasattr(o, "produces"):
                continue
            has_p = bool(o.produces)
            has_c = bool(o.consumes_fresh)
            ir = getattr(o, "compiler_ir", None)
            ir_has_rules = bool(
                ir and any(layer.ffn.rules for layer in ir.layers)
            )
            if has_p and has_c:
                key = "both"
            elif has_p:
                key = "produces_only"
            elif has_c:
                key = "consumes_only"
            else:
                key = "none"
                if ir_has_rules:
                    none_with_ir.append(o.name)
                else:
                    none_without_ir.append(o.name)
            buckets[key] += 1
            by_module.setdefault(mod_name, {
                "both": 0, "produces_only": 0,
                "consumes_only": 0, "none": 0,
            })[key] += 1

    print(f"total_factories_examined: {total}")
    print(f"skipped_factories: {skipped}")
    print(f"ops_classified: {sum(buckets.values())}")
    for k, v in buckets.items():
        print(f"  {k}: {v}")
    print(f"none_with_ir (auto-derivable): {len(none_with_ir)}")
    print(f"none_without_ir (manual / imperative-only): {len(none_without_ir)}")
    print("\nper-module breakdown:")
    for mod in sorted(by_module):
        row = by_module[mod]
        print(f"  {mod.rsplit('.', 1)[1]}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
