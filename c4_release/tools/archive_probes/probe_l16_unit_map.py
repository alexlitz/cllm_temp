#!/usr/bin/env python3
"""Map L20 (block-30) FFN hidden-unit indices to l16 rule names, and print the
rule whose unit writes OUTPUT_LO+8 + OUTPUT_HI_THIS_STEP+13 (the d8 misfire).

Builds the lowered IR for make_layer16_lev_routing and inspects unit ownership.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
from neural_vm.unified_compiler.ops.l16_ops import _layer16_lev_routing_rules

rules = _layer16_lev_routing_rules(100.0)
print(f"total rules: {len(rules)}")
# Each FFNRule -> one hidden unit, in order. Find rules writing LO+8 & HI+13.
target_units = {749}
unit = 0
for r in rules:
    name = getattr(r, "name", "?")
    writes = getattr(r, "writes", ()) or ()
    wnames = {w[0] if isinstance(w, (tuple, list)) else w for w in writes}
    has_lo8 = any("OUTPUT_LO+8" == (w[0] if isinstance(w, (tuple, list)) else w)
                  for w in writes)
    has_hi13 = any("OUTPUT_HI_THIS_STEP+13" == (w[0] if isinstance(w, (tuple, list)) else w)
                   for w in writes)
    if unit in target_units or (has_lo8 and has_hi13):
        print(f"  unit {unit:5d}: {name}")
        print(f"      writes: {[w for w in writes][:6]}")
    unit += 1
print(f"\n(units counted = {unit})")
