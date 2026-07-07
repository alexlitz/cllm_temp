#!/usr/bin/env python3
"""writer_index for OUTPUT_LO+0 (the dim the wrong var_three SI-store byte-0
comes from). Lists every FFN rule that writes OUTPUT_LO offset 0, with its op,
threshold, gate, and write weight — to show the competing writers and ground the
correct-by-construction store-only discriminator fix. Reuses the built gate
context (flat_ffn_ops) so the layout is the production one.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE); _ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import warnings; warnings.filterwarnings("ignore")
from tools.interp_oracle_gate import build_gate_context
from neural_vm.verification.writer_index import _collect_ffn_rules_from_op

TARGET = ("OUTPUT_LO", 0)
ctx = build_gate_context(verbose=True)
hits = []
for op in ctx.flat_ffn_ops:
    op_name = getattr(op, "name", "<anon>")
    for r in _collect_ffn_rules_from_op(op):
        for wt in getattr(r, "writes", ()):
            if wt.weight == 0.0:
                continue
            if (wt.dim.name, wt.dim.offset) == TARGET:
                gate = getattr(r, "gate", None)
                gate_name = getattr(gate, "name", gate)
                conds = ", ".join(f"{c.dim.name}[{c.dim.offset}]={c.weight}" for c in getattr(r, "conditions", ()))
                hits.append((op_name, getattr(r,'name','?'), wt.weight, float(getattr(r,'threshold',0)), gate_name, conds))
print(f"=== writers of {TARGET[0]}+{TARGET[1]} (override={os.environ.get('C4_STORE_AX_B0_OVERRIDE')}): {len(hits)} rules ===")
for op_name, rname, w, thr, gate_name, conds in sorted(hits, key=lambda h: -abs(h[2])):
    print(f"  w={w:+.4f} thr={thr:.1f} gate={gate_name}  {op_name}::{rname}")
    if abs(w) >= 0.2:
        print(f"       conds: {conds}")
