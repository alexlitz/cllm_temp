#!/usr/bin/env python3
"""Map block-41 unit 1790 (the OUTPUT-HI-flood overflow guard) back to its
OWNING declarative rule in _tail_bit32_result_correction_rules. GPU-free.

Signature (from probe_expr_tail_overflow_unit): b_up ~= -500500, gate dim ==
MARK_AX, W_down[OUTPUT_LO+0]=-100, W_up reads the OUTPUT_HI band at +100.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_vm.unified_compiler.ops.l10_ops import _tail_bit32_result_correction_rules


def kof(term):
    return term.dim.key(), term.weight


def main():
    rules = _tail_bit32_result_correction_rules()
    print(f"tail rules: {len(rules)}")
    hits = 0
    for i, r in enumerate(rules):
        wmap = {term.dim.key(): term.weight for term in r.writes}
        if wmap.get("OUTPUT_LO+0", 0.0) >= 0:
            continue
        cmap = {term.dim.key(): term.weight for term in r.conditions}
        reads_hi = any(k.startswith("OUTPUT_HI") for k in cmap)
        if not reads_hi:
            continue
        # require a large-magnitude threshold (the -500500 overflow bias)
        if abs(r.threshold) < 1000:
            continue
        hits += 1
        gate_name = r.gate.key() if r.gate is not None else None
        print(f"\n[{i}] {r.name}")
        print(f"    threshold={r.threshold}  gate={gate_name}  gate_bias={r.gate_bias}")
        print(f"    gate_terms={[kof(t) for t in r.gate_terms]}")
        print(f"    writes OUTPUT_LO+0={wmap.get('OUTPUT_LO+0')}")
        print(f"    conditions:")
        for k, w in cmap.items():
            print(f"      {k:30} {w:+.2f}")
    print(f"\nMATCHES: {hits}")


if __name__ == "__main__":
    main()
