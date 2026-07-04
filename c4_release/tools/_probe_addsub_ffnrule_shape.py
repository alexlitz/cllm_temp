"""Characterize the hand-authored L8/L9 add/sub FFNRule builders precisely.

Dumps the (conditions, threshold, gate, writes, scope, dominates_at) of the
FIRST FEW and a couple of representative rules from each add/sub builder so a
byte-identical generator spec can be authored. READ-only (no bake).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from c4_release.neural_vm.unified_compiler.ops import l8_ops, l9_ops

S = 100.0


def _fmt_rule(r):
    conds = tuple((c.dim if hasattr(c, "dim") else c[0],
                   c.weight if hasattr(c, "weight") else c[1])
                  for c in r.conditions)
    writes = tuple((w.dim if hasattr(w, "dim") else w[0],
                    w.weight if hasattr(w, "weight") else w[1])
                   for w in r.writes)
    return {
        "name": r.name,
        "conditions": conds,
        "threshold": r.threshold,
        "gate": r.gate,
        "gate_bias": getattr(r, "gate_bias", None),
        "gate_weight": getattr(r, "gate_weight", None),
        "writes": writes,
        "scope": r.scope,
        "dominates_at": r.dominates_at,
    }


def dump(label, rules, n=3):
    print(f"\n===== {label}  (n={len(rules)}) =====")
    for r in rules[:n]:
        d = _fmt_rule(r)
        for k, v in d.items():
            print(f"  {k}: {v}")
        print("  ---")


dump("L8 add_lo", l8_ops._layer8_alu_add_lo_rules(S))
dump("L8 sub_lo", l8_ops._layer8_alu_sub_lo_rules(S))
dump("L8 add_carry", l8_ops._layer8_alu_add_carry_rules(S))
dump("L8 sub_borrow", l8_ops._layer8_alu_sub_borrow_rules(S))
dump("L9 add_hi", l9_ops._add_hi_nibble_rules(S))
dump("L9 sub_hi", l9_ops._sub_hi_nibble_rules(S))
dump("L9 add_carry_out", l9_ops._add_carry_out_rules(S))
dump("L9 sub_borrow_out", l9_ops._sub_borrow_out_rules(S))

# Print the last carry_in==1 rule of add_hi to see carry-in branch
addhi = l9_ops._add_hi_nibble_rules(S)
print("\n===== L9 add_hi carry_in=1 sample (index 256) =====")
d = _fmt_rule(addhi[256])
for k, v in d.items():
    print(f"  {k}: {v}")
