#!/usr/bin/env python3
"""Map blk34 hidden unit indices to their owning FFNRule names (campaign cfg).

The L20/blk34 ffn is the lowered ``layer16_lev_routing`` op. Walk its IR rules
in order; the unit index == rule index (start_unit=0). Prints the rule name for
the units that explode OUTPUT at the BP[0] row (236) + neighbors.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_unit_name.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from neural_vm.unified_compiler.ops.l16_ops import _layer16_lev_routing_rules  # noqa: E402

TARGETS = [int(x) for x in os.environ.get("PROBE_UNITS", "234,235,236,237,238").split(",")]


def main():
    rules = _layer16_lev_routing_rules(100.0)
    print(f"total rules = {len(rules)}")
    for u in TARGETS:
        if u < len(rules):
            r = rules[u]
            print(f"  unit {u:5d} -> {getattr(r, 'name', '?')}")
    # also print all rules whose name contains 'e8_marker_from_alu' and index
    print("  -- e8/e0/f8 marker_from_alu rules --")
    for i, r in enumerate(rules):
        nm = getattr(r, "name", "")
        if "marker_from_alu" in nm:
            print(f"   idx {i:5d}  {nm}")


if __name__ == "__main__":
    main()
