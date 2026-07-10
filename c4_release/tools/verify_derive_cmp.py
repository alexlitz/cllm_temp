"""Byte-identity proof for the DERIVED comparison family (task #446).

Proves that ``building_blocks_dsl.derived_comparison_rules`` — the single
generator that realizes BLOG_SPEC §576-590 (all of EQ/NE/LT/GT/LE/GE reduced
to ONE zero-detector + sign) — reproduces the two hand-authored L10
cmp-combine banks rule-for-rule:

  * ``_l10_comparison_combine_rules``   (ComparisonCombine decode path)
  * ``_layer10_alu_cmp_combine_rules``  (L10-main ALU cmp lane)

Structural byte-identity is checked at the ``FFNRule`` level (conditions /
threshold / gate / writes, ignoring the diagnostic name/scope). The
whole-model ``tools/_isa_golden_hash.py`` gives the definitive weight-level
proof (unchanged under ``C4_DERIVE_CMP=1``); this tool is the fast,
model-free unit check.

Usage::

    CUDA_VISIBLE_DEVICES="" python tools/verify_derive_cmp.py
"""

import dataclasses
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _l10_comparison_combine_rules,
    _layer10_alu_cmp_combine_rules,
    _derived_cmp_combine_rules,
)
from neural_vm.unified_compiler.ops.shared import (  # noqa: E402
    cmp_gt_lo_lt_hieq_guard_enabled,
)

S = 100.0


def _sig(rule):
    d = dataclasses.asdict(rule)
    d.pop("name", None)
    d.pop("scope", None)
    return d


def _compare(label, hand, derived):
    ok = len(hand) == len(derived)
    mism = 0
    for i, (h, dv) in enumerate(zip(hand, derived)):
        if _sig(h) != _sig(dv):
            mism += 1
            if mism <= 4:
                sh, sd = _sig(h), _sig(dv)
                print(f"  [{label}] MISMATCH idx {i} ({h.name}):")
                for k in sh:
                    if sh[k] != sd.get(k):
                        print(f"    {k}: hand={sh[k]!r} derived={sd.get(k)!r}")
    ok = ok and mism == 0
    print(f"[{label}] {'OK' if ok else 'FAIL'} "
          f"(hand={len(hand)} derived={len(derived)} mismatches={mism})")
    return ok


def main() -> int:
    guard = 2.75 if cmp_gt_lo_lt_hieq_guard_enabled() else 2.5

    # Path 1: ComparisonCombine decode bank.
    d1 = _derived_cmp_combine_rules(
        S,
        opcode_gate_fmt="OP_{op}",
        default_threshold=1.5,
        override3_threshold=2.5,
        override_include_blocker=True,
        name_prefix_fmt="l10_cmp_{op}",
        gt_ge_guard_threshold=guard,
    )
    # _l10_comparison_combine_rules may append the C4_CMP_COMBINE_MARGIN clamp
    # bank; compare only the leading 18 core cmp units.
    ok1 = _compare("comparison_combine", _l10_comparison_combine_rules(S)[:18], d1)

    # Path 2: L10-main ALU cmp lane.
    d2 = _derived_cmp_combine_rules(
        S,
        opcode_gate_fmt="SE_OP_{op}+0",
        default_threshold=2.5,
        override3_threshold=4.0,
        override_include_blocker=False,
        name_prefix_fmt="l10_alu_cmp_{op}",
        gt_ge_guard_threshold=4.0,
    )
    ok2 = _compare("alu_cmp_combine", _layer10_alu_cmp_combine_rules(S), d2)

    all_ok = ok1 and ok2
    print("RESULT:", "BYTE-IDENTICAL" if all_ok else "DIVERGED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
