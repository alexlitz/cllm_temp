"""Spec-consistency gate: wide_alu_dsl comparator == the L9 CMP combine.

Proves the compact-spec ``nibble_compare_lane_rules`` generator (driven by the
per-lane spec authored INDEPENDENTLY here — a re-statement of the ORIGINAL
hand-authored ``multi_way_and_rule`` loops for hi_eq / lo_eq / hi_lt / lo_lt)
reproduces the LIVE :func:`l9_ops._layer9_cmp_rules` output FIELD-FOR-FIELD
(name, conditions, threshold, gate, writes).

Sibling of ``tools/verify_l8l9_addsub_derived.py`` (which does the same for the
ADD/SUB nibble lanes via ``nibble_alu_lane_rules``). Since the 2026-07
derivation the ops builder ITSELF calls the generator, so this asserts the
spec kwargs the ops file passes match the independent HAND spec here — a
regression tripwire if either side drifts. The byte-identity vs the DELETED
hand-authored loops is now permanently gated by the WHOLE-MODEL golden hash
(``tools/_isa_golden_hash.py`` == 91f55411, unchanged across the derivation).

READ-only (no bake). Exit 0 iff every lane matches.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from c4_release.neural_vm.dim_registry import dim_ref
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import multi_way_and_rule
from c4_release.neural_vm.unified_compiler.ops import l9_ops

S = 100.0

# The SE-tagged CMP gate + cmp-cascade output dims, resolved exactly as the
# live op does (l9_ops._layer9_cmp_rules).
GATE_CMP_GROUP = "SE_CMP_GROUP+0"
CMP_BYTE0 = dim_ref("cmp_flag", "cascade", 0)
CMP_BYTE1 = dim_ref("cmp_flag", "cascade", 1)
CMP_BYTE2 = dim_ref("cmp_flag", "cascade", 2)
CMP_BYTE3 = dim_ref("cmp_flag", "cascade", 3)


def _key(r):
    conds = tuple(
        (c.dim.name, c.dim.offset, round(c.weight, 12)) for c in r.conditions
    )
    writes = tuple(
        (w.dim.name, w.dim.offset, round(w.weight, 12)) for w in r.writes
    )
    gate = None if r.gate is None else (r.gate.name, r.gate.offset)
    gate_terms = tuple(sorted(
        (t.dim.name, t.dim.offset, round(t.weight, 12)) for t in r.gate_terms
    ))
    return (
        r.name,
        conds,
        round(r.threshold, 12),
        gate,
        round(r.gate_weight, 12),
        r.gate_bias if r.gate_bias is None else round(r.gate_bias, 12),
        gate_terms,
        writes,
    )


def _cmp(label, hand, derived):
    if len(hand) != len(derived):
        print(f"  FAIL {label}: length {len(derived)} != hand {len(hand)}")
        return False
    for i, (h, d) in enumerate(zip(hand, derived)):
        kh, kd = _key(h), _key(d)
        if kh != kd:
            print(f"  FAIL {label}[{i}] ({h.name}):")
            for fh, fd in zip(kh, kd):
                if fh != fd:
                    print(f"      hand:    {fh}")
                    print(f"      derived: {fd}")
            return False
    print(f"  ok {label}: {len(hand)} rules byte-identical")
    return True


# ---------------------------------------------------------------------------
# The INDEPENDENT hand spec: the ORIGINAL per-value ``multi_way_and_rule``
# loops that ``_layer9_cmp_rules`` used before the comparator derivation.
# ---------------------------------------------------------------------------


def _hand_hi_eq():
    rules = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"l9_cmp_hi_eq_{k}_step_end",
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("MARK_PC", -2.0),
                (f"SE_ALU_HI+{k}", 1.0),
                (f"SE_AX_CARRY_HI+{k}", 1.0),
            ),
            threshold=2.5,
            gate=GATE_CMP_GROUP,
            writes=((CMP_BYTE1, 2.0 / S),),
        ))
    return rules


def _hand_lo_eq():
    rules = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"l9_cmp_lo_eq_{k}_step_end",
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("MARK_PC", -2.0),
                (f"SE_ALU_LO+{k}", 1.0),
                (f"SE_AX_CARRY_LO+{k}", 1.0),
            ),
            threshold=2.5,
            gate=GATE_CMP_GROUP,
            writes=((CMP_BYTE2, 8.0 / S),),
        ))
    return rules


def _hand_hi_lt():
    rules = []
    for a in range(16):
        for b in range(a + 1, 16):
            rules.append(multi_way_and_rule(
                name=f"l9_cmp_hi_lt_a{a}_b{b}_step_end",
                conditions=(
                    ("MARK_SE_ONLY", 1.0),
                    ("MARK_PC", -2.0),
                    (f"SE_ALU_HI+{a}", 1.0),
                    (f"SE_AX_CARRY_HI+{b}", 1.0),
                ),
                threshold=2.5,
                gate=GATE_CMP_GROUP,
                writes=((CMP_BYTE0, 2.0 / S),),
            ))
    return rules


def _hand_lo_lt():
    rules = []
    for a in range(16):
        for b in range(a + 1, 16):
            rules.append(multi_way_and_rule(
                name=f"l9_cmp_lo_lt_a{a}_b{b}_step_end",
                conditions=(
                    ("MARK_SE_ONLY", 1.0),
                    ("MARK_PC", -2.0),
                    (f"SE_ALU_LO+{a}", 1.0),
                    (f"SE_AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=GATE_CMP_GROUP,
                writes=((CMP_BYTE3, 2.0 / S),),
            ))
    return rules


def main() -> int:
    # The live op now emits all four CMP lanes contiguously via the
    # comparator generator (hi_eq 16 + lo_eq 16 + hi_lt 120 + lo_lt 120).
    live = list(l9_ops._layer9_cmp_rules(S))
    hi_eq, lo_eq = live[0:16], live[16:32]
    hi_lt, lo_lt = live[32:152], live[152:272]

    ok = True
    print("L9 CMP comparator derivation byte-identity check:")
    ok &= _cmp("hi_eq", _hand_hi_eq(), hi_eq)
    ok &= _cmp("lo_eq", _hand_lo_eq(), lo_eq)
    ok &= _cmp("hi_lt", _hand_hi_lt(), hi_lt)
    ok &= _cmp("lo_lt", _hand_lo_lt(), lo_lt)

    if len(live) != 272:
        print(f"  FAIL: total {len(live)} != 272 CMP units")
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
