"""Byte-identity gate: derived (wide_alu_dsl) add/sub == hand-authored L8/L9.

Proves the compact-spec ``nibble_alu_lane_rules`` generator reproduces each of
the 8 hand-authored L8/L9 add/sub FFNRule builders FIELD-FOR-FIELD (name,
conditions, threshold, gate, writes). This is the authoring-time gate that the
DERIVED add/sub is byte-identical to the DELETED hand-authored form; the golden
hash (tools/_isa_golden_hash.py) is the whole-model gate.

READ-only (no bake). Exit 0 iff every builder matches.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from c4_release.neural_vm.unified_compiler.ops import l8_ops, l9_ops
from c4_release.neural_vm.unified_compiler.wide_alu_dsl import nibble_alu_lane_rules

S = 100.0


def _key(r):
    conds = tuple(sorted(
        (c.dim.name, c.dim.offset, round(c.weight, 12)) for c in r.conditions
    ))
    writes = tuple(sorted(
        (w.dim.name, w.dim.offset, round(w.weight, 12)) for w in r.writes
    ))
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


def _derived_l8_add_lo():
    return nibble_alu_lane_rules(
        op="add", emit="result",
        operand_a_band="ALU_LO", operand_b_band="AX_CARRY_LO",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-4.0,
        gate="OP_ADD", threshold_no_carry=2.5,
        result_band="OUTPUT_LO", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"l8_alu_add_lo_a{a}_b{b}_step_end",
    )


def _derived_l8_sub_lo():
    return nibble_alu_lane_rules(
        op="sub", emit="result",
        operand_a_band="ALU_LO", operand_b_band="AX_CARRY_LO",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-4.0,
        gate="OP_SUB", threshold_no_carry=2.5,
        result_band="OUTPUT_LO", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"l8_alu_sub_lo_a{a}_b{b}_step_end",
    )


def _derived_l8_add_carry():
    return nibble_alu_lane_rules(
        op="add", emit="carry_flag",
        operand_a_band="ALU_LO", operand_b_band="AX_CARRY_LO",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-4.0,
        gate="OP_ADD", threshold_no_carry=2.5,
        carry_flag_dim="CARRY+0", carry_flag_scale=2.0 / (S * 5.0),
        name_fn=lambda c, a, b: f"l8_alu_add_carry_a{a}_b{b}_step_end",
    )


def _derived_l8_sub_borrow():
    return nibble_alu_lane_rules(
        op="sub", emit="carry_flag",
        operand_a_band="ALU_LO", operand_b_band="AX_CARRY_LO",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-4.0,
        gate="OP_SUB", threshold_no_carry=2.5,
        carry_flag_dim="CARRY+0", carry_flag_scale=2.0 / (S * 5.0),
        name_fn=lambda c, a, b: f"l8_alu_sub_borrow_a{a}_b{b}_step_end",
    )


def _derived_l9_add_hi():
    return nibble_alu_lane_rules(
        op="add", emit="result",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate="OP_ADD", threshold_no_carry=2.5, threshold_with_carry=4.5,
        carry_in_dim="CARRY+0", carry_in_weight=2.0,
        result_band="OUTPUT_HI_THIS_STEP", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"l9_add_hi_c{c}_a{a}_b{b}",
    )


def _derived_l9_sub_hi():
    return nibble_alu_lane_rules(
        op="sub", emit="result",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate="OP_SUB", threshold_no_carry=2.5, threshold_with_carry=4.5,
        carry_in_dim="CARRY+0", carry_in_weight=2.0,
        result_band="OUTPUT_HI_THIS_STEP", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"sub_hi_b{c}_a{a}_b{b}",
    )


def _derived_l9_add_carry_out():
    return nibble_alu_lane_rules(
        op="add", emit="carry_flag",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate="OP_ADD", threshold_no_carry=2.5, threshold_with_carry=2.9,
        carry_in_dim="CARRY+0", carry_in_weight=0.01 / S,
        carry_flag_dim="CARRY+1", carry_flag_scale=2.0 / S,
        name_fn=lambda c, a, b: f"add_carry_out_c{c}_a{a}_b{b}",
    )


def _derived_l9_sub_borrow_out():
    return nibble_alu_lane_rules(
        op="sub", emit="carry_flag",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate="OP_SUB", threshold_no_carry=2.5, threshold_with_carry=2.9,
        carry_in_dim="CARRY+0", carry_in_weight=0.01 / S,
        carry_flag_dim="CARRY+2", carry_flag_scale=2.0 / S,
        name_fn=lambda c, a, b: f"sub_borrow_out_b{c}_a{a}_b{b}",
    )


def main():
    cases = [
        ("L8 add_lo", l8_ops._layer8_alu_add_lo_rules(S), _derived_l8_add_lo()),
        ("L8 sub_lo", l8_ops._layer8_alu_sub_lo_rules(S), _derived_l8_sub_lo()),
        ("L8 add_carry", l8_ops._layer8_alu_add_carry_rules(S),
         _derived_l8_add_carry()),
        ("L8 sub_borrow", l8_ops._layer8_alu_sub_borrow_rules(S),
         _derived_l8_sub_borrow()),
        ("L9 add_hi", l9_ops._add_hi_nibble_rules(S), _derived_l9_add_hi()),
        ("L9 sub_hi", l9_ops._sub_hi_nibble_rules(S), _derived_l9_sub_hi()),
        ("L9 add_carry_out", l9_ops._add_carry_out_rules(S),
         _derived_l9_add_carry_out()),
        ("L9 sub_borrow_out", l9_ops._sub_borrow_out_rules(S),
         _derived_l9_sub_borrow_out()),
    ]
    ok = True
    for label, hand, derived in cases:
        ok = _cmp(label, hand, derived) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
