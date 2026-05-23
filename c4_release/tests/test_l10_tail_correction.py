"""Regression coverage for declarative L10 tail correction rules."""

from c4_release.neural_vm.unified_compiler.ir import CompilerIR
from c4_release.neural_vm.unified_compiler.ops.l10_ops import (
    _tail_bit32_result_correction_rules,
)


def _tail_rule(name: str):
    for rule in _tail_bit32_result_correction_rules():
        if rule.name == name:
            return rule
    raise AssertionError(f"missing tail rule {name}")


def _single_rule_ir(rule) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.append(rule)
    return ir


def test_tail_bp_byte2_preserve_requires_bp_span_signal():
    ir = _single_rule_ir(_tail_rule("tail_bp_byte2_preserve_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "BYTE_INDEX_1": 0.013,
        "OUTPUT_LO+0": 0.95,
        "OUTPUT_HI+0": 29.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_bp_byte2_preserve_still_fires_on_bp_byte1():
    ir = _single_rule_ir(_tail_rule("tail_bp_byte2_preserve_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_1": 1.0,
        "OUTPUT_LO+1": 1.0,
        "OUTPUT_HI+0": 1.0,
    })

    assert out["OUTPUT_LO+1"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0


def test_tail_does_not_emit_ax_add_carry_rules():
    names = {rule.name for rule in _tail_bit32_result_correction_rules()}

    assert "tail_ax_add_carry_byte1_02" not in names


def test_tail_sub_borrow_requires_sub_marker():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+2": 20.0,
        "OUTPUT_HI+0": 20.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_sub_borrow_still_fires_for_sub_marker():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+0": 1700.0,
        "OUTPUT_HI+0": 24.0,
    })

    assert out["OUTPUT_LO+0"] > 1700.0


def test_tail_sub_borrow_blocks_mul_high_byte_shape():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+0": 43.0,
        "OUTPUT_HI+1": 40.0,
    })

    assert out.get("OUTPUT_HI+1", 0.0) == 40.0


def test_tail_sub_borrow_blocks_add_carry_shape():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "OUTPUT_LO+0": 1700.0,
        "OUTPUT_HI+0": 26.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 1700.0


def test_tail_sub_borrow_blocks_non_borrowing_sub():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+1": 6.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_sub_borrow_blocks_preserved_non_unit_high_byte():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+5": 24.0,
        "OUTPUT_HI+0": 24.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_sub_borrow_blocks_ax_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 4.0,
        "ALU_LO+1": 7.0,
        "OUTPUT_LO+0": 24.0,
        "OUTPUT_HI+0": 24.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 24.0


def test_tail_sub_borrow_blocks_imm_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "OP_IMM": 5.0,
        "OUTPUT_LO+0": 520.0,
        "OUTPUT_HI+0": 33.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 520.0


def test_tail_sub_borrow_preserves_nonzero_high_byte():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+5": 23.0,
        "OUTPUT_HI+0": 26.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_sub_borrow_blocks_unit_high_byte_without_low_byte_underflow_signature():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+1": 24.0,
        "OUTPUT_LO+0": 24.0,
        "OUTPUT_HI+0": 24.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 24.0


def test_tail_shr_marker_correction_blocks_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_shr_marker_byte0_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "TEMP+7": 1.0,
        "OP_SHR": 1.0,
        "OUTPUT_LO+6": 24.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_shr_marker_correction_still_fires_on_marker():
    ir = _single_rule_ir(_tail_rule("tail_shr_marker_byte0_01"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "TEMP+7": 1.0,
        "OP_SHR": 1.0,
        "OUTPUT_LO+6": 6.0,
    })

    assert out["OUTPUT_LO+1"] > 0.0


def test_tail_sp_pop_carry_blocks_ax_byte_row():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 2000.0,
        "OUTPUT_HI+0": 25.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 0.0
