"""Parity tests for L6 FFN bands migrated to CompilerIR rules."""

import torch

from neural_vm.vm_step import _SetDim, _set_layer6_routing_ffn
from neural_vm.unified_compiler.ops.l6_ops import (
    L6_ADJ_AX_ROUTE_END_UNIT,
    L6_ADJ_AX_ROUTE_START_UNIT,
    L6_ADJ_SP_WRITEBACK_END_UNIT,
    L6_ADJ_SP_WRITEBACK_START_UNIT,
    L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT,
    L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
    L6_BNZ_AX_ROUTE_END_UNIT,
    L6_BNZ_AX_ROUTE_START_UNIT,
    L6_BNZ_PC_OVERRIDE_END_UNIT,
    L6_BNZ_PC_OVERRIDE_START_UNIT,
    L6_BZ_AX_ROUTE_END_UNIT,
    L6_BZ_AX_ROUTE_START_UNIT,
    L6_BZ_PC_OVERRIDE_END_UNIT,
    L6_BZ_PC_OVERRIDE_START_UNIT,
    L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT,
    L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT,
    L6_EXIT_AX_ROUTE_END_UNIT,
    L6_EXIT_AX_ROUTE_START_UNIT,
    L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT,
    L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT,
    L6_CMP3_CLEANUP_END_UNIT,
    L6_CMP3_CLEANUP_START_UNIT,
    L6_GETCHAR_AX_ROUTE_END_UNIT,
    L6_GETCHAR_AX_ROUTE_START_UNIT,
    L6_HALT_DETECT_END_UNIT,
    L6_HALT_DETECT_START_UNIT,
    L6_ENT_SP_WRITEBACK_END_UNIT,
    L6_ENT_SP_WRITEBACK_START_UNIT,
    L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT,
    L6_ENT_FIRST_STEP_SP_BYTE0_START_UNIT,
    L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT,
    L6_ENT_FIRST_STEP_SP_BYTES_START_UNIT,
    L6_IMM_CARRY_REFRESH_END_UNIT,
    L6_IMM_CARRY_REFRESH_START_UNIT,
    L6_IMM_FETCH_ROUTE_END_UNIT,
    L6_IMM_FETCH_ROUTE_START_UNIT,
    L6_JMP_AX_ROUTE_END_UNIT,
    L6_JMP_AX_ROUTE_START_UNIT,
    L6_JSR_AX_ROUTE_END_UNIT,
    L6_JSR_AX_ROUTE_START_UNIT,
    L6_JSR_SP_BYTES_END_UNIT,
    L6_JSR_SP_BYTES_START_UNIT,
    L6_JSR_SP_DECREMENT_END_UNIT,
    L6_JSR_SP_DECREMENT_START_UNIT,
    L6_JSR_SP_FIXUP_END_UNIT,
    L6_JSR_SP_FIXUP_START_UNIT,
    L6_NOP_AX_ROUTE_END_UNIT,
    L6_NOP_AX_ROUTE_START_UNIT,
    L6_ALU_CLEAR_END_UNIT,
    L6_ALU_CLEAR_START_UNIT,
    L6_MEM_LEAKAGE_CLEANUP_END_UNIT,
    L6_MEM_LEAKAGE_CLEANUP_START_UNIT,
    L6_OPCODE_CONTAMINATION_CLEANUP_END_UNIT,
    L6_OPCODE_CONTAMINATION_CLEANUP_START_UNIT,
    L6_PSH_AX_ROUTE_END_UNIT,
    L6_PSH_AX_ROUTE_START_UNIT,
    L6_PSH_SP_DECREMENT_END_UNIT,
    L6_PSH_SP_DECREMENT_START_UNIT,
    L6_PSH_STACK0_WRITEBACK_END_UNIT,
    L6_PSH_STACK0_WRITEBACK_START_UNIT,
    L6_STACK_IDENTITY_END_UNIT,
    L6_STACK_IDENTITY_START_UNIT,
    L6_TEMP_CLEANUP_END_UNIT,
    L6_TEMP_CLEANUP_START_UNIT,
    L6_TEMP_CLEANUP_RULE_START_UNIT,
    _bake_layer6_routing_ffn,
    _layer6_adj_ax_route_rules,
    _layer6_adj_sp_writeback_rules,
    _layer6_bnz_ax_route_rules,
    _layer6_bnz_pc_override_rules,
    _layer6_bz_ax_route_rules,
    _layer6_bz_pc_override_rules,
    _layer6_cmp3_cleanup_rules,
    _layer6_delayed_jmp_pc_override_rules,
    _layer6_exit_ax_route_rules,
    _layer6_first_step_jmp_pc_override_rules,
    _layer6_getchar_ax_route_rules,
    _layer6_halt_detect_rules,
    _layer6_ent_sp_writeback_rules,
    _layer6_ent_first_step_sp_byte0_rules,
    _layer6_ent_first_step_sp_bytes_rules,
    _layer6_imm_carry_refresh_rules,
    _layer6_imm_fetch_route_rules,
    _layer6_jmp_ax_route_rules,
    _layer6_jsr_ax_route_rules,
    _layer6_jsr_sp_bytes_rules,
    _layer6_jsr_sp_decrement_rules,
    _layer6_jsr_sp_fixup_rules,
    _layer6_nop_ax_route_rules,
    _layer6_psh_ax_route_rules,
    _layer6_psh_sp_decrement_rules,
    _layer6_psh_stack0_writeback_rules,
    _layer6_stack_identity_rules,
    _layer6_tail_cleanup_rules,
    _layer6_temp_cleanup_rules,
    _layer6_all_step_jmp_pc_override_rules,
    _lower_layer6_cmp3_cleanup_ir,
    _lower_layer6_delayed_jmp_pc_override_ir,
    _lower_layer6_first_step_jmp_pc_override_ir,
    _lower_layer6_halt_detect_ir,
    _lower_layer6_imm_carry_refresh_ir,
    _lower_layer6_late_ax_output_route_ir,
    _lower_layer6_branch_pc_override_ir,
    _lower_layer6_ent_first_step_ir,
    _lower_layer6_stack_arithmetic_ir,
    _lower_layer6_stack_writeback_ir,
    _lower_layer6_tail_cleanup_ir,
    _lower_layer6_stack_identity_ir,
    _lower_layer6_temp_cleanup_ir,
    _lower_layer6_ax_output_route_ir,
    _lower_layer6_imm_fetch_route_ir,
    _lower_layer6_all_step_jmp_pc_override_ir,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 1600):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn_units(
    actual: _StubFFN,
    expected: _StubFFN,
    start: int,
    end: int,
):
    assert torch.equal(actual.W_up[start:end], expected.W_up[start:end])
    assert torch.equal(actual.b_up[start:end], expected.b_up[start:end])
    assert torch.equal(actual.W_gate[start:end], expected.W_gate[start:end])
    assert torch.equal(actual.b_gate[start:end], expected.b_gate[start:end])
    assert torch.equal(actual.W_down[:, start:end], expected.W_down[:, start:end])


def test_layer6_all_step_jmp_pc_override_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    end = _lower_layer6_all_step_jmp_pc_override_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert end == L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT
    assert len(_layer6_all_step_jmp_pc_override_rules(100.0)) == 64
    _assert_same_ffn_units(
        actual,
        expected,
        L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
        L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT,
    )


def test_layer6_delayed_jmp_pc_override_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    end = _lower_layer6_delayed_jmp_pc_override_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert end == L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT
    assert len(_layer6_delayed_jmp_pc_override_rules(100.0)) == 64
    _assert_same_ffn_units(
        actual,
        expected,
        L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT,
        L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT,
    )


def test_layer6_first_step_jmp_pc_override_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    end = _lower_layer6_first_step_jmp_pc_override_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert end == L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT
    assert len(_layer6_first_step_jmp_pc_override_rules(100.0)) == 64
    _assert_same_ffn_units(
        actual,
        expected,
        L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT,
        L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT,
    )


def test_layer6_imm_fetch_route_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    end = _lower_layer6_imm_fetch_route_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert end == L6_IMM_FETCH_ROUTE_END_UNIT
    assert len(_layer6_imm_fetch_route_rules(100.0)) == 32
    _assert_same_ffn_units(
        actual,
        expected,
        L6_IMM_FETCH_ROUTE_START_UNIT,
        L6_IMM_FETCH_ROUTE_END_UNIT,
    )


def test_layer6_imm_carry_refresh_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    end = _lower_layer6_imm_carry_refresh_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert end == L6_IMM_CARRY_REFRESH_END_UNIT
    assert len(_layer6_imm_carry_refresh_rules(100.0)) == 32
    _assert_same_ffn_units(
        actual,
        expected,
        L6_IMM_CARRY_REFRESH_START_UNIT,
        L6_IMM_CARRY_REFRESH_END_UNIT,
    )


def test_layer6_halt_cleanup_and_identity_ir_match_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    halt_end = _lower_layer6_halt_detect_ir(actual, 100.0, _SetDim)
    temp_end = _lower_layer6_temp_cleanup_ir(actual, 100.0, _SetDim)
    cmp3_end = _lower_layer6_cmp3_cleanup_ir(actual, 100.0, _SetDim)
    stack_end = _lower_layer6_stack_identity_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert halt_end == L6_HALT_DETECT_END_UNIT
    assert temp_end == L6_TEMP_CLEANUP_END_UNIT
    assert cmp3_end == L6_CMP3_CLEANUP_END_UNIT
    assert stack_end == L6_STACK_IDENTITY_END_UNIT
    assert len(_layer6_halt_detect_rules(100.0)) == 1
    assert len(_layer6_temp_cleanup_rules(100.0)) == 31
    assert len(_layer6_cmp3_cleanup_rules(100.0)) == 1
    assert len(_layer6_stack_identity_rules(100.0)) == 96
    for start, end in (
        (L6_HALT_DETECT_START_UNIT, L6_HALT_DETECT_END_UNIT),
        (L6_TEMP_CLEANUP_RULE_START_UNIT, L6_TEMP_CLEANUP_END_UNIT),
        (L6_CMP3_CLEANUP_START_UNIT, L6_CMP3_CLEANUP_END_UNIT),
        (L6_STACK_IDENTITY_START_UNIT, L6_STACK_IDENTITY_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)

    # The legacy TEMP[0] slot is intentionally blank and not claimed by IR.
    assert torch.equal(
        actual.W_up[
            L6_TEMP_CLEANUP_START_UNIT:L6_TEMP_CLEANUP_RULE_START_UNIT
        ],
        torch.zeros_like(
            actual.W_up[
                L6_TEMP_CLEANUP_START_UNIT:L6_TEMP_CLEANUP_RULE_START_UNIT
            ]
        ),
    )


def test_layer6_ax_output_route_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    route_ends = _lower_layer6_ax_output_route_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert route_ends == (
        L6_EXIT_AX_ROUTE_END_UNIT,
        L6_NOP_AX_ROUTE_END_UNIT,
        L6_JSR_AX_ROUTE_END_UNIT,
        L6_JMP_AX_ROUTE_END_UNIT,
    )
    assert len(_layer6_exit_ax_route_rules(100.0)) == 32
    assert len(_layer6_nop_ax_route_rules(100.0)) == 32
    assert len(_layer6_jsr_ax_route_rules(100.0)) == 32
    assert len(_layer6_jmp_ax_route_rules(100.0)) == 32
    for start, end in (
        (L6_EXIT_AX_ROUTE_START_UNIT, L6_EXIT_AX_ROUTE_END_UNIT),
        (L6_NOP_AX_ROUTE_START_UNIT, L6_NOP_AX_ROUTE_END_UNIT),
        (L6_JSR_AX_ROUTE_START_UNIT, L6_JSR_AX_ROUTE_END_UNIT),
        (L6_JMP_AX_ROUTE_START_UNIT, L6_JMP_AX_ROUTE_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_late_ax_output_route_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    route_ends = _lower_layer6_late_ax_output_route_ir(
        actual,
        100.0,
        _SetDim,
    )
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert route_ends == (
        L6_GETCHAR_AX_ROUTE_END_UNIT,
        L6_BZ_AX_ROUTE_END_UNIT,
        L6_BNZ_AX_ROUTE_END_UNIT,
        L6_PSH_AX_ROUTE_END_UNIT,
        L6_ADJ_AX_ROUTE_END_UNIT,
    )
    assert len(_layer6_getchar_ax_route_rules(100.0)) == 32
    assert len(_layer6_bz_ax_route_rules(100.0)) == 32
    assert len(_layer6_bnz_ax_route_rules(100.0)) == 32
    assert len(_layer6_psh_ax_route_rules(100.0)) == 32
    assert len(_layer6_adj_ax_route_rules(100.0)) == 32
    for start, end in (
        (L6_GETCHAR_AX_ROUTE_START_UNIT, L6_GETCHAR_AX_ROUTE_END_UNIT),
        (L6_BZ_AX_ROUTE_START_UNIT, L6_BZ_AX_ROUTE_END_UNIT),
        (L6_BNZ_AX_ROUTE_START_UNIT, L6_BNZ_AX_ROUTE_END_UNIT),
        (L6_PSH_AX_ROUTE_START_UNIT, L6_PSH_AX_ROUTE_END_UNIT),
        (L6_ADJ_AX_ROUTE_START_UNIT, L6_ADJ_AX_ROUTE_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_stack_writeback_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    route_ends = _lower_layer6_stack_writeback_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert route_ends == (
        L6_ADJ_SP_WRITEBACK_END_UNIT,
        L6_ENT_SP_WRITEBACK_END_UNIT,
    )
    assert len(_layer6_adj_sp_writeback_rules(100.0)) == 32
    assert len(_layer6_ent_sp_writeback_rules(100.0)) == 32
    for start, end in (
        (L6_ADJ_SP_WRITEBACK_START_UNIT, L6_ADJ_SP_WRITEBACK_END_UNIT),
        (L6_ENT_SP_WRITEBACK_START_UNIT, L6_ENT_SP_WRITEBACK_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_stack_arithmetic_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    route_ends = _lower_layer6_stack_arithmetic_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert route_ends == (
        L6_PSH_SP_DECREMENT_END_UNIT,
        L6_JSR_SP_DECREMENT_END_UNIT,
        L6_JSR_SP_FIXUP_END_UNIT,
        L6_JSR_SP_BYTES_END_UNIT,
        L6_PSH_STACK0_WRITEBACK_END_UNIT,
    )
    assert len(_layer6_psh_sp_decrement_rules(100.0)) == 32
    assert len(_layer6_jsr_sp_decrement_rules(100.0)) == 32
    assert len(_layer6_jsr_sp_fixup_rules(100.0)) == 2
    assert len(_layer6_jsr_sp_bytes_rules(100.0)) == 4
    assert len(_layer6_psh_stack0_writeback_rules(100.0)) == 32
    for start, end in (
        (L6_PSH_SP_DECREMENT_START_UNIT, L6_PSH_SP_DECREMENT_END_UNIT),
        (L6_JSR_SP_DECREMENT_START_UNIT, L6_JSR_SP_DECREMENT_END_UNIT),
        (L6_JSR_SP_FIXUP_START_UNIT, L6_JSR_SP_FIXUP_END_UNIT),
        (L6_JSR_SP_BYTES_START_UNIT, L6_JSR_SP_BYTES_END_UNIT),
        (
            L6_PSH_STACK0_WRITEBACK_START_UNIT,
            L6_PSH_STACK0_WRITEBACK_END_UNIT,
        ),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_ent_first_step_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    route_ends = _lower_layer6_ent_first_step_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert route_ends == (
        L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT,
        L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT,
    )
    assert len(_layer6_ent_first_step_sp_byte0_rules(100.0)) == 32
    assert len(_layer6_ent_first_step_sp_bytes_rules(100.0)) == 6
    for start, end in (
        (
            L6_ENT_FIRST_STEP_SP_BYTE0_START_UNIT,
            L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT,
        ),
        (
            L6_ENT_FIRST_STEP_SP_BYTES_START_UNIT,
            L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT,
        ),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_branch_pc_override_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    route_ends = _lower_layer6_branch_pc_override_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert route_ends == (L6_BZ_PC_OVERRIDE_END_UNIT, L6_BNZ_PC_OVERRIDE_END_UNIT)
    assert len(_layer6_bz_pc_override_rules(100.0)) == 64
    assert len(_layer6_bnz_pc_override_rules(100.0)) == 128
    for start, end in (
        (L6_BZ_PC_OVERRIDE_START_UNIT, L6_BZ_PC_OVERRIDE_END_UNIT),
        (L6_BNZ_PC_OVERRIDE_START_UNIT, L6_BNZ_PC_OVERRIDE_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_tail_cleanup_ir_matches_legacy_units():
    actual = _StubFFN()
    expected = _StubFFN()

    end = _lower_layer6_tail_cleanup_ir(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    assert end == L6_ALU_CLEAR_END_UNIT
    assert len(_layer6_tail_cleanup_rules(100.0)) == 66
    for start, end in (
        (
            L6_OPCODE_CONTAMINATION_CLEANUP_START_UNIT,
            L6_OPCODE_CONTAMINATION_CLEANUP_END_UNIT,
        ),
        (
            L6_MEM_LEAKAGE_CLEANUP_START_UNIT,
            L6_MEM_LEAKAGE_CLEANUP_END_UNIT,
        ),
        (L6_ALU_CLEAR_START_UNIT, L6_ALU_CLEAR_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_routing_bake_keeps_ir_bands_at_legacy_parity():
    actual = _StubFFN()
    expected = _StubFFN()

    _bake_layer6_routing_ffn(actual, 100.0, _SetDim)
    _set_layer6_routing_ffn(expected, 100.0, _SetDim)

    for start, end in (
        (L6_IMM_FETCH_ROUTE_START_UNIT, L6_IMM_FETCH_ROUTE_END_UNIT),
        (L6_IMM_CARRY_REFRESH_START_UNIT, L6_IMM_CARRY_REFRESH_END_UNIT),
        (L6_EXIT_AX_ROUTE_START_UNIT, L6_EXIT_AX_ROUTE_END_UNIT),
        (L6_NOP_AX_ROUTE_START_UNIT, L6_NOP_AX_ROUTE_END_UNIT),
        (L6_JSR_AX_ROUTE_START_UNIT, L6_JSR_AX_ROUTE_END_UNIT),
        (L6_JMP_AX_ROUTE_START_UNIT, L6_JMP_AX_ROUTE_END_UNIT),
        (
            L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT,
            L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT,
        ),
        (
            L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT,
            L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT,
        ),
        (L6_HALT_DETECT_START_UNIT, L6_HALT_DETECT_END_UNIT),
        (L6_TEMP_CLEANUP_RULE_START_UNIT, L6_TEMP_CLEANUP_END_UNIT),
        (L6_CMP3_CLEANUP_START_UNIT, L6_CMP3_CLEANUP_END_UNIT),
        (L6_STACK_IDENTITY_START_UNIT, L6_STACK_IDENTITY_END_UNIT),
        (L6_PSH_SP_DECREMENT_START_UNIT, L6_PSH_SP_DECREMENT_END_UNIT),
        (L6_JSR_SP_DECREMENT_START_UNIT, L6_JSR_SP_DECREMENT_END_UNIT),
        (L6_JSR_SP_FIXUP_START_UNIT, L6_JSR_SP_FIXUP_END_UNIT),
        (L6_JSR_SP_BYTES_START_UNIT, L6_JSR_SP_BYTES_END_UNIT),
        (
            L6_PSH_STACK0_WRITEBACK_START_UNIT,
            L6_PSH_STACK0_WRITEBACK_END_UNIT,
        ),
        (L6_GETCHAR_AX_ROUTE_START_UNIT, L6_GETCHAR_AX_ROUTE_END_UNIT),
        (L6_BZ_AX_ROUTE_START_UNIT, L6_BZ_AX_ROUTE_END_UNIT),
        (L6_BNZ_AX_ROUTE_START_UNIT, L6_BNZ_AX_ROUTE_END_UNIT),
        (L6_PSH_AX_ROUTE_START_UNIT, L6_PSH_AX_ROUTE_END_UNIT),
        (L6_ADJ_AX_ROUTE_START_UNIT, L6_ADJ_AX_ROUTE_END_UNIT),
        (L6_ADJ_SP_WRITEBACK_START_UNIT, L6_ADJ_SP_WRITEBACK_END_UNIT),
        (L6_ENT_SP_WRITEBACK_START_UNIT, L6_ENT_SP_WRITEBACK_END_UNIT),
        (
            L6_ENT_FIRST_STEP_SP_BYTE0_START_UNIT,
            L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT,
        ),
        (
            L6_ENT_FIRST_STEP_SP_BYTES_START_UNIT,
            L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT,
        ),
        (L6_BZ_PC_OVERRIDE_START_UNIT, L6_BZ_PC_OVERRIDE_END_UNIT),
        (L6_BNZ_PC_OVERRIDE_START_UNIT, L6_BNZ_PC_OVERRIDE_END_UNIT),
        (
            L6_OPCODE_CONTAMINATION_CLEANUP_START_UNIT,
            L6_OPCODE_CONTAMINATION_CLEANUP_END_UNIT,
        ),
        (
            L6_MEM_LEAKAGE_CLEANUP_START_UNIT,
            L6_MEM_LEAKAGE_CLEANUP_END_UNIT,
        ),
        (L6_ALU_CLEAR_START_UNIT, L6_ALU_CLEAR_END_UNIT),
    ):
        _assert_same_ffn_units(actual, expected, start, end)
    _assert_same_ffn_units(
        actual,
        expected,
        L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
        L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT,
    )
