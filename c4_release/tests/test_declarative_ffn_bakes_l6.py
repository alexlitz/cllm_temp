"""Parity tests for L6 FFN bands migrated to CompilerIR rules."""

import torch

from neural_vm.vm_step import _SetDim, _set_layer6_routing_ffn
from neural_vm.unified_compiler.ir import CompilerIR
from neural_vm.unified_compiler.ops.l6_ops import (
    L6_ADJ_AX_ROUTE_END_UNIT,
    L6_ADJ_AX_ROUTE_START_UNIT,
    L6_ADJ_SP_WRITEBACK_END_UNIT,
    L6_ADJ_SP_WRITEBACK_START_UNIT,
    L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT,
    L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
    L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT,
    L6_ALL_STEP_JSR_PC_OVERRIDE_START_UNIT,
    L6_BNZ_AX_ROUTE_END_UNIT,
    L6_BNZ_AX_ROUTE_START_UNIT,
    L6_BNZ_PC_OVERRIDE_END_UNIT,
    L6_BNZ_PC_OVERRIDE_START_UNIT,
    L6_BINARY_POP_SP_INCREMENT_END_UNIT,
    L6_BINARY_POP_SP_INCREMENT_START_UNIT,
    L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT,
    L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT,
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
    L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT,
    L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_START_UNIT,
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
    _bake_layer6_attn_spec,
    _bake_layer6_relay_heads_spec,
    _layer6_adj_ax_route_rules,
    _layer6_adj_sp_writeback_rules,
    _layer6_bnz_ax_route_rules,
    _layer6_bnz_pc_override_rules,
    _layer6_binary_pop_sp_increment_rules,
    _layer6_branch_pc_byte1_override_rules,
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
    _layer6_ent_after_jsr_sp_byte0_fixup_rules,
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
    _layer6_all_step_jsr_pc_override_rules,
    _lower_layer6_cmp3_cleanup_ir,
    _lower_layer6_delayed_jmp_pc_override_ir,
    _lower_layer6_first_step_jmp_pc_override_ir,
    _lower_layer6_halt_detect_ir,
    _lower_layer6_imm_carry_refresh_ir,
    _lower_layer6_late_ax_output_route_ir,
    _lower_layer6_branch_pc_override_ir,
    _lower_layer6_branch_pc_byte1_override_ir,
    _lower_layer6_ent_first_step_ir,
    _lower_layer6_ent_after_jsr_sp_byte0_fixup_ir,
    _lower_layer6_stack_arithmetic_ir,
    _lower_layer6_stack_writeback_ir,
    _lower_layer6_tail_cleanup_ir,
    _lower_layer6_stack_identity_ir,
    _lower_layer6_temp_cleanup_ir,
    _lower_layer6_ax_output_route_ir,
    _lower_layer6_imm_fetch_route_ir,
    _lower_layer6_all_step_jmp_pc_override_ir,
    _lower_layer6_all_step_jsr_pc_override_ir,
    _lower_layer6_binary_pop_sp_increment_ir,
)
from neural_vm.unified_compiler.ops.model_ops import make_function_call_weights_op


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 1600):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


class _StubAttn:
    def __init__(self, *, d_model: int = 512, num_heads: int = 8):
        self.num_heads = num_heads
        self.W_q = torch.zeros(d_model, d_model)
        self.W_k = torch.zeros(d_model, d_model)
        self.W_v = torch.zeros(d_model, d_model)
        self.W_o = torch.zeros(d_model, d_model)


def _apply_stub_ffn(ffn: _StubFFN, x: torch.Tensor) -> torch.Tensor:
    up = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up)
    gate = x @ ffn.W_gate.t() + ffn.b_gate
    return x + (up * gate) @ ffn.W_down.t()


def test_layer6_relay_heads_preserve_psh_jsr_opcode_relay_slots():
    attn = _StubAttn()

    _bake_layer6_relay_heads_spec(attn, _SetDim, 64)

    base = 6 * 64
    assert attn.W_q[base, _SetDim.MARK_SP] == 50.0
    assert attn.W_q[base, _SetDim.H1 + 2] == 50.0
    assert attn.W_q[base, _SetDim.MARK_STACK0] == 50.0
    assert attn.W_q[base, _SetDim.MARK_AX] == -50.0
    assert attn.W_k[base, _SetDim.MARK_AX] == 50.0

    assert attn.W_v[base + 1, _SetDim.OP_PSH] == 0.2
    assert attn.W_o[_SetDim.PSH_AT_SP, base + 1] == 1.0
    assert attn.W_o[_SetDim.CMP + 0, base + 1] == 1.0
    assert attn.W_v[base + 0, _SetDim.OP_LEV] == 0.1
    assert attn.W_o[_SetDim.OP_LEV, base + 0] == 10.0
    assert attn.W_v[base + 2, _SetDim.OP_ADJ] == 0.2
    assert attn.W_o[_SetDim.CMP + 1, base + 2] == 1.0
    assert attn.W_v[base + 3, _SetDim.OP_ADD] == 0.04
    assert attn.W_o[_SetDim.CMP + 3, base + 3] == 5.0
    assert attn.W_v[base + 5, _SetDim.OP_JSR] == 0.2
    assert attn.W_o[_SetDim.CMP + 4, base + 5] == 1.0
    assert attn.W_o[_SetDim.OP_JSR, base + 5] == 5.0
    assert attn.W_v[base + 6, _SetDim.OP_SI] == 0.2
    assert attn.W_v[base + 7, _SetDim.OP_SC] == 0.2
    assert attn.W_o[_SetDim.MEM_ADDR_SRC, base + 7] == 1.0


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


def test_layer6_all_step_jsr_pc_override_decodes_pc_opcode_directly():
    ffn = _StubFFN()

    end = _lower_layer6_all_step_jsr_pc_override_ir(ffn, 100.0, _SetDim)

    assert end == L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT
    assert len(_layer6_all_step_jsr_pc_override_rules(100.0)) == 80
    assert L6_ALL_STEP_JSR_PC_OVERRIDE_START_UNIT >= (
        L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT
    )

    x = torch.zeros(512)
    x[_SetDim.MARK_PC] = 1.0
    x[_SetDim.OPCODE_BYTE_LO + 3] = 1.0
    x[_SetDim.OPCODE_BYTE_HI + 0] = 1.0
    x[_SetDim.FETCH_LO + 3] = 1.0
    x[_SetDim.OUTPUT_LO + 0] = 1.0
    x[_SetDim.OUTPUT_HI + 0] = 1.0

    y = _apply_stub_ffn(ffn, x)

    assert y[_SetDim.OUTPUT_LO + 0] < x[_SetDim.OUTPUT_LO + 0]
    assert y[_SetDim.OUTPUT_LO + 10] > 0.9
    assert y[_SetDim.OUTPUT_HI + 1] > 0.9
    assert y[_SetDim.OUTPUT_HI + 0] < x[_SetDim.OUTPUT_HI + 0]

    x = torch.zeros(512)
    x[_SetDim.MARK_PC] = 1.0
    x[_SetDim.OPCODE_BYTE_LO + 3] = 1.0
    x[_SetDim.OPCODE_BYTE_HI + 0] = 1.0
    x[_SetDim.FETCH_LO + 4] = 1.0
    x[_SetDim.FETCH_HI + 1] = 1.0
    x[_SetDim.OUTPUT_HI + 2] = 1.0

    y = _apply_stub_ffn(ffn, x)

    assert y[_SetDim.OUTPUT_LO + 2] > 0.9
    assert y[_SetDim.OUTPUT_HI + 10] > 0.9
    assert y[_SetDim.OUTPUT_HI + 2] < x[_SetDim.OUTPUT_HI + 2]


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


def test_layer6_attn_relays_ent_immediate_from_ax_to_sp_marker():
    attn = _StubAttn()
    head_dim = attn.W_q.shape[0] // attn.num_heads

    _bake_layer6_attn_spec(attn, _SetDim, head_dim)

    base = 5 * head_dim
    relay_row = base + 51
    assert attn.W_q[relay_row, _SetDim.MARK_SP] == 500.0
    assert attn.W_q[relay_row, _SetDim.HAS_SE] == 500.0
    assert attn.W_q[relay_row, _SetDim.CONST] == -500.0
    assert attn.W_k[relay_row, _SetDim.MARK_AX] == 5.0
    assert attn.W_k[relay_row, _SetDim.OP_ENT] == 5.0
    assert attn.W_o[_SetDim.FETCH_LO + 0, base + 0] == 1.0
    assert attn.W_o[_SetDim.FETCH_HI + 1, base + 17] == 1.0


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


def test_layer6_stack_identity_fires_with_compact_marker_scale():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer6_stack_identity_rules(100.0))

    out = ir.symbolic_ffn({
        "MARK_BP": 1.0,
        "HAS_SE": 0.997,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+15": 1.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0

    first_step = ir.symbolic_ffn({
        "MARK_BP": 1.0,
        "HAS_SE": 0.0,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+15": 1.0,
    })

    assert first_step.get("OUTPUT_LO+0", 0.0) == 0.0
    assert first_step.get("OUTPUT_HI+15", 0.0) == 0.0


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
        (L6_JSR_SP_FIXUP_START_UNIT, L6_JSR_SP_FIXUP_END_UNIT),
        (L6_JSR_SP_BYTES_START_UNIT, L6_JSR_SP_BYTES_END_UNIT),
        (
            L6_PSH_STACK0_WRITEBACK_START_UNIT,
            L6_PSH_STACK0_WRITEBACK_END_UNIT,
        ),
    ):
        _assert_same_ffn_units(actual, expected, start, end)


def test_layer6_psh_sp_decrement_does_not_invert_no_borrow_hi_nibble():
    ffn = _StubFFN()
    _lower_layer6_stack_arithmetic_ir(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.PSH_AT_SP] = 1.0
    x[..., _SetDim.EMBED_LO + 8] = 1.0
    x[..., _SetDim.EMBED_HI + 14] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 0] > 0.9
    assert y[_SetDim.OUTPUT_LO + 8] < -0.9
    assert abs(float(y[_SetDim.OUTPUT_HI + 13])) < 1e-6
    assert abs(float(y[_SetDim.OUTPUT_HI + 14])) < 1e-6


def test_layer6_psh_sp_decrement_borrows_when_low_nibble_is_below_8():
    ffn = _StubFFN()
    _lower_layer6_stack_arithmetic_ir(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.PSH_AT_SP] = 1.0
    x[..., _SetDim.EMBED_LO + 7] = 1.0
    x[..., _SetDim.EMBED_HI + 14] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 15] > 0.9
    assert y[_SetDim.OUTPUT_LO + 7] < -0.9
    assert y[_SetDim.OUTPUT_HI + 13] > 0.9
    assert y[_SetDim.OUTPUT_HI + 14] < -0.9


def test_layer6_binary_pop_sp_increment_band_follows_function_call_band():
    function_call_op = make_function_call_weights_op()

    assert function_call_op.ffn_units_used == 2294
    assert L6_BINARY_POP_SP_INCREMENT_START_UNIT >= function_call_op.ffn_units_used
    assert (
        L6_BINARY_POP_SP_INCREMENT_END_UNIT
        == L6_BINARY_POP_SP_INCREMENT_START_UNIT + 32
    )
    assert len(_layer6_binary_pop_sp_increment_rules(100.0)) == 32


def test_layer6_binary_pop_sp_increment_emits_pre_l15_sp_byte0():
    ffn = _StubFFN(hidden_dim=L6_BINARY_POP_SP_INCREMENT_END_UNIT)

    end = _lower_layer6_binary_pop_sp_increment_ir(ffn, 100.0, _SetDim)

    assert end == L6_BINARY_POP_SP_INCREMENT_END_UNIT

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.CMP + 3] = 1.0
    x[..., _SetDim.EMBED_LO + 0] = 1.0
    x[..., _SetDim.EMBED_HI + 14] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 8] > 0.9
    assert y[_SetDim.OUTPUT_LO + 0] < -0.9
    assert abs(float(y[_SetDim.OUTPUT_HI + 15])) < 1e-6

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.CMP + 3] = 1.0
    x[..., _SetDim.EMBED_LO + 8] = 1.0
    x[..., _SetDim.EMBED_HI + 14] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 0] > 0.9
    assert y[_SetDim.OUTPUT_LO + 8] < -0.9
    assert y[_SetDim.OUTPUT_HI + 15] > 0.9
    assert y[_SetDim.OUTPUT_HI + 14] < -0.9


def test_layer6_binary_pop_sp_increment_blocks_non_sp_rows():
    ffn = _StubFFN(hidden_dim=L6_BINARY_POP_SP_INCREMENT_END_UNIT)
    _lower_layer6_binary_pop_sp_increment_ir(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.MARK_PC] = 1.0
    x[..., _SetDim.CMP + 3] = 1.0
    x[..., _SetDim.EMBED_LO + 0] = 1.0
    x[..., _SetDim.EMBED_HI + 14] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 8])) < 1e-6
    assert abs(float(y[_SetDim.OUTPUT_LO + 0])) < 1e-6

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.CMP + 3] = 1.0
    x[..., _SetDim.EMBED_LO + 0] = 1.0
    x[..., _SetDim.EMBED_HI + 14] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 8])) < 1e-6
    assert abs(float(y[_SetDim.OUTPUT_LO + 0])) < 1e-6


def test_layer6_jsr_sp_marker_decrement_handles_later_calls():
    ffn = _StubFFN()
    _lower_layer6_stack_arithmetic_ir(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.CMP + 4] = 1.0
    x[..., _SetDim.OP_JSR] = 5.0
    x[..., _SetDim.HAS_SE] = 1.0
    x[..., _SetDim.EMBED_LO + 0] = 1.0
    x[..., _SetDim.EMBED_HI + 13] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 8] > 0.9
    assert y[_SetDim.OUTPUT_HI + 12] > 0.9


def test_layer6_jsr_sp_marker_decrement_still_handles_first_step():
    ffn = _StubFFN()
    _lower_layer6_stack_arithmetic_ir(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_SP] = 1.0
    x[..., _SetDim.CMP + 4] = 1.0
    x[..., _SetDim.OP_JSR] = 5.0
    x[..., _SetDim.HAS_SE] = 0.0
    x[..., _SetDim.EMBED_LO + 0] = 1.0
    x[..., _SetDim.EMBED_HI + 0] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 8] > 0.9
    assert y[_SetDim.OUTPUT_HI + 15] > 0.9


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


def test_layer6_ent_after_jsr_sp_byte0_fixup_lowers_to_reserved_unit():
    actual = _StubFFN(hidden_dim=L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT)

    end = _lower_layer6_ent_after_jsr_sp_byte0_fixup_ir(
        actual,
        100.0,
        _SetDim,
    )

    assert end == L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT
    assert len(_layer6_ent_after_jsr_sp_byte0_fixup_rules(100.0)) == 6
    assert actual.W_up[
        L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_START_UNIT,
        _SetDim.OP_ENT,
    ] != 0.0


def test_layer6_ent_after_jsr_sp_byte0_fixup_emits_e8():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _layer6_ent_after_jsr_sp_byte0_fixup_rules(100.0)
    )

    out = ir.symbolic_ffn({
        "OP_ENT": 5.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+15": 1.0,
    })

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+14"] > 0.0
    assert out["OUTPUT_LO+10"] < 0.0
    assert out["OUTPUT_HI+1"] < 0.0


def test_layer6_ent_after_jsr_bp_byte0_fixup_emits_f0():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _layer6_ent_after_jsr_sp_byte0_fixup_rules(100.0)
    )

    out = ir.symbolic_ffn({
        "OP_ENT": 5.0,
        "MARK_BP": 1.0,
        "HAS_SE": 1.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+8"] < 0.0
    assert out["OUTPUT_HI+1"] < 0.0


def test_layer6_ent_after_jsr_bp_byte1_fixup_emits_ff():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _layer6_ent_after_jsr_sp_byte0_fixup_rules(100.0)
    )

    out = ir.symbolic_ffn({
        "OP_ENT": 5.0,
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_0": 1.0,
        "HAS_SE": 1.0,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_layer6_ent_after_jsr_bp_high_byte_fixup_emits_00():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _layer6_ent_after_jsr_sp_byte0_fixup_rules(100.0)
    )

    for byte_index in ("BYTE_INDEX_1", "BYTE_INDEX_2"):
        out = ir.symbolic_ffn({
            "OP_ENT": 5.0,
            "IS_BYTE": 1.0,
            "H1+3": 1.0,
            byte_index: 1.0,
            "HAS_SE": 1.0,
        })

        assert out["OUTPUT_LO+0"] > 0.0
        assert out["OUTPUT_HI+0"] > 0.0
        assert out["OUTPUT_LO+1"] < 0.0
        assert out["OUTPUT_LO+15"] < 0.0


def test_layer6_ent_after_jsr_stack0_byte0_fixup_emits_00():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _layer6_ent_after_jsr_sp_byte0_fixup_rules(100.0)
    )

    out = ir.symbolic_ffn({
        "OP_ENT": 5.0,
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+2"] < 0.0
    assert out["OUTPUT_HI+1"] < 0.0


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


def test_layer6_branch_pc_byte1_override_symbolically_emits_target_high_byte():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer6_branch_pc_byte1_override_rules(100.0))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_BZ": 5.0,
        "CMP+4": 1.0,
        "CMP+5": 1.0,
        "FETCH_HI+2": 40.0,
        "OUTPUT_LO+0": 1.0,
        "OUTPUT_HI+0": 1.0,
        "CONST": 1.0,
    })

    assert out["OUTPUT_LO+1"] > 0.0
    assert out["OUTPUT_LO+0"] < 1.0
    assert out["OUTPUT_HI+0"] > 1.0


def test_layer6_branch_pc_byte1_override_handles_jsr_target_high_byte():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer6_branch_pc_byte1_override_rules(100.0))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OPCODE_BYTE_LO+3": 1.0,
        "OPCODE_BYTE_HI+0": 1.0,
        "FETCH_HI+2": 40.0,
        "OUTPUT_LO+0": 1.0,
        "OUTPUT_HI+0": 1.0,
        "CONST": 1.0,
    })

    assert out["OUTPUT_LO+1"] > 0.0
    assert out["OUTPUT_LO+0"] < 1.0
    assert out["OUTPUT_HI+0"] > 1.0


def test_layer6_branch_pc_byte1_override_blocks_untaken_bz():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer6_branch_pc_byte1_override_rules(100.0))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_BZ": 5.0,
        "CMP+4": 1.0,
        "CMP+5": 0.0,
        "FETCH_HI+2": 40.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_layer6_branch_pc_byte1_override_ir_lowers_into_reserved_band():
    actual = _StubFFN()

    end = _lower_layer6_branch_pc_byte1_override_ir(actual, 100.0, _SetDim)

    assert end == L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT
    assert len(_layer6_branch_pc_byte1_override_rules(100.0)) == 196
    assert torch.count_nonzero(
        actual.W_up[
            L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT:
            L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT
        ]
    )


def test_layer6_routing_bake_includes_branch_pc_byte1_override_band():
    actual = _StubFFN()

    _bake_layer6_routing_ffn(actual, 100.0, _SetDim)

    assert torch.count_nonzero(
        actual.W_up[
            L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT:
            L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT
        ]
    )
    assert torch.count_nonzero(
        actual.W_down[
            :,
            L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT:
            L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT,
        ]
    )


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
