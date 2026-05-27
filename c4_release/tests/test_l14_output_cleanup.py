"""Focused L14 cleanup weight checks."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.setup_helpers import (  # noqa: E402
    _set_layer14_add_byte1_high_zero_cleanup,
    _set_layer14_clear_addsub_temp_negative_residue,
    _set_layer14_clear_output_corruption,
)
from neural_vm.unified_compiler.ops.l14_ops import (  # noqa: E402
    _block_l14_jsr_ax_zero_on_stack0_bytes,
    _clear_l14_mem_generation_overbroad_sp_suppression,
    _disable_l14_stack0_jsr_hi0_default,
    _guard_l14_output_units_on_step_boundary,
    _layer14_alu_high_byte_relay_spec,
)
from neural_vm.vm_step import (  # noqa: E402
    _SetDim,
    _set_layer14_alu_nocarry_ax_bytes_zero,
    _set_layer14_clear_mem_marker_output,
    _set_layer14_jsr_ax_bytes_zero,
    _set_layer14_mem_generation,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 8):
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
        self.alibi_slopes = torch.zeros(num_heads)


def _projection(terms, row: torch.Tensor) -> torch.Tensor:
    slots = max(term.slot for term in terms) + 1
    projected = torch.zeros(slots)
    for term in terms:
        projected[term.slot] += row[term.dim] * term.weight
    return projected


def _attention_score(spec, query_row: torch.Tensor, key_row: torch.Tensor) -> float:
    q = _projection(spec.q, query_row)
    k = _projection(spec.k, key_row)
    return float(q @ k)


def _value_output(spec, key_row: torch.Tensor) -> torch.Tensor:
    value = _projection(spec.v, key_row)
    output = torch.zeros_like(key_row)
    for term in spec.o:
        output[term.out_dim] += value[term.slot] * term.weight
    return output


def _ax_byte0_query_row() -> torch.Tensor:
    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.IS_BYTE] = 1.0
    row[_SetDim.H1 + 1] = 1.0
    row[_SetDim.BYTE_INDEX_0] = 1.0
    return row


def _wide_alu_source_row(op_dim: int) -> torch.Tensor:
    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.MARK_AX] = 1.0
    row[op_dim] = 1.0
    row[_SetDim.AX_FULL_LO + 2] = 0.25
    row[_SetDim.AX_FULL_HI + 0] = 0.75
    return row


def _apply_stub_ffn(ffn: _StubFFN, x: torch.Tensor) -> torch.Tensor:
    up = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up)
    gate = x @ ffn.W_gate.t() + ffn.b_gate
    return x + (up * gate) @ ffn.W_down.t()


def test_l14_stack0_zero_cleanup_suppresses_pop_group_rows():
    ffn = _StubFFN()

    end = _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    assert end >= 2
    assert ffn.W_up[0, _SetDim.CMP + 3] < -1000.0
    assert ffn.W_up[1, _SetDim.CMP + 3] < -1000.0
    assert ffn.W_up[0, _SetDim.H1 + 4] < -1000.0
    assert ffn.W_up[1, _SetDim.H1 + 4] < -1000.0


def test_l14_stack0_zero_cleanup_suppresses_mem_marker_rows():
    ffn = _StubFFN()

    _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    assert ffn.W_up[0, _SetDim.MARK_MEM] < -1000.0
    assert ffn.W_up[1, _SetDim.MARK_MEM] < -1000.0


def test_l14_stack0_zero_cleanup_suppresses_mem_address_rows():
    ffn = _StubFFN()

    _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    mem_i = 4
    assert ffn.W_up[0, _SetDim.H3 + mem_i] < -1000.0
    assert ffn.W_up[1, _SetDim.H3 + mem_i] < -1000.0


def test_l14_stack0_zero_cleanup_suppresses_register_byte_rows():
    ffn = _StubFFN()

    _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    pc_i = 0
    ax_i = 1
    sp_i = 2
    for unit in (0, 1):
        assert ffn.W_up[unit, _SetDim.H1 + pc_i] < -1000.0
        assert ffn.W_up[unit, _SetDim.H1 + ax_i] < -1000.0
        assert ffn.W_up[unit, _SetDim.H1 + sp_i] < -1000.0


def test_l14_stack0_zero_cleanup_preserves_nonzero_nibble_support():
    ffn = _StubFFN()

    _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    assert ffn.W_up[0, _SetDim.OUTPUT_LO + 0] == 0.0
    assert torch.all(ffn.W_up[0, _SetDim.OUTPUT_LO + 1:_SetDim.OUTPUT_LO + 16] < 0.0)
    assert ffn.W_up[1, _SetDim.OUTPUT_HI + 0] == 0.0
    assert torch.all(ffn.W_up[1, _SetDim.OUTPUT_HI + 1:_SetDim.OUTPUT_HI + 16] < 0.0)


def test_l14_stack0_zero_cleanup_blocks_add_sub_low_nibble_rows():
    ffn = _StubFFN()

    _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)

    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.IS_BYTE] = 1.0
    row[_SetDim.H1 + 1] = 1.0
    row[_SetDim.BYTE_INDEX_0] = 0.97
    row[_SetDim.TEMP + 9] = 1.0
    row[_SetDim.OUTPUT_LO + 5] = 1.7e18
    row[_SetDim.OUTPUT_LO + 3] = -1.36e18
    row[_SetDim.OUTPUT_LO + 6] = -1.82e18

    assert float(ffn.W_up[0] @ row + ffn.b_up[0]) < 0.0


def test_l14_addsub_temp_clamp_removes_negative_dust_only():
    ffn = _StubFFN(hidden_dim=2)
    _set_layer14_clear_addsub_temp_negative_residue(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.TEMP + 8] = -2.0e-22
    x[..., _SetDim.TEMP + 9] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.TEMP + 8])) < 1e-28
    assert abs(float(y[_SetDim.TEMP + 9]) - 1.0) < 1e-6


def test_l14_add_byte1_high_zero_cleanup_blocks_tail_mul_signature():
    ffn = _StubFFN(hidden_dim=1)
    _set_layer14_add_byte1_high_zero_cleanup(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.97
    x[..., _SetDim.TEMP + 8] = 1.0
    x[..., _SetDim.OUTPUT_LO + 3] = 6300.0
    x[..., _SetDim.OUTPUT_HI + 0] = 8700.0
    x[..., _SetDim.OUTPUT_HI + 1] = 1000.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 3] == x[0, 0, _SetDim.OUTPUT_LO + 3]
    assert y[_SetDim.OUTPUT_HI + 0] > x[0, 0, _SetDim.OUTPUT_HI + 0]
    assert y[_SetDim.OUTPUT_HI + 1] < 0.0


def test_l14_add_byte1_high_zero_cleanup_requires_add_relay():
    ffn = _StubFFN(hidden_dim=1)
    _set_layer14_add_byte1_high_zero_cleanup(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.97
    x[..., _SetDim.OUTPUT_HI + 1] = 1000.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_HI + 1] == x[0, 0, _SetDim.OUTPUT_HI + 1]


def test_l14_add_byte1_high_zero_cleanup_blocks_local_addr_high_nibble():
    ffn = _StubFFN(hidden_dim=1)
    end = _set_layer14_add_byte1_high_zero_cleanup(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.9701380133628845
    x[..., _SetDim.TEMP + 8] = 0.3037
    x[..., _SetDim.AX_CARRY_HI + 15] = 3.174
    x[..., _SetDim.OUTPUT_LO + 15] = 9.0
    x[..., _SetDim.OUTPUT_HI + 15] = 9.0
    x[..., _SetDim.OUTPUT_HI + 0] = 25.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 15] == x[0, 0, _SetDim.OUTPUT_LO + 15]
    assert y[_SetDim.OUTPUT_HI + 15] == x[0, 0, _SetDim.OUTPUT_HI + 15]
    assert y[_SetDim.OUTPUT_HI + 0] == x[0, 0, _SetDim.OUTPUT_HI + 0]


def test_l14_boundary_guard_does_not_invert_mem_value_blockers():
    ffn = _StubFFN(hidden_dim=80)

    end = _set_layer14_clear_mem_marker_output(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)

    mem_value_row = torch.zeros(512)
    mem_value_row[_SetDim.CONST] = 1.0
    mem_value_row[_SetDim.OP_JSR] = 5.0
    mem_value_row[_SetDim.IS_BYTE] = 1.0
    mem_value_row[_SetDim.MEM_VAL_B0] = 1.0

    mem_marker_row = torch.zeros(512)
    mem_marker_row[_SetDim.CONST] = 1.0
    mem_marker_row[_SetDim.OP_JSR] = 5.0
    mem_marker_row[_SetDim.MARK_MEM] = 1.0

    assert float(ffn.W_up[0] @ mem_value_row + ffn.b_up[0]) < 0.0
    assert float(ffn.W_up[0] @ mem_marker_row + ffn.b_up[0]) > 0.0


def test_l14_boundary_guard_blocks_empty_rows_with_negative_output_residue():
    ffn = _StubFFN()

    end = _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)

    boundary_row = torch.zeros(512)
    boundary_row[_SetDim.CONST] = 1.0
    boundary_row[_SetDim.OUTPUT_LO + 1:_SetDim.OUTPUT_LO + 16] = -240.0

    stack0_row = torch.zeros(512)
    stack0_row[_SetDim.CONST] = 1.0
    stack0_row[_SetDim.H4 + 3] = 1.0

    assert float(ffn.W_up[0] @ boundary_row + ffn.b_up[0]) < 0.0
    assert float(ffn.W_up[0] @ stack0_row + ffn.b_up[0]) > 0.0


def test_l14_boundary_guard_blocks_next_stack0_marker_transition():
    ffn = _StubFFN()

    end = _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)

    row = torch.zeros(512)
    row[_SetDim.CONST] = -1.0
    row[_SetDim.H1 + 3] = 1.0
    row[_SetDim.BYTE_INDEX_3] = 1.0
    row[_SetDim.NEXT_STACK0] = 1.0
    row[_SetDim.OUTPUT_LO + 1:_SetDim.OUTPUT_LO + 16] = -220.0

    assert float(ffn.W_up[0] @ row + ffn.b_up[0]) < 0.0


def test_l14_stack0_jsr_hi0_default_is_removed_for_return_address_rows():
    ffn = _StubFFN(hidden_dim=80)

    end = _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)
    _disable_l14_stack0_jsr_hi0_default(ffn, _SetDim, 0, end)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.MARK_STACK0] = 1.0
    x[..., _SetDim.OP_JSR] = 11.0
    x[..., _SetDim.OUTPUT_HI + 8] = 3.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_HI + 0] == x[0, 0, _SetDim.OUTPUT_HI + 0]
    assert y[_SetDim.OUTPUT_HI + 8] == x[0, 0, _SetDim.OUTPUT_HI + 8]


def test_l14_stack0_jsr_hi0_default_no_longer_amplifies_zero_high_nibble():
    ffn = _StubFFN(hidden_dim=80)

    end = _set_layer14_clear_output_corruption(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)
    _disable_l14_stack0_jsr_hi0_default(ffn, _SetDim, 0, end)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.MARK_STACK0] = 1.0
    x[..., _SetDim.OP_JSR] = 11.0
    x[..., _SetDim.OUTPUT_HI + 0] = 8.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_HI + 0] == x[0, 0, _SetDim.OUTPUT_HI + 0]


def test_l14_jsr_ax_zero_blocks_stack0_return_address_rows():
    ffn = _StubFFN(hidden_dim=4)

    end = _set_layer14_jsr_ax_bytes_zero(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)
    _block_l14_jsr_ax_zero_on_stack0_bytes(ffn, _SetDim, 100.0, 0, end)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.00002
    x[..., _SetDim.STACK0_BYTE0] = 1.0
    x[..., _SetDim.OP_JSR] = 5.0
    x[..., _SetDim.OUTPUT_LO + 10] = 3.0
    x[..., _SetDim.OUTPUT_HI + 8] = 3.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 10] == x[0, 0, _SetDim.OUTPUT_LO + 10]
    assert y[_SetDim.OUTPUT_HI + 8] == x[0, 0, _SetDim.OUTPUT_HI + 8]
    assert y[_SetDim.OUTPUT_HI + 0] == x[0, 0, _SetDim.OUTPUT_HI + 0]


def test_l14_jsr_ax_zero_still_fires_on_ax_byte_rows():
    ffn = _StubFFN(hidden_dim=4)

    end = _set_layer14_jsr_ax_bytes_zero(ffn, 100.0, _SetDim)
    _guard_l14_output_units_on_step_boundary(ffn, _SetDim, 100.0, 0, end)
    _block_l14_jsr_ax_zero_on_stack0_bytes(ffn, _SetDim, 100.0, 0, end)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.00002
    x[..., _SetDim.OP_JSR] = 1.0
    x[..., _SetDim.OUTPUT_LO + 5] = 2.0
    x[..., _SetDim.OUTPUT_HI + 8] = 2.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 0] > x[0, 0, _SetDim.OUTPUT_LO + 0]
    assert y[_SetDim.OUTPUT_HI + 0] > x[0, 0, _SetDim.OUTPUT_HI + 0]
    assert y[_SetDim.OUTPUT_LO + 5] < x[0, 0, _SetDim.OUTPUT_LO + 5]
    assert y[_SetDim.OUTPUT_HI + 8] < x[0, 0, _SetDim.OUTPUT_HI + 8]


def test_l14_nocarry_zero_ignores_temp7_residue_on_add_rows():
    ffn = _StubFFN()
    _set_layer14_alu_nocarry_ax_bytes_zero(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.97
    x[..., _SetDim.TEMP + 7] = 0.304
    x[..., _SetDim.OUTPUT_LO + 1] = 2.0
    x[..., _SetDim.OUTPUT_HI + 0] = 2.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 1]) - 2.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_LO + 0])) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_HI + 0]) - 2.0) < 1e-3


def test_l14_nocarry_zero_still_fires_with_full_temp7_relay():
    ffn = _StubFFN()
    _set_layer14_alu_nocarry_ax_bytes_zero(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.TEMP + 7] = 1.0
    x[..., _SetDim.OUTPUT_LO + 5] = 2.0
    x[..., _SetDim.OUTPUT_HI + 2] = 2.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 0] > 1.5
    assert y[_SetDim.OUTPUT_HI + 0] > 1.5
    assert y[_SetDim.OUTPUT_LO + 5] < 0.0
    assert y[_SetDim.OUTPUT_HI + 2] < 0.0


def test_l15_wide_alu_relay_blocks_add_sub_rows():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}
    k_terms = {(term.slot, term.dim, term.weight) for term in spec.k}

    assert (35, _SetDim.TEMP + 8, 10000.0) in q_terms
    assert (35, _SetDim.TEMP + 9, 10000.0) in q_terms
    assert (35, _SetDim.CONST, -20.0) in k_terms
    assert (35, _SetDim.MARK_AX, -10000.0) in k_terms
    assert (35, _SetDim.OP_MUL, 2000.0) in k_terms
    assert (35, _SetDim.OP_SHL, 2000.0) in k_terms


def test_l15_wide_alu_relay_requires_mul_or_shl_source():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}
    k_terms = {(term.slot, term.dim, term.weight) for term in spec.k}

    assert (36, _SetDim.IS_BYTE, 1000.0) in q_terms
    assert (36, _SetDim.H1 + 1, 1000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_0, 1000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_1, -3000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_2, -3000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_3, -3000.0) in q_terms
    assert (36, _SetDim.MARK_AX, -1000.0) in q_terms
    assert (36, _SetDim.CONST, -100.0) in k_terms
    assert (36, _SetDim.OP_MUL, 200.0) in k_terms
    assert (36, _SetDim.OP_SHL, 200.0) in k_terms


def test_l15_wide_alu_relay_has_non_ax_query_blockers():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}

    assert (36, _SetDim.MARK_PC, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 0, -6000.0) in q_terms
    assert (36, _SetDim.MARK_SP, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 2, -6000.0) in q_terms
    assert (36, _SetDim.H4 + 2, -6000.0) in q_terms
    assert (36, _SetDim.MARK_BP, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 3, -6000.0) in q_terms
    assert (36, _SetDim.H4 + 3, -6000.0) in q_terms
    assert (36, _SetDim.MARK_STACK0, -6000.0) in q_terms
    assert (36, _SetDim.STACK0_BYTE0, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 4, -6000.0) in q_terms
    assert (36, _SetDim.H3 + 4, -6000.0) in q_terms
    assert (36, _SetDim.H4 + 4, -6000.0) in q_terms
    assert (36, _SetDim.MARK_MEM, -6000.0) in q_terms


def test_l15_wide_alu_relay_scores_intended_ax_byte0_mul_and_shl_sources():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    query = _ax_byte0_query_row()

    for op_dim in (_SetDim.OP_MUL, _SetDim.OP_SHL):
        source = _wide_alu_source_row(op_dim)

        assert _attention_score(spec, query, source) > 0.0

        output = _value_output(spec, source)
        assert output[_SetDim.OUTPUT_LO + 2] == 5.0
        assert output[_SetDim.OUTPUT_HI + 0] == 15.0


def test_l15_wide_alu_relay_allows_mul_sources_with_temp8_residue():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    query = _ax_byte0_query_row()
    query[_SetDim.TEMP + 8] = 0.305

    mul_source = _wide_alu_source_row(_SetDim.OP_MUL)
    mul_source[_SetDim.OP_MUL] = 5.0
    non_source = _wide_alu_source_row(_SetDim.OP_MUL)
    non_source[_SetDim.OP_MUL] = 0.0

    assert _attention_score(spec, query, mul_source) > 0.0
    assert _attention_score(spec, query, non_source) < 0.0


def test_l15_wide_alu_relay_blocks_stack0_and_bp_byte_positions():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    source = _wide_alu_source_row(_SetDim.OP_MUL)

    stack0_byte0 = _ax_byte0_query_row()
    stack0_byte0[_SetDim.H4 + 3] = 1.0
    stack0_byte0[_SetDim.STACK0_BYTE0] = 1.0

    bp_byte0 = _ax_byte0_query_row()
    bp_byte0[_SetDim.H1 + 3] = 1.0
    bp_byte0[_SetDim.H4 + 3] = 1.0

    assert _attention_score(spec, stack0_byte0, source) < 0.0
    assert _attention_score(spec, bp_byte0, source) < 0.0


def test_l15_wide_alu_relay_blocks_mem_byte_positions():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    source = _wide_alu_source_row(_SetDim.OP_MUL)

    mem_byte0 = _ax_byte0_query_row()
    mem_byte0[_SetDim.H1 + 4] = 1.0
    mem_byte0[_SetDim.H3 + 4] = 1.0
    mem_byte0[_SetDim.H4 + 4] = 1.0

    assert _attention_score(spec, mem_byte0, source) < 0.0


def test_l15_wide_alu_relay_blocks_pc_and_sp_byte_positions():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    source = _wide_alu_source_row(_SetDim.OP_MUL)

    pc_byte0 = _ax_byte0_query_row()
    pc_byte0[_SetDim.H1 + 0] = 1.0

    sp_byte0 = _ax_byte0_query_row()
    sp_byte0[_SetDim.H1 + 2] = 1.0
    sp_byte0[_SetDim.H4 + 2] = 1.0

    assert _attention_score(spec, pc_byte0, source) < 0.0
    assert _attention_score(spec, sp_byte0, source) < 0.0


def test_l15_wide_alu_relay_blocks_non_source_ax_rows():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    query = _ax_byte0_query_row()
    non_source = _wide_alu_source_row(_SetDim.OP_MUL)
    non_source[_SetDim.OP_MUL] = 0.0

    assert _attention_score(spec, query, non_source) < 0.0


def test_l14_mem_value_selector_keeps_sp_neutral_for_jsr_sources():
    attn = _StubAttn()
    head_dim = attn.W_q.shape[0] // attn.num_heads

    _set_layer14_mem_generation(attn, 100.0, _SetDim, head_dim)
    _clear_l14_mem_generation_overbroad_sp_suppression(attn, _SetDim, head_dim)

    ax_i = 1
    sp_i = 2
    bp_i = 3
    mem_i = 4
    for head in range(4, 8):
        base = head * head_dim
        assert attn.W_k[base + 36, _SetDim.H1 + sp_i] == 0.0
        assert attn.W_k[base + 36, _SetDim.H1 + ax_i] > 0.0
        assert attn.W_k[base + 36, _SetDim.H4 + bp_i] < 0.0
        assert attn.W_k[base + 36, _SetDim.MARK_MEM] == 0.0
        assert attn.W_k[base + 36, _SetDim.H3 + mem_i] == 0.0
        assert attn.W_q[base + 38, _SetDim.MARK_BP] <= -5000.0
        assert attn.W_q[base + 38, _SetDim.MARK_SP] <= -5000.0
        assert attn.W_q[base + 38, _SetDim.H1 + bp_i] <= -5000.0
        assert attn.W_q[base + 38, _SetDim.H4 + bp_i] <= -5000.0
        assert attn.W_q[base + 38, _SetDim.MARK_MEM] <= -15000.0
        assert attn.W_q[base + 38, _SetDim.BYTE_INDEX_0] <= -15000.0
        assert attn.W_q[base + 38, _SetDim.BYTE_INDEX_1] <= -15000.0
        assert attn.W_q[base + 38, _SetDim.BYTE_INDEX_2] <= -15000.0
        own_value_idx = head - 4
        for idx, dim in enumerate(
            (
                _SetDim.MEM_VAL_B0,
                _SetDim.MEM_VAL_B1,
                _SetDim.MEM_VAL_B2,
                _SetDim.MEM_VAL_B3,
            )
        ):
            if idx == own_value_idx:
                assert attn.W_q[base + 38, dim] == 0.0
            else:
                assert attn.W_q[base + 38, dim] <= -15000.0
        assert attn.W_q[base + 44, _SetDim.OP_ENT] > 0.0
        assert attn.W_k[base + 44, _SetDim.OP_JSR] > 0.0
        assert attn.W_k[base + 44, _SetDim.H1 + bp_i] > 0.0
        assert attn.W_k[
            base + 44,
            (
                _SetDim.BYTE_INDEX_0,
                _SetDim.BYTE_INDEX_1,
                _SetDim.BYTE_INDEX_2,
                _SetDim.BYTE_INDEX_3,
            )[own_value_idx],
        ] > 0.0
        assert attn.W_q[base + 45, _SetDim.CONST] < 0.0
        assert attn.W_q[
            base + 45,
            (
                _SetDim.MEM_VAL_B0,
                _SetDim.MEM_VAL_B1,
                _SetDim.MEM_VAL_B2,
                _SetDim.MEM_VAL_B3,
            )[own_value_idx],
        ] > 0.0
        assert attn.W_k[base + 45, _SetDim.CONST] > 0.0


def test_l14_ent_addr_heads_source_bp_for_frame_store():
    attn = _StubAttn()
    head_dim = attn.W_q.shape[0] // attn.num_heads

    _set_layer14_mem_generation(attn, 100.0, _SetDim, head_dim)
    _clear_l14_mem_generation_overbroad_sp_suppression(attn, _SetDim, head_dim)

    bp_i = 3
    for head in range(4):
        base = head * head_dim
        assert attn.W_q[base + 39, _SetDim.OP_ENT] > 0.0
        assert attn.W_q[base + 39, _SetDim.HAS_SE] > 0.0
        assert attn.W_q[base + 39, _SetDim.CONST] < 0.0
        if head == 0:
            assert attn.W_q[base + 39, _SetDim.IS_BYTE] < 0.0
            assert attn.W_k[base + 39, _SetDim.MARK_BP] > 0.0
            assert attn.W_q[base + 41, _SetDim.MARK_MEM] == 0.0
        else:
            assert attn.W_k[base + 39, _SetDim.H1 + bp_i] == 0.0
            assert attn.W_q[base + 40, _SetDim.OP_ENT] > 0.0
            assert attn.W_q[base + 40, _SetDim.HAS_SE] == 0.0
            assert attn.W_q[base + 40, _SetDim.CONST] == 0.0
            for dim in (
                _SetDim.BYTE_INDEX_0,
                _SetDim.BYTE_INDEX_1,
                _SetDim.BYTE_INDEX_2,
            ):
                allowed_query_dim = (
                    _SetDim.BYTE_INDEX_0,
                    _SetDim.BYTE_INDEX_1,
                    _SetDim.BYTE_INDEX_2,
                )[head - 1]
                if dim == allowed_query_dim:
                    assert attn.W_q[base + 40, dim] == 0.0
                else:
                    assert attn.W_q[base + 40, dim] < 0.0
            assert attn.W_q[base + 40, _SetDim.BYTE_INDEX_3] < 0.0
            assert attn.W_k[base + 40, _SetDim.H1 + bp_i] > 0.0
            assert attn.W_k[
                base + 40,
                (
                    _SetDim.BYTE_INDEX_1,
                    _SetDim.BYTE_INDEX_2,
                    _SetDim.BYTE_INDEX_3,
                )[head - 1],
            ] > 0.0
            assert attn.W_q[base + 41, _SetDim.MARK_MEM] < 0.0
            assert attn.W_k[base + 41, _SetDim.CONST] > 0.0
            assert attn.W_q[base + 42, _SetDim.CONST] == 0.0
            assert attn.W_q[base + 42, _SetDim.OP_ENT] == 0.0
            assert attn.W_q[
                base + 42,
                (
                    _SetDim.BYTE_INDEX_0,
                    _SetDim.BYTE_INDEX_1,
                    _SetDim.BYTE_INDEX_2,
                )[head - 1],
            ] == 0.0
            assert attn.W_k[base + 42, _SetDim.CONST] == 0.0
            assert attn.W_q[base + 43, _SetDim.CONST] < 0.0
            assert attn.W_q[base + 43, _SetDim.H3 + 4] > 0.0
            assert attn.W_k[base + 43, _SetDim.CONST] > 0.0


def test_l14_si_addr_heads_block_current_store_stack0_rows():
    attn = _StubAttn()
    head_dim = attn.W_q.shape[0] // attn.num_heads

    _set_layer14_mem_generation(attn, 100.0, _SetDim, head_dim)
    _clear_l14_mem_generation_overbroad_sp_suppression(attn, _SetDim, head_dim)

    bp_i = 3
    for head in range(4):
        base = head * head_dim
        assert attn.W_k[base + 2, _SetDim.MEM_STORE] < 0.0
        if head == 0:
            assert attn.W_k[base + 2, _SetDim.MARK_STACK0] == 0.0
            assert attn.W_k[base + 2, _SetDim.STACK0_BYTE0] > 0.0
            assert attn.W_k[base + 2, _SetDim.L1H4 + bp_i] == 0.0
            assert attn.W_k[base + 2, _SetDim.H1 + bp_i] == 0.0
        else:
            assert attn.W_k[base + 2, _SetDim.MARK_STACK0] < 0.0


def test_l14_addr_heads_block_mem_value_lanes():
    attn = _StubAttn()
    head_dim = attn.W_q.shape[0] // attn.num_heads

    _set_layer14_mem_generation(attn, 100.0, _SetDim, head_dim)
    _clear_l14_mem_generation_overbroad_sp_suppression(attn, _SetDim, head_dim)

    sp_i = 2
    for head in range(4):
        base = head * head_dim
        assert attn.W_q[base + 38, _SetDim.H1 + sp_i] < 0.0
        for dim in (
            _SetDim.MEM_VAL_B0,
            _SetDim.MEM_VAL_B1,
            _SetDim.MEM_VAL_B2,
            _SetDim.MEM_VAL_B3,
        ):
            assert attn.W_q[base + 38, dim] < 0.0


def test_l14_mem_generation_zero_nibbles_have_positive_margin():
    attn = _StubAttn()
    head_dim = attn.W_q.shape[0] // attn.num_heads

    _set_layer14_mem_generation(attn, 100.0, _SetDim, head_dim)
    _clear_l14_mem_generation_overbroad_sp_suppression(attn, _SetDim, head_dim)

    for head in range(8):
        base = head * head_dim
        assert attn.W_o[_SetDim.OUTPUT_LO + 0, base + 0] == -0.5
        assert attn.W_o[_SetDim.OUTPUT_HI + 0, base + 0] == -0.5
