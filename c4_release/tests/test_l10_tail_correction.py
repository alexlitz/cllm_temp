"""Regression coverage for declarative L10 tail correction rules."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _layer10_bp_byte_passthrough_head_spec,
    _layer10_byte_passthrough_ir,
    _layer10_nonbitwise_stack0_byte_relay_head_spec,
    _layer10_stack0_byte_relay_head_spec,
    _layer10_stack0_persistence_head_spec,
    make_l10_post_ops_combined,
    _suppress_ffn_on_step_boundary,
    _tail_bit32_result_correction_rules,
)
from neural_vm.vm_step import (  # noqa: E402
    BitwiseBytePropagationPostOp,
    CarryPropagationPostOp,
    _SetDim,
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


def _tail_rules_ir(*names: str) -> CompilerIR:
    ir = CompilerIR()
    for name in names:
        ir.layer(0).ffn.append(_tail_rule(name))
    return ir


def _tail_rules_with_prefix(prefix: str):
    rules = [
        rule
        for rule in _tail_bit32_result_correction_rules()
        if rule.name.startswith(prefix)
    ]
    if not rules:
        raise AssertionError(f"missing tail rules with prefix {prefix}")
    return rules


def _tail_prefix_ir(prefix: str) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_tail_rules_with_prefix(prefix))
    return ir


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 2048):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _apply_stub_ffn(ffn: _StubFFN, x: torch.Tensor) -> torch.Tensor:
    up = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up)
    gate = x @ ffn.W_gate.t() + ffn.b_gate
    return x + (up * gate) @ ffn.W_down.t()


def _projection_weights(spec, section: str, slot: int, dim: int) -> list[float]:
    return [
        write.weight
        for write in getattr(spec, section)
        if write.slot == slot and write.dim == dim
    ]


def _has_output(spec, out_dim: int, *, positive: bool) -> bool:
    return any(
        write.out_dim == out_dim
        and ((write.weight > 0.0) if positive else (write.weight < 0.0))
        for write in spec.o
    )


def test_l10_combined_post_ops_block_pc_byte_rows_with_negative_output_residue():
    ffn = _StubFFN()
    op = make_l10_post_ops_combined()
    op.bake_fn(ffn, {"H1": _SetDim.H1}, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 0] = 1.0
    x[..., _SetDim.H1 + 1] = 1e-9
    x[..., _SetDim.BYTE_INDEX_0] = 0.97
    x[..., _SetDim.OUTPUT_LO + 0] = -233.0
    x[..., _SetDim.OUTPUT_LO + 1] = 234.0
    x[..., _SetDim.OUTPUT_HI + 0] = 235.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 0]) - -233.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_LO + 1]) - 234.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_HI + 0]) - 235.0) < 1e-3


def test_step_boundary_guard_preserves_negative_structural_blockers():
    ffn = _StubFFN(hidden_dim=1)
    ffn.W_up[0, _SetDim.IS_BYTE] = -100_000_000.0
    ffn.W_up[0, _SetDim.MARK_STACK0] = 100.0

    _suppress_ffn_on_step_boundary(ffn, _SetDim, 100.0)

    assert ffn.W_up[0, _SetDim.IS_BYTE] == -100_000_000.0
    assert ffn.W_up[0, _SetDim.MARK_STACK0] > 100.0


def test_l10_combined_post_ops_do_not_rerun_late_add_carry():
    ffn = _StubFFN()
    op = make_l10_post_ops_combined()
    op.bake_fn(ffn, {"H1": _SetDim.H1}, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.HAS_SE] = 0.9984797239303589
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.9701380133628845
    x[..., _SetDim.BYTE_INDEX_1] = 0.013296706601977348
    x[..., _SetDim.TEMP + 8] = 1.304825782775879
    x[..., _SetDim.TEMP + 10] = 0.9962686896324158
    x[..., _SetDim.CARRY + 1] = 2.0
    x[..., _SetDim.CARRY + 3] = 103.5
    x[..., _SetDim.OUTPUT_LO + 0] = 2.363e14
    x[..., _SetDim.OUTPUT_LO + 1] = 8.193e12
    x[..., _SetDim.OUTPUT_LO + 2] = -2.036e13
    x[..., _SetDim.OUTPUT_LO + 3] = 2.199e13
    x[..., _SetDim.OUTPUT_LO + 4] = -2.075e13
    x[..., _SetDim.OUTPUT_HI + 0] = 4.041e12

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.CARRY + 3]) - 103.5) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_LO + 0]) - 2.363e14) < 1e7
    assert abs(float(y[_SetDim.OUTPUT_LO + 3]) - 2.199e13) < 1e7


def test_tail_mem_store_local_offset_rule_ignores_negative_output_residue():
    rules = _tail_rules_with_prefix("tail_mem_store_addr0_e0_from_local_offset_exact")
    ffn = _StubFFN(hidden_dim=len(rules))
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ir.lower_ffn(
        ffn,
        {name: getattr(_SetDim, name) for name in dir(_SetDim) if name.isupper()},
        S=100.0,
    )

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.HAS_SE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.OUTPUT_LO + 8] = -6.0e19
    x[..., _SetDim.OUTPUT_HI + 14] = 1.0e20

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert torch.isfinite(y).all()
    assert y[_SetDim.OUTPUT_LO + 8] == x[0, 0, _SetDim.OUTPUT_LO + 8]
    assert y[_SetDim.OUTPUT_HI + 14] == x[0, 0, _SetDim.OUTPUT_HI + 14]


def test_l10_bitwise_byte_post_op_blocks_pc_byte_span_pollution():
    ffn = BitwiseBytePropagationPostOp(512, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 0] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 1.0
    x[..., _SetDim.TEMP + 4] = 1.0
    x[..., _SetDim.ALU_LO + 0] = 1.0
    x[..., _SetDim.OUTPUT_LO + 1] = 234.0

    y = ffn(x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 1]) - 234.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_LO + 0])) < 1e-3


def test_l10_bitwise_byte_post_op_blocks_add_carry_row_residue():
    ffn = BitwiseBytePropagationPostOp(512, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 1.0
    x[..., _SetDim.TEMP + 8] = 1.0
    x[..., _SetDim.TEMP + 10] = 0.996
    x[..., _SetDim.CARRY + 1] = 2.0
    x[..., _SetDim.OUTPUT_LO + 0] = -132.0
    x[..., _SetDim.OUTPUT_LO + 1] = 134.0
    x[..., _SetDim.OUTPUT_HI + 0] = 321.0
    x[..., _SetDim.OUTPUT_HI + 1] = 13.0
    x[..., _SetDim.ALU_HI + 0] = 6.0

    y = ffn(x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 0]) - -132.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_LO + 1]) - 134.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_HI + 0]) - 321.0) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_HI + 1]) - 13.0) < 1e-3


def test_l10_bitwise_byte_post_op_still_fires_on_bitwise_row():
    ffn = BitwiseBytePropagationPostOp(512, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 1.0
    x[..., _SetDim.TEMP + 4] = 1.0
    x[..., _SetDim.OUTPUT_LO + 3] = 234.0
    x[..., _SetDim.ALU_LO + 1] = 1.0

    y = ffn(x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 1] > 100.0
    assert y[_SetDim.OUTPUT_LO + 3] < -100.0


def test_l10_bitwise_byte_post_op_allows_temp10_on_bitwise_row():
    ffn = BitwiseBytePropagationPostOp(512, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 1.0
    x[..., _SetDim.TEMP + 5] = 1.0
    x[..., _SetDim.TEMP + 10] = 1.0
    x[..., _SetDim.OUTPUT_LO + 0] = 2.5
    x[..., _SetDim.ALU_LO + 15] = 2.7

    y = ffn(x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 15] > 0.0
    assert y[_SetDim.OUTPUT_LO + 0] < 2.5


def test_l10_carry_post_op_requires_real_carry_relay():
    ffn = CarryPropagationPostOp(512, 100.0, byte_idx=0, cascade=False)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.97
    x[..., _SetDim.OUTPUT_LO + 0] = 8.8
    x[..., _SetDim.OUTPUT_LO + 1] = -8.8
    x[..., _SetDim.OUTPUT_HI + 0] = 8.8
    x[..., _SetDim.OUTPUT_HI + 1] = -8.8

    y = ffn(x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 0]) - 8.8) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_LO + 1]) - -8.8) < 1e-3
    assert abs(float(y[_SetDim.OUTPUT_HI + 0]) - 8.8) < 1e-3


def test_l10_carry_post_op_still_fires_with_real_carry_relay():
    ffn = CarryPropagationPostOp(512, 100.0, byte_idx=0, cascade=False)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 1.0
    x[..., _SetDim.CARRY + 1] = 2.0
    x[..., _SetDim.OUTPUT_LO + 0] = 2.0
    x[..., _SetDim.OUTPUT_HI + 0] = 2.0

    y = ffn(x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 1] > 10.0
    assert y[_SetDim.OUTPUT_LO + 0] < -10.0
    assert y[_SetDim.OUTPUT_HI + 0] > 10.0


def test_tail_bit32_rules_all_block_pc_byte_span():
    for rule in _tail_bit32_result_correction_rules():
        if (
            rule.name == "tail_clear_output_after_byte3"
            or rule.name.startswith(
                "tail_pc_byte1_01_from_long_initial_pc_exact"
            )
            or rule.name.startswith(
                "tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact"
            )
            or rule.name.startswith(
                "tail_pc_byte0_12_from_initial_jmp_exact"
            )
            or rule.name.startswith(
                "tail_pc_byte0_1a_from_taken_branch_index3_exact"
            )
        ):
            continue
        assert any(
            term.dim.name == "H1"
            and term.dim.offset == 0
            and term.weight <= -1_000_000.0
            for term in rule.conditions
        ), rule.name


def test_tail_bit32_rules_block_step_end_transition_except_clear():
    for rule in _tail_bit32_result_correction_rules():
        if rule.name == "tail_clear_output_before_step_end":
            continue
        assert any(
            term.dim.name == "NEXT_SE"
            and term.weight <= -1_000_000.0
            for term in rule.conditions
        ), rule.name


def test_layer10_stack0_store_relay_specs_key_from_store_rows():
    for spec in (
        _layer10_stack0_byte_relay_head_spec(_SetDim, 100.0),
        _layer10_nonbitwise_stack0_byte_relay_head_spec(_SetDim, 100.0),
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "k", 1, _SetDim.MEM_STORE)
        )
        assert any(
            weight < 0.0
            for weight in _projection_weights(spec, "k", 1, _SetDim.MARK_MEM)
        )
        assert _has_output(spec, _SetDim.ALU_LO + 0, positive=True)
        assert _has_output(spec, _SetDim.ALU_HI + 0, positive=True)


def test_layer10_stack0_relay_specs_key_direct_stack_bytes():
    for spec in (
        _layer10_stack0_byte_relay_head_spec(_SetDim, 100.0),
        _layer10_nonbitwise_stack0_byte_relay_head_spec(_SetDim, 100.0),
    ):
        for slot, dim in (
            (31, _SetDim.STACK0_BYTE1),
            (32, _SetDim.STACK0_BYTE2),
            (34, _SetDim.STACK0_BYTE3),
        ):
            assert any(
                weight > 0.0
                for weight in _projection_weights(spec, "k", slot, dim)
            ), (slot, dim)


def test_layer10_stack0_store_route_reads_current_ax_bytes():
    spec = _layer10_stack0_persistence_head_spec(_SetDim, 100.0)

    for q_slot, target_dim, source_byte_dim in (
        (7, _SetDim.MARK_STACK0, _SetDim.BYTE_INDEX_0),
        (8, _SetDim.STACK0_BYTE0, _SetDim.BYTE_INDEX_1),
        (9, _SetDim.STACK0_BYTE1, _SetDim.BYTE_INDEX_2),
        (10, _SetDim.STACK0_BYTE2, _SetDim.BYTE_INDEX_3),
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "q", q_slot, target_dim)
        )
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "k", q_slot, source_byte_dim)
        )

    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "k", 11, _SetDim.H1 + 1)
    )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "q", 11, _SetDim.MEM_STORE)
    )
    assert any(
        weight < 0.0
        for weight in _projection_weights(spec, "q", 11, _SetDim.MEM_ADDR_SRC)
    )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "q", 11, _SetDim.CMP + 3)
    )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "q", 11, _SetDim.HAS_SE)
    )
    assert any(
        weight < 0.0
        for weight in _projection_weights(spec, "q", 11, _SetDim.CONST)
    )


def test_layer10_stack0_persistence_direct_route_blocks_binary_pop_rows():
    spec = _layer10_stack0_persistence_head_spec(_SetDim, 100.0)

    for slot, target_dim in (
        (4, _SetDim.STACK0_BYTE0),
        (5, _SetDim.STACK0_BYTE1),
        (6, _SetDim.STACK0_BYTE2),
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "q", slot, target_dim)
        )
        assert any(
            weight < 0.0
            for weight in _projection_weights(spec, "q", slot, _SetDim.CMP + 3)
        )

    state = {
        _SetDim.STACK0_BYTE0: 0.9734056,
        _SetDim.STACK0_BYTE1: 0.0132971,
        _SetDim.STACK0_BYTE2: 5.9604645e-7,
        _SetDim.CMP + 3: 4.0,
    }

    def q_score(slot: int) -> float:
        return sum(
            write.weight * state.get(write.dim, 0.0)
            for write in spec.q
            if write.slot == slot
        )

    # This mirrors the strict-audit ADD row that predicts STACK0 byte 1 from
    # the preceding STACK0 byte-0 position.  Binary-pop rows must not carry the
    # old stack byte through this persistence route.
    assert q_score(4) < 0.0
    assert q_score(5) < 0.0
    assert q_score(6) < 0.0


def test_layer10_stack0_store_route_dominates_partial_mem_store_residue():
    spec = _layer10_stack0_persistence_head_spec(_SetDim, 100.0)

    state = {
        _SetDim.CONST: 1.0,
        _SetDim.HAS_SE: 0.997859,
        _SetDim.CMP + 3: 4.0,
        _SetDim.MEM_STORE: 1.52938,
        _SetDim.STACK0_BYTE0: 0.973406,
        _SetDim.STACK0_BYTE1: 0.0132971,
    }

    def q_score(slot: int) -> float:
        return sum(
            write.weight * state.get(write.dim, 0.0)
            for write in spec.q
            if write.slot == slot
        )

    # Predicting STACK0 byte 1 happens at the STACK0 byte-0 position.  On SI/SC
    # store rows the AX-byte store route must beat ordinary STACK0 persistence,
    # otherwise the previous address byte leaks into STACK0 byte 1.
    assert q_score(8) > q_score(4)


def test_layer10_stack0_store_route_blocks_stack_source_residue():
    spec = _layer10_stack0_persistence_head_spec(_SetDim, 100.0)

    state = {
        _SetDim.CONST: 1.0,
        _SetDim.HAS_SE: 0.99794,
        _SetDim.CMP + 3: 4.0,
        _SetDim.MEM_STORE: 0.40725404,
        _SetDim.MEM_ADDR_SRC: 0.40725404,
        _SetDim.MARK_STACK0: 1.0,
    }

    def q_score(slot: int) -> float:
        return sum(
            write.weight * state.get(write.dim, 0.0)
            for write in spec.q
            if write.slot == slot
        )

    assert q_score(11) < 0.0

    state[_SetDim.MEM_ADDR_SRC] = 0.0
    assert q_score(11) > 0.0


def test_layer10_bp_head_isolates_stack_source_top_store_ax_byte0():
    spec = _layer10_bp_byte_passthrough_head_spec(_SetDim, 100.0)

    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "k", 40, _SetDim.H1 + 1)
    )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "k", 41, _SetDim.BYTE_INDEX_0)
    )
    for slot, byte_index_dim in (
        (43, _SetDim.BYTE_INDEX_1),
        (45, _SetDim.BYTE_INDEX_2),
        (47, _SetDim.BYTE_INDEX_3),
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "k", slot, byte_index_dim)
        )

    top_store_state = {
        _SetDim.CONST: 1.0,
        _SetDim.MARK_STACK0: 1.0,
        _SetDim.HAS_SE: 0.998318076133728,
        _SetDim.CMP + 3: 4.0,
        _SetDim.MEM_STORE: 0.40725404024124146,
        _SetDim.MEM_ADDR_SRC: 0.40725404024124146,
        _SetDim.ADDR_B0_LO + 0: 1.000007152557373,
        _SetDim.ADDR_B0_HI + 14: 1.0,
    }

    def projection_score(writes, slot: int, state: dict[int, float]) -> float:
        return sum(
            write.weight * state.get(write.dim, 0.0)
            for write in writes
            if write.slot == slot
        )

    q40 = projection_score(spec.q, 40, top_store_state)
    q41 = projection_score(spec.q, 41, top_store_state)
    assert q40 > 0.0
    assert q41 > 0.0

    stale_e8_marker_state = {
        _SetDim.CONST: 1.0,
        _SetDim.MARK_STACK0: 1.0,
        _SetDim.HAS_SE: 0.997939944267273,
        _SetDim.CMP + 3: 4.0,
        _SetDim.MEM_STORE: 0.40725404024124146,
        _SetDim.MEM_ADDR_SRC: 0.40725404024124146,
        _SetDim.ADDR_B0_LO + 8: 1.0,
        _SetDim.ADDR_B0_LO + 0: 7.191520126070827e-06,
        _SetDim.ADDR_B0_HI + 14: 6.436998295297086e-16,
        _SetDim.ADDR_B0_HI + 15: 1.9453605091257486e-06,
        _SetDim.H1 + 3: 0.003357573179528117,
    }
    assert projection_score(spec.q, 40, stale_e8_marker_state) < 0.0
    assert projection_score(spec.q, 41, stale_e8_marker_state) < 0.0

    ax_byte0 = {
        _SetDim.H1 + 1: 1.0,
        _SetDim.BYTE_INDEX_0: 0.9701380133628845,
    }
    ax_byte1 = {
        _SetDim.H1 + 1: 1.0,
        _SetDim.BYTE_INDEX_0: 0.013296706601977348,
    }
    score_byte0 = (
        q40 * projection_score(spec.k, 40, ax_byte0)
        + q41 * projection_score(spec.k, 41, ax_byte0)
    )
    score_byte1 = (
        q40 * projection_score(spec.k, 40, ax_byte1)
        + q41 * projection_score(spec.k, 41, ax_byte1)
    )
    assert score_byte0 > score_byte1

    stack0_byte1_state = {
        _SetDim.CONST: 1.0,
        _SetDim.STACK0_BYTE0: 0.9734055995941162,
        _SetDim.HAS_SE: 0.9983121752738953,
        _SetDim.CMP + 3: 4.0,
        _SetDim.MEM_STORE: 1.5293787717819214,
        _SetDim.MEM_ADDR_SRC: 1.5293787717819214,
        _SetDim.ADDR_B0_LO + 0: 0.999792218208313,
        _SetDim.ADDR_B0_HI + 14: 0.9997877478599548,
    }
    q42 = projection_score(spec.q, 42, stack0_byte1_state)
    q43 = projection_score(spec.q, 43, stack0_byte1_state)
    assert q42 > 0.0
    assert q43 > 0.0

    ax_byte1 = {
        _SetDim.H1 + 1: 1.0,
        _SetDim.BYTE_INDEX_1: 0.9701374769210815,
    }
    ax_byte0_for_byte1_route = {
        _SetDim.H1 + 1: 1.0,
        _SetDim.BYTE_INDEX_1: 0.013296706601977348,
    }
    score_ax_byte1 = (
        q42 * projection_score(spec.k, 42, ax_byte1)
        + q43 * projection_score(spec.k, 43, ax_byte1)
    )
    score_ax_byte0 = (
        q42 * projection_score(spec.k, 42, ax_byte0_for_byte1_route)
        + q43 * projection_score(spec.k, 43, ax_byte0_for_byte1_route)
    )
    assert score_ax_byte1 > score_ax_byte0

    bp_byte3_store_residue = {
        _SetDim.CONST: 1.0,
        _SetDim.H1 + 3: 0.9999992847442627,
        _SetDim.HAS_SE: 0.9983298778533936,
        _SetDim.CMP + 3: 3.003380298614502,
        _SetDim.MEM_STORE: 1.6414873600006104,
        _SetDim.MEM_ADDR_SRC: 1.6414873600006104,
        _SetDim.ADDR_B0_LO + 0: 0.9998024702072144,
        _SetDim.ADDR_B0_HI + 14: 0.9997925162315369,
        _SetDim.STACK0_BYTE0: 7.152557373046875e-07,
    }
    assert projection_score(spec.q, 42, bp_byte3_store_residue) < 0.0
    assert projection_score(spec.q, 43, bp_byte3_store_residue) < 0.0


def test_layer10_ax_byte_passthrough_blocks_binary_result_rows():
    dim_positions = {
        name: value for name, value in vars(_SetDim).items()
        if isinstance(value, int)
    }
    spec = (
        _layer10_byte_passthrough_ir(dim_positions, 64)
        .layer(0)
        .attention
        .heads[0]
        .spec
    )

    for dim in (
        _SetDim.OP_IMM,
        _SetDim.OP_LI_RELAY,
        _SetDim.OP_LC_RELAY,
        _SetDim.CMP + 3,
        _SetDim.TEMP + 3,
    ):
        assert any(
            weight < 0.0
            for weight in _projection_weights(spec, "q", 0, dim)
        ), dim


def test_layer10_ax_byte_passthrough_allows_addsub_operand_rows():
    dim_positions = {
        name: value for name, value in vars(_SetDim).items()
        if isinstance(value, int)
    }
    spec = (
        _layer10_byte_passthrough_ir(dim_positions, 64)
        .layer(0)
        .attention
        .heads[0]
        .spec
    )

    for dim in (_SetDim.TEMP + 8, _SetDim.TEMP + 9):
        assert not any(
            weight < 0.0
            for weight in _projection_weights(spec, "q", 0, dim)
        ), dim


def test_layer10_ax_byte_passthrough_has_li_reload_mem_value_route():
    dim_positions = {
        name: value for name, value in vars(_SetDim).items()
        if isinstance(value, int)
    }
    spec = (
        _layer10_byte_passthrough_ir(dim_positions, 64)
        .layer(0)
        .attention
        .heads[0]
        .spec
    )

    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "q", 39, _SetDim.OP_LI_RELAY)
    )
    for source_dim in (
        _SetDim.MEM_VAL_B0,
        _SetDim.MEM_VAL_B1,
        _SetDim.MEM_VAL_B2,
        _SetDim.MEM_VAL_B3,
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "k", 39, source_dim)
        )

    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "q", 40, _SetDim.MARK_AX)
    )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "k", 40, _SetDim.MEM_VAL_B1)
    )

    for slot, byte_dim, source_dim in (
        (40, _SetDim.MARK_AX, _SetDim.MEM_VAL_B1),
        (41, _SetDim.BYTE_INDEX_0, _SetDim.MEM_VAL_B2),
        (42, _SetDim.BYTE_INDEX_1, _SetDim.MEM_VAL_B3),
        (43, _SetDim.BYTE_INDEX_2, _SetDim.MEM_VAL_B3),
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "q", slot, _SetDim.OP_LI_RELAY)
        )
        assert any(
            weight < 0.0
            for weight in _projection_weights(spec, "q", slot, _SetDim.CONST)
        )
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "q", slot, byte_dim)
        )
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "k", slot, source_dim)
        )

    for slot, byte_dim in (
        (44, _SetDim.MARK_AX),
        (45, _SetDim.BYTE_INDEX_0),
        (46, _SetDim.BYTE_INDEX_1),
        (47, _SetDim.BYTE_INDEX_2),
    ):
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "q", slot, _SetDim.OP_LI_RELAY)
        )
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "q", slot, byte_dim)
        )
        assert any(
            weight < 0.0
            for weight in _projection_weights(spec, "q", slot, _SetDim.CONST)
        )
        assert any(
            weight > 0.0
            for weight in _projection_weights(spec, "k", slot, _SetDim.MEM_STORE)
        )
        if slot != 44:
            assert any(
                weight > 0.0
                for weight in _projection_weights(
                    spec,
                    "k",
                    slot,
                    _SetDim.MEM_ADDR_SRC,
                )
            )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "q", 48, _SetDim.MARK_AX)
    )
    assert any(
        weight > 0.0
        for weight in _projection_weights(spec, "k", 48, _SetDim.MEM_ADDR_SRC)
    )

    def projection_score(writes, slot: int, state: dict[int, float]) -> float:
        return sum(
            write.weight * state.get(write.dim, 0.0)
            for write in writes
            if write.slot == slot
        )

    li_marker = {
        _SetDim.CONST: 1.0,
        _SetDim.OP_LI_RELAY: 1.0,
        _SetDim.MARK_AX: 1.0,
    }
    li_byte0 = {
        _SetDim.CONST: 1.0,
        _SetDim.OP_LI_RELAY: 1.0,
        _SetDim.BYTE_INDEX_0: 0.9701380133628845,
        _SetDim.BYTE_INDEX_1: 0.013296706601977348,
    }
    non_li_byte0 = {
        _SetDim.CONST: 1.0,
        _SetDim.BYTE_INDEX_0: 0.9701380133628845,
        _SetDim.BYTE_INDEX_1: 0.013296706601977348,
    }
    mem_value0_source = {
        _SetDim.MEM_STORE: 2.0,
        _SetDim.MEM_ADDR_SRC: 2.0,
        _SetDim.MEM_VAL_B1: 0.9701380133628845,
    }
    mem_value2_source = {
        _SetDim.MEM_STORE: 2.0,
        _SetDim.MEM_ADDR_SRC: 1.0,
        _SetDim.MEM_VAL_B3: 0.9701380133628845,
    }
    psh_value0_source = {
        _SetDim.MEM_STORE: 2.0,
        _SetDim.MEM_ADDR_SRC: 1.0,
        _SetDim.MEM_VAL_B1: 0.9701380133628845,
    }
    nonstore_value0_source = {
        _SetDim.MEM_VAL_B1: 0.9701380133628845,
    }

    def route_score(
        query_state: dict[int, float],
        source_state: dict[int, float],
    ) -> float:
        return sum(
            projection_score(spec.q, slot, query_state)
            * projection_score(spec.k, slot, source_state)
            for slot in range(39, 49)
        )

    assert projection_score(spec.q, 39, li_marker) > 0.0
    assert projection_score(spec.q, 39, non_li_byte0) == pytest.approx(0.0)
    assert projection_score(spec.q, 40, li_marker) > 0.0
    assert projection_score(spec.q, 40, {_SetDim.CONST: 1.0}) < 0.0
    assert projection_score(spec.q, 41, li_marker) < 0.0
    assert projection_score(spec.q, 41, non_li_byte0) < 0.0
    assert projection_score(spec.q, 41, li_byte0) > 0.0
    assert projection_score(spec.q, 44, li_marker) > 0.0
    assert projection_score(spec.q, 45, li_marker) == pytest.approx(0.0)
    assert projection_score(spec.q, 45, li_byte0) > 0.0
    assert projection_score(spec.q, 48, li_marker) > 0.0
    assert projection_score(spec.q, 48, li_byte0) == pytest.approx(0.0)
    assert projection_score(spec.q, 42, li_byte0) < projection_score(
        spec.q, 41, li_byte0
    )
    assert route_score(li_marker, mem_value0_source) > route_score(
        li_marker, mem_value2_source
    )
    assert route_score(li_marker, mem_value0_source) > route_score(
        li_marker, nonstore_value0_source
    )
    assert route_score(li_marker, mem_value0_source) > route_score(
        li_marker, psh_value0_source
    )
    assert route_score(li_byte0, mem_value2_source) > route_score(
        non_li_byte0, mem_value2_source
    )


def test_id250_teacher_forced_critical_bytes_survive_l10_tail():
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.unified_compiler.decl_verifier import (
        build_teacher_forced_symbolic_trace,
        verify_teacher_forced_token_support,
    )
    from src.compiler import compile_c

    bytecode, data = compile_c("int main() { int x; x = 990; return x; }")
    trace = build_teacher_forced_symbolic_trace(bytecode, data)
    runner = BatchedPureNeuralRunner(max_seq_len=512)

    for step, slot in (
        (0, "STACK0_byte0"),
        (3, "AX_byte1"),
        (7, "AX_byte0"),
        (7, "AX_byte1"),
        (7, "AX_byte2"),
        (7, "AX_byte3"),
        (7, "STACK0_byte0"),
    ):
        report = verify_teacher_forced_token_support(
            runner.model,
            trace.context,
            token_index=trace.token_index(step, slot),
            prefix_len=trace.prefix_len,
            mem_store_positions=trace.mem_store_positions,
            max_context_window=512,
            min_margin=0.0,
            probe_name=f"id=0250:{step}:{slot}",
        )
        assert report.supported, report.format()


def test_tail_mem_store_addr_rules_block_stack0_marker_residue():
    ir = _tail_prefix_ir("tail_mem_store_addr")
    row = {
        "CONST": 1.0,
        "MARK_STACK0": 1.0,
        "H1+10": 1.0,
        "H3+10": 1.0,
        "MEM_STORE": 0.40725401043891907,
        "OP_JSR": 11.082738876342773,
    }
    for lane in range(16):
        row[f"OUTPUT_LO+{lane}"] = -507622.5625
        row[f"OUTPUT_HI+{lane}"] = -507622.5625
    row["OUTPUT_LO+10"] = 507625.5625
    row["OUTPUT_HI+0"] = 507625.5625

    out = ir.symbolic_ffn(row)

    for lane in range(16):
        assert out[f"OUTPUT_LO+{lane}"] == pytest.approx(row[f"OUTPUT_LO+{lane}"])
        assert out[f"OUTPUT_HI+{lane}"] == pytest.approx(row[f"OUTPUT_HI+{lane}"])


def test_tail_ax_add_byte1_hi_zero_blocks_stack0_byte_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_hi_zero_lo_f"))
    row = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9982175827026367,
        "H1+10": 0.9999994039535522,
        "BYTE_INDEX_0": 0.9734055995941162,
        "BYTE_INDEX_1": 0.013297183439135551,
        "STACK0_BYTE0": 0.9734055995941162,
        "STACK0_BYTE1": 0.013297183439135551,
        "TEMP+8": 0.020851222798228264,
        "ALU_HI+0": 1.0,
        "AX_CARRY_HI+0": 1.0,
        "OUTPUT_LO+0": -326.9977111816406,
        "OUTPUT_LO+15": 332.94451904296875,
        "OUTPUT_HI+0": -326.9977111816406,
        "OUTPUT_HI+15": 332.94451904296875,
    }

    out = ir.symbolic_ffn(row)

    assert out["OUTPUT_LO+0"] == pytest.approx(row["OUTPUT_LO+0"])
    assert out["OUTPUT_LO+15"] == pytest.approx(row["OUTPUT_LO+15"])
    assert out["OUTPUT_HI+0"] == pytest.approx(row["OUTPUT_HI+0"])
    assert out["OUTPUT_HI+15"] == pytest.approx(row["OUTPUT_HI+15"])


def test_tail_sp_initial_stack_exact_blocks_stack0_marker_residue():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")
    row = {
        "CONST": 1.0,
        "HAS_SE": 0.9985107183456421,
        "MARK_STACK0": 1.0,
        "H1+10": 1.0,
        "H3+10": 1.0,
        "H1+3": 0.003357573412358761,
        "OUTPUT_LO+8": -1883940.625,
        "OUTPUT_HI+14": -1883940.625,
        "OUTPUT_HI+15": -1887999.875,
        "ALU_LO+14": 0.0,
    }
    for lane in range(16):
        row.setdefault(f"OUTPUT_LO+{lane}", -1888000.0)
        row.setdefault(f"OUTPUT_HI+{lane}", -1888000.0)
    row["OUTPUT_LO+14"] = 1888000.0
    row["OUTPUT_HI+13"] = 1888000.0

    out = ir.symbolic_ffn(row)

    for lane in range(16):
        assert out[f"OUTPUT_LO+{lane}"] == pytest.approx(row[f"OUTPUT_LO+{lane}"])
        assert out[f"OUTPUT_HI+{lane}"] == pytest.approx(row[f"OUTPUT_HI+{lane}"])


def test_id550_stack0_byte1_high_nibble_survives_l10_tail():
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.unified_compiler.decl_verifier import (
        build_teacher_forced_symbolic_trace,
        verify_teacher_forced_token_support,
    )
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs

    source, _, _ = generate_test_programs()[550]
    bytecode, data = compile_c(source)
    trace = build_teacher_forced_symbolic_trace(bytecode, data)
    runner = BatchedPureNeuralRunner(max_seq_len=512)
    token_index = trace.token_index(5, "STACK0_byte1")

    report = verify_teacher_forced_token_support(
        runner.model,
        trace.context,
        token_index=token_index,
        prefix_len=trace.prefix_len,
        mem_store_positions=trace.mem_store_positions,
        max_context_window=512,
        min_margin=0.0,
        probe_name="id=0550:5:STACK0_byte1",
    )

    assert report.supported, report.format()


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


def test_tail_bp_byte2_preserve_blocks_large_ax_byte_residue():
    ir = _single_rule_ir(_tail_rule("tail_bp_byte2_preserve_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_1": 0.013,
        "OUTPUT_LO+1": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+1"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_bp_byte2_preserve_blocks_byte0_residue():
    ir = _single_rule_ir(_tail_rule("tail_bp_byte2_preserve_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_0": 1.0,
        "BYTE_INDEX_1": 0.013,
        "OUTPUT_LO+1": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+1"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


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


def test_tail_bp_byte2_preserve_tolerates_adjacent_index_residue():
    ir = _single_rule_ir(_tail_rule("tail_bp_byte2_preserve_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9982135891914368,
        "H1+3": 1.0,
        "BYTE_INDEX_1": 0.9701374769210815,
        "BYTE_INDEX_2": 0.013297064229846,
        "OUTPUT_LO+1": 2.0,
        "OUTPUT_LO+0": 2.5493369102478027,
        "OUTPUT_HI+0": 4.549336910247803,
    })

    assert out["OUTPUT_LO+1"] > 2.0
    assert out["OUTPUT_HI+0"] > 4.549336910247803


def test_tail_bp_byte2_preserve_blocks_lea_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_bp_byte2_preserve_01"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_LEA": 5.0,
        "HAS_SE": 1.0,
        "H1+3": 20.0,
        "BYTE_INDEX_1": 1.0,
        "OUTPUT_LO+1": 10000.0,
        "OUTPUT_HI+0": 10000.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 10000.0
    assert out.get("OUTPUT_HI+0", 0.0) == 10000.0


def test_tail_ax_byte0_rules_block_lea_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_wide_shl_byte1_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "MARK_AX": 1.0,
        "OP_LEA": 5.0,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+0": 1.0,
        "CARRY+3": 1000.0,
        "OUTPUT_LO+1": 1000.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 1000.0


def test_tail_wide_shl_byte1_blocks_add_residue_without_op_shl():
    ir = _single_rule_ir(_tail_rule("tail_wide_shl_byte1_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 1.304825782775879,
        "CARRY+3": 103.5004,
        "OUTPUT_LO+1": 8.192543157368e12,
        "OUTPUT_LO+3": 2.1993707531384e13,
        "OUTPUT_HI+0": 4.041023946752e12,
    })

    assert out["OUTPUT_LO+1"] == 8.192543157368e12
    assert out["OUTPUT_LO+3"] == 2.1993707531384e13
    assert out["OUTPUT_HI+0"] == 4.041023946752e12


def test_tail_ax_byte0_rules_block_jmp_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_JMP": 5.0,
        "OUTPUT_LO+1": -10000.0,
        "OUTPUT_HI+0": 10000.0,
    })

    assert out["OUTPUT_LO+1"] == -10000.0
    assert out["OUTPUT_HI+0"] == 10000.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_zero_blocks_imm_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "OP_IMM": 5.0,
        "ALU_HI+0": 800.0,
        "AX_CARRY_HI+0": 40000.0,
        "OUTPUT_LO+13": 520.0,
        "OUTPUT_HI+0": 580.0,
    })

    assert out["OUTPUT_LO+13"] == 520.0
    assert out["OUTPUT_HI+0"] == 580.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_zero_blocks_lea_address_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_LEA": 5.0,
        "TEMP+8": 100.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 20.0,
        "AX_CARRY_HI+0": 20.0,
        "OUTPUT_LO+15": 9.0,
        "OUTPUT_HI+15": 9.0,
    })

    assert out["OUTPUT_LO+15"] == 9.0
    assert out["OUTPUT_HI+15"] == 9.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_zero_blocks_si_ax_byte_row_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984007477760315,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "OP_SI": 5.0,
        "MEM_STORE": 0.004305745009332895,
        "TEMP+8": 183.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 20.0,
        "AX_CARRY_HI+0": 20.0,
        "OUTPUT_LO+3": 2.0,
        "OUTPUT_HI+0": 2.94,
    })

    assert out["OUTPUT_LO+3"] == 2.0
    assert out["OUTPUT_HI+0"] == 2.94
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_zero_blocks_sp_marker_transition():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701374769210815,
        "NEXT_SP": 1.368794322013855,
        "TEMP+8": 240.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 20.0,
        "AX_CARRY_HI+0": 20.0,
        "OUTPUT_LO+0": -218.64,
        "OUTPUT_HI+0": -218.64,
    })

    assert out["OUTPUT_LO+0"] == -218.64
    assert out["OUTPUT_HI+0"] == -218.64


def test_tail_ax_add_no_carry_zero_ignores_tiny_negative_blocker_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 183.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 20.0,
        "AX_CARRY_HI+0": 20.0,
        "MARK_AX": 1.0,
        "OP_IMM": 5.0,
        "OP_SI": -1.0e-21,
        "MEM_STORE": -1.0e-21,
        "OUTPUT_LO+11": 533.0,
        "OUTPUT_HI+12": 520.0,
    })

    assert out["OUTPUT_LO+11"] == 533.0
    assert out["OUTPUT_HI+12"] == 520.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_byte1_high_zero_blocks_lea_address_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_hi_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_LEA": 5.0,
        "TEMP+8": 100.0,
        "ALU_HI+0": 20.0,
        "AX_CARRY_HI+0": 20.0,
        "OUTPUT_LO+15": 9.0,
        "OUTPUT_HI+15": 9.0,
    })

    assert out["OUTPUT_LO+15"] == 9.0
    assert out["OUTPUT_HI+15"] == 9.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_ax_add_byte1_high_zero_blocks_mul_owned_byte_row():
    ir = _tail_prefix_ir("tail_ax_add_byte1_hi_zero")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9980560541152954,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.0132971,
        "TEMP+8": 0.3048,
        "TEMP+10": 0.996,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 6.237,
        "OUTPUT_LO+1": 40.0,
        "OUTPUT_HI+0": 2.9402759,
        "OUTPUT_HI+1": 40.0,
    })

    assert out["OUTPUT_HI+0"] == pytest.approx(2.9402759)
    assert out["OUTPUT_HI+1"] == pytest.approx(40.0)


def test_tail_ax_add_byte1_carry_high2_beats_drafted_stale_low8_row():
    ir = _tail_rules_ir(
        "tail_ax_add_byte1_hi_zero_lo_8",
        "tail_ax_add_byte1_carry_high2_03",
        "tail_ax_add_byte1_missing_stack_high_02",
    )
    ffn = _StubFFN(hidden_dim=ir.required_ffn_units())
    ir.lower_ffn(
        ffn,
        {name: getattr(_SetDim, name) for name in dir(_SetDim) if name.isupper()},
        S=100.0,
    )

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.HAS_SE] = 0.9984797239303589
    x[..., _SetDim.H1 + 1] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 0.9701380133628845
    x[..., _SetDim.BYTE_INDEX_1] = 0.013296706601977348
    x[..., _SetDim.TEMP + 8] = 1.304825782775879
    x[..., _SetDim.TEMP + 10] = 0.9962686896324158
    x[..., _SetDim.CARRY + 1] = 2.0
    x[..., _SetDim.CARRY + 3] = 3_984_912.5
    x[..., _SetDim.ALU_HI + 0] = 6.237
    x[..., _SetDim.AX_CARRY_HI + 0] = 3.728
    x[..., _SetDim.OUTPUT_LO + 0] = -2.3769082260665315e24
    x[..., _SetDim.OUTPUT_LO + 1] = -2.594176860070444e24
    x[..., _SetDim.OUTPUT_LO + 2] = -2.594176860070444e24
    x[..., _SetDim.OUTPUT_LO + 3] = -2.5941739777666826e24
    x[..., _SetDim.OUTPUT_LO + 4] = -2.594176860070444e24
    x[..., _SetDim.OUTPUT_LO + 5] = -2.5941759953793157e24
    x[..., _SetDim.OUTPUT_LO + 6] = -2.594176860070444e24
    x[..., _SetDim.OUTPUT_LO + 7] = -2.594176860070444e24
    x[..., _SetDim.OUTPUT_LO + 8] = 8.875321636683635e25
    x[..., _SetDim.OUTPUT_LO + 14] = 2.3760402202887507e24
    x[..., _SetDim.OUTPUT_HI + 0] = -1.3888250596682745e26
    x[..., _SetDim.OUTPUT_HI + 1] = 6.540475849392202e24
    x[..., _SetDim.OUTPUT_HI + 13] = 1.0285807650094059e26

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 3] > y[_SetDim.OUTPUT_LO + 8]
    assert y[_SetDim.OUTPUT_HI + 0] > y[_SetDim.OUTPUT_HI + 1]


def test_tail_ax_add_byte1_no_carry_low1_materializes_02():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_no_carry_low1_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.304825782775879,
        "TEMP+10": 0.9962686896324158,
        "ALU_LO+1": 6.0,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 3.728,
        "OUTPUT_LO+1": -357.7730712890625,
        "OUTPUT_LO+2": -17.386804580688477,
        "OUTPUT_HI+0": -1997.99951171875,
        "OUTPUT_HI+1": -1818.688720703125,
    })

    assert out["OUTPUT_LO+2"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+1"] < -357.7730712890625


def test_tail_ax_add_byte1_no_carry_low1_requires_boosted_add_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_no_carry_low1_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0,
        "TEMP+10": 0.9962686896324158,
        "ALU_LO+1": 6.0,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 3.728,
        "OUTPUT_LO+1": 8.965822219848633,
        "OUTPUT_HI+0": -1997.99951171875,
    })

    assert out["OUTPUT_LO+1"] == 8.965822219848633
    assert out["OUTPUT_HI+0"] == -1997.99951171875
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_no_carry_low2_materializes_03():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_no_carry_low2_03"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.304825782775879,
        "TEMP+10": 0.9962686896324158,
        "ALU_LO+2": 6.0,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 3.728,
        "OUTPUT_LO+2": -25.737689971923828,
        "OUTPUT_LO+3": -17.386804580688477,
        "OUTPUT_HI+0": -1997.99951171875,
        "OUTPUT_HI+1": -1818.688720703125,
    })

    assert out["OUTPUT_LO+3"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+2"] < -25.737689971923828


def test_tail_ax_add_byte1_no_carry_low2_requires_boosted_add_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_no_carry_low2_03"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0,
        "TEMP+10": 0.9962686896324158,
        "ALU_LO+2": 6.0,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 3.728,
        "OUTPUT_LO+2": 8.965362548828125,
        "OUTPUT_HI+0": -1997.99951171875,
    })

    assert out["OUTPUT_LO+2"] == 8.965362548828125
    assert out["OUTPUT_HI+0"] == -1997.99951171875
    assert out.get("OUTPUT_LO+3", 0.0) == 0.0


def test_tail_ax_add_byte1_carry_low2_beats_missing_stack_high_02():
    ir = _tail_rules_ir(
        "tail_ax_add_byte1_carry_low2_03",
        "tail_ax_add_byte1_missing_stack_high_02",
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.304825782775879,
        "TEMP+10": 0.9962686896324158,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 1.221,
        "ALU_LO+2": 6.0,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 3.728,
        "OUTPUT_LO+1": -110.91671752929688,
        "OUTPUT_LO+2": -95.88512420654297,
        "OUTPUT_LO+3": -110.91671752929688,
        "OUTPUT_HI+0": -1997.99951171875,
        "OUTPUT_HI+1": -1818.688720703125,
    })

    assert out["OUTPUT_LO+3"] > out["OUTPUT_LO+2"]
    assert out["OUTPUT_HI+0"] > out["OUTPUT_HI+1"]


def test_tail_ax_add_byte1_carry_low2_requires_boosted_add_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_carry_low2_03"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0,
        "TEMP+10": 0.9962686896324158,
        "CARRY+1": 2.0,
        "ALU_LO+2": 6.0,
        "ALU_HI+0": 6.237,
        "AX_CARRY_HI+0": 3.728,
        "OUTPUT_LO+5": 2616.27001953125,
        "OUTPUT_HI+0": -1997.99951171875,
    })

    assert out["OUTPUT_LO+5"] == 2616.27001953125
    assert out["OUTPUT_HI+0"] == -1997.99951171875
    assert out.get("OUTPUT_LO+3", 0.0) == 0.0


def test_tail_ax_lea_local_addr_byte1_preserves_ff_after_e8():
    ir = _single_rule_ir(_tail_rule("tail_ax_lea_local_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "FETCH_HI+15": 1.0,
        "OUTPUT_LO+15": 9.0,
        "OUTPUT_HI+15": 2.0,
        "OUTPUT_HI+0": 25.0,
    })

    assert out["OUTPUT_LO+15"] > 9.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_ax_lea_local_addr_byte1_accepts_temp10_local_signature():
    ir = _single_rule_ir(_tail_rule("tail_ax_lea_local_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.997,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "FETCH_HI+15": 2.44211264543992e-06,
        "TEMP+10": 0.9984737038612366,
        "OUTPUT_LO+0": 2.9402759075164795,
        "OUTPUT_HI+0": 2.9402759075164795,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_ax_lea_local_addr_byte1_requires_staged_high_nibble():
    ir = _single_rule_ir(_tail_rule("tail_ax_lea_local_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+15": 9.0,
        "OUTPUT_HI+0": 25.0,
    })

    assert out["OUTPUT_LO+15"] == 9.0
    assert out["OUTPUT_HI+0"] == 25.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_ax_lea_local_addr_byte1_blocks_mem_address_rows():
    ir = _single_rule_ir(_tail_rule("tail_ax_lea_local_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H3+4": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+15": 9.0,
        "OUTPUT_HI+15": 2.0,
        "OUTPUT_HI+0": 25.0,
    })

    assert out["OUTPUT_LO+15"] == 9.0
    assert out["OUTPUT_HI+15"] == 2.0
    assert out["OUTPUT_HI+0"] == 25.0


def test_tail_cmp_eq_false_blocks_equal_true_marker():
    ir = _single_rule_ir(_tail_rule("tail_cmp_eq_false_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_EQ": 5.0,
        "CMP+1": 1.0,
        "CMP+2": 1.0,
        "OUTPUT_LO+1": 19.0,
        "OUTPUT_HI+0": 18.0,
    })

    assert out["OUTPUT_LO+1"] == 19.0
    assert out["OUTPUT_HI+0"] == 18.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_cmp_eq_false_still_fires_when_low_compare_mismatches():
    ir = _single_rule_ir(_tail_rule("tail_cmp_eq_false_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_EQ": 5.0,
        "CMP+1": 1.0,
        "OUTPUT_LO+1": 19.0,
        "OUTPUT_HI+0": 18.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_LO+1"] < 19.0


def test_tail_cmp_le_eq_prefix_false_clears_spurious_true():
    ir = _single_rule_ir(_tail_rule("tail_cmp_le_eq_prefix_false_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_LE": 5.0,
        "CMP+1": 2.5,
        "OUTPUT_LO+1": 83.0,
        "OUTPUT_HI+0": 18.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_LO+1"] < 83.0
    assert out["OUTPUT_HI+0"] > 18.0


def test_tail_cmp_le_eq_prefix_false_preserves_equal_true():
    ir = _single_rule_ir(_tail_rule("tail_cmp_le_eq_prefix_false_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_LE": 5.0,
        "CMP+1": 2.5,
        "CMP+2": 1.0,
        "OUTPUT_LO+1": 83.0,
        "OUTPUT_HI+0": 18.0,
    })

    assert out["OUTPUT_LO+1"] == 83.0
    assert out["OUTPUT_HI+0"] == 18.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_cmp_le_eq_prefix_false_preserves_lt_true_at_marker():
    ir = _single_rule_ir(_tail_rule("tail_cmp_le_eq_prefix_false_00"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_LE": 5.0,
        "CMP+0": 1.0,
        "OUTPUT_LO+1": 36.0,
        "OUTPUT_HI+0": 19.0,
    })

    assert out["OUTPUT_LO+1"] == 36.0
    assert out["OUTPUT_HI+0"] == 19.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_does_not_emit_ax_add_carry_rules():
    names = {rule.name for rule in _tail_bit32_result_correction_rules()}

    assert "tail_ax_add_carry_byte1_02" not in names


def test_tail_does_not_emit_ambiguous_sp_byte3_carry_rules():
    names = {rule.name for rule in _tail_bit32_result_correction_rules()}

    assert not any(name.startswith("tail_sp_pop_carry_byte3_") for name in names)


def test_tail_stack0_pop_marker_zero_blocks_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_marker_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 1.0,
        "IS_BYTE": 1.0,
        "OUTPUT_LO+8": 2.0,
        "OUTPUT_HI+14": 2.0,
    })

    assert out["OUTPUT_LO+8"] == 2.0
    assert out["OUTPUT_HI+14"] == 2.0


def test_tail_stack0_pop_marker_zero_blocks_sp_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_marker_zero"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+8": 2.0,
        "OUTPUT_HI+14": 2.0,
    })

    assert out["OUTPUT_LO+8"] == 2.0
    assert out["OUTPUT_HI+14"] == 2.0


def test_tail_stack0_pop_marker_zero_blocks_mem_store_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_marker_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "OUTPUT_LO+14": 2.0,
        "OUTPUT_HI+13": 2.0,
    })

    assert out["OUTPUT_LO+14"] == 2.0
    assert out["OUTPUT_HI+13"] == 2.0


def test_tail_stack0_pop_marker_zero_still_fires_without_mem_store():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_marker_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+14": 2.0,
        "OUTPUT_HI+13": 2.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+14"] < 0.0
    assert out["OUTPUT_HI+13"] < 0.0


def test_tail_mem_store_addr0_f8_exact_overrides_stale_zero_lanes():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_f8_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9961883425712585,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+2": 0.9940742254257202,
        "OUTPUT_LO+0": 2.921051025390625,
        "OUTPUT_LO+8": 0.9968036413192749,
        "OUTPUT_HI+0": 2.9213929176330566,
        "OUTPUT_HI+15": 0.9968036413192749,
    })

    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"] + 1000.0
    assert out["OUTPUT_HI+15"] > out["OUTPUT_HI+0"] + 1000.0


def test_tail_mem_store_addr0_f8_exact_requires_staged_low_eight():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_f8_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9980560541152954,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+14": 0.9937615394592285,
        "OUTPUT_LO+0": 3.8322701454162598,
        "OUTPUT_HI+0": 3.503345012664795,
        "OUTPUT_HI+15": 0.9901931285858154,
    })

    assert out["OUTPUT_LO+0"] == 3.8322701454162598
    assert out["OUTPUT_HI+0"] == 3.503345012664795
    assert out["OUTPUT_HI+15"] == 0.9901931285858154


def test_tail_mem_store_addr0_f8_exact_blocks_second_push_residue():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_f8_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9980560541152954,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+0": 3.8322701454162598,
        "OUTPUT_LO+8": 219616.703125,
        "OUTPUT_LO+14": 0.6614807844161987,
        "ALU_LO+14": 0.9937615394592285,
        "OUTPUT_HI+15": 0.9901931285858154,
    })

    assert out["OUTPUT_LO+8"] == 219616.703125
    assert out["OUTPUT_HI+15"] == 0.9901931285858154


def test_tail_mem_store_addr0_f8_exact_blocks_negative_inactive_alu_lane():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_f8_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9982885122299194,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "CMP+3": 2.996619462966919,
        "ALU_LO+2": 0.00031229533487930894,
        "ALU_LO+14": -247.27655029296875,
        "ADDR_B0_LO+8": 0.9888219833374023,
        "ADDR_B0_HI+14": 0.9716129302978516,
        "OUTPUT_LO+8": 1282.8973388671875,
        "OUTPUT_HI+14": 1361.1473388671875,
        "OUTPUT_HI+15": -6.25,
    })

    assert out["OUTPUT_LO+8"] == 1282.8973388671875
    assert out["OUTPUT_HI+14"] == 1361.1473388671875
    assert out["OUTPUT_HI+15"] == -6.25


def test_tail_mem_store_addr1_ff_from_stack_store_exacts_nibbles():
    ir = _tail_prefix_ir("tail_mem_store_addr1_ff_from_stack_store_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 3.78348,
        "OUTPUT_HI+0": 3.78348,
    })

    assert out["OUTPUT_LO+15"] == pytest.approx(500.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(500.0)
    for lane in range(16):
        if lane != 15:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr1_ff_from_stack_store_blocks_non_stack_addr0():
    ir = _tail_prefix_ir("tail_mem_store_addr1_ff_from_stack_store_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 1.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.78348,
        "OUTPUT_HI+0": 3.78348,
    })

    assert out["OUTPUT_LO+0"] == 3.78348
    assert out["OUTPUT_HI+0"] == 3.78348
    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_mem_store_addr0_f8_initial_jsr_exacts_nibbles():
    ir = _tail_prefix_ir("tail_mem_store_addr0_f8_initial_jsr_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "MEM_STORE": 1.9999995231628418,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "OUTPUT_LO+8": -839271360.0,
        "OUTPUT_HI+15": -14807114752.0,
        "OUTPUT_HI+0": 881834262528.0,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(5000.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(5000.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(0.0)


def test_tail_mem_store_addr0_f8_initial_jsr_requires_first_step():
    ir = _tail_prefix_ir("tail_mem_store_addr0_f8_initial_jsr_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 1.0,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "MEM_STORE": 1.9999995231628418,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "OUTPUT_LO+8": 36939.66015625,
        "OUTPUT_HI+0": 4668.30078125,
    })

    assert out["OUTPUT_LO+8"] == 36939.66015625
    assert out["OUTPUT_HI+0"] == 4668.30078125


def test_tail_mem_store_addr0_initial_jsr_blocks_local_addr_collisions():
    ir = _tail_prefix_ir("tail_mem_store_addr0_")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "MEM_STORE": 2.0,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "ALU_LO+0": -109.212,
        "ALU_LO+7": -109.915,
        "ALU_LO+8": -109.915,
        "ALU_LO+10": -109.866,
        "ALU_LO+14": -109.75,
        "OUTPUT_LO+0": -1717824640.0,
        "OUTPUT_LO+8": 1717824896.0,
        "OUTPUT_HI+0": -25613821952.0,
        "OUTPUT_HI+14": 1706504832.0,
        "OUTPUT_HI+15": 1722568704.0,
    })

    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+15"] > out["OUTPUT_HI+14"]


def test_tail_mem_store_addr0_f0_exact_overrides_second_push_residue():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_f0_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9980560541152954,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+0": 3.8322701454162598,
        "OUTPUT_LO+8": 219616.703125,
        "OUTPUT_LO+14": 0.6614807844161987,
        "ALU_LO+14": 0.9937615394592285,
        "OUTPUT_HI+0": 3.503345012664795,
        "OUTPUT_HI+15": 0.9901931285858154,
    })

    assert out["OUTPUT_LO+0"] > out["OUTPUT_LO+8"] + 1000.0
    assert out["OUTPUT_HI+15"] > out["OUTPUT_HI+0"] + 1000.0


def test_tail_mem_store_addr0_f0_exact_blocks_jsr_initial_store():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_f0_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "OP_JSR": 12.491548538208008,
        "CMP+0": 2.996619701385498,
        "ALU_LO+14": 16377.4033203125,
        "OUTPUT_LO+8": 19929.138671875,
        "OUTPUT_HI+15": 19874.861328125,
    })

    assert out["OUTPUT_LO+8"] == 19929.138671875
    assert out["OUTPUT_HI+15"] == 19874.861328125
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_mem_store_addr0_00_from_global_exacts_nibbles():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_00_from_global_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.998,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "MEM_ADDR_SRC": 4.0,
        "OUTPUT_LO+0": 13.9,
        "OUTPUT_HI+0": 16.9,
    })

    assert out["OUTPUT_LO+0"] > 1_000_000.0
    assert out["OUTPUT_HI+0"] > 1_000_000.0
    assert out["OUTPUT_LO+8"] < -1_000_000.0
    assert out["OUTPUT_HI+15"] < -1_000_000.0


def test_tail_mem_store_addr0_00_from_global_blocks_initial_jsr_store():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_00_from_global_exact"))

    row = {
        "MARK_MEM": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "OP_JSR": 12.491547584533691,
        "OUTPUT_LO+0": 14941.8525390625,
        "OUTPUT_HI+0": 1853.215576171875,
        "OUTPUT_HI+14": -1425.1607666015625,
        "OUTPUT_HI+15": 1773.0650634765625,
    }

    out = ir.symbolic_ffn(row)

    assert out["OUTPUT_LO+0"] == row["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] == row["OUTPUT_HI+0"]
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == row["OUTPUT_HI+15"]


def test_tail_mem_store_addr0_00_from_global_blocks_local_e0_shape():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_00_from_global_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.998,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "MEM_ADDR_SRC": 4.0,
        "OUTPUT_LO+0": 13.9,
        "OUTPUT_HI+14": 16.9,
    })

    assert out["OUTPUT_LO+0"] == 13.9
    assert out["OUTPUT_HI+14"] == 16.9
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_mem_store_addr0_00_from_global_blocks_psh_stack_store():
    ir = _single_rule_ir(_tail_rule("tail_mem_store_addr0_00_from_global_exact"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.998,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "MEM_ADDR_SRC": 4.0,
        "PSH_AT_SP": 1.0,
        "OUTPUT_LO+0": 13.9,
        "OUTPUT_HI+0": 16.9,
    })

    assert out["OUTPUT_LO+0"] == 13.9
    assert out["OUTPUT_HI+0"] == 16.9
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_mem_store_addr2_zero_from_global_exacts_after_addr1_residue():
    ir = _tail_prefix_ir("tail_mem_store_addr2_zero_from_global_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        "BYTE_INDEX_1": 0.97,
        "OUTPUT_LO+0": 1.775,
        "OUTPUT_LO+2": 4.669,
        "OUTPUT_HI+0": 6.722,
        "OUTPUT_HI+1": 1.669,
    })

    assert out["OUTPUT_LO+0"] > 100.0
    assert out["OUTPUT_HI+0"] > 100.0
    assert out["OUTPUT_LO+2"] == pytest.approx(0.0)
    assert out["OUTPUT_HI+1"] == pytest.approx(0.0)


def test_tail_mem_store_addr2_zero_from_global_blocks_addr1_row():
    ir = _tail_prefix_ir("tail_mem_store_addr2_zero_from_global_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        "BYTE_INDEX_0": 1.0,
        "OUTPUT_LO+0": 1.0,
        "OUTPUT_LO+2": 2.0,
        "OUTPUT_HI+0": 1.0,
    })

    assert out["OUTPUT_LO+2"] == pytest.approx(2.0)
    assert out["OUTPUT_LO+0"] == pytest.approx(1.0)


def test_tail_mem_store_addr2_zero_from_global_blocks_local_high_byte():
    ir = _tail_prefix_ir("tail_mem_store_addr2_zero_from_global_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        "BYTE_INDEX_1": 1.0,
        "OUTPUT_LO+15": 4.0,
        "OUTPUT_HI+15": 4.0,
    })

    assert out["OUTPUT_LO+15"] == pytest.approx(4.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(4.0)


def test_tail_mem_store_addr_high_zero_from_global_blocks_bp_byte3_residue():
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _tail_rules_with_prefix("tail_mem_store_addr2_zero_from_global_exact")
    )
    ir.layer(0).ffn.rules.extend(
        _tail_rules_with_prefix("tail_mem_store_addr3_zero_from_global_exact")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_2": 1.0,
        "OUTPUT_LO+0": 1.03e15,
        "OUTPUT_HI+0": -6.81e15,
        "OUTPUT_HI+1": 0.03,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(1.03e15)
    assert out["OUTPUT_HI+0"] == pytest.approx(-6.81e15)
    assert out["OUTPUT_HI+1"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("prefix", "byte_index"),
    [
        ("tail_mem_store_addr2_zero_from_global_exact", "BYTE_INDEX_1"),
        ("tail_mem_store_addr3_zero_from_global_exact", "BYTE_INDEX_2"),
    ],
)
def test_tail_mem_store_addr_high_zero_from_global_blocks_stack0_byte_rows(
    prefix, byte_index
):
    ir = _tail_prefix_ir(prefix)

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        byte_index: 1.0,
        "STACK0_BYTE0": 1.0,
        "OUTPUT_LO+0": 123.0,
        "OUTPUT_HI+0": 456.0,
        "OUTPUT_LO+15": -7.0,
        "OUTPUT_HI+15": -8.0,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(123.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(456.0)
    assert out["OUTPUT_LO+15"] == pytest.approx(-7.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(-8.0)


@pytest.mark.parametrize(
    ("prefix", "byte_index"),
    [
        ("tail_mem_store_addr2_zero_from_global_exact", "BYTE_INDEX_1"),
        ("tail_mem_store_addr3_zero_from_global_exact", "BYTE_INDEX_2"),
        ("tail_sp_marker_byte0_f8_from_initial_stack_exact", "BYTE_INDEX_0"),
    ],
)
def test_tail_stack0_span_blockers_dominate_authoritative_output_bands(
    prefix, byte_index
):
    ir = _tail_prefix_ir(prefix)

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        byte_index: 1.0,
        "OUTPUT_LO+0": -441_904_000.0,
        "OUTPUT_LO+8": -441_899_936.0,
        "OUTPUT_LO+9": 441_904_032.0,
        "OUTPUT_HI+0": -441_904_000.0,
        "OUTPUT_HI+3": 441_904_032.0,
        "OUTPUT_HI+14": -441_899_936.0,
        "OUTPUT_HI+15": -441_904_000.0,
    })

    assert out["OUTPUT_LO+9"] == pytest.approx(441_904_032.0)
    assert out["OUTPUT_HI+3"] == pytest.approx(441_904_032.0)
    assert out["OUTPUT_LO+0"] == pytest.approx(-441_904_000.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(-441_904_000.0)


def test_tail_mem_store_addr0_f8_from_mod_local_exacts_nibbles():
    ir = _tail_prefix_ir("tail_mem_store_addr0_f8_from_mod_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9961883425712585,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+7": 0.9937863945960999,
        "ALU_LO+10": 0.0,
        "ALU_LO+14": 0.0,
        "OUTPUT_LO+0": 2.921051025390625,
        "OUTPUT_LO+7": 0.572,
        "OUTPUT_LO+8": 0.9968036413192749,
        "OUTPUT_HI+0": 2.9213929176330566,
        "OUTPUT_HI+15": 1.1968036413192749,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(500.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(500.0)
    for lane in range(16):
        if lane != 8:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 15:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr0_f8_from_mod_local_blocks_second_push():
    ir = _tail_prefix_ir("tail_mem_store_addr0_f8_from_mod_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9980560541152954,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+7": 0.0,
        "ALU_LO+14": 0.9937615394592285,
        "OUTPUT_LO+0": 3.8322701454162598,
        "OUTPUT_LO+8": 219616.703125,
        "OUTPUT_HI+15": 0.9901931285858154,
    })

    assert out["OUTPUT_LO+0"] == 3.8322701454162598
    assert out["OUTPUT_LO+8"] == 219616.703125
    assert out["OUTPUT_HI+15"] == 0.9901931285858154


def test_tail_mem_store_addr0_f8_from_mod_local_blocks_psh_sp_store():
    ir = _tail_prefix_ir("tail_mem_store_addr0_f8_from_mod_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9978197813034058,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "MEM_STORE": 2.0,
        "PSH_AT_SP": 1.498309850692749,
        "CMP+0": 2.996619701385498,
        "ALU_LO+7": 0.0005193031392991543,
        "ALU_LO+10": 0.0,
        "ALU_LO+14": 1.1776190149248578e-05,
        "OUTPUT_LO+0": 2.566706895828247,
        "OUTPUT_LO+8": 0.5347899794578552,
        "OUTPUT_HI+14": 13285903.0,
        "OUTPUT_HI+15": 11668551.0,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(2.566706895828247)
    assert out["OUTPUT_LO+8"] == pytest.approx(0.5347899794578552)
    assert out["OUTPUT_HI+14"] == pytest.approx(13285903.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(11668551.0)


def test_tail_mem_store_addr0_f8_from_mod_local_blocks_stack_source_store():
    ir = _tail_prefix_ir("tail_mem_store_addr0_f8_from_mod_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9982885122299194,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        "CMP+0": 2.996619462966919,
        "ALU_LO+7": 0.0,
        "ALU_LO+10": 0.07819131016731262,
        "ALU_LO+14": 0.0,
        "OUTPUT_LO+8": 85991448.0,
        "OUTPUT_HI+14": 90491448.0,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(85991448.0)
    assert out["OUTPUT_HI+14"] == pytest.approx(90491448.0)
    assert out.get("OUTPUT_LO+0", 0.0) == pytest.approx(0.0)
    assert out.get("OUTPUT_HI+0", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr0_e8_from_nested_local_exacts_nibbles():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_nested_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9966400265693665,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+10": 0.9940742254257202,
        "ALU_LO+7": 0.0,
        "ALU_LO+14": 0.0,
        "OUTPUT_LO+0": 2.958621025085449,
        "OUTPUT_LO+8": 0.9968036413192749,
        "OUTPUT_LO+10": 0.536,
        "OUTPUT_HI+0": 3.503345012664795,
        "OUTPUT_HI+14": 1.1968036413192749,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(500.0)
    assert out["OUTPUT_HI+14"] == pytest.approx(500.0)
    for lane in range(16):
        if lane != 8:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 14:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr0_e8_from_nested_local_blocks_stack_source_store():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_nested_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9982885122299194,
        "H1+4": 1.0,
        "H1+11": 1.0,
        "MEM_STORE": 2.0,
        "MEM_ADDR_SRC": 2.0,
        "CMP+0": 2.996619462966919,
        "ALU_LO+10": 0.07819131016731262,
        "ALU_LO+7": 0.0,
        "ALU_LO+14": 0.0,
        "OUTPUT_LO+8": 85991448.0,
        "OUTPUT_HI+14": 90491448.0,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(85991448.0)
    assert out["OUTPUT_HI+14"] == pytest.approx(90491448.0)
    assert out.get("OUTPUT_LO+0", 0.0) == pytest.approx(0.0)
    assert out.get("OUTPUT_HI+0", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr0_e0_from_local_offset_exacts_nibbles():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e0_from_local_offset_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+8": 0.9937615394592285,
        "ALU_LO+7": 0.0,
        "ALU_LO+10": 0.0,
        "ALU_LO+14": 0.0,
        "OUTPUT_LO+0": 2.574,
        "OUTPUT_LO+8": 0.535,
        "OUTPUT_HI+0": -298.0,
        "OUTPUT_HI+14": 161.262,
        "OUTPUT_HI+15": 141.692,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(50000.0)
    assert out["OUTPUT_HI+14"] == pytest.approx(50000.0)
    for lane in range(16):
        if lane != 0:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 14:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr0_e0_from_local_offset_blocks_e8_row():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e0_from_local_offset_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "ALU_LO+8": 0.0,
        "ALU_LO+10": 0.9940742254257202,
        "OUTPUT_LO+8": 12.2,
        "OUTPUT_HI+14": 83.8,
    })

    assert out["OUTPUT_LO+8"] == 12.2
    assert out["OUTPUT_HI+14"] == 83.8
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_mem_store_addr0_e0_from_local_offset_blocks_strong_e8_output():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e0_from_local_offset_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9982885122299194,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "CMP+3": 2.996619462966919,
        "ALU_LO+8": 0.40735572576522827,
        "ALU_LO+14": -247.27655029296875,
        "ADDR_B0_LO+8": 0.9888219833374023,
        "ADDR_B0_HI+14": 0.9716129302978516,
        "OUTPUT_LO+8": 1282.8973388671875,
        "OUTPUT_HI+14": 1361.1473388671875,
    })

    assert out["OUTPUT_LO+8"] == 1282.8973388671875
    assert out["OUTPUT_HI+14"] == 1361.1473388671875
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_mem_store_addr0_local_frame_rules_block_raw_psh_store_row():
    ir = CompilerIR()
    for prefix in (
        "tail_mem_store_addr0_f8_from_mod_local_exact",
        "tail_mem_store_addr0_e0_from_local_offset_exact",
        "tail_mem_store_addr0_e8_from_nested_local_exact",
        "tail_mem_store_addr0_e8_from_local_frame_addr_exact",
        "tail_mem_store_addr0_e8_from_local_frame_output_exact",
    ):
        ir.layer(0).ffn.rules.extend(_tail_rules_with_prefix(prefix))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9969,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "PSH_AT_SP": 1.498,
        "CMP+0": 2.996619701385498,
        "ALU_LO+7": 0.9937863945960999,
        "ALU_LO+8": 0.9937615394592285,
        "ALU_LO+10": 0.9940742254257202,
        "ALU_LO+14": 0.0,
        "ADDR_B0_LO+8": 0.9904117584228516,
        "ADDR_B0_HI+14": 0.9812884330749512,
        "OUTPUT_LO+0": 13.946,
        "OUTPUT_LO+8": -10.0,
        "OUTPUT_HI+14": -10.0,
        "OUTPUT_HI+15": -10.0,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(500.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(500.0)
    assert out.get("OUTPUT_LO+0", 0.0) == pytest.approx(0.0)
    assert out.get("OUTPUT_HI+14", 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "prefix",
    [
        "tail_mem_store_addr0_e0_from_local_offset_exact",
        "tail_mem_store_addr0_e8_from_nested_local_exact",
        "tail_mem_store_addr0_e8_from_local_frame_addr_exact",
        "tail_mem_store_addr0_e8_from_local_frame_output_exact",
    ],
)
def test_tail_mem_store_addr0_local_frame_rules_require_no_psh_at_sp(prefix):
    ir = _tail_prefix_ir(prefix)

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "PSH_AT_SP": 1.498,
        "CMP+0": 2.996619701385498,
        "ALU_LO+8": 0.9937615394592285,
        "ALU_LO+10": 0.9940742254257202,
        "ADDR_B0_LO+8": 0.9904117584228516,
        "ADDR_B0_HI+14": 0.9812884330749512,
        "OUTPUT_LO+0": 13.946,
        "OUTPUT_LO+8": -10.0,
        "OUTPUT_HI+14": -10.0,
    })

    assert out["OUTPUT_LO+0"] == 13.946
    assert out["OUTPUT_LO+8"] == -10.0
    assert out["OUTPUT_HI+14"] == -10.0


def test_tail_mem_store_addr0_e0_from_jsr_local_exacts_nibbles():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e0_from_jsr_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "ALU_LO+14": 5360.0,
        "OUTPUT_LO+0": -10600.0,
        "OUTPUT_LO+8": 10900.0,
        "OUTPUT_HI+0": -354000.0,
        "OUTPUT_HI+14": 30600.0,
        "OUTPUT_HI+15": 11000.0,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(5000.0)
    assert out["OUTPUT_HI+14"] == pytest.approx(5000.0)
    for lane in range(16):
        if lane != 0:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 14:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_mem_store_addr0_e0_from_jsr_local_blocks_initial_f8():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e0_from_jsr_local_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "H1+4": 1.0,
        "MEM_STORE": 1.9999995231628418,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "ALU_LO+14": -110.0,
        "OUTPUT_LO+8": -839271360.0,
        "OUTPUT_HI+0": 881834262528.0,
        "OUTPUT_HI+15": -14807114752.0,
    })

    assert out["OUTPUT_LO+8"] == -839271360.0
    assert out["OUTPUT_HI+0"] == 881834262528.0
    assert out["OUTPUT_HI+15"] == -14807114752.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_mem_store_addr0_e0_from_jsr_local_strong_overrides_residue():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e0_from_jsr_local_strong")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "ALU_LO+14": 5360.0,
        "OUTPUT_LO+1": 8_867_526_213_632.0,
        "OUTPUT_HI+0": 18_137_942_589_440.0,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(5000.0)
    assert out["OUTPUT_HI+14"] == pytest.approx(5000.0)
    assert out["OUTPUT_LO+1"] == pytest.approx(0.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(0.0)


def test_tail_mem_store_addr0_local_frame_output_e8_dominates_f8():
    ir = CompilerIR()
    for prefix in (
        "tail_mem_store_addr0_f8_from_mod_local_exact",
        "tail_mem_store_addr0_e8_from_nested_local_exact",
        "tail_mem_store_addr0_e8_from_local_frame_output_exact",
    ):
        ir.layer(0).ffn.rules.extend(_tail_rules_with_prefix(prefix))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9971901178359985,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+0": -56.29740524291992,
        "OUTPUT_LO+8": 60.241825103759766,
        "OUTPUT_HI+0": -98.06285858154297,
        "OUTPUT_HI+14": 83.8114242553711,
        "OUTPUT_HI+15": -0.0019485921366140246,
    })

    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+14"] > out["OUTPUT_HI+15"]


def test_tail_mem_store_addr0_local_frame_addr_e8_dominates_f8():
    ir = CompilerIR()
    for prefix in (
        "tail_mem_store_addr0_f8_from_mod_local_exact",
        "tail_mem_store_addr0_e8_from_nested_local_exact",
        "tail_mem_store_addr0_e8_from_local_frame_addr_exact",
    ):
        ir.layer(0).ffn.rules.extend(_tail_rules_with_prefix(prefix))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 4.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+0": 16.946823120117188,
        "OUTPUT_LO+8": -10.0,
        "OUTPUT_HI+0": 16.946823120117188,
        "OUTPUT_HI+14": -10.0,
        "OUTPUT_HI+15": -10.0,
        "ADDR_B0_LO+8": 0.9904117584228516,
        "ADDR_B0_HI+14": 0.9812884330749512,
    })

    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+14"] > out["OUTPUT_HI+15"]


def test_tail_mem_store_addr0_local_frame_e8_blocks_initial_f8_shape():
    ir = CompilerIR()
    for prefix in (
        "tail_mem_store_addr0_e8_from_local_frame_addr_exact",
        "tail_mem_store_addr0_e8_from_local_frame_output_exact",
    ):
        ir.layer(0).ffn.rules.extend(_tail_rules_with_prefix(prefix))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9980560541152954,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+8": 200.0,
        "OUTPUT_HI+15": 200.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
    })

    assert out["OUTPUT_LO+8"] == 200.0
    assert out["OUTPUT_HI+15"] == 200.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_mem_store_addr0_local_frame_output_e8_blocks_e0_offset_row():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_local_frame_output_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+0": -1.124,
        "OUTPUT_LO+8": 4.226,
        "OUTPUT_HI+14": 161.262,
        "OUTPUT_HI+15": 141.692,
        "ALU_LO+8": 0.9937615394592285,
    })

    assert out["OUTPUT_LO+8"] == 4.226
    assert out["OUTPUT_HI+14"] == 161.262
    assert out.get("OUTPUT_LO+0", 0.0) == -1.124


def test_tail_mem_store_addr0_local_frame_output_e8_blocks_large_e0_offset_row():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_local_frame_output_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+0": 2.574,
        "OUTPUT_LO+8": 0.535,
        "OUTPUT_HI+0": -298.0,
        "OUTPUT_HI+14": 161.262,
        "OUTPUT_HI+15": 141.692,
        "ALU_LO+8": 0.9937615394592285,
    })

    assert out["OUTPUT_LO+0"] == 2.574
    assert out["OUTPUT_LO+8"] == 0.535
    assert out["OUTPUT_HI+14"] == 161.262


def test_tail_mem_store_addr0_local_frame_output_e8_blocks_jsr_e0_row():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_local_frame_output_exact")

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "OP_JSR": 12.491547584533691,
        "CMP+4": 1.497720718383789,
        "ALU_LO+14": 5360.0,
        "OUTPUT_LO+0": -10600.0,
        "OUTPUT_LO+8": 10900.0,
        "OUTPUT_HI+0": -354000.0,
        "OUTPUT_HI+14": 30600.0,
        "OUTPUT_HI+15": 11000.0,
    })

    assert out["OUTPUT_LO+0"] == -10600.0
    assert out["OUTPUT_LO+8"] == 10900.0
    assert out["OUTPUT_HI+14"] == 30600.0


def test_tail_mem_store_addr0_local_frame_output_e8_blocks_ax_imm_residue():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_local_frame_output_exact")

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "OUTPUT_LO+0": -1.34e9,
        "OUTPUT_LO+8": 1.34e9,
        "OUTPUT_HI+0": -1.35e9,
        "OUTPUT_HI+12": 1.35e9,
        "OUTPUT_HI+14": 10.0,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(1.34e9)
    assert out["OUTPUT_HI+12"] == pytest.approx(1.35e9)
    assert out["OUTPUT_LO+0"] == pytest.approx(-1.34e9)


def test_tail_mem_store_addr0_local_frame_output_e8_blocks_bp_byte3_residue():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_local_frame_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_2": 1.0,
        "OUTPUT_LO+0": 1.03e15,
        "OUTPUT_HI+0": -6.81e15,
        "OUTPUT_HI+1": 0.03,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(1.03e15)
    assert out["OUTPUT_HI+0"] == pytest.approx(-6.81e15)
    assert out["OUTPUT_HI+1"] == pytest.approx(0.03)


def test_tail_mem_store_addr0_local_frame_output_e8_blocks_sub_byte_residue():
    ir = _tail_prefix_ir("tail_mem_store_addr0_e8_from_local_frame_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "OUTPUT_LO+0": -1.47e22,
        "OUTPUT_LO+8": 1.46e22,
        "OUTPUT_HI+0": -1.81e27,
        "OUTPUT_HI+14": 1.14e26,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(-1.47e22)
    assert out["OUTPUT_LO+8"] == pytest.approx(1.46e22)
    assert out["OUTPUT_HI+0"] == pytest.approx(-1.81e27)
    assert out["OUTPUT_HI+14"] == pytest.approx(1.14e26)


def test_tail_stack0_f8_byte1_from_output_exacts_nibbles():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9965417981147766,
        "H1+10": 1.0,
        "BYTE_INDEX_0": 0.9734055995941162,
        "STACK0_BYTE0": 0.9734055995941162,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "OUTPUT_LO+0": 154.01010131835938,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+0": 157.01010131835938,
    })

    assert out["OUTPUT_LO+2"] == pytest.approx(20_000.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(20_000.0)
    for lane in range(16):
        if lane != 2:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 0:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_stack0_f8_byte1_from_output_blocks_zero_byte1():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9980018734931946,
        "H1+10": 1.0,
        "BYTE_INDEX_0": 0.9734055995941162,
        "STACK0_BYTE0": 0.9734055995941162,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "OUTPUT_LO+0": 3.9468109607696533,
        "OUTPUT_HI+0": 42.94681167602539,
    })

    assert out["OUTPUT_LO+0"] == 3.9468109607696533
    assert out["OUTPUT_HI+0"] == 42.94681167602539
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_stack0_f8_byte1_from_output_blocks_store_pop_rows():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998,
        "H1+10": 1.0,
        "BYTE_INDEX_0": 0.973,
        "STACK0_BYTE0": 0.973,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "MEM_STORE": 1.53,
        "OUTPUT_LO+0": 0.947,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+0": 0.947,
        "OUTPUT_HI+1": 3.0,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(0.947)
    assert out["OUTPUT_LO+2"] == pytest.approx(3.0)
    assert out["OUTPUT_HI+1"] == pytest.approx(3.0)


def test_tail_stack0_f8_byte1_from_output_blocks_binary_pop_rows():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9983319044113159,
        "H1+10": 1.0,
        "CMP+3": 4.0,
        "BYTE_INDEX_0": 0.9734055995941162,
        "STACK0_BYTE0": 0.9734055995941162,
        "ADDR_B0_LO+8": 0.99979,
        "ADDR_B0_HI+15": 0.99979,
        "OUTPUT_LO+0": 104.87068939208984,
        "OUTPUT_LO+2": 4.767031669616699,
        "OUTPUT_HI+0": 116.17977142333984,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(104.87068939208984)
    assert out["OUTPUT_LO+2"] == pytest.approx(4.767031669616699)
    assert out["OUTPUT_HI+0"] == pytest.approx(116.17977142333984)


def test_tail_stack0_f8_byte1_from_output_blocks_ax_byte_row():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9981348514556885,
        "H1+1": 1.0,
        "H1+10": 1.0,
        "BYTE_INDEX_0": 0.9701374769210815,
        "STACK0_BYTE0": 0.9701374769210815,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "OUTPUT_LO+1": -6140.89453125,
        "OUTPUT_LO+2": 5581.818359375,
        "OUTPUT_HI+0": 7892.8798828125,
        "OUTPUT_HI+1": 607.6930541992188,
    })

    assert out["OUTPUT_LO+1"] == -6140.89453125
    assert out["OUTPUT_LO+2"] == 5581.818359375
    assert out["OUTPUT_HI+0"] == 7892.8798828125
    assert out["OUTPUT_HI+1"] == 607.6930541992188


def test_tail_stack0_f8_byte1_from_output_requires_has_se_margin():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+10": 1.0,
        "BYTE_INDEX_0": 0.9734055995941162,
        "OUTPUT_LO+0": 142.07675170898438,
        "OUTPUT_HI+0": 142.07675170898438,
    })

    assert out["OUTPUT_LO+0"] == 142.07675170898438
    assert out["OUTPUT_HI+0"] == 142.07675170898438
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_pc_byte0_12_from_initial_jmp_exacts_nibbles():
    ir = _tail_prefix_ir("tail_pc_byte0_12_from_initial_jmp_exact")

    out = ir.symbolic_ffn({
        "MARK_PC": 1.0,
        "OP_JMP": 10.0,
        "FETCH_LO+2": 64.6191,
        "FETCH_HI+0": 68.7496,
        "OUTPUT_LO+2": 1520.0493,
        "OUTPUT_HI+0": 1593.2112,
    })

    assert out["OUTPUT_LO+2"] == pytest.approx(1520.0493)
    assert out["OUTPUT_HI+1"] == pytest.approx(5000.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(-3406.7888)


def test_tail_pc_byte0_12_from_initial_jmp_blocks_non_pc_rows():
    ir = _tail_prefix_ir("tail_pc_byte0_12_from_initial_jmp_exact")

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_JMP": 10.0,
        "FETCH_LO+2": 64.6191,
        "FETCH_HI+0": 68.7496,
        "OUTPUT_LO+2": 1520.0493,
        "OUTPUT_HI+0": 1593.2112,
    })

    assert out["OUTPUT_LO+2"] == 1520.0493
    assert out["OUTPUT_HI+0"] == 1593.2112
    assert out.get("OUTPUT_HI+1", 0.0) == 0.0


@pytest.mark.parametrize("op_dim", ["OP_BZ", "OP_BNZ"])
def test_tail_pc_byte0_1a_from_taken_branch_index3_exacts_nibbles(op_dim):
    ir = _tail_prefix_ir("tail_pc_byte0_1a_from_taken_branch_index3_exact")

    out = ir.symbolic_ffn({
        "MARK_PC": 1.0,
        op_dim: 5.0,
        "FETCH_LO+3": 40.0,
        "FETCH_HI+0": 41.0,
        "OUTPUT_LO+3": 40.0,
        "OUTPUT_HI+0": 41.0,
    })

    assert out["OUTPUT_LO+10"] == pytest.approx(5000.0)
    assert out["OUTPUT_LO+3"] == pytest.approx(-4960.0)
    assert out["OUTPUT_HI+1"] == pytest.approx(5000.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(-4959.0)


@pytest.mark.parametrize("op_dim", ["OP_BZ", "OP_BNZ"])
def test_tail_pc_byte0_1a_from_taken_branch_index3_blocks_fallthrough(op_dim):
    ir = _tail_prefix_ir("tail_pc_byte0_1a_from_taken_branch_index3_exact")

    out = ir.symbolic_ffn({
        "MARK_PC": 1.0,
        op_dim: 5.0,
        "FETCH_LO+3": 40.0,
        "FETCH_HI+0": 41.0,
        "OUTPUT_LO+2": 0.99,
        "OUTPUT_HI+1": 0.98,
    })

    assert out["OUTPUT_LO+2"] == 0.99
    assert out["OUTPUT_HI+1"] == 0.98
    assert out.get("OUTPUT_LO+10", 0.0) == 0.0
    assert out.get("OUTPUT_LO+3", 0.0) == 0.0


@pytest.mark.parametrize("op_dim", ["OP_BZ", "OP_BNZ"])
def test_tail_pc_byte0_1a_from_taken_branch_index3_blocks_plain_jmp(op_dim):
    ir = _tail_prefix_ir("tail_pc_byte0_1a_from_taken_branch_index3_exact")

    out = ir.symbolic_ffn({
        "MARK_PC": 1.0,
        "OP_JMP": 10.0,
        op_dim: -1e-21,
        "FETCH_LO+3": 2.31,
        "FETCH_HI+0": 68.75,
        "OUTPUT_LO+2": 1520.05,
        "OUTPUT_LO+3": 54.4,
        "OUTPUT_LO+10": -24.0,
        "OUTPUT_HI+0": 1593.21,
    })

    assert out["OUTPUT_LO+2"] == pytest.approx(1520.05)
    assert out["OUTPUT_LO+3"] == pytest.approx(54.4)
    assert out["OUTPUT_LO+10"] == pytest.approx(-24.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(1593.21)


def test_tail_pc_byte1_01_from_long_initial_pc_exacts_nibbles():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_long_initial_pc_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_0": 0.9701393842697144,
        "BYTE_INDEX_2": 0.0,
        "BYTE_INDEX_3": 0.0,
        "FETCH_HI+2": 39.99980163574219,
        "AX_CARRY_HI+9": 3.0,
        "NEXT_AX": 6.358210521284491e-05,
        "OUTPUT_LO+0": 0.9402,
        "OUTPUT_LO+1": -0.0001,
        "OUTPUT_HI+0": 0.9402,
    })

    assert out["OUTPUT_LO+1"] == pytest.approx(4.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(4.0)
    for lane in range(16):
        if lane != 1:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 0:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_pc_byte1_01_from_initial_jsr_fetch_hi_exacts_nibbles():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_0": 0.9701393842697144,
        "BYTE_INDEX_2": 0.0,
        "BYTE_INDEX_3": 0.0,
        "FETCH_HI+2": 39.99981689453125,
        "AX_CARRY_HI+1": 2.679687738418579,
        "NEXT_AX": 6.358210521284491e-05,
        "OUTPUT_LO+0": 2.46177339553833,
        "OUTPUT_LO+1": 37.60724639892578,
        "OUTPUT_HI+0": 4.757164478302002,
    })

    assert out["OUTPUT_LO+1"] > 37.60724639892578
    assert out["OUTPUT_HI+0"] > 4.757164478302002
    assert out["OUTPUT_LO+0"] < 2.46177339553833
    assert out.get("OUTPUT_HI+1", 0.0) < 0.0


def test_tail_pc_byte1_01_from_initial_jsr_fetch_hi_requires_initial_step():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_0": 0.9701393842697144,
        "FETCH_HI+2": 39.99981689453125,
        "AX_CARRY_HI+1": 2.679687738418579,
        "OUTPUT_LO+2": 12.0,
        "OUTPUT_HI+1": 13.0,
    })

    assert out["OUTPUT_LO+2"] == 12.0
    assert out["OUTPUT_HI+1"] == 13.0
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_pc_byte1_01_from_initial_jsr_fetch_hi_requires_byte_index0():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_1": 0.9701393842697144,
        "FETCH_HI+2": 39.99981689453125,
        "AX_CARRY_HI+1": 2.679687738418579,
        "OUTPUT_LO+0": 12.0,
        "OUTPUT_HI+0": 13.0,
    })

    assert out["OUTPUT_LO+0"] == 12.0
    assert out["OUTPUT_HI+0"] == 13.0
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_pc_byte1_01_from_initial_jsr_fetch_hi_requires_carry_sentinel():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_0": 0.9701393842697144,
        "OUTPUT_LO+0": 40.067691802978516,
        "OUTPUT_LO+1": 0.0014171712100505829,
        "OUTPUT_HI+0": 4.757174491882324,
    })

    assert out["OUTPUT_LO+0"] == 40.067691802978516
    assert out["OUTPUT_HI+0"] == 4.757174491882324
    assert out["OUTPUT_LO+1"] == 0.0014171712100505829


def test_tail_pc_byte1_01_from_initial_jsr_blocks_plain_initial_pc_byte():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_0": 0.9701393842697144,
        "BYTE_INDEX_1": 0.0132961,
        "FETCH_HI+2": 40.0016,
        "AX_CARRY_HI+1": 0.0,
        "OUTPUT_LO+0": 0.940187,
        "OUTPUT_LO+1": 0.0,
        "OUTPUT_HI+0": 0.940187,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(0.940187)
    assert out["OUTPUT_HI+0"] == pytest.approx(0.940187)
    assert out.get("OUTPUT_LO+1", 0.0) == pytest.approx(0.0)


def test_tail_pc_byte1_01_from_long_initial_pc_blocks_plain_initial_pc_byte():
    ir = _tail_prefix_ir("tail_pc_byte1_01_from_long_initial_pc_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+0": 0.9999545812606812,
        "H1+7": 0.9999545812606812,
        "H1+14": 0.9999545812606812,
        "BYTE_INDEX_0": 0.9701393842697144,
        "BYTE_INDEX_1": 0.0132961,
        "FETCH_HI+2": 40.0016,
        "AX_CARRY_HI+9": 0.0,
        "OUTPUT_LO+0": 0.940187,
        "OUTPUT_LO+1": 0.0,
        "OUTPUT_HI+0": 0.940187,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(0.940187)
    assert out["OUTPUT_HI+0"] == pytest.approx(0.940187)
    assert out.get("OUTPUT_LO+1", 0.0) == pytest.approx(0.0)


def test_tail_sp_marker_byte0_f8_from_initial_stack_exacts_nibbles():
    """B7-6: rule reads SP_BYTE0_IS_F8 + IN_STEP_FRESH structural dims as
    formal consumers (vanishingly small weights, 0.001 each) so the new
    B7-1/B7-2 dims have a downstream reader without perturbing the v3
    CMP+4 firing balance.  CMP+4 remains the dominant JSR-bootstrap
    context narrower; HAS_SE -100 retained as a soft step-0 narrowing.
    """

    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "CMP+4": 1.0,
        "SP_BYTE0_IS_F8": 1.0,
        "IN_STEP_FRESH": 0.9,
        "OUTPUT_LO+0": 0.9734733700752258,
        "OUTPUT_LO+8": 0.026526624336838722,
        "OUTPUT_HI+0": 0.97722327709198,
        "OUTPUT_HI+15": 0.02277671918272972,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(4.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(4.0)
    for lane in range(16):
        if lane != 8:
            assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        if lane != 15:
            assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_sp_marker_byte0_f8_requires_jsr_bootstrap_context():
    """B7-6: CMP+4 (JSR-bootstrap relay) remains the primary trigger.  The
    new SP_BYTE0_IS_F8 + IN_STEP_FRESH evidence is wired at vanishing
    weights (0.001), so without CMP+4 the rule does not fire even with
    those positives fully present."""

    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "SP_BYTE0_IS_F8": 1.0,
        "IN_STEP_FRESH": 0.9,
        # CMP+4 absent → no JSR-bootstrap context.
        "OUTPUT_LO+0": 4.0,
        "OUTPUT_HI+0": 4.0,
    })

    # Without CMP+4 the activation ≈ 10.02 + 0.0009 + 0.001 = 10.022;
    # threshold 10.04; rule does not fire (margin -0.018).
    assert out["OUTPUT_LO+0"] == pytest.approx(4.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(4.0)
    assert out.get("OUTPUT_LO+8", 0.0) == pytest.approx(0.0)
    assert out.get("OUTPUT_HI+15", 0.0) == pytest.approx(0.0)


def test_tail_sp_marker_byte0_f8_blocks_pure_output_residue():
    """B7-6: Stale OUTPUT 0xF8 residue alone (no CMP+4) must not relock
    the SP marker byte (the original circular self-amp bug)."""

    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        # No CMP+4, no SP_BYTE0_IS_F8, no IN_STEP_FRESH — only OUTPUT
        # residue.
        "OUTPUT_LO+0": 4.0,
        "OUTPUT_LO+8": 4.0,
        "OUTPUT_HI+0": 4.0,
        "OUTPUT_HI+15": 4.0,
    })

    # Base activation 10.02 < threshold 10.04; rule does not fire.
    assert out["OUTPUT_LO+0"] == pytest.approx(4.0)
    assert out["OUTPUT_LO+8"] == pytest.approx(4.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(4.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(4.0)


def test_tail_sp_marker_byte0_f8_consumes_sp_byte0_is_f8_dim():
    """B7-6: smoke test that the rule's conditions explicitly reference
    the new B7-1/B7-2 structural dims so the compiler dependency tracker
    sees this rule as a consumer of SP_BYTE0_IS_F8 and IN_STEP_FRESH."""
    rules = [
        rule for rule in _tail_bit32_result_correction_rules()
        if rule.name.startswith("tail_sp_marker_byte0_f8_from_initial_stack_exact")
    ]
    assert rules, "expected at least one tail_sp_marker_byte0_f8 rule"
    condition_dims = {
        term.dim.name for rule in rules for term in rule.conditions
    }
    assert "SP_BYTE0_IS_F8" in condition_dims
    assert "IN_STEP_FRESH" in condition_dims


def test_tail_sp_marker_byte0_f8_blocks_zero_stack_marker():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "OUTPUT_LO+0": 1.0,
        "OUTPUT_HI+0": 1.0,
    })

    assert out["OUTPUT_LO+0"] == 1.0
    assert out["OUTPUT_HI+0"] == 1.0
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_sp_marker_byte0_f8_blocks_initial_jsr_stack0_marker():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "OP_JSR": 11.0,
        "MARK_STACK0": 1.0,
        "H1+2": 1_000_000_000.0,
        "H1+9": 1_000_000_000.0,
        "OUTPUT_LO+0": 3.9,
        "OUTPUT_LO+8": 0.0,
        "OUTPUT_LO+10": 4.0,
        "OUTPUT_HI+0": 4.0,
        "OUTPUT_HI+15": 0.0,
    })

    assert out["OUTPUT_LO+10"] == 4.0
    assert out["OUTPUT_HI+0"] == 4.0
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_sp_marker_byte0_f8_blocks_ax_marker_output_eight_residue():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "HAS_SE": 0.997884,
        "OP_IMM": 5.0,
        "OUTPUT_LO+8": 13.0,
        "OUTPUT_HI+0": 13.0,
        "OUTPUT_HI+15": 4.0,
    })

    assert out["OUTPUT_LO+8"] == 13.0
    assert out["OUTPUT_HI+0"] == 13.0
    assert out["OUTPUT_HI+15"] == 4.0


def test_tail_sp_marker_byte0_f8_blocks_ax_imm_ff_residue():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "OP_IMM": 5.0,
        "OUTPUT_LO+15": 520.0,
        "OUTPUT_HI+15": 1_345_005_440.0,
        "OUTPUT_HI+0": -1_345_004_928.0,
        "OUTPUT_HI+14": 10.0,
        "ALU_LO+14": -50.0,
    })

    assert out["OUTPUT_LO+15"] == 520.0
    assert out["OUTPUT_HI+15"] == 1_345_005_440.0
    assert out["OUTPUT_HI+0"] == -1_345_004_928.0
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_marker_byte0_f8_blocks_lea_local_frame_marker_residue():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "HAS_SE": 0.9975294470787048,
        "OP_LEA": 5.0,
        "OUTPUT_LO+0": -54813.03125,
        "OUTPUT_LO+8": 55586.0,
        "OUTPUT_HI+0": -34047084.0,
        "OUTPUT_HI+14": 2290023.25,
        "OUTPUT_HI+15": 2290019.25,
    })

    assert out["OUTPUT_LO+8"] == 55586.0
    assert out["OUTPUT_HI+14"] == 2290023.25
    assert out["OUTPUT_HI+15"] == 2290019.25
    assert out["OUTPUT_HI+0"] == -34047084.0


def test_tail_sp_marker_byte0_f8_blocks_ent_frame_setup():
    ir = _tail_prefix_ir("tail_sp_marker_byte0_f8_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 0.9896095991134644,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "OP_ENT": 6.528390884399414,
        "OUTPUT_LO+0": 3969.28271484375,
        "OUTPUT_HI+15": 3963.10498046875,
    })

    assert out["OUTPUT_LO+0"] == 3969.28271484375
    assert out["OUTPUT_HI+15"] == 3963.10498046875
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_byte1_ff_from_initial_stack_exacts_nibbles():
    ir = _tail_prefix_ir("tail_sp_byte1_ff_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "BYTE_INDEX_0": 0.970139,
        "BYTE_INDEX_1": 0.013297,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 0.94,
        "OUTPUT_LO+15": -0.025,
        "OUTPUT_HI+0": 0.94,
        "OUTPUT_HI+15": -0.025,
    })

    assert out["OUTPUT_LO+15"] == pytest.approx(50.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(50.0)
    for lane in range(15):
        assert out.get(f"OUTPUT_LO+{lane}", 0.0) == pytest.approx(0.0)
        assert out.get(f"OUTPUT_HI+{lane}", 0.0) == pytest.approx(0.0)


def test_tail_sp_byte1_ff_from_initial_stack_requires_prior_f8_byte():
    ir = _tail_prefix_ir("tail_sp_byte1_ff_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "BYTE_INDEX_0": 0.970139,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 0.94,
        "OUTPUT_HI+0": 0.94,
    })

    assert out["OUTPUT_LO+0"] == 0.94
    assert out["OUTPUT_HI+0"] == 0.94
    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_sp_byte1_ff_from_initial_stack_reinforces_existing_ff_lane():
    ir = _tail_prefix_ir("tail_sp_byte1_ff_from_initial_stack_exact")

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "BYTE_INDEX_0": 0.970139,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+15": 5.88,
        "OUTPUT_HI+15": 5.88,
    })

    assert out["OUTPUT_LO+15"] == pytest.approx(50.0)
    assert out["OUTPUT_HI+15"] == pytest.approx(50.0)


def test_tail_sp_store_pop_byte1_zero_blocks_stored_value_high_byte():
    ir = _single_rule_ir(_tail_rule("tail_sp_store_pop_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9982727766036987,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "CMP+3": 4.0,
        "MEM_STORE": 1.2824475765228271,
        "OUTPUT_LO+0": 2.703033924102783,
        "OUTPUT_HI+0": 2.703033924102783,
        "OUTPUT_LO+2": 2.9000067710876465,
        "OUTPUT_HI+1": 2.9000067710876465,
    })

    assert out["OUTPUT_LO+0"] > 1000.0
    assert out["OUTPUT_HI+0"] > 1000.0
    assert out["OUTPUT_LO+2"] < -1000.0
    assert out["OUTPUT_HI+1"] < -1000.0


def test_tail_sp_store_pop_byte1_zero_requires_store_pop_signal():
    ir = _single_rule_ir(_tail_rule("tail_sp_store_pop_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "MEM_STORE": 1.3,
        "OUTPUT_LO+2": 2.9,
        "OUTPUT_HI+1": 2.9,
    })

    assert out["OUTPUT_LO+2"] == 2.9
    assert out["OUTPUT_HI+1"] == 2.9
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_sp_store_pop_byte1_zero_blocks_stack_source_store():
    ir = _single_rule_ir(_tail_rule("tail_sp_store_pop_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9979930520057678,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "CMP+3": 4.0,
        "MEM_STORE": 1.2824475765228271,
        "MEM_ADDR_SRC": 1.2824475765228271,
        "OUTPUT_LO+0": 5.599205493927002,
        "OUTPUT_HI+0": 5.599205493927002,
        "OUTPUT_LO+15": 4.121987196015198e-09,
        "OUTPUT_HI+15": 4.121987196015198e-09,
    })

    assert out["OUTPUT_LO+0"] == 5.599205493927002
    assert out["OUTPUT_HI+0"] == 5.599205493927002
    assert out["OUTPUT_LO+15"] == 4.121987196015198e-09
    assert out["OUTPUT_HI+15"] == 4.121987196015198e-09


def test_tail_sp_store_pop_byte1_zero_blocks_ax_add_byte1_row():
    ir = _single_rule_ir(_tail_rule("tail_sp_store_pop_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998,
        "H1+1": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 0.97,
        "CMP+3": 4.0,
        "MEM_STORE": 1.5,
        "OUTPUT_LO+0": -9117.0,
        "OUTPUT_LO+1": 9128.9,
        "OUTPUT_HI+0": -8.65e10,
        "OUTPUT_HI+1": 1.15e10,
    })

    assert out["OUTPUT_LO+1"] == pytest.approx(9128.9)
    assert out["OUTPUT_HI+1"] == pytest.approx(1.15e10)
    assert out["OUTPUT_LO+0"] == pytest.approx(-9117.0)


def test_tail_sp_store_pop_byte1_zero_blocks_ax_sub_borrow_residue():
    ir = _single_rule_ir(_tail_rule("tail_sp_store_pop_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.97,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "CMP+3": 0.008,
        "OUTPUT_LO+0": -1.47e22,
        "OUTPUT_LO+2": -1.85e18,
        "OUTPUT_HI+0": -1.81e27,
        "OUTPUT_HI+1": 1.89e26,
    })

    assert out["OUTPUT_LO+0"] == pytest.approx(-1.47e22)
    assert out["OUTPUT_LO+2"] == pytest.approx(-1.85e18)
    assert out["OUTPUT_HI+0"] == pytest.approx(-1.81e27)
    assert out["OUTPUT_HI+1"] == pytest.approx(1.89e26)


@pytest.mark.parametrize(
    "prefix",
    [
        "tail_pc_byte1_01_from_long_initial_pc_exact",
        "tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact",
        "tail_sp_marker_byte0_f8_from_initial_stack_exact",
        "tail_sp_byte1_ff_from_initial_stack_exact",
        "tail_mem_store_addr0_f8_from_mod_local_exact",
        "tail_mem_store_addr0_e0_from_local_offset_exact",
        "tail_mem_store_addr0_e0_from_jsr_local_exact",
        "tail_mem_store_addr0_e8_from_nested_local_exact",
    ],
)
def test_tail_exact_rules_block_bp_byte3_span(prefix):
    ir = _tail_prefix_ir(prefix)

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+3": 0.9999994039535522,
        "H1+10": 1.0,
        "BYTE_INDEX_2": 0.9701374769210815,
        "BYTE_INDEX_3": 0.013297064229846,
        "OUTPUT_LO+0": 0.9402737021446228,
        "OUTPUT_HI+0": 7_639_167.5,
    })

    assert out["OUTPUT_LO+0"] == 0.9402737021446228
    assert out["OUTPUT_HI+0"] == 7_639_167.5
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0
    assert out.get("OUTPUT_HI+1", 0.0) == 0.0


@pytest.mark.parametrize(
    "name",
    [
        "tail_mem_store_addr0_f8_exact",
        "tail_mem_store_addr0_f0_exact",
    ],
)
def test_tail_mem_store_addr0_exact_blocks_register_byte_span_residue(name):
    ir = _single_rule_ir(_tail_rule(name))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "BYTE_INDEX_0": 0.970139,
        "BYTE_INDEX_1": 0.013297,
        "OP_JSR": 5.0,
        "MEM_STORE": 0.656,
        "ALU_LO+2": -35.0,
        "ALU_LO+14": -35.0,
        "OUTPUT_LO+0": 0.94,
        "OUTPUT_HI+0": 0.94,
    })

    assert out["OUTPUT_LO+0"] == 0.94
    assert out["OUTPUT_HI+0"] == 0.94
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


@pytest.mark.parametrize(
    "prefix",
    [
        "tail_mem_store_addr0_f8_from_mod_local_exact",
        "tail_mem_store_addr0_e0_from_local_offset_exact",
        "tail_mem_store_addr0_e0_from_jsr_local_exact",
        "tail_mem_store_addr0_e8_from_nested_local_exact",
    ],
)
def test_tail_mem_store_addr0_band_exact_blocks_sp_byte_span_residue(prefix):
    ir = _tail_prefix_ir(prefix)

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "BYTE_INDEX_0": 0.970139,
        "BYTE_INDEX_1": 0.013297,
        "OP_JSR": 5.0,
        "MEM_STORE": 0.656,
        "ALU_LO+7": -35.0,
        "ALU_LO+10": -35.0,
        "ALU_LO+14": -35.0,
        "OUTPUT_LO+0": 0.94,
        "OUTPUT_HI+0": 0.94,
        "OUTPUT_HI+15": -0.025,
    })

    assert out["OUTPUT_LO+0"] == 0.94
    assert out["OUTPUT_HI+0"] == 0.94
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == -0.025


def test_tail_mem_store_addr0_rules_block_step_start_transition():
    for name in (
        "tail_mem_store_addr0_f8_exact",
        "tail_mem_store_addr0_f0_exact",
    ):
        ir = _single_rule_ir(_tail_rule(name))

        out = ir.symbolic_ffn({
            "NEXT_PC": 1.4,
            "MARK_MEM": 1.0,
            "HAS_SE": 1.0,
            "H1+4": 1.0,
            "MEM_STORE": 2.0,
            "CMP+0": 3.0,
            "OUTPUT_LO+0": -1000.0,
            "OUTPUT_LO+8": 1000.0,
            "OUTPUT_LO+14": -1000.0,
            "ALU_LO+14": 0.0,
            "OUTPUT_HI+15": 1000.0,
        })

        assert out["OUTPUT_LO+8"] == 1000.0
        assert out["OUTPUT_HI+15"] == 1000.0


def test_tail_stack0_pop_reveals_saved_addr_e8_after_d8_pop():
    ir = CompilerIR()
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_pop_marker_zero"))
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_pop_reveals_saved_addr_e8"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+13": 1.0,
    })

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+14"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_stack0_pop_reveals_saved_addr_e8_from_e0_addr():
    ir = CompilerIR()
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_pop_marker_zero"))
    ir.layer(0).ffn.append(
        _tail_rule("tail_stack0_pop_reveals_saved_addr_e8_from_e0_addr")
    )

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "ADDR_B0_HI+13": -1.0,
    })

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+14"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_stack0_pop_reveals_saved_addr_e8_blocks_d0_pop():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_reveals_saved_addr_e8"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+13": 1.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_stack0_pop_reveals_saved_addr_e8_requires_pop_signal():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_reveals_saved_addr_e8"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 10.0,
        "ADDR_B0_HI+13": 5.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_stack0_pop_reveals_saved_addr_e8_blocks_ent_frame_marker():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_reveals_saved_addr_e8"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9956,
        "OP_ENT": 8.574,
        "ADDR_B0_LO+8": 26.419,
        "ADDR_B0_HI+13": -2.0,
        "ADDR_B0_LO+0": -24.332,
        "MEM_STORE": 0.407,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_stack0_store_correction_is_owned_by_attention_route():
    names = {rule.name for rule in _tail_bit32_result_correction_rules()}

    assert "tail_stack0_store_byte_de" not in names
    assert "tail_stack0_store_byte_09" not in names
    assert "tail_stack0_store_byte_90" not in names
    assert "tail_stack0_store_byte_00" not in names


def test_tail_stack0_store_non_top_zero_blocks_local_store_value():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+13": 1.0,
        "OUTPUT_LO+5": 3.0,
        "OUTPUT_HI+1": 3.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+5"] < 3.0
    assert out["OUTPUT_HI+1"] < 3.0


def test_tail_stack0_store_non_top_zero_preserves_strong_loaded_stack_value():
    ir = CompilerIR()
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_non_top_zero"))
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_loaded_byte_01"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+13": 1.0,
        "OUTPUT_LO+1": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+1"] > 40.0
    assert out["OUTPUT_HI+0"] > 40.0
    assert out["OUTPUT_LO+0"] < 0.0


def test_tail_stack0_store_loaded_blocks_byte_rows_with_large_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_loaded_byte_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+1": 2_000_000_000.0,
        "OUTPUT_HI+0": 2_000_000_000.0,
    })

    assert out["OUTPUT_LO+1"] == 2_000_000_000.0
    assert out["OUTPUT_HI+0"] == 2_000_000_000.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_stack0_store_loaded_blocks_bp_byte_span_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_loaded_byte_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_2": 0.97,
        "OUTPUT_LO+2": 8_000_000.0,
        "OUTPUT_HI+0": 8_000_000.0,
    })

    assert out["OUTPUT_LO+2"] == 8_000_000.0
    assert out["OUTPUT_HI+0"] == 8_000_000.0


def test_tail_sp_pop_byte3_zero_blocks_ax_byte_span_with_large_residue():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 2_000_000_000.0,
        "OUTPUT_HI+0": 2_000_000_000.0,
    })

    assert out["OUTPUT_LO+0"] == 2_000_000_000.0
    assert out["OUTPUT_HI+0"] == 2_000_000_000.0
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_stack0_store_non_top_zero_allows_postpop_e8_top_store():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+3": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out["OUTPUT_LO+2"] == 3.0
    assert out["OUTPUT_HI+3"] == 3.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_blocks_local_store():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "OUTPUT_LO+5": 3.0,
        "OUTPUT_HI+1": 3.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+5"] < 3.0
    assert out["OUTPUT_HI+1"] < 3.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_preserves_loaded_stack_value():
    ir = CompilerIR()
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_loaded_byte_01"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "OUTPUT_LO+1": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+1"] > 40.0
    assert out["OUTPUT_HI+0"] > 40.0
    assert out["OUTPUT_LO+0"] < 0.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_requires_store_signal():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "EMBED_LO+0": 1.0,
        "ADDR_B0_LO+0": 7.0,
        "ADDR_B0_HI+13": 2.0,
        "ADDR_B0_HI+14": -2.0,
        "ADDR_B0_LO+8": -1.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+14": 3.0,
    })

    assert out["OUTPUT_LO+0"] == 3.0
    assert out["OUTPUT_HI+14"] == 3.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_blocks_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "MEM_STORE": 1.3,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+15": 1.0,
        "OUTPUT_LO+15": 3.0,
        "OUTPUT_HI+15": 3.0,
    })

    assert out["OUTPUT_LO+15"] == 3.0
    assert out["OUTPUT_HI+15"] == 3.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_blocks_mem_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "OUTPUT_LO+8": 3.0,
        "OUTPUT_HI+15": 3.0,
    })

    assert out["OUTPUT_LO+8"] == 3.0
    assert out["OUTPUT_HI+15"] == 3.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_blocks_stack_source_store():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9984076023101807,
        "CMP+3": 4.0,
        "MEM_STORE": 0.40725404024124146,
        "MEM_ADDR_SRC": 0.40725404024124146,
        "EMBED_LO+8": 5.3777512221131474e-08,
        "EMBED_HI+14": 3.57113009386012e-07,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "ADDR_B0_LO+8": -1.0,
        "OUTPUT_LO+15": 3.0000078678131104,
        "OUTPUT_HI+2": 3.0,
    })

    assert out["OUTPUT_LO+15"] == 3.0000078678131104
    assert out["OUTPUT_HI+2"] == 3.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_stack0_store_non_top_zero_allows_top_store_value():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+14": 1.0,
        "OUTPUT_LO+12": 3.0,
        "OUTPUT_HI+1": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out["OUTPUT_LO+12"] == 3.0
    assert out["OUTPUT_HI+1"] == 3.0


def test_tail_stack0_store_non_top_zero_blocks_e0_top_signature():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+13": 1.0,
        "OUTPUT_LO+5": 3.0,
        "OUTPUT_HI+1": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out["OUTPUT_LO+5"] == 3.0
    assert out["OUTPUT_HI+1"] == 3.0


def test_tail_stack0_store_non_top_zero_e0_blocks_middle_local_store():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e0"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+14": 1.0,
        "OUTPUT_LO+6": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 3.0
    assert out["OUTPUT_LO+6"] < 3.0


def test_tail_stack0_store_non_top_zero_e0_preserves_loaded_stack_value():
    ir = CompilerIR()
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_non_top_zero_e0"))
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_loaded_byte_01"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+14": 1.0,
        "OUTPUT_LO+1": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+1"] > 40.0
    assert out["OUTPUT_HI+0"] > 40.0
    assert out["OUTPUT_LO+0"] < 0.0


def test_tail_stack0_store_non_top_zero_e0_blocks_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e0"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+14": 1.0,
        "OUTPUT_LO+6": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+6"] == 3.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_stack0_store_non_top_zero_e8_from_e0_blocks_e0_top_signature():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_non_top_zero_e8_from_e0"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "OUTPUT_LO+5": 3.0,
        "OUTPUT_HI+1": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out["OUTPUT_LO+5"] == 3.0
    assert out["OUTPUT_HI+1"] == 3.0


def test_tail_stack0_store_top_e0_restores_staged_nonzero_value():
    ir = CompilerIR()
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_non_top_zero_e0"))
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_store_top_e0_byte_01"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+14": 1.0,
        "ALU_LO+1": 1.0,
        "ALU_HI+0": 7.0,
        "OUTPUT_LO+1": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+1"] > 3.0
    assert out["OUTPUT_HI+0"] > 3.0
    assert out["OUTPUT_LO+0"] < 0.0


def test_tail_stack0_store_top_e0_requires_strong_e0_store_address():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_top_e0_byte_02"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.5,
        "EMBED_LO+0": 0.004,
        "EMBED_HI+14": 1.0,
        "ALU_LO+2": 1.0,
        "ALU_HI+0": 7.0,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+0": 43.0,
    })

    assert out["OUTPUT_LO+2"] == 3.0
    assert out["OUTPUT_HI+0"] == 43.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_stack0_pop_loaded_blocks_large_ax_byte_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_loaded_byte_09"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+9": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+9"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_stack0_pop_loaded_blocks_store_row_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_loaded_byte_39"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9985874891281128,
        "CMP+3": 4.0,
        "MEM_STORE": 0.40725404024124146,
        "OUTPUT_LO+9": 334.4128723144531,
        "OUTPUT_HI+3": 518.2252197265625,
    })

    assert out["OUTPUT_LO+9"] == 334.4128723144531
    assert out["OUTPUT_HI+3"] == 518.2252197265625
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_stack0_pop_loaded_blocks_bp_byte_span_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pop_loaded_byte_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_2": 0.97,
        "OUTPUT_LO+2": 8_000_000.0,
        "OUTPUT_HI+0": 8_000_000.0,
    })

    assert out["OUTPUT_LO+2"] == 8_000_000.0
    assert out["OUTPUT_HI+0"] == 8_000_000.0


def test_tail_stack0_store_top_e0_blocks_large_ax_byte_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_top_e0_byte_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 1.0,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+14": 1.0,
        "ALU_LO+1": 1.0,
        "ALU_HI+0": 7.0,
        "OUTPUT_LO+1": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+1"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_stack0_store_top_e0_does_not_restore_stale_e0_address():
    names = {rule.name for rule in _tail_bit32_result_correction_rules()}

    assert "tail_stack0_store_top_e0_byte_e0" not in names


def test_tail_stack0_store_top_e8_from_e0_restores_current_store_value():
    ir = _tail_rules_ir(
        "tail_stack0_store_non_top_zero_e8_from_e0",
        "tail_stack0_store_top_e8_from_e0_byte_de",
    )

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998318076133728,
        "CMP+3": 4.0,
        "MEM_STORE": 0.40725404024124146,
        "EMBED_LO+8": 0.9920187592506409,
        "EMBED_HI+14": 0.9920188188552856,
        "ADDR_B0_LO+0": 7.442606449127197,
        "ADDR_B0_HI+0": 4.009108066558838,
        "ADDR_B0_HI+14": 1.97,
        "H1+3": 0.003357573179528117,
        "OUTPUT_LO+14": 3.0,
        "OUTPUT_HI+13": 3.0,
    })

    assert out["OUTPUT_LO+14"] > 3.0
    assert out["OUTPUT_HI+13"] > 3.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_stack0_store_top_e8_from_e0_restores_39_from_e8_addr():
    ir = _tail_rules_ir(
        "tail_stack0_pop_loaded_byte_30",
        "tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr",
    )

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9985874891281128,
        "CMP+3": 4.0,
        "MEM_STORE": 0.40725404024124146,
        "MEM_ADDR_SRC": 0.40725404024124146,
        "EMBED_LO+8": 3.582539704893861e-07,
        "EMBED_HI+14": 4.122032919440244e-07,
        "ADDR_B0_LO+0": -2.008293867111206,
        "ADDR_B0_HI+0": -1.9909181594848633,
        "ADDR_B0_LO+8": 1.990378737449646,
        "ADDR_B0_HI+14": 0.9730035066604614,
        "OUTPUT_LO+0": 1128.9244384765625,
        "OUTPUT_HI+0": 1095.1925048828125,
        "OUTPUT_LO+9": 334.4128723144531,
        "OUTPUT_HI+3": 518.2252197265625,
    })

    assert out["OUTPUT_LO+9"] > 334.4128723144531
    assert out["OUTPUT_HI+3"] > 518.2252197265625
    assert out["OUTPUT_LO+0"] < 1128.9244384765625
    assert out["OUTPUT_HI+0"] < 1095.1925048828125


def test_tail_stack0_store_top_e8_from_e0_ignores_initial_stack0_value():
    ir = _tail_prefix_ir("tail_stack0_store_top_e8_from_e0_byte_")

    state = {
        "MARK_STACK0": 1.0,
        "MEM_STORE": 0.40725401043891907,
        "ADDR_B0_LO+8": 1.0000001192092896,
        "ADDR_B0_HI+15": 1.0000003576278687,
        "OUTPUT_LO+10": 507625.5625,
        "OUTPUT_HI+0": 507625.5625,
    }
    for i in range(16):
        state.setdefault(f"OUTPUT_LO+{i}", -507622.5625)
        state.setdefault(f"OUTPUT_HI+{i}", -507622.5625)

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+10"] == 507625.5625
    assert out["OUTPUT_HI+0"] == 507625.5625
    assert out["OUTPUT_HI+1"] == -507622.5625


def test_tail_stack0_store_top_e8_from_e0_39_requires_low9_evidence():
    ir = _single_rule_ir(
        _tail_rule("tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr")
    )

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9981152415275574,
        "ADDR_B0_LO+8": 7.971285343170166,
        "ADDR_B0_HI+14": 7.971285343170166,
        "ADDR_B0_LO+0": -1.9924825429916382,
        "ADDR_B0_HI+0": -1.9924836158752441,
        "MEM_STORE": 1.2841345778724644e-05,
        "MEM_ADDR_SRC": 1.2841345778724644e-05,
        "OUTPUT_LO+2": 4058.505859375,
        "OUTPUT_LO+9": 3.938452982597199e-18,
        "OUTPUT_HI+3": 4058.5029296875,
    })

    assert out["OUTPUT_LO+2"] == 4058.505859375
    assert out["OUTPUT_HI+3"] == 4058.5029296875
    assert out.get("OUTPUT_LO+9", 0.0) == 3.938452982597199e-18


def test_tail_stack0_store_top_e8_from_e0_requires_l15_top_signature():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_top_e8_from_e0_byte_de"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998318076133728,
        "CMP+3": 4.0,
        "MEM_STORE": 0.40725404024124146,
        "EMBED_LO+8": 0.9920187592506409,
        "EMBED_HI+14": 0.9920188188552856,
        "ADDR_B0_LO+0": 7.442606449127197,
        "ADDR_B0_HI+14": 4.009108066558838,
        "OUTPUT_LO+14": 3.0,
        "OUTPUT_HI+13": 3.0,
    })

    assert out["OUTPUT_LO+14"] == 3.0
    assert out["OUTPUT_HI+13"] == 3.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_stack0_store_top_e8_from_e0_blocks_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_top_e8_from_e0_byte_de"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 1.0,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+14": 1.0,
        "ADDR_B0_LO+0": 8.0,
        "ADDR_B0_HI+0": 4.0,
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "OUTPUT_LO+14": 3.0,
        "OUTPUT_HI+13": 3.0,
    })

    assert out["OUTPUT_LO+14"] == 3.0
    assert out["OUTPUT_HI+13"] == 3.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_stack0_store_top_e8_from_e0_requires_positive_address_signature():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_top_e8_from_e0_byte_30"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9956295490264893,
        "MEM_STORE": 0.40726688504219055,
        "EMBED_LO+8": 3.26611079799477e-05,
        "EMBED_HI+14": 3.2605676096864045e-05,
        "ADDR_B0_LO+0": -12.62271499633789,
        "ADDR_B0_HI+0": -0.5151978731155396,
        "H1+3": 0.003357573179528117,
        "OUTPUT_LO+0": 14.835540771484375,
        "OUTPUT_HI+3": 4.382873726171965e-07,
    })

    assert out["OUTPUT_LO+0"] == 14.835540771484375
    assert out["OUTPUT_HI+3"] == 4.382873726171965e-07
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_stack0_store_top_e8_from_e0_requires_store_signal():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_top_e8_from_e0_byte_0f"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.997279,
        "CMP+8": 0.697795,
        "EMBED_LO+0": 0.995722,
        "EMBED_HI+0": 0.99689,
        "ADDR_B0_LO+0": 7.975916,
        "ADDR_B0_HI+0": 7.975923,
        "OUTPUT_LO+0": 1.980587,
        "OUTPUT_HI+0": 1.982908,
    })

    assert out["OUTPUT_LO+0"] == 1.980587
    assert out["OUTPUT_HI+0"] == 1.982908
    assert out.get("OUTPUT_LO+15", 0.0) == 0.0


def test_tail_stack0_store_loaded_byte11_blocks_bp_byte3_zero_row():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_loaded_byte_11"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_2": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 4.0,
        "MARK_STACK0": 1.0,
        "OUTPUT_LO+0": 8_000_000.0,
        "OUTPUT_HI+0": 8_000_000.0,
        "OUTPUT_LO+1": 8_000_000.0,
        "OUTPUT_HI+1": 8_000_000.0,
    })

    assert out["OUTPUT_LO+0"] == 8_000_000.0
    assert out["OUTPUT_HI+0"] == 8_000_000.0
    assert out["OUTPUT_LO+1"] == 8_000_000.0
    assert out["OUTPUT_HI+1"] == 8_000_000.0


def test_tail_stack0_store_loaded_byte11_still_fires_on_stack0_marker():
    ir = _single_rule_ir(_tail_rule("tail_stack0_store_loaded_byte_11"))

    out = ir.symbolic_ffn({
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 4.0,
        "MARK_STACK0": 1.0,
        "OUTPUT_LO+1": 8_000_000.0,
        "OUTPUT_HI+1": 8_000_000.0,
    })

    assert out["OUTPUT_LO+1"] > 8_000_000.0
    assert out["OUTPUT_HI+1"] > 8_000_000.0
    assert out.get("OUTPUT_LO+0", 0.0) < 0.0
    assert out.get("OUTPUT_HI+0", 0.0) < 0.0


def test_tail_stack0_store_top_value_2f_from_alu_stays_inactive():
    with pytest.raises(AssertionError, match="tail_stack0_store_top_value_2f_from_alu"):
        _tail_rule("tail_stack0_store_top_value_2f_from_alu")


def test_tail_pop_mem_marker_zero_blocks_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_pop_mem_marker_zero"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 1.0,
        "IS_BYTE": 1.0,
        "OUTPUT_LO+6": 2.0,
        "OUTPUT_HI+7": 2.0,
    })

    assert out["OUTPUT_LO+6"] == 2.0
    assert out["OUTPUT_HI+7"] == 2.0


def test_tail_pop_mem_marker_zero_blocks_sp_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_pop_mem_marker_zero"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+8": 2.0,
        "OUTPUT_HI+14": 2.0,
    })

    assert out["OUTPUT_LO+8"] == 2.0
    assert out["OUTPUT_HI+14"] == 2.0


def test_tail_sp_pop_marker_increments_staged_sp_byte():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 2.8,
        "OUTPUT_HI+14": 1.0,
    })

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+14"] > 1.0


def test_tail_sp_pop_marker_increment_blocks_non_sp_markers():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 2.8,
        "OUTPUT_HI+14": 1.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out["OUTPUT_HI+14"] == 1.0


def test_tail_sp_pop_marker_increment_blocks_ent_frames():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OP_ENT": 1.0,
        "OUTPUT_LO+0": 4000.0,
        "OUTPUT_HI+14": 4000.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_pop_marker_increment_blocks_store_steps():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OP_SI": 1.0,
        "MEM_STORE": 1.0,
        "OUTPUT_LO+0": 4000.0,
        "OUTPUT_HI+14": 4000.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_pop_marker_increment_requires_e0_high_nibble():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 42.0,
        "OUTPUT_HI+13": 1.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_pop_marker_increment_requires_pop_flag():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+14": 3.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out["OUTPUT_HI+14"] == 3.0


def test_tail_sp_pop_marker_increment_blocks_d8_embed_signature():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+14": 3.0,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+13": 1.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 3.0
    assert out["OUTPUT_HI+14"] == 3.0


def test_tail_sp_pop_marker_increment_blocks_mem_store_residue():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_e0_to_e8"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.001,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+14": 1.0,
    })

    assert out.get("OUTPUT_LO+8", 0.0) == 0.0
    assert out["OUTPUT_HI+14"] == 1.0


def test_tail_sp_pop_marker_d8_to_e0_restores_store_pop():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_d8_to_e0"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+13": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+0"] > 2.0
    assert out["OUTPUT_HI+14"] > 0.0
    assert out["OUTPUT_HI+0"] < 2.0


def test_tail_sp_pop_marker_output_d8_to_e0_restores_store_pop():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_output_d8_to_e0"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+8": 1.0,
        "OUTPUT_HI+13": 1.0,
        "OUTPUT_LO+0": 42.0,
        "OUTPUT_HI+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] > 42.0
    assert out["OUTPUT_HI+14"] > 0.0
    assert out["OUTPUT_LO+8"] < 1.0
    assert out["OUTPUT_HI+13"] < 1.0


def test_tail_sp_pop_marker_output_d8_to_e0_requires_output_low_nibble():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_output_d8_to_e0"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 42.0,
        "OUTPUT_HI+13": 1.0,
        "OUTPUT_HI+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] == 42.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_sp_pop_marker_output_d8_to_e0_requires_output_high_nibble():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_output_d8_to_e0"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+8": 1.0,
        "OUTPUT_LO+0": 42.0,
        "OUTPUT_HI+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] == 42.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_sp_pop_marker_output_d8_to_e0_blocks_f8_marker():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_output_d8_to_e0"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+8": 1.0,
        "OUTPUT_HI+15": 1.0,
        "OUTPUT_LO+0": 42.0,
        "OUTPUT_HI+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] == 42.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_sp_pop_marker_output_d8_to_e0_blocks_ax_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_output_d8_to_e0"))

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+8": 1_000_000.0,
        "OUTPUT_HI+13": 1_000_000.0,
        "OUTPUT_LO+0": 42.0,
        "OUTPUT_HI+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] == 42.0
    assert out["OUTPUT_HI+0"] == 42.0
    assert out["OUTPUT_LO+8"] == 1_000_000.0
    assert out["OUTPUT_HI+13"] == 1_000_000.0


def test_tail_sp_pop_marker_d8_to_e0_requires_d8_low_nibble():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_d8_to_e0"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+13": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+0"] == 2.0
    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_sp_pop_marker_d0_to_d8_restores_larger_frame_pop():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_d0_to_d8"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+13": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+13": 2.0,
    })

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+13"] > 2.0
    assert out["OUTPUT_LO+0"] < 2.0


def test_tail_sp_pop_marker_d0_to_d8_blocks_d8_embed_signature():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_d0_to_d8"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "EMBED_LO+0": 0.01,
        "EMBED_LO+8": 1.0,
        "EMBED_HI+13": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+13": 2.0,
    })

    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+13"] == 2.0
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_pop_marker_f0_to_f8_restores_larger_frame_pop():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_f0_to_f8"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+15": 2.0,
    })

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+15"] > 2.0
    assert out["OUTPUT_LO+0"] < 2.0


def test_tail_sp_pop_marker_f0_to_f8_blocks_mem_store_residue():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_f0_to_f8"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "MEM_STORE": 0.001,
        "EMBED_LO+0": 1.0,
        "EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+15": 2.0,
    })

    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+15"] == 2.0
    assert out.get("OUTPUT_LO+8", 0.0) == 0.0


def test_tail_sp_pop_byte1_ff_after_f8_preserves_stack_high_byte():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_f8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+0"] < 2.0
    assert out["OUTPUT_HI+0"] < 2.0


def test_tail_sp_pop_marker_d8_to_e0_blocks_push_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_marker_d8_to_e0"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        "PSH_AT_SP": 1.0,
        "EMBED_LO+8": 1000.0,
        "EMBED_HI+13": 1000.0,
        "OUTPUT_HI+13": 2.0,
    })

    assert out.get("OUTPUT_HI+14", 0.0) == 0.0


def test_tail_sp_pop_byte1_preserves_ff_after_e8():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+2": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+2"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_sp_pop_byte1_preserves_ff_after_e0():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_e0"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_sp_pop_byte1_preserves_ff_after_d8():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_d8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+13": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_sp_pop_byte3_zero_blocks_byte1_residue():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.997,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 0.97,
        "BYTE_INDEX_2": 0.000001,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 42.9,
        "OUTPUT_HI+0": 2.9,
        "OUTPUT_LO+15": 7.0,
        "OUTPUT_HI+15": 7.0,
    })

    assert out["OUTPUT_LO+0"] == 42.9
    assert out["OUTPUT_HI+0"] == 2.9
    assert out["OUTPUT_LO+15"] == 7.0
    assert out["OUTPUT_HI+15"] == 7.0


def test_tail_sp_pop_byte3_zero_fires_on_byte3_signature():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.997,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 0.97,
        "BYTE_INDEX_3": 0.013,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 2.9,
        "OUTPUT_HI+0": 2.9,
        "OUTPUT_LO+1": 7.0,
        "OUTPUT_HI+1": 7.0,
    })

    assert out["OUTPUT_LO+0"] > 2.9
    assert out["OUTPUT_HI+0"] > 2.9
    assert out["OUTPUT_LO+1"] < 7.0
    assert out["OUTPUT_HI+1"] < 7.0


def test_tail_sp_pop_byte1_ff_after_e8_blocks_ax_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+2": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+2"] == 2.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_sp_pop_byte1_ff_after_e8_blocks_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+2": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+2"] == 2.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_sp_pop_byte1_ff_after_e8_requires_exact_low_nibble():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+4": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_sp_pop_byte1_ff_after_e8_blocks_stack0_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "STACK0_BYTE0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_stack0_pushed_addr_byte1_preserves_ff_after_e8():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pushed_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+15"] > 0.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0


def test_tail_stack0_pushed_addr_byte1_blocks_e4_high_nibble_match():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pushed_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+4": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 3.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_stack0_pushed_addr_byte1_blocks_e4_with_e8_residue():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pushed_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+4": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 3.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_stack0_pushed_addr_byte1_blocks_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pushed_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0
    assert out["OUTPUT_LO+0"] == 3.0
    assert out["OUTPUT_HI+0"] == 3.0


def test_tail_stack0_pushed_addr_byte1_blocks_store_rows():
    ir = _single_rule_ir(_tail_rule("tail_stack0_pushed_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9970132112503052,
        "STACK0_BYTE0": 0.9734055995941162,
        "BYTE_INDEX_0": 0.9734055995941162,
        "MEM_STORE": 1.5293787717819214,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+0": 5.053187847137451,
    })

    assert out["OUTPUT_LO+2"] == pytest.approx(3.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(5.053187847137451)
    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_stack0_pushed_addr_byte1_store_preserves_ff_after_e8():
    ir = _single_rule_ir(
        _tail_rule("tail_stack0_pushed_addr_byte1_store_ff_after_e8")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9972294569015503,
        "H1+10": 0.9999994039535522,
        "STACK0_BYTE0": 0.9734055995941162,
        "BYTE_INDEX_0": 0.9734055995941162,
        "MEM_STORE": 1.5293787717819214,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.9468109607696533,
        "OUTPUT_HI+0": 3.9468109607696533,
        "OUTPUT_LO+15": 3.0,
        "OUTPUT_HI+15": 3.3088135719299316,
    })

    assert out["OUTPUT_LO+15"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+15"] > out["OUTPUT_HI+0"]


def test_tail_stack0_pushed_addr_byte1_store_requires_ff_residue():
    ir = _single_rule_ir(
        _tail_rule("tail_stack0_pushed_addr_byte1_store_ff_after_e8")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9970132112503052,
        "H1+10": 1.0,
        "STACK0_BYTE0": 0.9734055995941162,
        "BYTE_INDEX_0": 0.9734055995941162,
        "MEM_STORE": 1.5293787717819214,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+0": 5.053187847137451,
    })

    assert out["OUTPUT_LO+2"] == pytest.approx(3.0)
    assert out["OUTPUT_HI+0"] == pytest.approx(5.053187847137451)
    assert out.get("OUTPUT_LO+15", 0.0) == 0.0
    assert out.get("OUTPUT_HI+15", 0.0) == 0.0


def test_tail_stack0_pushed_addr_byte1_store_preserves_ff_after_e0():
    ir = _single_rule_ir(
        _tail_rule("tail_stack0_pushed_addr_byte1_store_ff_after_e0")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9982361793518066,
        "H1+10": 0.9999994039535522,
        "STACK0_BYTE0": 0.9734055995941162,
        "BYTE_INDEX_0": 0.9734055995941162,
        "MEM_STORE": 1.5293787717819214,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 3.9468109607696533,
        "OUTPUT_HI+0": 3.9468109607696533,
        "OUTPUT_LO+15": 3.0,
        "OUTPUT_HI+15": 3.3088135719299316,
    })

    assert out["OUTPUT_LO+15"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+15"] > out["OUTPUT_HI+0"]


def test_tail_stack0_pushed_addr_byte1_loses_to_exact_existing_byte2():
    ir = _tail_prefix_ir("tail_stack0_f8_byte1_from_output_exact")
    ir.layer(0).ffn.append(_tail_rule("tail_stack0_pushed_addr_byte1_ff_after_e8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9980018734931946,
        "H1+10": 1.0,
        "STACK0_BYTE0": 0.9734055995941162,
        "BYTE_INDEX_0": 0.9734055995941162,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "CLEAN_EMBED_LO+8": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+2": 3.0,
        "OUTPUT_HI+0": 42.94681167602539,
    })

    assert out["OUTPUT_LO+2"] > out.get("OUTPUT_LO+15", 0.0)
    assert out["OUTPUT_HI+0"] > out.get("OUTPUT_HI+15", 0.0)


def test_tail_ax_add_no_carry_byte1_zeros_stale_address_output():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_LO+0": 6.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "OUTPUT_LO+0": 12.0,
        "OUTPUT_LO+1": 12.0,
        "OUTPUT_HI+15": 12.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_LO+1"] < 12.0
    assert out["OUTPUT_HI+15"] < 12.0


def test_tail_ax_add_no_carry_byte1_blocks_nonzero_high_byte_low_nibble():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "ALU_LO+5": 1.0,
        "OUTPUT_LO+5": 5.0,
        "OUTPUT_HI+0": 5.0,
    })

    assert out["OUTPUT_LO+5"] == 5.0
    assert out["OUTPUT_HI+0"] == 5.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_byte1_blocks_nonzero_second_operand_high_byte():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "AX_CARRY_LO+2": 1.0,
        "OUTPUT_LO+2": 5.0,
        "OUTPUT_HI+0": 5.0,
    })

    assert out["OUTPUT_LO+2"] == 5.0
    assert out["OUTPUT_HI+0"] == 5.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_byte1_blocks_real_carry():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "CARRY+1": 1.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "OUTPUT_LO+1": 12.0,
        "OUTPUT_HI+15": 12.0,
    })

    assert out["OUTPUT_LO+1"] == 12.0
    assert out["OUTPUT_HI+15"] == 12.0


def test_tail_ax_add_no_carry_byte1_blocks_real_carry_huge_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296722434461117,
        "TEMP+8": 1.304825782775879,
        "CARRY+1": 2.0,
        "ALU_LO+0": 6.236810207366943,
        "ALU_HI+0": 6.236800670623779,
        "AX_CARRY_HI+0": 3.7277987003326416,
        "OUTPUT_LO+0": 2.36312338633608e14,
        "OUTPUT_LO+3": 2.1993707531384e13,
        "OUTPUT_HI+0": 4.041023946752e12,
    })

    assert out["OUTPUT_LO+0"] == 2.36312338633608e14
    assert out["OUTPUT_LO+3"] == 2.1993707531384e13
    assert out["OUTPUT_HI+0"] == 4.041023946752e12


def test_tail_ax_add_no_carry_byte1_blocks_sp_frame_row_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    row = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H1+2": 1.0,
        "MARK_SP": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 10.0,
        "AX_CARRY_HI+0": 10.0,
        "OUTPUT_LO+8": 12.0,
        "OUTPUT_HI+15": 13.0,
    }
    for other in range(1, 16):
        row[f"ALU_LO+{other}"] = -100.0

    out = ir.symbolic_ffn(row)

    assert out["OUTPUT_LO+8"] == 12.0
    assert out["OUTPUT_HI+15"] == 13.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_byte1_blocks_stack0_frame_row_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    row = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "MARK_STACK0": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 10.0,
        "AX_CARRY_HI+0": 10.0,
        "OUTPUT_LO+10": 12.0,
        "OUTPUT_HI+0": 13.0,
    }
    for other in range(1, 16):
        row[f"ALU_LO+{other}"] = -100.0

    out = ir.symbolic_ffn(row)

    assert out["OUTPUT_LO+10"] == 12.0
    assert out["OUTPUT_HI+0"] == 13.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_byte1_blocks_ent_mem_value_row_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    row = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "OP_ENT": 5.0,
        "MEM_STORE": 1.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 10.0,
        "AX_CARRY_HI+0": 10.0,
        "OUTPUT_LO+1": 12.0,
        "OUTPUT_HI+0": 13.0,
    }
    for other in range(1, 16):
        row[f"ALU_LO+{other}"] = -35.0

    out = ir.symbolic_ffn(row)

    assert out["OUTPUT_LO+1"] == 12.0
    assert out["OUTPUT_HI+0"] == 13.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_byte1_blocks_weak_high_byte_source():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 1.0000001192092896,
        "ALU_LO+0": 6.236810207366943,
        "ALU_HI+0": 6.236800670623779,
        "AX_CARRY_LO+2": 0.07984784990549088,
        "AX_CARRY_HI+0": 3.7277987003326416,
        "OUTPUT_LO+0": -4.184763431549072,
        "OUTPUT_LO+2": 7.125039100646973,
        "OUTPUT_HI+0": 12.77362060546875,
    })

    assert out["OUTPUT_LO+2"] == 7.125039100646973
    assert out["OUTPUT_HI+0"] == 12.77362060546875
    assert out.get("OUTPUT_LO+0", 0.0) == -4.184763431549072


def test_tail_ax_add_no_carry_byte1_blocks_bitwise_relay():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+3": 1.0,
        "TEMP+5": 1.0,
        "TEMP+8": 1.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 10.0,
        "AX_CARRY_HI+0": 10.0,
        "OUTPUT_LO+15": 3.8,
        "OUTPUT_HI+0": 9.5,
    })

    assert out["OUTPUT_LO+15"] == 3.8
    assert out["OUTPUT_HI+0"] == 9.5
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_ax_add_no_carry_byte1_blocks_sub_relay():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_no_carry_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "TEMP+9": 1.0,
        "OP_SUB": 1.0,
        "ALU_LO+0": 10.0,
        "ALU_HI+0": 10.0,
        "AX_CARRY_HI+0": 10.0,
        "OUTPUT_LO+15": 7.0,
        "OUTPUT_HI+15": 8.0,
    })

    assert out["OUTPUT_LO+15"] == 7.0
    assert out["OUTPUT_HI+15"] == 8.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_ax_add_byte1_hi_zero_repairs_stale_high_nibble():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_hi_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "OUTPUT_LO+2": 10.0,
        "OUTPUT_HI+1": 4.0,
    })

    assert out["OUTPUT_LO+2"] == 10.0
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_HI+1"] < 4.0


def test_tail_ax_add_byte1_hi_zero_overrides_block27_scale():
    ir = _tail_rules_ir(
        "tail_ax_add_byte1_hi_zero_lo_0",
        "tail_ax_add_byte1_hi_zero_lo_3",
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0000001192092896,
        "ALU_HI+0": 6.236852645874023,
        "AX_CARRY_HI+0": 3.941866397857666,
        "OUTPUT_LO+0": 2000.0,
        "OUTPUT_LO+3": 3000.0,
        "OUTPUT_HI+0": 100.0,
        "OUTPUT_HI+1": 4000.0,
    })

    assert out["OUTPUT_LO+3"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] > out["OUTPUT_HI+1"]


def test_tail_ax_add_byte1_hi_zero_blocks_sub_rows():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_hi_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "TEMP+9": 1.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "OUTPUT_HI+1": 4.0,
    })

    assert out["OUTPUT_HI+1"] == 4.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_ax_add_byte1_hi_zero_blocks_bp_byte_rows():
    ir = _tail_rules_ir(
        "tail_ax_add_byte1_hi_zero",
        "tail_ax_add_byte1_hi_zero_lo_b",
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_0": 1e-6,
        "BYTE_INDEX_2": 1.0,
        "TEMP+8": 1.0,
        "ALU_HI+0": 6.0,
        "AX_CARRY_HI+0": 4.0,
        "OUTPUT_LO+11": 3.0,
        "OUTPUT_HI+1": 4.0,
    })

    assert out["OUTPUT_LO+11"] == 3.0
    assert out["OUTPUT_HI+1"] == 4.0
    assert out.get("OUTPUT_HI+0", 0.0) == 0.0


def test_tail_ax_add_byte1_hi_zero_lo_blocks_bp_frame_byte0_row():
    ir = _tail_rules_ir(
        "tail_ax_add_byte1_hi_zero",
        "tail_ax_add_byte1_hi_zero_lo_f",
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9973672032356262,
        "H1+3": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.020405665040016174,
        "ALU_HI+0": 0.7190718054771423,
        "AX_CARRY_HI+0": 0.0003145717200823128,
        "OUTPUT_LO+15": 50.375244140625,
        "OUTPUT_HI+0": -47.434967041015625,
        "OUTPUT_HI+15": 50.375244140625,
    })

    assert out["OUTPUT_LO+15"] == 50.375244140625
    assert out["OUTPUT_HI+0"] == -47.434967041015625
    assert out["OUTPUT_HI+15"] == 50.375244140625


def test_tail_ax_add_byte1_hi_zero_lo_blocks_huge_sub_output():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_hi_zero_lo_5"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "ALU_HI+0": 6.236852645874023,
        "AX_CARRY_HI+0": 3.941866397857666,
        "OUTPUT_LO+5": 1.71e18,
        "OUTPUT_HI+0": 4.82e18,
    })

    assert out["OUTPUT_LO+5"] == 1.71e18
    assert out["OUTPUT_HI+0"] == 4.82e18


def test_tail_ax_sub_byte1_hi_zero_repairs_non_borrow_sub_high_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_byte1_hi_zero_lo_5"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "OUTPUT_LO+5": 11.52,
        "OUTPUT_HI+0": -43.67,
        "OUTPUT_HI+8": 4.17,
    })

    assert out["OUTPUT_HI+0"] > out["OUTPUT_HI+8"]
    assert out["OUTPUT_LO+5"] > out.get("OUTPUT_LO+0", 0.0)


def test_tail_ax_sub_byte1_hi_zero_blocks_borrow_rows():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_byte1_hi_zero_lo_f"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 1.0,
        "OUTPUT_LO+15": 12.0,
        "OUTPUT_HI+15": 7.0,
        "OUTPUT_HI+0": 8.0,
    })

    assert out["OUTPUT_HI+15"] == 7.0
    assert out["OUTPUT_HI+0"] == 8.0


def test_tail_ax_sub_byte1_hi_zero_blocks_huge_underflow_borrow_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_byte1_hi_zero_lo_8"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "CARRY+3": 9.00263e7,
        "OUTPUT_LO+8": 1.136948985422256e27,
        "OUTPUT_HI+13": 1.2506424672545367e27,
        "OUTPUT_HI+0": -1.81e27,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(1.136948985422256e27)
    assert out["OUTPUT_HI+13"] == pytest.approx(1.2506424672545367e27)
    assert out["OUTPUT_HI+0"] == pytest.approx(-1.81e27)


def test_tail_ax_sub_full_underflow_emits_bounded_huge_borrow_repair():
    ir = _tail_rules_ir(
        "tail_ax_sub_byte1_hi_zero_lo_8",
        "tail_ax_sub_full_underflow_byte1_ff",
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "OUTPUT_LO+0": -1.47e22,
        "OUTPUT_LO+8": 1.46e22,
        "OUTPUT_HI+0": -1.81e27,
        "OUTPUT_HI+15": 1.46e26,
    })

    assert out["OUTPUT_LO+15"] > 900_000_000.0
    assert out["OUTPUT_HI+15"] > 900_000_000.0
    assert out["OUTPUT_LO+8"] == pytest.approx(1.46e22)
    assert out["OUTPUT_HI+0"] == pytest.approx(-1.81e27)


def test_tail_ax_sub_byte1_hi_zero_blocks_imm_marker_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_byte1_hi_zero_lo_8"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "OUTPUT_LO+0": -1.34e9,
        "OUTPUT_LO+8": 1.34e9,
        "OUTPUT_HI+0": -1.35e9,
        "OUTPUT_HI+12": 1.35e9,
    })

    assert out["OUTPUT_LO+8"] == pytest.approx(1.34e9)
    assert out["OUTPUT_HI+12"] == pytest.approx(1.35e9)
    assert out["OUTPUT_LO+0"] == pytest.approx(-1.34e9)


def test_tail_ax_sub_full_underflow_byte1_materializes_ff():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_full_underflow_byte1_ff"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "OUTPUT_LO+15": 12.0,
        "OUTPUT_HI+0": 8.0,
    })

    assert out["OUTPUT_LO+15"] > 12.0
    assert out["OUTPUT_HI+15"] > 0.0
    assert out["OUTPUT_HI+0"] < 8.0


def test_tail_ax_sub_borrow_decrements_l15_restored_high_byte():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_borrow_byte1_5_to_4"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+5": 6.0,
        "OUTPUT_LO+4": -245612544.0,
        "OUTPUT_LO+5": 146251312.0,
        "OUTPUT_HI+0": 449082016.0,
    })

    assert out["OUTPUT_LO+4"] > out["OUTPUT_LO+5"]
    assert out["OUTPUT_HI+0"] > 449082016.0


def test_tail_ax_sub_borrow_decrement_requires_borrow_relay():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_borrow_byte1_5_to_4"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "ALU_LO+5": 6.0,
        "OUTPUT_LO+5": 146251312.0,
        "OUTPUT_HI+0": 449082016.0,
    })

    assert out["OUTPUT_LO+5"] == 146251312.0
    assert out["OUTPUT_HI+0"] == 449082016.0
    assert out.get("OUTPUT_LO+4", 0.0) == 0.0


def test_tail_ax_sub_borrow_decrement_uses_unborrowed_alu_source():
    ir = _single_rule_ir(_tail_rule("tail_ax_sub_borrow_byte1_5_to_4"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+6": 6.0,
        "OUTPUT_LO+5": 1.71e18,
        "OUTPUT_HI+0": 4.82e18,
    })

    assert out["OUTPUT_LO+5"] == 1.71e18
    assert out.get("OUTPUT_LO+4", 0.0) == 0.0


def test_tail_sp_pop_byte3_asserts_zero():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_LO+1": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 3.0
    assert out["OUTPUT_LO+1"] < 0.0


def test_tail_sp_pop_byte3_zero_requires_staged_zero():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+1": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out["OUTPUT_HI+0"] == 3.0
    assert out["OUTPUT_LO+1"] == 3.0


def test_tail_sp_pop_byte3_zero_blocks_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "MARK_BP": 1.0,
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 1.0,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_LO+1": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+0"] == 3.0
    assert out["OUTPUT_LO+1"] == 3.0


def test_tail_sp_pop_byte3_zero_blocks_stack0_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 1.0,
        "CMP+3": 4.0,
        "STACK0_BYTE0": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_LO+1": 3.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+0"] == 3.0
    assert out["OUTPUT_LO+1"] == 3.0


def test_tail_sp_pop_byte3_zero_blocks_ax_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_byte3_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 0.000001,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 10_000.0,
        "OUTPUT_LO+7": 10_000.0,
        "OUTPUT_HI+0": 10_000.0,
    })

    assert out["OUTPUT_LO+0"] == 10_000.0
    assert out["OUTPUT_LO+7"] == 10_000.0
    assert out["OUTPUT_HI+0"] == 10_000.0

    huge = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 7.152557373046875e-7,
        "CMP+3": 4.0,
        "OUTPUT_LO+0": 7.085196304539714e18,
        "OUTPUT_LO+5": 1.7100452833515274e18,
        "OUTPUT_HI+0": 4.818260064030687e18,
    })

    assert huge["OUTPUT_LO+0"] == 7.085196304539714e18
    assert huge["OUTPUT_LO+5"] == 1.7100452833515274e18
    assert huge["OUTPUT_HI+0"] == 4.818260064030687e18


def test_tail_si_ax_byte1_requires_real_mem_store_signal():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_SI": 5.0,
        "MEM_STORE": 0.004,
        "OUTPUT_LO+2": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out.get("OUTPUT_HI+0", 0.0) == 3.0


def test_tail_si_ax_byte1_blocks_large_non_store_residue():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_SI": 5.0,
        "OUTPUT_LO+2": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
        "OUTPUT_LO+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] == 42.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_si_ax_byte1_blocks_byte3_residue():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_3": 1.0,
        "OP_SI": 5.0,
        "MEM_STORE": 0.001,
        "OUTPUT_LO+2": 1_000_000_000.0,
        "OUTPUT_HI+0": 1_000_000_000.0,
        "OUTPUT_LO+0": 42.0,
    })

    assert out["OUTPUT_LO+0"] == 42.0
    assert out["OUTPUT_HI+0"] == 1_000_000_000.0


def test_tail_si_ax_byte1_blocks_stack0_marker_residue():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_00"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "MEM_STORE": 0.40725401043891907,
        "OUTPUT_LO+2": -507622.5625,
        "OUTPUT_HI+0": 507625.5625,
        "OUTPUT_HI+1": -507622.5625,
        "OUTPUT_LO+10": 507625.5625,
    })

    assert out["OUTPUT_LO+10"] == 507625.5625
    assert out["OUTPUT_HI+0"] == 507625.5625
    assert out["OUTPUT_HI+1"] == -507622.5625


def test_tail_si_ax_byte1_still_fires_for_store_signal():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_SI": 5.0,
        "MEM_STORE": 1.0,
        "OUTPUT_LO+2": 2.0,
        "OUTPUT_HI+0": 3.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0
    assert out["OUTPUT_HI+0"] > 3.0


def test_tail_si_ax_byte1_blocks_lea_local_frame_row():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_12"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OP_LEA": 5.0,
        "MEM_STORE": 0.023,
        "OUTPUT_LO+2": 5_690_513_408.0,
        "OUTPUT_HI+1": 5_692_803_072.0,
        "OUTPUT_LO+8": 55_586.3984375,
        "OUTPUT_HI+14": 2_290_062.0,
    })

    assert out["OUTPUT_LO+8"] == 55_586.3984375
    assert out["OUTPUT_HI+14"] == 2_290_062.0
    assert out.get("OUTPUT_LO+2", 0.0) == 5_690_513_408.0
    assert out.get("OUTPUT_HI+1", 0.0) == 5_692_803_072.0


def test_tail_si_ax_byte1_blocks_mem_addr_row():
    ir = _single_rule_ir(_tail_rule("tail_si_ax_byte1_12"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 0.9976623058319092,
        "H1+4": 1.0,
        "MEM_STORE": 2.0,
        "CMP+0": 2.996619701385498,
        "OUTPUT_LO+2": 54_250_434_560.0,
        "OUTPUT_HI+1": 1_676_348_500.0,
        "OUTPUT_LO+8": 1_646_635.125,
        "OUTPUT_HI+14": 1_131_778.875,
    })

    assert out["OUTPUT_LO+8"] == 1_646_635.125
    assert out["OUTPUT_HI+14"] == 1_131_778.875
    assert out["OUTPUT_LO+2"] == 54_250_434_560.0
    assert out["OUTPUT_HI+1"] == 1_676_348_500.0


def test_tail_wide_mul_byte1_preserve_blocks_without_temp10_gate():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+9"] == 40.0
    assert out["OUTPUT_HI+0"] == 40.0


def test_tail_wide_mul_byte1_preserve_blocks_byte3_rows():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_3": 1.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+9"] == 40.0
    assert out["OUTPUT_HI+0"] == 40.0


def test_tail_wide_mul_byte1_preserve_blocks_non_ax_spans():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_f9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+15": 40.0,
    })

    assert out["OUTPUT_LO+9"] == 40.0
    assert out["OUTPUT_HI+15"] == 40.0


def test_tail_wide_mul_byte1_preserve_blocks_without_step_history():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "BYTE_INDEX_1": 0.0,
        "BYTE_INDEX_2": 0.0,
        "BYTE_INDEX_3": 0.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+9"] == 40.0
    assert out["OUTPUT_HI+0"] == 40.0


def test_tail_wide_mul_byte1_preserve_blocks_ent_rows_with_temp_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 100.0,
        "OP_ENT": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+9"] == 40.0
    assert out["OUTPUT_HI+0"] == 40.0


def test_tail_wide_mul_byte1_preserve_blocks_add_rows_with_temp_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 100.0,
        "TEMP+8": 1.0,
        "OP_ADD": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+0": 40.0,
    })

    assert out["OUTPUT_LO+9"] == 40.0
    assert out["OUTPUT_HI+0"] == 40.0


def test_tail_wide_mul_byte1_preserve_blocks_huge_add_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_4"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+4": 1_000_000_000_000_000_000.0,
        "OUTPUT_HI+0": 1_000_000_000_000_000_000.0,
    })

    assert out["OUTPUT_LO+4"] == 1_000_000_000_000_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000_000_000_000_000.0


def test_tail_wide_mul_byte1_preserve_blocks_sub_borrow_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_f0"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "TEMP+10": 1.3010942935943604,
        "OUTPUT_LO+0": 7.085196304539714e18,
        "OUTPUT_LO+5": 1.7100452833515274e18,
        "OUTPUT_HI+0": 4.818260064030687e18,
        "OUTPUT_HI+15": 6.338251601425203e17,
    })

    assert out["OUTPUT_LO+0"] == 7.085196304539714e18
    assert out["OUTPUT_LO+5"] == 1.7100452833515274e18
    assert out["OUTPUT_HI+15"] == 6.338251601425203e17


def test_tail_wide_mul_byte1_preserve_handles_nonzero_high_nibble():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_f9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "BYTE_INDEX_1": 0.0,
        "BYTE_INDEX_2": 0.0,
        "BYTE_INDEX_3": 0.0,
        "TEMP+10": 1.0,
        "OP_MUL": 1.0,
        "OUTPUT_LO+9": 40.0,
        "OUTPUT_HI+15": 40.0,
    })

    assert out["OUTPUT_LO+9"] > 40.0
    assert out["OUTPUT_HI+15"] > 40.0


def test_tail_wide_mul_byte1_preserve_blocks_weak_staged_byte():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_0"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+0": 13.0,
        "OUTPUT_HI+0": 13.0,
    })

    assert out["OUTPUT_LO+0"] == 13.0
    assert out["OUTPUT_HI+0"] == 13.0


def test_tail_wide_mul_byte1_preserve_still_fires_with_byte0_gate():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_9"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "OP_MUL": 1.0,
        "OUTPUT_LO+9": 100.0,
        "OUTPUT_HI+0": 100.0,
    })

    assert out["OUTPUT_LO+9"] > 100.0
    assert out["OUTPUT_HI+0"] > 100.0


def test_tail_ax_add_mul_byte1_materialize_repairs_low_one():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_mul_byte1_materialize_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998542845249176,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.30489400029182434,
        "TEMP+10": 0.9961900115013123,
        "OP_MUL": 1.0,
        "EMBED_HI+0": 1.0,
        "FETCH_HI+0": 0.3991380035877228,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+1"] > 0.0
    assert out["OUTPUT_LO+0"] < 2.9402761459350586
    assert out["OUTPUT_HI+0"] > 2.9402761459350586


def test_tail_ax_add_mul_byte1_materialize_repairs_low_two_from_hi2():
    ir = _single_rule_ir(
        _tail_rule("tail_ax_add_mul_byte1_materialize_02_from_hi2")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998542845249176,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.3049095869064331,
        "TEMP+10": 0.9961900115013123,
        "OP_MUL": 1.0,
        "EMBED_HI+2": 1.0,
        "FETCH_HI+2": 1.2207061052322388,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+2"] > 0.0
    assert out["OUTPUT_LO+0"] < 2.9402761459350586
    assert out["OUTPUT_HI+0"] > 2.9402761459350586


def test_tail_ax_add_mul_byte1_materialize_repairs_low_two_from_hid():
    ir = _single_rule_ir(
        _tail_rule("tail_ax_add_mul_byte1_materialize_02_from_hid")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998542845249176,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.30489400029182434,
        "TEMP+10": 0.9961900115013123,
        "OP_MUL": 1.0,
        "EMBED_HI+13": 1.0,
        "FETCH_HI+2": 0.9340206980705261,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+2"] > 0.0
    assert out["OUTPUT_LO+0"] < 2.9402761459350586
    assert out["OUTPUT_HI+0"] > 2.9402761459350586


def test_tail_ax_add_mul_byte1_materialize_requires_add_relay():
    ir = _single_rule_ir(
        _tail_rule("tail_ax_add_mul_byte1_materialize_02_from_hi2")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998542845249176,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+10": 1.3010942935943604,
        "EMBED_HI+2": 1.0,
        "FETCH_HI+2": 1.2207061052322388,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+0"] == 2.9402761459350586
    assert out["OUTPUT_HI+0"] == 2.9402761459350586
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_mul_byte1_materialize_blocks_zero_high_product():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_mul_byte1_materialize_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998542845249176,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.30489400029182434,
        "TEMP+10": 0.9961900115013123,
        "FETCH_HI+0": 0.11243800073862076,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+0"] == 2.9402761459350586
    assert out["OUTPUT_HI+0"] == 2.9402761459350586
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_ax_add_mul_byte1_materialize_blocks_plain_mul_zero_byte1():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_mul_byte1_materialize_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9974556565284729,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 0.30365103483200073,
        "TEMP+10": 0.9980799555778503,
        "EMBED_HI+0": 1.0000574588775635,
        "FETCH_HI+0": 1.332642912864685,
        "OUTPUT_LO+0": 2.9402759075164795,
        "OUTPUT_HI+0": 2.9402759075164795,
    })

    assert out["OUTPUT_LO+0"] == 2.9402759075164795
    assert out["OUTPUT_HI+0"] == 2.9402759075164795
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_ax_add_mul_byte1_materialize_requires_mul_ownership():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_mul_byte1_materialize_01"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9986177086830139,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "TEMP+10": 0.014740509912371635,
        "EMBED_HI+0": 1.000071406364441,
        "FETCH_HI+0": 0.3991560637950897,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+0"] == 2.9402761459350586
    assert out["OUTPUT_HI+0"] == 2.9402761459350586
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_ax_add_mul_byte1_materialize_blocks_plain_add_byte1_zero():
    ir = _single_rule_ir(
        _tail_rule("tail_ax_add_mul_byte1_materialize_02_from_hi2")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998542845249176,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.3049095869064331,
        "TEMP+10": 0.9961900115013123,
        "EMBED_HI+2": 1.0,
        "FETCH_HI+2": 1.2207061052322388,
        "OUTPUT_LO+0": 2.9402761459350586,
        "OUTPUT_HI+0": 2.9402761459350586,
    })

    assert out["OUTPUT_LO+0"] == 2.9402761459350586
    assert out["OUTPUT_HI+0"] == 2.9402761459350586
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_mul_byte1_materialize_blocks_psh_preserve_row():
    ir = _single_rule_ir(
        _tail_rule("tail_ax_add_mul_byte1_materialize_02_from_hi2")
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 0.304,
        "TEMP+10": 0.997,
        "OP_PSH": 5.0,
        "EMBED_HI+2": 0.0,
        "FETCH_HI+2": 0.934,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+15": 2.0,
        "OUTPUT_HI+15": 2.0,
    })

    assert out["OUTPUT_LO+15"] == 2.0
    assert out["OUTPUT_HI+15"] == 2.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_carry_high2_repairs_huge_byte3_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_carry_high2_03"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 1.304825782775879,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+0": 2.36312338633608e14,
        "OUTPUT_LO+3": 2.1993707531384e13,
        "OUTPUT_HI+3": 4.688537e12,
    })

    assert out["OUTPUT_LO+3"] > 2.1993707531384e13
    assert out["OUTPUT_LO+0"] < 2.36312338633608e14
    assert out["OUTPUT_HI+0"] > 0.0
    assert out["OUTPUT_HI+3"] < 4.688537e12


def test_tail_ax_add_byte1_carry_high2_blocks_existing_byte7():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_carry_high2_03"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+3": 185.0347,
        "OUTPUT_LO+7": 2616.264,
        "OUTPUT_HI+0": 8464.702,
    })

    assert out["OUTPUT_LO+7"] == 2616.264
    assert out.get("OUTPUT_LO+3", 0.0) == 185.0347


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


def test_tail_sub_borrow_blocks_add_residue_with_huge_output():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+8": 1.304825782775879,
        "CARRY+1": 2.0,
        "ALU_LO+1": 6.595503032613692e-10,
        "OUTPUT_LO+0": 2.36312338633608e14,
        "OUTPUT_LO+3": 2.1993707531384e13,
        "OUTPUT_HI+0": 4.041023946752e12,
    })

    assert out["OUTPUT_LO+0"] == 2.36312338633608e14
    assert out["OUTPUT_LO+3"] == 2.1993707531384e13
    assert out["OUTPUT_HI+0"] == 4.041023946752e12


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


def test_tail_sub_borrow_blocks_sp_marker_row():
    ir = _single_rule_ir(_tail_rule("tail_sub_borrow_byte1_00"))

    out = ir.symbolic_ffn({
        "MARK_SP": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+9": 1.0,
        "CARRY+2": 2.0,
        "ALU_LO+1": 6.0,
        "OUTPUT_LO+0": 4000.0,
        "OUTPUT_LO+1": -4000.0,
        "OUTPUT_HI+0": -4000.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 4000.0


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


def test_tail_shr_marker_correction_blocks_mem_marker_rows():
    ir = _single_rule_ir(_tail_rule("tail_shr_marker_byte0_01"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "OUTPUT_LO+6": 153.0,
        "OUTPUT_LO+8": 153.0,
        "OUTPUT_HI+15": 153.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_shr_marker_correction_still_fires_on_marker():
    ir = _single_rule_ir(_tail_rule("tail_shr_marker_byte0_01"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "TEMP+7": 1.0,
        "OP_SHR": 5.0,
        "OUTPUT_LO+10": -0.84,
        "OUTPUT_HI+0": 2.56,
        "OUTPUT_HI+2": -0.84,
    })

    assert out["OUTPUT_LO+1"] > 0.0


def test_tail_shr_marker_correction_preserves_shr_by_one_result():
    ir = _single_rule_ir(_tail_rule("tail_shr_marker_byte0_01"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "TEMP+7": 1.0,
        "OP_SHR": 5.0,
        "OUTPUT_LO+6": -0.84,
        "OUTPUT_LO+10": 1.16,
        "OUTPUT_HI+2": 1.16,
    })

    assert out["OUTPUT_LO+10"] == 1.16
    assert out["OUTPUT_HI+2"] == 1.16
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_shr_marker_correction_requires_shr_opcode():
    ir = _single_rule_ir(_tail_rule("tail_shr_marker_byte0_01"))

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "H1+1": 1.0,
        "TEMP+7": 1.0,
        "OUTPUT_LO+6": 40.0,
        "OUTPUT_HI+7": 40.0,
    })

    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_shr_byte1_zero_requires_full_temp7_relay():
    ir = _single_rule_ir(_tail_rule("tail_shr_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+7": 0.305,
        "OUTPUT_LO+2": 12.0,
        "OUTPUT_HI+0": 15.0,
    })

    assert out.get("OUTPUT_LO+0", 0.0) == 0.0
    assert out["OUTPUT_LO+2"] == 12.0


def test_tail_shr_byte1_zero_still_fires_with_full_temp7_relay():
    ir = _single_rule_ir(_tail_rule("tail_shr_byte1_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+7": 1.0,
        "OUTPUT_LO+1": 1.0,
    })

    assert out["OUTPUT_LO+0"] > 0.0


def test_tail_ax_add_byte1_hi_zero_lo_rules_use_finite_strength():
    rule = _tail_rule("tail_ax_add_byte1_hi_zero_lo_3")

    assert max(abs(write.weight) for write in rule.writes) == 10_000.0


def test_tail_wide_mul_byte1_preserve_rules_use_bounded_conditions():
    rule = _tail_rule("tail_wide_mul_byte1_preserve_43")

    assert max(abs(term.weight) for term in rule.conditions) <= 1000000.0
    assert rule.threshold == 220.0
    assert rule.gate.name == "OP_MUL"
    assert rule.gate.offset == 0


def test_tail_rules_do_not_use_trillion_scale_numeric_bounds():
    for rule in _tail_bit32_result_correction_rules():
        condition_max = max((abs(term.weight) for term in rule.conditions), default=0.0)
        write_max = max((abs(write.weight) for write in rule.writes), default=0.0)
        gate_term_max = max((abs(term.weight) for term in rule.gate_terms), default=0.0)

        assert condition_max <= 1_000_000_000.0, rule.name
        assert write_max <= 1_000_000_000.0, rule.name
        assert gate_term_max <= 1_000_000_000.0, rule.name
        assert abs(rule.threshold) <= 1_000_000_000.0, rule.name


def test_tail_wide_mul_byte1_preserve_blocks_add_relay_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_43"))
    state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+8": 1.0,
        "TEMP+10": 1.0,
        "OUTPUT_LO+3": 2.2e13,
        "OUTPUT_HI+4": 4.6e12,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+3"] == state["OUTPUT_LO+3"]
    assert out["OUTPUT_HI+4"] == state["OUTPUT_HI+4"]
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_wide_mul_byte1_preserve_blocks_tiny_negative_op_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_43"))
    state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "OP_MUL": -1e-21,
        "OP_ADD": -1e-21,
        "OUTPUT_LO+3": 2.2e13,
        "OUTPUT_HI+4": 4.6e12,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+3"] == state["OUTPUT_LO+3"]
    assert out["OUTPUT_HI+4"] == state["OUTPUT_HI+4"]


def test_tail_wide_mul_byte1_preserve_blocks_add_even_with_mul_residue():
    ir = _single_rule_ir(_tail_rule("tail_wide_mul_byte1_preserve_43"))
    state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 1.0,
        "TEMP+10": 1.0,
        "OP_MUL": 1.0,
        "OP_ADD": 1.0,
        "OUTPUT_LO+3": 100.0,
        "OUTPUT_HI+4": 100.0,
    }

    out = ir.symbolic_ffn(state)

    assert out["OUTPUT_LO+3"] == state["OUTPUT_LO+3"]
    assert out["OUTPUT_HI+4"] == state["OUTPUT_HI+4"]


def test_tail_ax_add_byte1_missing_stack_high_repairs_fetch_hi1_case():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.97,
        "TEMP+8": 1.0,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 1.22,
        "OUTPUT_LO+1": 253.0,
        "OUTPUT_HI+0": 254.0,
    })

    assert out["OUTPUT_LO+2"] > 0.0
    assert out["OUTPUT_LO+1"] < 253.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_plain_carry_case():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.97,
        "TEMP+8": 1.0,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.934,
        "OUTPUT_LO+1": 9128.0,
        "OUTPUT_HI+0": 9311.0,
    })

    assert out["OUTPUT_LO+1"] == 9128.0
    assert out["OUTPUT_HI+0"] == 9311.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_requires_carry():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.97,
        "TEMP+8": 1.0,
        "FETCH_HI+1": 1.22,
        "OUTPUT_LO+1": 12.0,
        "OUTPUT_HI+0": 15.0,
    })

    assert out["OUTPUT_LO+1"] == 12.0
    assert out["OUTPUT_HI+0"] == 15.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_existing_byte3():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.97,
        "TEMP+8": 1.0,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 1.22,
        "OUTPUT_LO+3": 4551.0,
        "OUTPUT_HI+0": 8773.0,
    })

    assert out["OUTPUT_LO+3"] == 4551.0
    assert out["OUTPUT_HI+0"] == 8773.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_existing_byte5():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+1": -251.51998901367188,
        "OUTPUT_LO+3": -188.16000366210938,
        "OUTPUT_LO+5": 2616.27001953125,
        "OUTPUT_HI+0": 8464.7353515625,
    })

    assert out["OUTPUT_LO+5"] == 2616.27001953125
    assert out["OUTPUT_HI+0"] == 8464.7353515625
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_negative_lane_evidence():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+0": 485.9805908203125,
        "OUTPUT_LO+1": -613.1743774414062,
        "OUTPUT_LO+2": 487.04034423828125,
        "OUTPUT_LO+3": -613.1743774414062,
        "OUTPUT_LO+4": -2611.14501953125,
        "OUTPUT_LO+5": 2616.270263671875,
        "OUTPUT_HI+0": 7681.05908203125,
    })

    assert out["OUTPUT_LO+5"] == 2616.270263671875
    assert out["OUTPUT_HI+0"] == 7681.05908203125
    assert out["OUTPUT_LO+2"] == 487.04034423828125


def test_tail_ax_add_byte1_missing_stack_high_blocks_sub_residue():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "TEMP+9": 1.0,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+4": 1.7281791163115766e18,
        "OUTPUT_HI+0": 4.260235372945998e18,
    })

    assert out["OUTPUT_LO+4"] == 1.7281791163115766e18
    assert out["OUTPUT_HI+0"] == 4.260235372945998e18
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_existing_byte4():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+1": -66.48534393310547,
        "OUTPUT_LO+2": -188.16000366210938,
        "OUTPUT_LO+3": -2611.144775390625,
        "OUTPUT_LO+4": 2616.27001953125,
        "OUTPUT_HI+0": 8464.7353515625,
    })

    assert out["OUTPUT_LO+4"] == 2616.27001953125
    assert out["OUTPUT_HI+0"] == 8464.7353515625
    assert out.get("OUTPUT_LO+2", 0.0) == -188.16000366210938


def test_tail_ax_add_byte1_missing_stack_high_blocks_existing_byte6():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+1": -215.0399932861328,
        "OUTPUT_LO+3": 149.1645965576172,
        "OUTPUT_LO+5": -1890.9033203125,
        "OUTPUT_LO+6": 1895.4190673828125,
        "OUTPUT_HI+0": 7860.3779296875,
    })

    assert out["OUTPUT_LO+6"] == 1895.4190673828125
    assert out["OUTPUT_HI+0"] == 7860.3779296875
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_existing_byte7():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 0.9340206980705261,
        "OUTPUT_LO+3": 185.0347,
        "OUTPUT_LO+6": -2611.139,
        "OUTPUT_LO+7": 2616.264,
        "OUTPUT_HI+0": 8464.702,
    })

    assert out["OUTPUT_LO+7"] == 2616.264
    assert out["OUTPUT_HI+0"] == 8464.702
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_bp_frame_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+1": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_0": 1.0,
        "BYTE_INDEX_1": 1.0,
        "TEMP+8": 1.0,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 1.22,
        "OUTPUT_LO+0": 4000.0,
        "OUTPUT_LO+1": -4000.0,
        "OUTPUT_LO+3": -4000.0,
        "OUTPUT_LO+4": -4000.0,
        "OUTPUT_LO+5": -4000.0,
        "OUTPUT_LO+6": -4000.0,
        "OUTPUT_LO+7": -4000.0,
        "OUTPUT_HI+0": 4000.0,
    })

    assert out["OUTPUT_LO+0"] == 4000.0
    assert out["OUTPUT_HI+0"] == 4000.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_pc_byte_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984797239303589,
        "H1+0": 1.0,
        "H1+1": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "TEMP+8": 1.0000001192092896,
        "CARRY+1": 2.0,
        "FETCH_HI+1": 40.00002670288086,
        "OUTPUT_LO+1": 253.0,
        "OUTPUT_HI+0": 254.0,
    })

    assert out["OUTPUT_LO+1"] == 253.0
    assert out["OUTPUT_HI+0"] == 254.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_stack0_byte_row():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9984540939331055,
        "BYTE_INDEX_0": 0.9734055995941162,
        "BYTE_INDEX_1": 0.013297183439135551,
        "STACK0_BYTE1": 0.013297183439135551,
        "OUTPUT_LO+1": -971.656494140625,
        "OUTPUT_LO+3": -971.656494140625,
        "OUTPUT_LO+4": -971.656494140625,
        "OUTPUT_LO+5": -971.656494140625,
        "OUTPUT_LO+6": -971.656494140625,
        "OUTPUT_LO+7": -971.656494140625,
        "OUTPUT_HI+0": 123.0,
    })

    assert out["OUTPUT_HI+0"] == 123.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_tail_ax_add_byte1_missing_stack_high_blocks_stack0_marker_signed_bands():
    ir = _single_rule_ir(_tail_rule("tail_ax_add_byte1_missing_stack_high_02"))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9985138177871704,
        "TEMP+8": 0.0008986028842628002,
        "OUTPUT_LO+3": -425984032.0,
        "OUTPUT_LO+4": -425984032.0,
        "OUTPUT_LO+5": -425984032.0,
        "OUTPUT_LO+6": -425984032.0,
        "OUTPUT_LO+7": -425984032.0,
        "OUTPUT_LO+15": 425984000.0,
        "OUTPUT_HI+2": 425984000.0,
    })

    assert out["OUTPUT_LO+15"] == 425984000.0
    assert out["OUTPUT_HI+2"] == 425984000.0
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


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


def test_tail_clear_output_after_byte3_clears_final_byte_residue():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_after_byte3"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "BYTE_INDEX_3": 1.0,
        "OUTPUT_LO+0": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
        "OUTPUT_LO+15": 1_000_000.0,
        "OUTPUT_HI+15": 1_000_000.0,
    })

    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0
    assert out["OUTPUT_LO+15"] < 0.0
    assert out["OUTPUT_HI+15"] < 0.0


def test_tail_clear_output_after_byte3_blocks_earlier_bytes():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_after_byte3"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "BYTE_INDEX_2": 1.0,
        "OUTPUT_LO+0": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+0"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_clear_output_after_byte3_blocks_mem_addr_to_value_transition():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_after_byte3"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "BYTE_INDEX_3": 1.0,
        "MEM_VAL_B0": 1.0,
        "OUTPUT_LO+10": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+10"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_clear_output_after_byte3_blocks_stack0_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_after_byte3"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "BYTE_INDEX_3": 1.0,
        "STACK0_BYTE3": 1.0,
        "OUTPUT_LO+10": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+10"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_clear_output_before_step_end_clears_mem_value_residue():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_before_step_end"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "NEXT_SE": 1.0,
        "OUTPUT_LO+0": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
        "OUTPUT_LO+15": 1_000_000.0,
        "OUTPUT_HI+15": 1_000_000.0,
    })

    assert out["OUTPUT_LO+0"] < 0.0
    assert out["OUTPUT_HI+0"] < 0.0
    assert out["OUTPUT_LO+15"] < 0.0
    assert out["OUTPUT_HI+15"] < 0.0


def test_tail_clear_output_before_step_end_blocks_stack0_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_before_step_end"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "NEXT_SE": 1.0,
        "STACK0_BYTE0": 1.0,
        "OUTPUT_LO+14": 1_000_000.0,
        "OUTPUT_HI+13": 1_000_000.0,
    })

    assert out["OUTPUT_LO+14"] == 1_000_000.0
    assert out["OUTPUT_HI+13"] == 1_000_000.0


def test_tail_clear_output_before_step_end_requires_next_se():
    ir = _single_rule_ir(_tail_rule("tail_clear_output_before_step_end"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "OUTPUT_LO+0": 1_000_000.0,
        "OUTPUT_HI+0": 1_000_000.0,
    })

    assert out["OUTPUT_LO+0"] == 1_000_000.0
    assert out["OUTPUT_HI+0"] == 1_000_000.0


def test_tail_sp_pop_carry_requires_clean_zero_previous_byte():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_08"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_1": 0.97,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+15": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+8": 20.2,
        "OUTPUT_HI+0": 21.8,
        "OUTPUT_LO+9": 3.0,
    })

    assert out.get("OUTPUT_LO+9", 0.0) == 3.0
    assert out["OUTPUT_HI+0"] == 21.8


def test_tail_sp_pop_carry_requires_exact_byte_index():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_2": 0.97,
        "BYTE_INDEX_3": 0.013,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 2.9,
        "OUTPUT_HI+0": 2.9,
    })

    assert out["OUTPUT_LO+0"] == 2.9
    assert out["OUTPUT_HI+0"] == 2.9
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_sp_pop_carry_byte1_zero_fires_on_measured_add_sp_row():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.99843,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 0.970138,
        "BYTE_INDEX_1": 0.0132967,
        "BYTE_INDEX_2": 5.96046e-07,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 0.953674,
        "OUTPUT_LO+3": 1.9866,
        "OUTPUT_HI+0": 2.94028,
    })

    assert out["OUTPUT_LO+0"] > out["OUTPUT_LO+3"]
    assert out["OUTPUT_HI+0"] > 2.94028


def test_tail_sp_pop_carry_byte1_zero_blocks_store_frame_row():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9979575872421265,
        "H1+2": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "MEM_STORE": 1.2824475765228271,
        "EMBED_LO+0": 1.0000532865524292,
        "EMBED_HI+14": 1.0000396966934204,
        "OUTPUT_LO+0": 0.08617234230041504,
        "OUTPUT_LO+15": 5.880698204040527,
        "OUTPUT_HI+0": 0.08617234230041504,
        "OUTPUT_HI+15": 5.880709648132324,
    })

    assert out["OUTPUT_LO+15"] == 5.880698204040527
    assert out["OUTPUT_HI+15"] == 5.880709648132324
    assert out["OUTPUT_LO+0"] == 0.08617234230041504
    assert out["OUTPUT_HI+0"] == 0.08617234230041504


def test_tail_sp_pop_carry_byte1_zero_blocks_e0_high_byte_row():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte1_zero"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998437,
        "H1+2": 1.0,
        "H1+9": 1.0,
        "BYTE_INDEX_0": 0.9701380133628845,
        "BYTE_INDEX_1": 0.013296706601977348,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+14": 1.0,
        "OUTPUT_LO+0": 0.9402758479118347,
        "OUTPUT_LO+15": 2.0,
        "OUTPUT_HI+0": 0.9402758479118347,
        "OUTPUT_HI+15": 2.0,
    })

    assert out["OUTPUT_LO+15"] == 2.0
    assert out["OUTPUT_HI+15"] == 2.0
    assert out["OUTPUT_LO+0"] == 0.9402758479118347
    assert out["OUTPUT_HI+0"] == 0.9402758479118347


def test_tail_sp_pop_carry_byte2_zero_fires_on_measured_add_sp_row():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998421,
        "H1+2": 1.0,
        "BYTE_INDEX_1": 0.970137,
        "BYTE_INDEX_2": 0.0132971,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+1"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] > 2.0


def test_tail_sp_pop_carry_blocks_branch_sp_byte_rows():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_1": 1.0,
        "CMP+3": 4.0,
        "OP_JMP": 1.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+0"] == 2.0
    assert out["OUTPUT_HI+0"] == 2.0
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_sp_pop_carry_blocks_startup_stack0_row_without_has_se():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "BYTE_INDEX_0": 0.9734055995941162,
        "BYTE_INDEX_1": 0.013297183439135551,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 142.0,
        "OUTPUT_HI+0": 142.0,
    })

    assert out["OUTPUT_LO+0"] == 142.0
    assert out["OUTPUT_HI+0"] == 142.0
    assert out.get("OUTPUT_LO+1", 0.0) == 0.0


def test_tail_sp_pop_carry_requires_real_pop_flag_margin():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9975,
        "H1+2": 1.0,
        "BYTE_INDEX_1": 0.9701,
        "BYTE_INDEX_2": 0.0133,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+1": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+1"] == 2.0
    assert out["OUTPUT_HI+0"] == 2.0
    assert out.get("OUTPUT_LO+0", 0.0) == 0.0


def test_tail_sp_pop_carry_byte2_blocks_nonpop_sp_zero_high_rules():
    ir = _tail_rules_ir(
        "tail_sp_pop_carry_byte2_00",
        "tail_sp_pop_carry_byte2_01",
        "tail_sp_pop_carry_byte2_0f",
        "tail_sp_pop_carry_byte2_ff",
    )

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 0.997515,
        "H1+2": 1.0,
        "BYTE_INDEX_1": 0.970137,
        "BYTE_INDEX_2": 0.013297,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+1": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+1"] == 2.0
    assert out["OUTPUT_HI+0"] == 2.0
    for idx in range(16):
        if idx != 1:
            assert out.get(f"OUTPUT_LO+{idx}", 0.0) == 0.0
        if idx != 0:
            assert out.get(f"OUTPUT_HI+{idx}", 0.0) == 0.0


def test_tail_sp_pop_carry_still_fires_after_clean_zero_previous_byte():
    ir = _single_rule_ir(_tail_rule("tail_sp_pop_carry_byte2_00"))

    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "H1+2": 1.0,
        "BYTE_INDEX_1": 1.0,
        "CMP+3": 4.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+0": 2.0,
        "OUTPUT_HI+0": 2.0,
    })

    assert out["OUTPUT_LO+1"] > 0.0
    assert out["OUTPUT_HI+0"] > 0.0


def test_tail_pop_mem_marker_zero_blocks_store_rows():
    ir = _single_rule_ir(_tail_rule("tail_pop_mem_marker_zero"))

    out = ir.symbolic_ffn({
        "MARK_MEM": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 3.0,
        "MEM_STORE": 2.0,
        "OUTPUT_LO+8": 2.0,
        "OUTPUT_HI+14": 2.0,
    })

    assert out["OUTPUT_LO+8"] == 2.0
    assert out["OUTPUT_HI+14"] == 2.0
