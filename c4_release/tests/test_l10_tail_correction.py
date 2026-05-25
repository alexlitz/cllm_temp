"""Regression coverage for declarative L10 tail correction rules."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _layer10_byte_passthrough_ir,
    _layer10_nonbitwise_stack0_byte_relay_head_spec,
    _layer10_stack0_byte_relay_head_spec,
    _layer10_stack0_persistence_head_spec,
    make_l10_post_ops_combined,
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
        if rule.name == "tail_clear_output_after_byte3":
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


def test_tail_sp_pop_marker_increment_tolerates_mem_store_residue():
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

    assert out["OUTPUT_LO+8"] > 0.0
    assert out["OUTPUT_HI+14"] > 1.0


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
        "OUTPUT_LO+0": 2.339727474688e12,
        "OUTPUT_LO+3": 2.1993707536384e13,
        "OUTPUT_HI+0": 4.041023946752e12,
        "OUTPUT_HI+1": 4.629028929536e12,
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
        "OUTPUT_LO+9": 100.0,
        "OUTPUT_HI+0": 100.0,
    })

    assert out["OUTPUT_LO+9"] > 100.0
    assert out["OUTPUT_HI+0"] > 100.0


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
        "OP_SHR": 1.0,
        "OUTPUT_LO+6": 6.0,
    })

    assert out["OUTPUT_LO+1"] > 0.0


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

    assert max(abs(write.weight) for write in rule.writes) == 1.0e12


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
