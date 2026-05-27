"""Focused tests for L3 PC byte1 carry/preserve repair."""

import torch

from neural_vm.vm_step import _SetDim
from neural_vm.unified_compiler.ops.l3_ops import (
    _add_layer3_pc_byte1_output_rules,
    _pc_byte1_prev_head_spec,
    _rewrite_layer3_initial_sp_marker_to_f8,
    _rewrite_layer3_initial_sp_byte2_to_zero,
    _stack0_carry_head_spec,
    _suppress_layer3_stack0_marker_carry_projection,
    make_layer3_ffn_op,
)
from neural_vm.unified_compiler.primitives import Primitives
from neural_vm.vm_step import _set_layer3_ffn


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 128):
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


class _StubBlock:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 200):
        self.ffn = _StubFFN(d_model=d_model, hidden_dim=hidden_dim)


def _apply_stub_ffn(ffn: _StubFFN, x: torch.Tensor) -> torch.Tensor:
    up = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up)
    gate = x @ ffn.W_gate.t() + ffn.b_gate
    return x + (up * gate) @ ffn.W_down.t()


def _pc_byte0_base_input() -> torch.Tensor:
    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.H1 + 0] = 1.0
    x[..., _SetDim.BYTE_INDEX_0] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.HAS_SE] = 1.0
    return x


def test_layer3_pc_byte1_head_relay_writes_previous_pc_byte1_to_temp():
    attn = _StubAttn()
    head_dim = 512 // attn.num_heads

    Primitives.generate_attention_head(attn, _pc_byte1_prev_head_spec(_SetDim), head_dim)

    base = 7 * head_dim
    assert attn.W_q[base, _SetDim.IS_BYTE] == 15.0
    assert attn.W_q[base, _SetDim.BYTE_INDEX_0] == 15.0
    assert attn.W_k[base, _SetDim.BYTE_INDEX_1] == 15.0
    assert attn.W_v[base + 2, _SetDim.CLEAN_EMBED_LO + 1] == 1.0
    assert attn.W_v[base + 18, _SetDim.CLEAN_EMBED_HI + 1] == 1.0
    assert attn.W_o[_SetDim.TEMP + 1, base + 2] == 1.0
    assert attn.W_o[_SetDim.TEMP + 17, base + 18] == 1.0


def test_stack0_marker_carry_head_must_not_write_stale_embed_residue():
    attn = _StubAttn()
    head_dim = 512 // attn.num_heads

    Primitives.generate_attention_head(attn, _stack0_carry_head_spec(_SetDim), head_dim)

    base = 4 * head_dim
    assert attn.W_q[base, _SetDim.MARK_STACK0] == 15.0
    assert attn.W_k[base, _SetDim.STACK0_BYTE0] == 15.0
    assert torch.count_nonzero(attn.W_o[:, base : base + head_dim]).item() == 0


def test_layer3_pc_byte1_wrap_rule_emits_one_at_0x102_byte0():
    ffn = _StubFFN()
    _add_layer3_pc_byte1_output_rules(ffn, 100.0, _SetDim)

    x = _pc_byte0_base_input()
    x[..., _SetDim.CLEAN_EMBED_LO + 2] = 1.0
    x[..., _SetDim.CLEAN_EMBED_HI + 0] = 1.0
    x[..., _SetDim.OUTPUT_LO + 0] = 1.0
    x[..., _SetDim.OUTPUT_HI + 0] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 1] > 100.0
    assert y[_SetDim.OUTPUT_LO + 0] < -100.0
    assert y[_SetDim.OUTPUT_HI + 0] > 100.0


def test_layer3_pc_byte1_preserve_rule_uses_previous_byte1_temp():
    ffn = _StubFFN()
    _add_layer3_pc_byte1_output_rules(ffn, 100.0, _SetDim)

    x = _pc_byte0_base_input()
    x[..., _SetDim.TEMP + 1] = 1.0
    x[..., _SetDim.TEMP + 16] = 1.0
    x[..., _SetDim.CLEAN_EMBED_HI + 3] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert y[_SetDim.OUTPUT_LO + 1] > 100.0
    assert y[_SetDim.OUTPUT_LO + 0] < -100.0
    assert y[_SetDim.OUTPUT_HI + 0] > 100.0


def test_layer3_pc_byte1_preserve_rule_blocks_known_low_branch_band():
    ffn = _StubFFN()
    _add_layer3_pc_byte1_output_rules(ffn, 100.0, _SetDim)

    x = _pc_byte0_base_input()
    x[..., _SetDim.TEMP + 1] = 1.0
    x[..., _SetDim.TEMP + 16] = 1.0
    x[..., _SetDim.CLEAN_EMBED_HI + 6] = 1.0

    y = _apply_stub_ffn(ffn, x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 1])) < 1e-12
    assert abs(float(y[_SetDim.OUTPUT_LO + 0])) < 1e-12
    assert abs(float(y[_SetDim.OUTPUT_HI + 0])) < 1e-12


def test_layer3_stack0_marker_projection_is_suppressed_after_carry():
    ffn = _StubFFN(hidden_dim=200)
    _set_layer3_ffn(ffn, 100.0, _SetDim)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_STACK0] = 1.0
    x[..., _SetDim.HAS_SE] = 1.0
    x[..., _SetDim.EMBED_LO + 10] = 1.0
    x[..., _SetDim.EMBED_HI + 0] = 1.0

    before = _apply_stub_ffn(ffn, x)[0, 0]
    assert before[_SetDim.OUTPUT_LO + 10] > 0.9
    assert before[_SetDim.OUTPUT_HI + 0] > 0.9

    suppressed = _suppress_layer3_stack0_marker_carry_projection(
        ffn, 100.0, _SetDim
    )
    after = _apply_stub_ffn(ffn, x)[0, 0]

    assert suppressed == 32
    assert abs(float(after[_SetDim.OUTPUT_LO + 10])) < 1e-12
    assert abs(float(after[_SetDim.OUTPUT_HI + 0])) < 1e-12
    assert after[_SetDim.EMBED_LO + 10] > 0.9
    assert after[_SetDim.EMBED_HI + 0] > 0.9


def test_zero_stack0_byte_must_not_inherit_scalar_value_residue():
    block = _StubBlock()
    make_layer3_ffn_op().bake_fn(block, {}, 100.0)

    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.MARK_STACK0] = 1.0
    x[..., _SetDim.HAS_SE] = 1.0
    x[..., _SetDim.EMBED_LO + 7] = 1.0
    x[..., _SetDim.EMBED_HI + 3] = 1.0

    y = _apply_stub_ffn(block.ffn, x)[0, 0]

    assert abs(float(y[_SetDim.OUTPUT_LO + 7])) < 1e-12
    assert abs(float(y[_SetDim.OUTPUT_HI + 3])) < 1e-12
    assert y[_SetDim.EMBED_LO + 7] > 0.9
    assert y[_SetDim.EMBED_HI + 3] > 0.9


def test_layer3_initial_sp_byte2_rewrite_emits_zero_without_touching_bp():
    ffn = _StubFFN(hidden_dim=200)
    _set_layer3_ffn(ffn, 100.0, _SetDim)

    sp = torch.zeros(1, 1, 512)
    sp[..., _SetDim.H1 + 2] = 1.0
    sp[..., _SetDim.BYTE_INDEX_1] = 1.0

    before = _apply_stub_ffn(ffn, sp)[0, 0]
    assert before[_SetDim.OUTPUT_LO + 1] > 0.9
    assert before[_SetDim.OUTPUT_HI + 0] > 0.9

    rewritten = _rewrite_layer3_initial_sp_byte2_to_zero(
        ffn, 100.0, _SetDim
    )
    after = _apply_stub_ffn(ffn, sp)[0, 0]

    assert rewritten == 1
    assert after[_SetDim.OUTPUT_LO + 0] > 0.9
    assert abs(float(after[_SetDim.OUTPUT_LO + 1])) < 1e-12
    assert after[_SetDim.OUTPUT_HI + 0] > 0.9

    bp = torch.zeros(1, 1, 512)
    bp[..., _SetDim.H1 + 3] = 1.0
    bp[..., _SetDim.BYTE_INDEX_1] = 1.0
    bp_after = _apply_stub_ffn(ffn, bp)[0, 0]

    assert bp_after[_SetDim.OUTPUT_LO + 1] > 0.9
    assert bp_after[_SetDim.OUTPUT_HI + 0] > 0.9


def test_layer3_initial_sp_marker_rewrite_emits_f8_without_touching_bp():
    ffn = _StubFFN(hidden_dim=200)
    _set_layer3_ffn(ffn, 100.0, _SetDim)

    sp = torch.zeros(1, 1, 512)
    sp[..., _SetDim.MARK_SP] = 1.0

    before = _apply_stub_ffn(ffn, sp)[0, 0]
    assert before[_SetDim.OUTPUT_LO + 0] > 0.9
    assert before[_SetDim.OUTPUT_HI + 0] > 0.9

    rewritten = _rewrite_layer3_initial_sp_marker_to_f8(
        ffn, 100.0, _SetDim
    )
    after = _apply_stub_ffn(ffn, sp)[0, 0]

    assert rewritten == 2
    assert after[_SetDim.OUTPUT_LO + 8] > 0.9
    assert after[_SetDim.OUTPUT_HI + 15] > 0.9
    assert abs(float(after[_SetDim.OUTPUT_LO + 0])) < 1e-12
    assert abs(float(after[_SetDim.OUTPUT_HI + 0])) < 1e-12

    bp = torch.zeros(1, 1, 512)
    bp[..., _SetDim.MARK_BP] = 1.0
    bp_after = _apply_stub_ffn(ffn, bp)[0, 0]

    assert bp_after[_SetDim.OUTPUT_LO + 0] > 0.9
    assert bp_after[_SetDim.OUTPUT_HI + 0] > 0.9


def test_layer3_bake_leaves_initial_sp_rewrites_out_of_l3():
    block = _StubBlock()
    make_layer3_ffn_op().bake_fn(block, {}, 100.0)

    sp_marker = torch.zeros(1, 1, 512)
    sp_marker[..., _SetDim.MARK_SP] = 1.0
    marker_after = _apply_stub_ffn(block.ffn, sp_marker)[0, 0]
    assert marker_after[_SetDim.OUTPUT_LO + 0] > 0.9
    assert marker_after[_SetDim.OUTPUT_HI + 0] > 0.9
    assert abs(float(marker_after[_SetDim.OUTPUT_LO + 8])) < 1e-12
    assert abs(float(marker_after[_SetDim.OUTPUT_HI + 15])) < 1e-12

    sp_byte2 = torch.zeros(1, 1, 512)
    sp_byte2[..., _SetDim.H1 + 2] = 1.0
    sp_byte2[..., _SetDim.BYTE_INDEX_1] = 1.0
    byte2_after = _apply_stub_ffn(block.ffn, sp_byte2)[0, 0]
    assert byte2_after[_SetDim.OUTPUT_LO + 1] > 0.9
    assert byte2_after[_SetDim.OUTPUT_HI + 0] > 0.9
    assert abs(float(byte2_after[_SetDim.OUTPUT_LO + 0])) < 1e-12
