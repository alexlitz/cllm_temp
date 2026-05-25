"""Focused tests for L3 PC byte1 carry/preserve repair."""

import torch

from neural_vm.vm_step import _SetDim
from neural_vm.unified_compiler.ops.l3_ops import (
    _add_layer3_pc_byte1_output_rules,
    _pc_byte1_prev_head_spec,
)
from neural_vm.unified_compiler.primitives import Primitives


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
