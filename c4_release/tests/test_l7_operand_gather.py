"""Regression checks for L7 operand gather routing."""

from types import SimpleNamespace

import torch

from neural_vm.base_layers import PureAttention, PureFFN
from neural_vm.unified_compiler.ops.l7_ops import _layer7_operand_gather_head_specs
from neural_vm.vm_step import _SetDim, _set_function_call_weights


def _q_weight(spec, slot: int, dim: int) -> float:
    return sum(write.weight for write in spec.q if write.slot == slot and write.dim == dim)


def _o_weight(spec, out_dim: int, slot: int) -> float:
    return sum(write.weight for write in spec.o if write.out_dim == out_dim and write.slot == slot)


def test_l7_stack0_gather_is_fully_blocked_for_address_ops():
    head0, _head1 = _layer7_operand_gather_head_specs(_SetDim)

    assert _q_weight(head0, 0, _SetDim.OP_LEA) == -15.0
    assert _q_weight(head0, 0, _SetDim.OP_ADJ) == -15.0
    assert _q_weight(head0, 0, _SetDim.OP_ENT) == -15.0

    assert _q_weight(head0, 33, _SetDim.OP_LEA) == -150.0
    assert _q_weight(head0, 33, _SetDim.OP_ADJ) == -150.0
    assert _q_weight(head0, 33, _SetDim.OP_ENT) == -150.0


def test_l7_bp_sp_gather_overcomes_l6_alu_clear_for_address_ops():
    _head0, head1 = _layer7_operand_gather_head_specs(_SetDim)

    assert _o_weight(head1, _SetDim.ALU_LO + 0xF, 1 + 0xF) == 6.0
    assert _o_weight(head1, _SetDim.ALU_HI + 0xF, 17 + 0xF) == 6.0


def test_l6_lea_first_step_seed_is_blocked_by_has_se():
    attn5 = PureAttention(dim=512, num_heads=8)
    attn6 = PureAttention(dim=512, num_heads=8)
    ffn6 = PureFFN(dim=512, hidden_dim=4096)
    blocks = [SimpleNamespace(attn=None, ffn=None) for _ in range(7)]
    blocks[5] = SimpleNamespace(attn=attn5, ffn=None)
    blocks[6] = SimpleNamespace(attn=attn6, ffn=ffn6)
    model = SimpleNamespace(blocks=blocks)

    with torch.no_grad():
        _set_function_call_weights(model, 100.0, _SetDim, 64)

    assert ffn6.W_up[1700, _SetDim.HAS_SE].item() == -1000.0
    assert ffn6.W_up[1701, _SetDim.HAS_SE].item() == -1000.0
