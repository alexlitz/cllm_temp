"""Parity checks for L9 lookup ALU declarative gate assumptions."""

import torch

from c4_release.neural_vm.vm_step import _SetDim, _set_layer9_alu


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 3600):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def test_layer9_lea_adj_ent_fetch_gates_use_one_hot_scale():
    ffn = _StubFFN()
    _set_layer9_alu(ffn, 100.0, _SetDim)

    # LEA hi, no carry, a=15, b=15
    lea_no_carry = 512 + 15 * 16 + 15
    assert ffn.W_up[lea_no_carry, _SetDim.MARK_AX].item() == 2000.0
    assert ffn.W_up[lea_no_carry, _SetDim.MARK_PC].item() == -100000.0
    assert ffn.W_up[lea_no_carry, _SetDim.FETCH_HI + 15].item() == 2000.0
    assert ffn.W_up[lea_no_carry, _SetDim.CARRY + 0].item() == -800.0
    assert ffn.b_up[lea_no_carry].item() == -4050.0

    # LEA hi, with carry, a=15, b=15
    lea_carry = 768 + 15 * 16 + 15
    assert ffn.W_up[lea_carry, _SetDim.CARRY + 0].item() == 800.0
    assert ffn.b_up[lea_carry].item() == -4850.0

    # ADJ hi uses the same one-hot FETCH scaling and carry discrimination.
    adj_no_carry = 1024 + 15 * 16 + 15
    assert ffn.W_up[adj_no_carry, _SetDim.MARK_AX].item() == 2000.0
    assert ffn.W_up[adj_no_carry, _SetDim.FETCH_HI + 15].item() == 2000.0
    assert ffn.W_up[adj_no_carry, _SetDim.CARRY + 0].item() == -800.0
    assert ffn.b_up[adj_no_carry].item() == -4200.0

    # ENT hi also needs the stronger SP-marker blocker after lowering threshold.
    ent_no_borrow = 2048 + 15 * 16 + 15
    assert ffn.W_up[ent_no_borrow, _SetDim.MARK_AX].item() == 2000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.MARK_SP].item() == -100000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.IS_BYTE].item() == -100000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.FETCH_HI + 15].item() == 2000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.CARRY + 0].item() == -800.0
    assert ffn.b_up[ent_no_borrow].item() == -4200.0
