"""Byte-equivalence test for the L3 carry-forward attention refactor.

`make_layer3_carry_forward_attn_op` previously called
`_set_carry_forward_attn` for heads 0-3 (PC/AX/SP/BP). After the
2026-05-10 refactor those four heads use
`Primitives.carry_forward_attention`. This test asserts the resulting
W_q/W_k/W_v/W_o tensors are byte-identical (`torch.equal`) to the
legacy helper output.

Heads 4-6 are not touched by the refactor and are excluded here so
this test stays focused on the primitive substitution.
"""
import torch
import torch.nn as nn

from neural_vm.vm_step import _set_carry_forward_attn, _SetDim
from neural_vm.unified_compiler.primitives import Primitives


D_MODEL = 512
N_HEADS = 8
HD = D_MODEL // N_HEADS  # 64


class _StubAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = N_HEADS
        self.W_q = nn.Parameter(torch.zeros(D_MODEL, D_MODEL))
        self.W_k = nn.Parameter(torch.zeros(D_MODEL, D_MODEL))
        self.W_v = nn.Parameter(torch.zeros(D_MODEL, D_MODEL))
        self.W_o = nn.Parameter(torch.zeros(D_MODEL, D_MODEL))


def _bake_legacy(attn):
    BD = _SetDim
    PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
    _set_carry_forward_attn(
        attn, 0, BD.MARK_PC, PC_I, PC_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )
    _set_carry_forward_attn(
        attn, 1, BD.MARK_AX, AX_I, AX_I, HD, BD.AX_CARRY_LO, BD.AX_CARRY_HI
    )
    _set_carry_forward_attn(
        attn, 2, BD.MARK_SP, SP_I, SP_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )
    _set_carry_forward_attn(
        attn, 3, BD.MARK_BP, BP_I, BP_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )


def _bake_primitive(attn):
    BD = _SetDim
    PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
    Primitives.carry_forward_attention(
        attn, 0, BD.MARK_PC, PC_I, PC_I,
        BD.EMBED_LO, BD.EMBED_HI, HD=HD,
    )
    Primitives.carry_forward_attention(
        attn, 1, BD.MARK_AX, AX_I, AX_I,
        BD.AX_CARRY_LO, BD.AX_CARRY_HI, HD=HD,
    )
    Primitives.carry_forward_attention(
        attn, 2, BD.MARK_SP, SP_I, SP_I,
        BD.EMBED_LO, BD.EMBED_HI, HD=HD,
    )
    Primitives.carry_forward_attention(
        attn, 3, BD.MARK_BP, BP_I, BP_I,
        BD.EMBED_LO, BD.EMBED_HI, HD=HD,
    )


def test_l3_carry_forward_primitive_byte_identical():
    legacy = _StubAttn()
    new = _StubAttn()
    with torch.no_grad():
        _bake_legacy(legacy)
        _bake_primitive(new)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        a = getattr(legacy, name).data
        b = getattr(new, name).data
        assert torch.equal(a, b), (
            f"L3 carry-forward {name} diverged between legacy helper and "
            f"Primitives.carry_forward_attention "
            f"(diff_count={(a != b).sum().item()})"
        )
