"""Parity tests for L8 FFN bands migrated to declarative rules."""

import torch

from c4_release.neural_vm.unified_compiler.ops.l8_ops import (
    _layer8_multibyte_fetch_head_spec,
    _layer8_multibyte_routing_rules,
    lower_layer8_multibyte_routing_ir,
)
from c4_release.neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
from c4_release.neural_vm.vm_step import (
    _SetDim,
    _set_layer8_alu,
    _set_layer8_multibyte_routing,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 2200):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def _project_head_terms(terms, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(64)
    for term in terms:
        out[term.slot] += x[term.dim] * term.weight
    return out


def test_layer8_multibyte_fetch_blocks_ax_byte_k_self_match():
    spec = _layer8_multibyte_fetch_head_spec(_SetDim)
    query = torch.zeros(512)
    correct_code_byte = torch.zeros(512)
    stale_ax_byte = torch.zeros(512)

    query[_SetDim.CONST] = 1.0
    query[_SetDim.IS_BYTE] = 1.0
    query[_SetDim.H1 + 1] = 1.0
    query[_SetDim.FETCH_LO + 2] = 1.0
    query[_SetDim.FETCH_HI + 0] = 1.0

    correct_code_byte[_SetDim.IS_BYTE] = 1.0
    correct_code_byte[_SetDim.ADDR_KEY + 2] = 1.0
    correct_code_byte[_SetDim.ADDR_KEY + 16] = 1.0
    correct_code_byte[_SetDim.ADDR_KEY + 32] = 1.0
    correct_code_byte[_SetDim.CLEAN_EMBED_LO + 2] = 1.0

    stale_ax_byte[_SetDim.IS_BYTE] = 1.0
    stale_ax_byte[_SetDim.H1 + 1] = 1.0
    stale_ax_byte[_SetDim.ADDR_KEY + 2] = 1.0
    stale_ax_byte[_SetDim.ADDR_KEY + 16] = 1.0
    stale_ax_byte[_SetDim.ADDR_KEY + 32] = 1.0
    stale_ax_byte[_SetDim.CLEAN_EMBED_LO + 0] = 1.0

    q = _project_head_terms(spec.q, query)
    correct_score = torch.dot(q, _project_head_terms(spec.k, correct_code_byte))
    stale_score = torch.dot(q, _project_head_terms(spec.k, stale_ax_byte))

    assert correct_score.item() == 1200.0
    assert stale_score.item() <= -1000.0


def test_layer8_multibyte_routing_ir_matches_legacy_helper():
    actual = _StubFFN()
    expected = _StubFFN()

    unit_start = _set_layer8_alu(actual, 100.0, _SetDim)
    end = lower_layer8_multibyte_routing_ir(
        actual,
        100.0,
        _SetDim,
        start_unit=unit_start,
    )
    _set_layer8_multibyte_routing(expected, 100.0, _SetDim)

    assert end == unit_start + 32
    assert len(_layer8_multibyte_routing_rules(100.0)) == 32
    _assert_same_ffn(actual, expected)


def test_layer8_lea_adj_ent_fetch_gates_use_one_hot_scale():
    ffn = _StubFFN()
    _set_layer8_alu(ffn, 100.0, _SetDim)

    # LEA lo, a=0, b=8
    assert ffn.W_up[256 + 8, _SetDim.FETCH_LO + 8].item() == 2000.0
    assert ffn.W_up[256 + 8, _SetDim.MARK_PC].item() == -100000.0
    assert ffn.b_up[256 + 8].item() == -8050.0

    # LEA carry, first firing pair a=1, b=15
    assert ffn.W_up[888, _SetDim.FETCH_LO + 15].item() == 2000.0
    assert ffn.b_up[888].item() == -8050.0

    # ADJ lo, a=0, b=8
    assert ffn.W_up[1008 + 8, _SetDim.FETCH_LO + 8].item() == 2000.0
    assert ffn.b_up[1008 + 8].item() == -8500.0

    # ENT lo, sp_lo=0, imm_lo=8
    assert ffn.W_up[1504 + 8, _SetDim.FETCH_LO + 8].item() == 2000.0
    assert ffn.b_up[1504 + 8].item() == -8500.0

    # ENT borrow, first firing pair sp_lo=0, imm_lo=0
    assert ffn.W_up[1760, _SetDim.FETCH_LO + 0].item() == 2000.0
    assert ffn.W_up[1760, _SetDim.IS_BYTE].item() == -100000.0
    assert ffn.b_up[1760].item() == -8500.0


def test_early_mem_to_alu_route_stays_disabled_until_pre_l8_addr_keys_exist():
    ops = {op.name: op for op in all_core_ops(alu_mode="efficient")}

    assert ops["layer4_sp_to_addr_key"].claims == set()
    assert ops["layer8_mem_to_alu"].claims == set()
