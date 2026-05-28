"""Parity tests for L15 FFN bands migrated to CompilerIR rules."""

import pytest
import torch

from c4_release.neural_vm.unified_compiler.ops.l15_ops import (
    _layer15_si_mem_addr0_from_stack0_spec,
    _suppress_l15_lookup_during_current_store_generation,
    lower_l15_nibble_copy_ir,
    make_l15_attention_resize_op,
    make_l15_nibble_copy_ir,
)
from c4_release.neural_vm.vm_step import (
    Token,
    _SetDim,
    _set_layer15_memory_lookup,
    _set_nibble_copy_ffn,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 128):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


class _StubAttn:
    def __init__(self, *, d_model: int = 512, num_heads: int = 4, head_dim: int = 64):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_q = torch.zeros(num_heads * head_dim, d_model)
        self.W_k = torch.zeros(num_heads * head_dim, d_model)
        self.W_v = torch.zeros(num_heads * head_dim, d_model)
        self.W_o = torch.zeros(d_model, num_heads * head_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_layer15_nibble_copy_ir_matches_legacy_helper():
    actual = _StubFFN()
    expected = _StubFFN()

    end = lower_l15_nibble_copy_ir(actual, _SetDim, S=100.0)
    _set_nibble_copy_ffn(expected, 100.0, _SetDim)

    assert end == 42
    assert make_l15_nibble_copy_ir().required_ffn_units() == 42
    _assert_same_ffn(actual, expected)


def test_layer15_lookup_blocks_store_opcodes_on_load_restore_row():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    for head in range(4):
        row = head * 64 + 31
        assert attn.W_q[row, _SetDim.OP_LI_RELAY] == 20000.0
        expected_mark_ax = 0.0 if head == 0 else -20000.0
        assert attn.W_q[row, _SetDim.MARK_AX] == expected_mark_ax
        assert attn.W_q[row, _SetDim.OP_SI] == -20000.0
        assert attn.W_q[row, _SetDim.OP_SC] == -20000.0
        assert attn.W_k[row, _SetDim.MEM_STORE] == 5.0


def test_layer15_lookup_blocks_non_load_marker_setup():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    assert attn.W_q[0, _SetDim.CONST] == -200000.0
    assert attn.W_q[0, _SetDim.OP_LI_RELAY] == 200000.0
    assert attn.W_q[0, _SetDim.OP_LC_RELAY] == 200000.0
    assert attn.W_q[0, _SetDim.OP_LI] == 200000.0
    assert attn.W_q[0, _SetDim.OP_LC] == 200000.0
    assert attn.W_q[0, _SetDim.CMP + 3] == 50000.0
    assert attn.W_q[0, _SetDim.OP_JSR] == -1000000.0
    assert attn.W_q[0, _SetDim.OP_ENT] == -1000000.0
    assert attn.W_q[0, _SetDim.OP_LEA] == -1000000.0
    assert attn.W_q[0, _SetDim.OP_IMM] == -1000000.0
    assert attn.W_q[0, _SetDim.MARK_STACK0] == 75000.0
    assert attn.W_q[0, _SetDim.HAS_SE] == 75000.0
    assert attn.W_q[0, _SetDim.ADDR_B0_LO + 8] == 75000.0
    assert attn.W_q[0, _SetDim.ADDR_B0_HI + 14] == 75000.0
    assert attn.W_q[0, _SetDim.ADDR_B0_HI + 15] == -100000.0
    assert attn.W_q[1, _SetDim.MARK_STACK0] == 50.0
    assert attn.W_q[1, _SetDim.HAS_SE] == 50.0
    assert attn.W_q[1, _SetDim.ADDR_B0_LO + 8] == 50.0
    assert attn.W_q[1, _SetDim.ADDR_B0_HI + 14] == 50.0
    assert attn.W_q[1, _SetDim.ADDR_B0_HI + 15] == -150.0
    assert attn.W_q[62, _SetDim.H1 + 2] == 100000.0
    assert attn.W_q[62, _SetDim.MARK_BP] == 100000.0
    assert attn.W_q[62, _SetDim.TEMP + 10] == 100000.0
    assert attn.W_q[62, _SetDim.TEMP + 24] == 100000.0
    assert attn.W_q[62, _SetDim.IS_BYTE] == 0.0
    assert attn.W_k[62, _SetDim.CONST] == -300000.0
    for head in range(1, 4):
        row = head * 64
        assert attn.W_q[row, _SetDim.OP_JSR] == 0.0
        assert attn.W_q[row, _SetDim.OP_ENT] == 0.0
        assert attn.W_q[row, _SetDim.OP_LEA] == 0.0
        assert attn.W_q[row, _SetDim.OP_IMM] == 0.0
        assert attn.W_q[row, _SetDim.MARK_STACK0] == -100000.0
        assert attn.W_q[row, _SetDim.MARK_SP] == -100000.0
        blocker_row = head * 64 + 62
        assert attn.W_q[blocker_row, _SetDim.H1 + 2] == 100000.0
        assert attn.W_q[blocker_row, _SetDim.MARK_BP] == 100000.0
        assert attn.W_q[blocker_row, _SetDim.TEMP + 10] == 100000.0
        assert attn.W_q[blocker_row, _SetDim.TEMP + 24] == 100000.0
        assert attn.W_q[blocker_row, _SetDim.IS_BYTE] == 0.0
        assert attn.W_k[blocker_row, _SetDim.CONST] == -300000.0


def test_layer15_lookup_blocks_top_store_stack0_marker_only():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 42
    assert attn.W_q[row, _SetDim.MARK_STACK0] == 10000.0
    assert attn.W_q[row, _SetDim.HAS_SE] == 10000.0
    assert attn.W_q[row, _SetDim.MEM_STORE] == 10000.0
    assert attn.W_q[row, _SetDim.EMBED_LO + 0] == 10000.0
    assert attn.W_q[row, _SetDim.EMBED_HI + 14] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_LO + 0] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_HI + 14] == 10000.0
    assert attn.W_q[row, _SetDim.OP_LI_RELAY] == 50000.0
    assert attn.W_q[row, _SetDim.OP_LC_RELAY] == 50000.0
    assert attn.W_k[row, _SetDim.CONST] == 20.0

    row = 60
    assert attn.W_q[row, _SetDim.CONST] == -30000.0
    assert attn.W_q[row, _SetDim.MARK_STACK0] == 10000.0
    assert attn.W_q[row, _SetDim.MARK_SP] == -100000.0
    assert attn.W_q[row, _SetDim.HAS_SE] == 10000.0
    assert attn.W_q[row, _SetDim.MEM_STORE] == 150000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_LO + 8] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_HI + 14] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_HI + 15] == -20000.0
    assert attn.W_k[row, _SetDim.CONST] == -20.0

    assert attn.W_q[59, _SetDim.CONST] == -17500.0
    assert attn.W_q[59, _SetDim.MARK_STACK0] == 5000.0
    assert attn.W_q[59, _SetDim.HAS_SE] == 5000.0
    assert attn.W_q[59, _SetDim.ADDR_B0_LO + 8] == 5000.0
    assert attn.W_q[59, _SetDim.ADDR_B0_HI + 14] == 5000.0
    assert attn.W_q[59, _SetDim.IS_BYTE] == -20000.0
    assert attn.W_q[59, _SetDim.MEM_STORE] == -25000.0
    assert attn.W_k[59, _SetDim.ADDR_B0_LO + 8] == 5000.0
    assert attn.W_k[59, _SetDim.ADDR_B0_HI + 14] == 5000.0
    assert attn.W_k[59, _SetDim.MEM_VAL_B1] == 0.0
    assert attn.W_k[59, _SetDim.L2H0 + 4] == 0.0
    assert attn.W_k[59, _SetDim.STACK0_BYTE0] == 25000.0
    assert attn.W_k[59, _SetDim.ADDR_B0_HI + 15] == 0.0
    assert attn.W_q[61, _SetDim.CONST] == -35000.0
    assert attn.W_q[61, _SetDim.MARK_AX] == 10000.0
    assert attn.W_q[61, _SetDim.OP_LI_RELAY] == 10000.0
    assert attn.W_q[61, _SetDim.OP_LC_RELAY] == 10000.0
    assert attn.W_q[61, _SetDim.MARK_STACK0] == 25000.0
    assert attn.W_q[61, _SetDim.ADDR_B0_LO + 8] == 10000.0
    assert attn.W_q[61, _SetDim.ADDR_B0_HI + 14] == 10000.0
    assert attn.W_q[61, _SetDim.IS_BYTE] == -40000.0
    assert attn.W_q[61, _SetDim.MEM_STORE] == -40000.0
    assert attn.W_k[61, _SetDim.MEM_VAL_B1] == 10000.0
    assert attn.W_k[61, _SetDim.ADDR_B0_LO + 8] == 10000.0
    assert attn.W_k[61, _SetDim.ADDR_B0_HI + 14] == 10000.0
    assert attn.W_k[61, _SetDim.STACK0_BYTE0] == -20000.0

    for head in range(1, 4):
        row = head * 64 + 42
        assert attn.W_q[row, _SetDim.MARK_STACK0] == 0.0
        assert attn.W_q[row, _SetDim.MEM_STORE] == 0.0
        assert attn.W_k[row, _SetDim.CONST] == 0.0


def test_layer15_legacy_lookup_neutralizes_top_store_miss_score_rows():
    attn = _StubAttn()

    _set_layer15_memory_lookup(attn, 100.0, _SetDim, 64)
    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 42
    assert attn.W_q[row, _SetDim.MARK_STACK0] == 10000.0
    assert attn.W_q[row, _SetDim.MEM_STORE] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_LO + 0] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_HI + 14] == 10000.0
    assert attn.W_k[row, _SetDim.CONST] == 20.0

    row = 59
    assert attn.W_q[row, _SetDim.CONST] == -17500.0
    assert attn.W_q[row, _SetDim.MARK_STACK0] == 5000.0
    assert attn.W_q[row, _SetDim.MEM_STORE] == -25000.0
    assert attn.W_k[row, _SetDim.MEM_VAL_B1] == 0.0
    assert attn.W_k[row, _SetDim.L2H0 + 4] == 0.0
    assert attn.W_k[row, _SetDim.STACK0_BYTE0] == 25000.0
    assert attn.W_k[row, _SetDim.ADDR_B0_HI + 15] == 0.0
    assert torch.count_nonzero(attn.W_v[row, :]) > 0
    assert torch.count_nonzero(attn.W_o[:, row]) > 0

    row = 60
    assert attn.W_q[row, _SetDim.MARK_STACK0] == 10000.0
    assert attn.W_q[row, _SetDim.MARK_SP] == -100000.0
    assert attn.W_q[row, _SetDim.MEM_STORE] == 150000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_LO + 8] == 10000.0
    assert attn.W_q[row, _SetDim.ADDR_B0_HI + 14] == 10000.0
    assert attn.W_k[row, _SetDim.CONST] == -20.0
    assert torch.count_nonzero(attn.W_v[row, :]) > 0
    assert torch.count_nonzero(attn.W_o[:, row]) > 0

    row = 61
    assert attn.W_q[row, _SetDim.OP_LI_RELAY] == 10000.0
    assert attn.W_q[row, _SetDim.MARK_STACK0] == 25000.0
    assert attn.W_k[row, _SetDim.MEM_VAL_B1] == 10000.0
    assert torch.count_nonzero(attn.W_v[row, :]) > 0
    assert torch.count_nonzero(attn.W_o[:, row]) > 0


def test_layer15_lookup_source_gate_blocks_bp_register_bytes():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    for head in range(4):
        row = head * 64 + 37
        if head == 0:
            assert attn.W_q[row, _SetDim.MARK_AX] == 0.0
            assert attn.W_q[row, _SetDim.MARK_STACK0] == 3000.0
        for dim in (
            _SetDim.H1 + 2,
            _SetDim.H2 + 2,
            _SetDim.H3 + 2,
            _SetDim.L2H0 + 2,
            _SetDim.H1 + 3,
            _SetDim.H2 + 3,
            _SetDim.H3 + 3,
            _SetDim.L2H0 + 3,
        ):
            assert attn.W_k[row, dim] == -80.0


def test_layer15_lookup_blocks_nonpop_stack0_marker_in_current_head():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    for head in range(4):
        row = head * 64 + 63
        assert attn.W_q[row, _SetDim.MARK_STACK0] == 60000.0
        assert attn.W_q[row, _SetDim.MEM_STORE] == 10000.0
        assert attn.W_q[row, _SetDim.HAS_SE] == 0.0
        assert attn.W_q[row, _SetDim.CMP + 3] == -15000.0
        assert attn.W_q[row, _SetDim.IS_BYTE] == 60000.0
        assert attn.W_q[row, _SetDim.OP_LI_RELAY] == -60000.0
        assert attn.W_q[row, _SetDim.OP_LC_RELAY] == -60000.0
        assert attn.W_q[row, _SetDim.ADDR_B0_LO + 8] == -40000.0
        assert attn.W_q[row, _SetDim.EMBED_LO + 8] == 10000.0
        assert attn.W_q[row, _SetDim.EMBED_HI + 14] == 10000.0
        assert attn.W_q[row, _SetDim.ADDR_B0_LO + 0] == 10000.0
        assert attn.W_q[row, _SetDim.ADDR_B0_HI + 14] == -20000.0
        assert attn.W_k[row, _SetDim.CONST] == -20.0
        assert attn.W_k[row, _SetDim.MEM_VAL_B1] == 0.0
        assert torch.count_nonzero(attn.W_v[row, :]) == 0
        assert torch.count_nonzero(attn.W_o[:, row]) == 0

    for head in range(1, 4):
        next_head_row0 = head * 64
        assert attn.W_q[next_head_row0, _SetDim.MARK_STACK0] == 0.0
        assert attn.W_q[next_head_row0, _SetDim.IS_BYTE] == 0.0
        assert attn.W_k[next_head_row0, _SetDim.CONST] == 0.0


def test_layer15_nonpop_stack0_marker_allows_e8_preserve_lookup():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 63
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.ADDR_B0_LO + 8] = 1.0
    query[_SetDim.ADDR_B0_HI + 14] = 1.0

    f8_query = query.clone()
    f8_query[_SetDim.ADDR_B0_HI + 14] = 0.0
    f8_query[_SetDim.ADDR_B0_HI + 15] = 1.0

    const_key = torch.zeros(512)
    const_key[_SetDim.CONST] = 1.0

    assert torch.dot(attn.W_q[row], query) == 0.0
    assert torch.dot(attn.W_q[row], f8_query) > 0.0
    assert torch.dot(attn.W_q[row], f8_query) * torch.dot(
        attn.W_k[row], const_key
    ) < 0.0

    activation_row = 0
    assert torch.dot(attn.W_q[activation_row], query) > 0.0
    assert torch.dot(attn.W_q[activation_row], f8_query) < 0.0
    store_anchor_row = 1
    assert torch.dot(attn.W_q[store_anchor_row], query) > 0.0
    assert torch.dot(attn.W_q[store_anchor_row], f8_query) < 0.0


def test_layer15_e8_preserve_row_reinforces_ax_li_loads():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 61
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_AX] = 1.0
    query[_SetDim.OP_LI_RELAY] = 1.0
    query[_SetDim.HAS_SE] = 1.0
    query[_SetDim.ADDR_B0_LO + 8] = 1.0
    query[_SetDim.ADDR_B0_HI + 14] = 1.0

    key_e8 = torch.zeros(512)
    key_e8[_SetDim.MEM_VAL_B1] = 1.0
    key_e8[_SetDim.ADDR_B0_LO + 8] = 1.0
    key_e8[_SetDim.ADDR_B0_HI + 14] = 1.0

    key_e0 = torch.zeros(512)
    key_e0[_SetDim.MEM_VAL_B1] = 1.0
    key_e0[_SetDim.ADDR_B0_LO + 0] = 1.0
    key_e0[_SetDim.ADDR_B0_HI + 14] = 1.0

    stack_key = key_e8.clone()
    stack_key[_SetDim.STACK0_BYTE0] = 1.0

    q_value = torch.dot(attn.W_q[row], query)
    assert q_value > 0.0
    assert q_value * torch.dot(attn.W_k[row], key_e8) > (
        q_value * torch.dot(attn.W_k[row], key_e0)
    )
    assert q_value * torch.dot(attn.W_k[row], key_e8) > (
        q_value * torch.dot(attn.W_k[row], stack_key)
    )

    stack0_query = torch.zeros(512)
    stack0_query[_SetDim.CONST] = 1.0
    stack0_query[_SetDim.MARK_STACK0] = 1.0
    stack0_query[_SetDim.ADDR_B0_LO + 0] = 1.0
    stack0_query[_SetDim.ADDR_B0_HI + 14] = 1.0

    assert torch.dot(attn.W_q[row], stack0_query) == 0.0


def test_layer15_early_ent_stack0_preserve_prefers_ent_source_only_early():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 58
    early_stack0_query = torch.zeros(512)
    early_stack0_query[_SetDim.CONST] = 1.0
    early_stack0_query[_SetDim.MARK_STACK0] = 1.0
    early_stack0_query[_SetDim.OP_ENT] = 0.001

    later_stack0_query = torch.zeros(512)
    later_stack0_query[_SetDim.CONST] = 1.0
    later_stack0_query[_SetDim.MARK_STACK0] = 1.0

    bp_query = early_stack0_query.clone()
    bp_query[_SetDim.MARK_STACK0] = 0.0
    bp_query[_SetDim.MARK_BP] = 1.0
    bp_query[_SetDim.OP_ENT] = 8.0
    byte_query = early_stack0_query.clone()
    byte_query[_SetDim.MARK_STACK0] = 0.0
    byte_query[_SetDim.IS_BYTE] = 1.0
    byte_query[_SetDim.OP_ENT] = 8.0

    ent_key = torch.zeros(512)
    ent_key[_SetDim.OP_ENT] = 8.0
    plain_key = torch.zeros(512)
    plain_key[_SetDim.CONST] = 1.0

    assert torch.dot(attn.W_q[row], early_stack0_query) > 0.0
    assert torch.dot(attn.W_q[row], later_stack0_query) == 0.0
    assert torch.dot(attn.W_q[row], bp_query) < 0.0
    assert torch.dot(attn.W_q[row], byte_query) < 0.0
    assert torch.dot(attn.W_k[row], ent_key) > 0.0
    assert torch.dot(attn.W_k[row], plain_key) == 0.0


def test_layer15_pop_d8_stack0_query_prefers_e0_mem_value():
    attn = _StubAttn(num_heads=10, head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    base = 9 * 64
    row = base + 63
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.HAS_SE] = 1.0
    query[_SetDim.CMP + 3] = 4.0
    query[_SetDim.ADDR_B0_LO + 8] = 1.0
    query[_SetDim.ADDR_B0_HI + 13] = 1.0

    nonpop_query = query.clone()
    nonpop_query[_SetDim.CMP + 3] = 0.0
    byte_query = query.clone()
    byte_query[_SetDim.IS_BYTE] = 1.0
    ax_query = query.clone()
    ax_query[_SetDim.MARK_STACK0] = 0.0
    ax_query[_SetDim.MARK_AX] = 1.0

    key_e0_value = torch.zeros(512)
    key_e0_value[_SetDim.CONST] = 1.0
    key_e0_value[_SetDim.MEM_VAL_B1] = 1.0
    key_e0_value[_SetDim.ADDR_B0_LO + 0] = 1.0
    key_e0_value[_SetDim.ADDR_B0_HI + 14] = 1.0
    key_e0_marker = torch.zeros(512)
    key_e0_marker[_SetDim.CONST] = 1.0
    key_e0_marker[_SetDim.ADDR_B0_LO + 0] = 1.0
    key_e0_marker[_SetDim.ADDR_B0_HI + 14] = 1.0
    plain_key = torch.zeros(512)
    plain_key[_SetDim.CONST] = 1.0

    q_value = torch.dot(attn.W_q[row], query)
    sink_q = torch.dot(attn.W_q[base], query)

    def score(q, k):
        return (
            torch.dot(attn.W_q[base], q) * torch.dot(attn.W_k[base], k)
            + torch.dot(attn.W_q[row], q) * torch.dot(attn.W_k[row], k)
        )

    assert q_value > 0.0
    assert sink_q > 0.0
    assert torch.dot(attn.W_q[row], nonpop_query) == 0.0
    assert torch.dot(attn.W_q[row], byte_query) < 0.0
    assert torch.dot(attn.W_q[row], ax_query) < 0.0
    assert score(query, key_e0_value) > score(query, key_e0_marker)
    assert score(query, plain_key) < 0.0
    assert score(ax_query, key_e0_value) < 0.0
    assert attn.W_v[base + 1 + 15, _SetDim.CLEAN_EMBED_LO + 15] == 1.0
    assert attn.W_v[base + 17 + 2, _SetDim.CLEAN_EMBED_HI + 2] == 1.0
    assert attn.W_o[_SetDim.OUTPUT_LO + 15, base + 1 + 15] == 40.0
    assert attn.W_o[_SetDim.OUTPUT_HI + 2, base + 17 + 2] == 40.0


def test_layer15_attention_resize_reapplies_post_resize_guards():
    class _Block:
        def __init__(self):
            self._n_layers_hint = 32
            self.attn = _StubAttn(num_heads=8, head_dim=64)

    block = _Block()
    dim_positions = {
        name: value
        for name, value in vars(_SetDim).items()
        if isinstance(value, int)
    }
    make_l15_attention_resize_op().bake_fn(block, dim_positions, 100.0)

    base = 9 * 64
    assert block.attn.num_heads == 14
    assert block.attn.head_dim == 64
    assert block.attn.W_q[base, _SetDim.CONST] == 1.0
    assert block.attn.W_k[base, _SetDim.CONST] == -1000.0
    row = base + 63
    assert block.attn.W_q[row, _SetDim.MARK_STACK0] == 50000.0
    assert block.attn.W_q[row, _SetDim.MARK_AX] == -250000.0
    assert block.attn.W_k[row, _SetDim.MEM_VAL_B1] == 1.0
    assert block.attn.W_o[_SetDim.OUTPUT_LO + 15, base + 1 + 15] == 40.0


def test_layer15_lookup_blocks_mem_address_byte_queries_by_head():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    byte_flags = {
        1: _SetDim.BYTE_INDEX_0,
        2: _SetDim.BYTE_INDEX_1,
        3: _SetDim.BYTE_INDEX_2,
    }
    for head, byte_flag in byte_flags.items():
        row = head * 64 + 36
        assert attn.W_q[row, _SetDim.CONST] == 0.0
        assert attn.W_q[row, _SetDim.H1 + 4] == 100000.0
        assert attn.W_q[row, _SetDim.IS_BYTE] == 0.0
        assert attn.W_q[row, byte_flag] == 0.0
        assert attn.W_k[row, _SetDim.CONST] == -20.0


def test_layer15_lookup_blocks_pc_byte_queries_by_head():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    for head in range(4):
        row = head * 64 + 35
        assert attn.W_q[row, _SetDim.H1 + 0] == 100000.0
        assert attn.W_q[row, _SetDim.MARK_PC] == 100000000.0
        assert attn.W_q[row, _SetDim.IS_BYTE] == 0.0
        assert attn.W_k[row, _SetDim.CONST] == -100000.0


def test_layer15_stack0_marker_preserve_reads_latest_stack0_byte0():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 36
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.HAS_SE] = 1.0

    pop_query = query.clone()
    pop_query[_SetDim.CMP + 3] = 4.0
    store_query = query.clone()
    store_query[_SetDim.MEM_STORE] = 1.0
    byte_query = query.clone()
    byte_query[_SetDim.IS_BYTE] = 1.0
    ax_li_query = torch.zeros(512)
    ax_li_query[_SetDim.CONST] = 1.0
    ax_li_query[_SetDim.MARK_AX] = 1.0
    ax_li_query[_SetDim.HAS_SE] = 1.0
    ax_li_query[_SetDim.OP_LI_RELAY] = 1.0

    stack0_byte0_key = torch.zeros(512)
    stack0_byte0_key[_SetDim.CONST] = 1.0
    stack0_byte0_key[_SetDim.H1 + 10] = 1.0
    stack0_byte0_key[_SetDim.BYTE_INDEX_0] = 1.0

    bp_byte0_key = stack0_byte0_key.clone()
    bp_byte0_key[_SetDim.H1 + 3] = 1.0
    ent_stack0_byte0_key = stack0_byte0_key.clone()
    ent_stack0_byte0_key[_SetDim.OP_ENT] = 10.0

    const_key = torch.zeros(512)
    const_key[_SetDim.CONST] = 1.0

    q_value = torch.dot(attn.W_q[row], query)
    assert q_value > 0.0
    assert torch.dot(attn.W_q[row], pop_query) < 0.0
    assert torch.dot(attn.W_q[row], store_query) < 0.0
    assert torch.dot(attn.W_q[row], byte_query) < 0.0
    assert torch.dot(attn.W_q[row], ax_li_query) == 0.0
    assert q_value * torch.dot(attn.W_k[row], stack0_byte0_key) > 0.0
    assert q_value * torch.dot(attn.W_k[row], const_key) < 0.0
    assert q_value * torch.dot(attn.W_k[row], bp_byte0_key) < 0.0
    assert q_value * torch.dot(attn.W_k[row], ent_stack0_byte0_key) < 0.0


def test_layer15_mem_addr0_query_sinks_head1_before_mem_addr1():
    attn = _StubAttn(head_dim=64)

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 1 * 64 + 36
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.H1 + 4] = 1.0
    query[_SetDim.IS_BYTE] = 1.0
    query[_SetDim.BYTE_INDEX_0] = 1.0

    non_mem_query = query.clone()
    non_mem_query[_SetDim.H1 + 4] = 0.0

    stack0_marker_query = torch.zeros(512)
    stack0_marker_query[_SetDim.CONST] = 1.0
    stack0_marker_query[_SetDim.MARK_STACK0] = 1.0

    const_key = torch.zeros(512)
    const_key[_SetDim.CONST] = 1.0

    q_value = torch.dot(attn.W_q[row], query)
    assert q_value == 100000.0
    assert torch.dot(attn.W_q[row], non_mem_query) == 0.0
    assert torch.dot(attn.W_q[row], stack0_marker_query) == 0.0
    assert q_value * torch.dot(attn.W_k[row], const_key) < -1_900_000.0


def test_layer15_lookup_strengthens_local_slot_byte0_match():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    for head in range(4):
        bit3_row = head * 64 + 7
        assert attn.W_q[bit3_row, _SetDim.ADDR_B0_LO + 8] == 100.0
        assert attn.W_k[bit3_row, _SetDim.ADDR_B0_LO + 8] == 100.0
        assert attn.W_q[bit3_row, _SetDim.ADDR_B0_LO + 0] == -100.0
        assert attn.W_k[bit3_row, _SetDim.ADDR_B0_LO + 0] == -100.0

        hi_bit1_row = head * 64 + 9
        assert attn.W_q[hi_bit1_row, _SetDim.ADDR_B0_HI + 14] == 100.0
        assert attn.W_k[hi_bit1_row, _SetDim.ADDR_B0_HI + 14] == 100.0
        assert attn.W_q[hi_bit1_row, _SetDim.ADDR_B0_HI + 13] == -100.0
        assert attn.W_k[hi_bit1_row, _SetDim.ADDR_B0_HI + 13] == -100.0

        exact_row = head * 64 + 43 + 8
        assert attn.W_q[exact_row, _SetDim.CONST] == -100.0
        assert attn.W_q[exact_row, _SetDim.ADDR_B0_LO + 8] == 100.0
        assert attn.W_q[exact_row, _SetDim.OP_LI_RELAY] == 100.0
        expected_lc_gate = 100.0 if head == 0 else 0.0
        assert attn.W_q[exact_row, _SetDim.OP_LC_RELAY] == expected_lc_gate
        assert attn.W_k[exact_row, _SetDim.ADDR_B0_LO + 8] == 100.0
        assert attn.W_q[exact_row, _SetDim.ADDR_B0_LO + 0] == 0.0
        assert attn.W_k[exact_row, _SetDim.ADDR_B0_LO + 0] == 0.0


def test_layer15_pop_stack0_marker_low0_query_prefers_low8_store_key():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 34
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.HAS_SE] = 1.0
    query[_SetDim.CMP + 3] = 4.0
    query[_SetDim.ADDR_B0_LO + 0] = 1.0

    byte_query = query.clone()
    byte_query[_SetDim.IS_BYTE] = 1.0
    sp_query = query.clone()
    sp_query[_SetDim.MARK_SP] = 1.0
    store_query = query.clone()
    store_query[_SetDim.MEM_STORE] = 1.0

    key_low8 = torch.zeros(512)
    key_low8[_SetDim.ADDR_B0_LO + 8] = 1.0

    key_low0 = torch.zeros(512)
    key_low0[_SetDim.ADDR_B0_LO + 0] = 1.0

    q_value = torch.dot(attn.W_q[row], query)
    assert q_value > 0.0
    assert torch.dot(attn.W_q[row], byte_query) < 0.0
    assert torch.dot(attn.W_q[row], sp_query) < 0.0
    assert torch.dot(attn.W_q[row], store_query) < 0.0
    assert torch.dot(attn.W_k[row], key_low8) > 0.0
    assert torch.dot(attn.W_k[row], key_low0) == 0.0
    assert q_value * torch.dot(attn.W_k[row], key_low8) > 1_000_000.0


def test_layer15_pop_low8_row_is_neutral_for_nonpop_low8_preserve():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 34
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.HAS_SE] = 1.0
    query[_SetDim.ADDR_B0_LO + 8] = 1.0

    key_low8 = torch.zeros(512)
    key_low8[_SetDim.ADDR_B0_LO + 8] = 1.0

    assert torch.dot(attn.W_q[row], query) == 0.0
    assert torch.dot(attn.W_k[row], key_low8) > 0.0


def test_layer15_pop_low8_row_is_neutral_for_nonpop_low0_preserve():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 34
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.HAS_SE] = 1.0
    query[_SetDim.ADDR_B0_LO + 0] = 1.0

    key_low8 = torch.zeros(512)
    key_low8[_SetDim.ADDR_B0_LO + 8] = 1.0

    assert torch.dot(attn.W_q[row], query) == 0.0
    assert torch.dot(attn.W_k[row], key_low8) > 0.0


def test_layer15_pop_stack0_marker_low8_query_sinks_empty_stack_slot():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row = 35
    query = torch.zeros(512)
    query[_SetDim.CONST] = 1.0
    query[_SetDim.MARK_STACK0] = 1.0
    query[_SetDim.HAS_SE] = 1.0
    query[_SetDim.CMP + 3] = 4.0
    query[_SetDim.ADDR_B0_LO + 8] = 1.0
    query[_SetDim.ADDR_B0_HI + 15] = 1.0

    non_stack0_query = query.clone()
    non_stack0_query[_SetDim.MARK_STACK0] = 0.0

    const_key = torch.zeros(512)
    const_key[_SetDim.CONST] = 1.0

    q_value = torch.dot(attn.W_q[row], query)
    assert q_value > 0.0
    assert torch.dot(attn.W_q[row], non_stack0_query) > 0.0
    assert q_value * torch.dot(attn.W_k[row], const_key) < -1_000_000.0


def test_layer15_local_slot_hi_nibble_mismatch_beats_adjacent_frame_slot():
    attn = _StubAttn()

    _suppress_l15_lookup_during_current_store_generation(attn, _SetDim, 64)

    row_base = 0

    def bit_score(nibble_q: int, nibble_k: int, *, base_dim: int) -> float:
        score = 0.0
        for bit in range(4):
            row = row_base + 4 + (4 if base_dim == _SetDim.ADDR_B0_HI else 0) + bit
            score += (
                float(attn.W_q[row, base_dim + nibble_q])
                * float(attn.W_k[row, base_dim + nibble_k])
            )
        return score

    # Adjacent call-frame slots such as return-pc 0xffe8 and arg 0xfff8
    # share the low nibble. The high-nibble exact match must dominate recency.
    exact_hi = bit_score(15, 15, base_dim=_SetDim.ADDR_B0_HI)
    adjacent_hi = bit_score(15, 14, base_dim=_SetDim.ADDR_B0_HI)

    assert exact_hi - adjacent_hi == 20000.0


def test_layer15_si_mem_addr0_head_reads_clean_stack0_byte0():
    spec = _layer15_si_mem_addr0_from_stack0_spec(_SetDim)

    assert spec.head_idx == 13

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}
    k_terms = {(term.slot, term.dim, term.weight) for term in spec.k}
    v_terms = {(term.slot, term.dim, term.weight) for term in spec.v}
    o_terms = {(term.out_dim, term.slot, term.weight) for term in spec.o}

    assert (0, _SetDim.MARK_MEM, 100.0) in q_terms
    assert (0, _SetDim.MEM_ADDR_SRC, 50.0) in q_terms
    assert (0, _SetDim.CONST, -140.0) in q_terms
    assert (0, _SetDim.STACK0_BYTE0, 100.0) in k_terms
    assert (0, _SetDim.MEM_STORE, -400.0) in k_terms

    assert (1 + 8, _SetDim.CLEAN_EMBED_LO + 8, 1.0) in v_terms
    assert (17 + 14, _SetDim.CLEAN_EMBED_HI + 14, 1.0) in v_terms
    assert not any(
        dim == _SetDim.OUTPUT_LO + 8
        for _, dim, _ in v_terms
    )
    assert (_SetDim.OUTPUT_LO + 15, 0, -10.0) in o_terms
    assert (_SetDim.OUTPUT_LO + 8, 1 + 8, 20.0) in o_terms


@pytest.mark.slow
def test_layer15_teacher_forced_test450_stack0_byte0_top_store_values():
    from c4_release.neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from c4_release.src.compiler import compile_c
    from c4_release.tests.test_1096_neural_declarative_diagnostic import (
        _STEP_SLOT_NAMES,
        _build_symbolic_expected_execution,
        _head_logits,
    )
    from c4_release.tests.test_suite_1000 import generate_test_programs

    source, _expected, _description = generate_test_programs()[450]
    bytecode, data = compile_c(source)
    expected = _build_symbolic_expected_execution(bytecode, data)
    stack0_byte0_offset = _STEP_SLOT_NAMES.index("STACK0_byte0")
    probes = {
        23: 0xE0,
        32: 0x01,
        48: 0x03,
    }

    target_indexes = {
        step: expected.prefix_len + step * Token.STEP_TOKENS + stack0_byte0_offset
        for step in probes
    }
    for step, target_index in target_indexes.items():
        assert expected.context[target_index] == probes[step]

    # The first two probes lock the teacher-forced oracle positions.  The
    # step-48 probe is the focused L15 stale top-store regression.
    runner = BatchedPureNeuralRunner(max_seq_len=4096)
    model = runner.model
    device = next(model.parameters()).device
    max_target_index = target_indexes[48]
    token_ids = torch.tensor(
        [expected.context[:max_target_index]],
        dtype=torch.long,
        device=device,
    )

    model.embed.set_mem_history_end(0)
    with torch.no_grad():
        x = model.embed(token_ids)
        for block_index, block in enumerate(model.blocks):
            x = block(x)
            if block_index == 24:
                break

    logit_pos = target_indexes[48] - 1
    logits = _head_logits(model, x[0, logit_pos])
    assert int(torch.argmax(logits).item()) == probes[48]
