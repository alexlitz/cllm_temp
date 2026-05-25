"""Parity tests for L15 FFN bands migrated to CompilerIR rules."""

import pytest
import torch

from c4_release.neural_vm.unified_compiler.ops.l15_ops import (
    _layer15_si_mem_addr0_from_stack0_spec,
    _suppress_l15_lookup_during_current_store_generation,
    lower_l15_nibble_copy_ir,
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
    assert attn.W_q[62, _SetDim.H1 + 2] == 100000.0
    assert attn.W_q[62, _SetDim.IS_BYTE] == 0.0
    assert attn.W_k[62, _SetDim.CONST] == -20.0
    for head in range(1, 4):
        row = head * 64
        assert attn.W_q[row, _SetDim.OP_JSR] == 0.0
        assert attn.W_q[row, _SetDim.OP_ENT] == 0.0
        assert attn.W_q[row, _SetDim.OP_LEA] == 0.0
        assert attn.W_q[row, _SetDim.OP_IMM] == 0.0
        blocker_row = head * 64 + 62
        assert attn.W_q[blocker_row, _SetDim.H1 + 2] == 100000.0
        assert attn.W_q[blocker_row, _SetDim.IS_BYTE] == 0.0
        assert attn.W_k[blocker_row, _SetDim.CONST] == -20.0


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

    for row in (59, 60, 61):
        assert attn.W_q[row, _SetDim.CONST] == 0.0
        assert attn.W_q[row, _SetDim.MARK_STACK0] == 0.0
        assert attn.W_q[row, _SetDim.MEM_STORE] == 0.0
        assert attn.W_k[row, _SetDim.CONST] == 0.0

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

    for row in (59, 60, 61):
        assert torch.count_nonzero(attn.W_q[row, :]) == 0
        assert torch.count_nonzero(attn.W_k[row, :]) == 0
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
        assert attn.W_q[row, _SetDim.EMBED_LO + 8] == 10000.0
        assert attn.W_q[row, _SetDim.EMBED_HI + 14] == 10000.0
        assert attn.W_q[row, _SetDim.ADDR_B0_LO + 0] == 10000.0
        assert attn.W_q[row, _SetDim.ADDR_B0_HI + 14] == 10000.0
        assert attn.W_k[row, _SetDim.CONST] == -20.0
        assert attn.W_k[row, _SetDim.MEM_VAL_B1] == 0.0
        assert torch.count_nonzero(attn.W_v[row, :]) == 0
        assert torch.count_nonzero(attn.W_o[:, row]) == 0

    for head in range(1, 4):
        next_head_row0 = head * 64
        assert attn.W_q[next_head_row0, _SetDim.MARK_STACK0] == 0.0
        assert attn.W_q[next_head_row0, _SetDim.IS_BYTE] == 0.0
        assert attn.W_k[next_head_row0, _SetDim.CONST] == 0.0


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
