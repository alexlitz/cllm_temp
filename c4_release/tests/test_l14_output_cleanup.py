"""Focused L14 cleanup weight checks (declarative wide-ALU high-byte relay).

The legacy imperative ``_set_layer14_*`` cleanup oracles that used to live
here have been retired: every L14 cleanup op is now guarded on the
declarative path by ``tests/test_l14_per_op.py`` (per-op drift +
fires-during-bake + the mem-generation V->O routing forward invariant),
so the imperative byte-identity fixtures were redundant. The remaining
tests below cover the ``_layer14_alu_high_byte_relay_spec`` attention op
directly from its declarative spec (no imperative helper).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ops.l14_ops import (  # noqa: E402
    _layer14_alu_high_byte_relay_spec,
)
from neural_vm.vm_step import _SetDim  # noqa: E402


def _projection(terms, row: torch.Tensor) -> torch.Tensor:
    slots = max(term.slot for term in terms) + 1
    projected = torch.zeros(slots)
    for term in terms:
        projected[term.slot] += row[term.dim] * term.weight
    return projected


def _attention_score(spec, query_row: torch.Tensor, key_row: torch.Tensor) -> float:
    q = _projection(spec.q, query_row)
    k = _projection(spec.k, key_row)
    return float(q @ k)


def _value_output(spec, key_row: torch.Tensor) -> torch.Tensor:
    value = _projection(spec.v, key_row)
    output = torch.zeros_like(key_row)
    for term in spec.o:
        output[term.out_dim] += value[term.slot] * term.weight
    return output


def _ax_byte0_query_row() -> torch.Tensor:
    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.IS_BYTE] = 1.0
    row[_SetDim.H1 + 1] = 1.0
    row[_SetDim.BYTE_INDEX_0] = 1.0
    return row


def _wide_alu_source_row(op_dim: int) -> torch.Tensor:
    row = torch.zeros(512)
    row[_SetDim.CONST] = 1.0
    row[_SetDim.MARK_AX] = 1.0
    row[op_dim] = 1.0
    row[_SetDim.AX_FULL_LO + 2] = 0.25
    row[_SetDim.AX_FULL_HI + 0] = 0.75
    return row


def test_l15_wide_alu_relay_blocks_add_sub_rows():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}
    k_terms = {(term.slot, term.dim, term.weight) for term in spec.k}

    assert (35, _SetDim.TEMP + 8, 10000.0) in q_terms
    assert (35, _SetDim.TEMP + 9, 10000.0) in q_terms
    assert (35, _SetDim.CONST, -20.0) in k_terms
    assert (35, _SetDim.MARK_AX, -10000.0) in k_terms
    assert (35, _SetDim.OP_MUL, 2000.0) in k_terms
    assert (35, _SetDim.OP_SHL, 2000.0) in k_terms


def test_l15_wide_alu_relay_requires_mul_or_shl_source():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}
    k_terms = {(term.slot, term.dim, term.weight) for term in spec.k}

    assert (36, _SetDim.IS_BYTE, 1000.0) in q_terms
    assert (36, _SetDim.H1 + 1, 1000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_0, 1000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_1, -3000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_2, -3000.0) in q_terms
    assert (36, _SetDim.BYTE_INDEX_3, -3000.0) in q_terms
    assert (36, _SetDim.MARK_AX, -1000.0) in q_terms
    assert (36, _SetDim.CONST, -100.0) in k_terms
    assert (36, _SetDim.OP_MUL, 200.0) in k_terms
    assert (36, _SetDim.OP_SHL, 200.0) in k_terms


def test_l15_wide_alu_relay_has_non_ax_query_blockers():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    q_terms = {(term.slot, term.dim, term.weight) for term in spec.q}

    assert (36, _SetDim.MARK_PC, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 0, -6000.0) in q_terms
    assert (36, _SetDim.MARK_SP, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 2, -6000.0) in q_terms
    assert (36, _SetDim.H4 + 2, -6000.0) in q_terms
    assert (36, _SetDim.MARK_BP, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 3, -6000.0) in q_terms
    assert (36, _SetDim.H4 + 3, -6000.0) in q_terms
    assert (36, _SetDim.MARK_STACK0, -6000.0) in q_terms
    assert (36, _SetDim.STACK0_BYTE0, -6000.0) in q_terms
    assert (36, _SetDim.H1 + 4, -6000.0) in q_terms
    assert (36, _SetDim.H3 + 4, -6000.0) in q_terms
    assert (36, _SetDim.H4 + 4, -6000.0) in q_terms
    assert (36, _SetDim.MARK_MEM, -6000.0) in q_terms


def test_l15_wide_alu_relay_scores_intended_ax_byte0_mul_and_shl_sources():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    query = _ax_byte0_query_row()

    for op_dim in (_SetDim.OP_MUL, _SetDim.OP_SHL):
        source = _wide_alu_source_row(op_dim)

        assert _attention_score(spec, query, source) > 0.0

        output = _value_output(spec, source)
        assert output[_SetDim.OUTPUT_LO + 2] == 5.0
        assert output[_SetDim.OUTPUT_HI + 0] == 15.0


def test_l15_wide_alu_relay_allows_mul_sources_with_temp8_residue():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    query = _ax_byte0_query_row()
    query[_SetDim.TEMP + 8] = 0.305

    mul_source = _wide_alu_source_row(_SetDim.OP_MUL)
    mul_source[_SetDim.OP_MUL] = 5.0
    non_source = _wide_alu_source_row(_SetDim.OP_MUL)
    non_source[_SetDim.OP_MUL] = 0.0

    assert _attention_score(spec, query, mul_source) > 0.0
    assert _attention_score(spec, query, non_source) < 0.0


def test_l15_wide_alu_relay_blocks_stack0_and_bp_byte_positions():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    source = _wide_alu_source_row(_SetDim.OP_MUL)

    stack0_byte0 = _ax_byte0_query_row()
    stack0_byte0[_SetDim.H4 + 3] = 1.0
    stack0_byte0[_SetDim.STACK0_BYTE0] = 1.0

    bp_byte0 = _ax_byte0_query_row()
    bp_byte0[_SetDim.H1 + 3] = 1.0
    bp_byte0[_SetDim.H4 + 3] = 1.0

    assert _attention_score(spec, stack0_byte0, source) < 0.0
    assert _attention_score(spec, bp_byte0, source) < 0.0


def test_l15_wide_alu_relay_blocks_mem_byte_positions():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    source = _wide_alu_source_row(_SetDim.OP_MUL)

    mem_byte0 = _ax_byte0_query_row()
    mem_byte0[_SetDim.H1 + 4] = 1.0
    mem_byte0[_SetDim.H3 + 4] = 1.0
    mem_byte0[_SetDim.H4 + 4] = 1.0

    assert _attention_score(spec, mem_byte0, source) < 0.0


def test_l15_wide_alu_relay_blocks_pc_and_sp_byte_positions():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)
    source = _wide_alu_source_row(_SetDim.OP_MUL)

    pc_byte0 = _ax_byte0_query_row()
    pc_byte0[_SetDim.H1 + 0] = 1.0

    sp_byte0 = _ax_byte0_query_row()
    sp_byte0[_SetDim.H1 + 2] = 1.0
    sp_byte0[_SetDim.H4 + 2] = 1.0

    assert _attention_score(spec, pc_byte0, source) < 0.0
    assert _attention_score(spec, sp_byte0, source) < 0.0


def test_l15_wide_alu_relay_blocks_non_source_ax_rows():
    spec = _layer14_alu_high_byte_relay_spec(_SetDim)

    query = _ax_byte0_query_row()
    non_source = _wide_alu_source_row(_SetDim.OP_MUL)
    non_source[_SetDim.OP_MUL] = 0.0

    assert _attention_score(spec, query, non_source) < 0.0
