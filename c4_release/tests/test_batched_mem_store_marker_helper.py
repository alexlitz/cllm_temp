"""Unit tests for BatchedPureNeuralRunner's current-step MEM marker helpers.

These helpers (introduced in Recovery B2-H) ensure that ``mem_store_positions``
includes the current step's MEM marker BEFORE the model predicts MEM_addr0 for
store ops (SI/SC/PSH/ENT/JSR). Without this, L15/L16 lookup rules miss the
MEM_STORE=1 / MEM_ADDR_SRC=1 flags at the just-emitted MEM marker and the
model picks the wrong address (e.g. ENT push-BP at 0xfff0 collapsing to
0xffe0).

The helpers are pure functions of the element state and context — no GPU /
model required.
"""

from __future__ import annotations

from typing import List

from neural_vm.batched_pure_neural import (
    BatchedPureNeuralRunner,
    _ElementState,
    _MEM_STORE_OPS,
)
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
from neural_vm.vm_step import Token
from src.compiler import Op


def _make_state(bytecode: List[int], context: List[int], last_pc: int) -> _ElementState:
    state = _ElementState(bytecode=bytecode, context=context, prefix_len=0)
    state.last_pc = last_pc
    return state


_marker = BatchedPureNeuralRunner._current_step_store_mem_marker
_add = BatchedPureNeuralRunner._add_current_step_marker


def _call_marker(state: _ElementState, context: List[int]):
    # B3-beta extended the return type to a (position, is_si_sc) tuple so the
    # caller can decide whether to also set MEM_ADDR_SRC=1. These tests only
    # care about the marker position itself; the is_si_sc flag is covered by
    # the higher-level lane-routing tests.
    pos, _is_si_sc = _marker(state, context)
    return pos


def _call_add(
    positions: List[int],
    marker_abs,
    windowed: List[int],
    *,
    absolute_to_windowed: int = 0,
):
    return _add(
        positions,
        marker_abs,
        windowed,
        absolute_to_windowed=absolute_to_windowed,
    )


def _psh_at_pc(pc: int) -> List[int]:
    """Minimal bytecode where ``pc`` indexes a PSH instruction."""

    bc = [int(Op.NOP)] * (pc // INSTR_WIDTH + 1)
    bc[pc // INSTR_WIDTH] = int(Op.PSH)
    return bc


def test_marker_returns_position_for_store_op_with_pending_mem():
    pc = 4 * INSTR_WIDTH + PC_OFFSET  # PSH at index 4
    bc = _psh_at_pc(pc)
    assert (bc[pc // INSTR_WIDTH] & 0xFF) == int(Op.PSH)
    assert int(Op.PSH) in _MEM_STORE_OPS
    context = [Token.REG_PC, 0, 0, 0, 0, Token.MEM]
    state = _make_state(bc, context, pc)
    pos = _call_marker(state, context)
    assert pos == len(context) - 1, (
        f"Expected helper to return MEM position {len(context)-1}, got {pos}"
    )


def test_marker_returns_none_when_step_already_ended():
    pc = 4 * INSTR_WIDTH + PC_OFFSET
    bc = _psh_at_pc(pc)
    # MEM is followed by STEP_END — that step is "done", not "pending".
    context = [Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0, Token.STEP_END]
    state = _make_state(bc, context, pc)
    assert _call_marker(state, context) is None


def test_marker_returns_none_for_non_store_op():
    pc = 5 * INSTR_WIDTH + PC_OFFSET
    bc = [int(Op.NOP)] * (pc // INSTR_WIDTH + 1)
    bc[pc // INSTR_WIDTH] = int(Op.IMM)  # IMM is not in _MEM_STORE_OPS
    assert int(Op.IMM) not in _MEM_STORE_OPS
    context = [Token.MEM]
    state = _make_state(bc, context, pc)
    assert _call_marker(state, context) is None


def test_marker_returns_none_when_no_mem_in_recent_window():
    pc = 4 * INSTR_WIDTH + PC_OFFSET
    bc = _psh_at_pc(pc)
    # No MEM marker present yet (we're still emitting registers).
    context = [Token.REG_PC, 0, 0, 0, 0, Token.REG_AX, 0, 0, 0, 0]
    state = _make_state(bc, context, pc)
    assert _call_marker(state, context) is None


def test_add_current_step_marker_inserts_into_sorted_unique_list():
    windowed = [Token.STEP_END, Token.MEM, Token.REG_PC, Token.MEM]
    positions = [3]  # already includes the final MEM
    # Marker at position 1; should be merged + sorted.
    result = _call_add(positions, 1, windowed)
    assert result == [1, 3]


def test_add_current_step_marker_is_noop_when_marker_is_none():
    windowed = [Token.MEM]
    assert _call_add([0], None, windowed) == [0]


def test_add_current_step_marker_skips_non_mem_position():
    windowed = [Token.REG_PC, Token.REG_AX]
    # marker_abs points at REG_AX (1) which is not MEM — defensive guard.
    assert _call_add([], 1, windowed) == []


def test_add_current_step_marker_skips_out_of_range():
    windowed = [Token.MEM]
    assert _call_add([], 50, windowed) == []


def test_add_current_step_marker_remaps_via_offset():
    # Simulate a windowed context where the absolute marker at 7 maps to
    # windowed position 7 + 3 = 10 (mem_history insertion offset).
    windowed = [0] * 11
    windowed[10] = Token.MEM
    result = _call_add([], 7, windowed, absolute_to_windowed=3)
    assert result == [10]


def test_add_current_step_marker_dedups_existing_position():
    windowed = [Token.MEM, Token.MEM]
    assert _call_add([0, 1], 0, windowed) == [0, 1]
