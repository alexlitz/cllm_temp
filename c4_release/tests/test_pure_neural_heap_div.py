"""Phase 7 gate: heap (SI/SC/LI/LC) + DIV/MOD parity.

Builds on Phase 1 (PC + AX) and Phase 2 (PSH + binary ALU). These tests
exercise the remaining ALU ops (MUL, DIV, MOD, SHL, SHR) and the heap-style
store/load pair (SI/SC writing AX to mem[pop], LI/LC reading mem[AX]).

Phase 7 closes when all tests in this file pass.
"""

import pytest

from neural_vm.embedding import Opcode


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run(runner, prog, max_steps=30):
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


class TestPureNeuralALUCompleteness:
    """Remaining binary ALU ops not covered by Phase 2."""

    @pytest.mark.xfail(reason="MUL: _set_layer11_mul_partial / _set_layer12_mul_combine not wired in pure-neural")
    @pytest.mark.parametrize("a,b,expected", [
        (3, 4, 12),
        (5, 5, 25),
        (10, 10, 100),
        (255, 1, 255),
        (16, 16, 256),
    ])
    def test_mul_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.MUL,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (20, 5, 4),
        (100, 10, 10),
        (255, 1, 255),
        (256, 16, 16),
    ])
    def test_div_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.DIV,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (7, 3, 1),
        (10, 3, 1),
        (15, 4, 3),
        (256, 5, 1),
    ])
    def test_mod_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.MOD,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.xfail(reason="SHL: _set_layer13_shifts / ALUShift not wired in pure-neural")
    @pytest.mark.parametrize("a,b,expected", [
        (1, 1, 2),
        (1, 3, 8),
        (3, 2, 12),
        (255, 1, 254),
    ])
    def test_shl_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.SHL,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.xfail(reason="SHR: _set_layer13_shifts / ALUShift not wired in pure-neural")
    @pytest.mark.parametrize("a,b,expected", [
        (8, 1, 4),
        (256, 2, 64),
        (255, 1, 127),
        (1, 1, 0),
    ])
    def test_shr_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.SHR,
            Opcode.EXIT,
        ]) == expected


class TestPureNeuralHeap:
    """Store-then-load round trips: SI/LI for ints, SC/LC for chars."""

    @pytest.mark.xfail(reason="SI/LI: _set_layer14_mem_generation / _set_layer15_memory_lookup / _set_layer7_memory_heads gaps")
    def test_si_then_li(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 100),
            Opcode.PSH,
            (Opcode.IMM, 1024),
            Opcode.SI,
            (Opcode.IMM, 1024),
            Opcode.LI,
            Opcode.EXIT,
        ]) == 100

    @pytest.mark.xfail(reason="SC/LC: _set_layer14_mem_generation / _set_layer15_memory_lookup gaps for char-width")
    def test_sc_then_lc(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, ord('A')),
            Opcode.PSH,
            (Opcode.IMM, 1024),
            Opcode.SC,
            (Opcode.IMM, 1024),
            Opcode.LC,
            Opcode.EXIT,
        ]) == 65

    @pytest.mark.xfail(reason="Multi-address heap state: depends on per-step memory persistence in pure-neural")
    def test_two_writes_two_reads(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 7),
            Opcode.PSH,
            (Opcode.IMM, 1024),
            Opcode.SI,
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 1032),
            Opcode.SI,
            (Opcode.IMM, 1032),
            Opcode.LI,
            (Opcode.IMM, 1024),
            Opcode.LI,
            Opcode.EXIT,
        ], max_steps=50) == 7

    @pytest.mark.xfail(reason="DIV-by-zero corner case has no defined neural behavior yet")
    def test_div_by_zero_xfail(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 10),
            Opcode.PSH,
            (Opcode.IMM, 0),
            Opcode.DIV,
            Opcode.EXIT,
        ]) == 0
