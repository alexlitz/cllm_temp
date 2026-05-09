"""Phase 2 gate: SP arithmetic + PSH + binary ALU.

Builds on Phase 1 (PC + AX coherence). These tests exercise:
  - SP increment/decrement on PSH (sp -= 8) and binary-pop ops (sp += 8)
  - PSH writing AX to mem[SP-8] via the model's MEM token sequence
  - Binary ALU ops (ADD/SUB/AND/OR/XOR/etc.) reading prev STACK0 from MEM

Phase 2 closes when all tests in this file pass.

Currently most fail. Each failure points at a specific gap in the
neural architecture.
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


class TestPureNeuralPSH:
    """PSH-then-EXIT — does the SP decrement work?"""

    def test_imm_psh_exit(self, pure_neural_runner):
        # AX should remain the IMM value (PSH preserves AX)
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            Opcode.PSH,
            Opcode.EXIT,
        ]) == 5


class TestPureNeuralBinaryOps:
    """Binary ALU ops on small operands (< 256)."""

    @pytest.mark.parametrize("a,b,expected", [
        (3, 4, 7),
        (10, 20, 30),
        (5, 0, 5),
        (0, 5, 5),
        (100, 50, 150),
    ])
    def test_add_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.ADD,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (50, 20, 30),
        (100, 50, 50),
        (200, 100, 100),
        (42, 42, 0),
    ])
    def test_sub_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.SUB,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (0xFF, 0x0F, 0x0F),
        (0xF0, 0x0F, 0x00),
        (0xFF, 0xFF, 0xFF),
    ])
    def test_and_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.AND,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (0xF0, 0x0F, 0xFF),
        (0xAA, 0x55, 0xFF),
        (0x00, 0x00, 0x00),
    ])
    def test_or_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.OR,
            Opcode.EXIT,
        ]) == expected
