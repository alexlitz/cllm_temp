"""Phase 1 gate: PC + AX coherence in pure-neural mode.

These tests run AutoregressiveVMRunner(pure_neural=True) with NO Python
overrides at all. The neural network must compute PC, AX, SP, BP entirely
from its forward pass.

Currently most of these fail. The first failing test pinpoints the smallest
broken behavior. Phase 1 closes when all tests in this file pass.
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


class TestPureNeuralSingleInstruction:
    """The smallest possible programs — must pass before anything else."""

    def test_imm_then_exit(self, pure_neural_runner):
        assert _run(pure_neural_runner, [(Opcode.IMM, 5), Opcode.EXIT]) == 5

    def test_imm_zero_then_exit(self, pure_neural_runner):
        assert _run(pure_neural_runner, [(Opcode.IMM, 0), Opcode.EXIT]) == 0

    @pytest.mark.parametrize("val", [1, 7, 42, 100, 200, 255])
    def test_imm_byte_values(self, pure_neural_runner, val):
        assert _run(pure_neural_runner, [(Opcode.IMM, val), Opcode.EXIT]) == val


class TestPureNeuralTwoInstructions:
    """Two-step programs — exercises PC+8 increment + AX carry-forward."""

    def test_two_imms(self, pure_neural_runner):
        # Step 1: AX = 5, PC = 10
        # Step 2: AX = 7, PC = 18
        # AX result returned by EXIT = 7
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7

    def test_imm_then_nop_then_exit(self, pure_neural_runner):
        # AX should persist across NOP
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 42),
            Opcode.NOP,
            Opcode.EXIT,
        ]) == 42

    def test_nop_then_imm(self, pure_neural_runner):
        # PC must increment past NOP
        assert _run(pure_neural_runner, [
            Opcode.NOP,
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ]) == 42


class TestPureNeuralThreeOrMoreInstructions:
    """Sustained PC arithmetic across multiple steps."""

    def test_three_imms(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            (Opcode.IMM, 7),
            (Opcode.IMM, 9),
            Opcode.EXIT,
        ]) == 9

    def test_five_imms(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 1),
            (Opcode.IMM, 2),
            (Opcode.IMM, 3),
            (Opcode.IMM, 4),
            (Opcode.IMM, 5),
            Opcode.EXIT,
        ]) == 5
