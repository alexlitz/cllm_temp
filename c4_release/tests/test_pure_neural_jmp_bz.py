"""Phase 4 gate: control flow (JMP/BZ/BNZ) in pure-neural mode.

Builds on Phase 1 (PC + AX) and Phase 2 (PSH + binary ALU). Exercises the
neural network's ability to redirect PC for unconditional and conditional
branches with NO Python overrides.

Phase 4 closes when all tests in this file pass.
"""

import pytest

from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
from neural_vm.embedding import Opcode


# Branch opcodes whose immediate is an instruction *index*. The neural model's
# L6 PC override sets PC = imm directly (no idx*INSTR_WIDTH multiply), and the
# C4 compiler emits imm = idx*INSTR_WIDTH + PC_OFFSET. So the tests must encode
# branch targets as PC values, not raw indices.
BRANCH_OPS = {Opcode.JMP, Opcode.BZ, Opcode.BNZ, Opcode.JSR}


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            if op in BRANCH_OPS:
                imm = imm * INSTR_WIDTH + PC_OFFSET
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run(runner, prog, max_steps=8):
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


class TestPureNeuralJMP:
    """Unconditional jumps."""

    def test_jmp_forward(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.JMP, 2),
            Opcode.NOP,
            (Opcode.IMM, 9),
            Opcode.EXIT,
        ]) == 9

    def test_jmp_from_step_2(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            (Opcode.JMP, 4),
            (Opcode.IMM, 99),
            Opcode.EXIT,
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7

    def test_jmp_backward(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            (Opcode.JMP, 4),
            (Opcode.IMM, 7),
            Opcode.EXIT,
            (Opcode.JMP, 2),
        ]) == 7


class TestPureNeuralBZ:
    """Branch-if-zero conditional."""

    def test_bz_taken(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 0),
            (Opcode.BZ, 4),
            (Opcode.IMM, 99),
            Opcode.EXIT,
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7

    @pytest.mark.parametrize("imm", [
        1,
        5,
        255,
    ])
    def test_bz_not_taken(self, pure_neural_runner, imm):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, imm),
            (Opcode.BZ, 4),
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7


class TestPureNeuralBNZ:
    """Branch-if-not-zero conditional."""

    @pytest.mark.parametrize("imm", [1, 5, 255])
    def test_bnz_taken(self, pure_neural_runner, imm):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, imm),
            (Opcode.BNZ, 4),
            (Opcode.IMM, 99),
            Opcode.EXIT,
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7

    def test_bnz_not_taken(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 0),
            (Opcode.BNZ, 4),
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7


class TestPureNeuralLoop:
    """Tiny loop combining PSH/SUB/BNZ to a backward target."""

    def test_countdown_loop(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 3),
            Opcode.PSH,
            (Opcode.IMM, 1),
            Opcode.SUB,
            (Opcode.BNZ, 1),
            Opcode.EXIT,
        ], max_steps=15) == 0
