"""Phase 4 gate: control flow (JMP/BZ/BNZ) in pure-neural mode.

Builds on Phase 1 (PC + AX) and Phase 2 (PSH + binary ALU). Exercises the
neural network's ability to redirect PC for unconditional and conditional
branches with NO Python overrides.

Phase 4 closes when all tests in this file pass. Currently all are xfail —
pure_neural mode does not yet redirect PC for any branch op.
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


def _run(runner, prog, max_steps=8):
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


class TestPureNeuralJMP:
    """Unconditional jumps."""

    @pytest.mark.xfail(reason="pure_neural JMP not yet supported (PC never redirects, hangs at step 1)")
    def test_jmp_forward(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.JMP, 2),
            Opcode.NOP,
            (Opcode.IMM, 9),
            Opcode.EXIT,
        ]) == 9

    @pytest.mark.xfail(reason="pure_neural JMP not yet supported at step >=2 (L5 head 3 fixed-address lookup)")
    def test_jmp_from_step_2(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            (Opcode.JMP, 4),
            (Opcode.IMM, 99),
            Opcode.EXIT,
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7

    @pytest.mark.xfail(reason="pure_neural backward JMP not yet supported (L5 fetch addr stale across steps)")
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

    @pytest.mark.xfail(reason="pure_neural BZ taken-path not yet supported (PC never redirects at step 1)")
    def test_bz_taken(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 0),
            (Opcode.BZ, 4),
            (Opcode.IMM, 99),
            Opcode.EXIT,
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7

    @pytest.mark.xfail(reason="pure_neural BZ fall-through not yet supported (hangs at BZ at step 1)")
    @pytest.mark.parametrize("imm", [1, 5, 255])
    def test_bz_not_taken(self, pure_neural_runner, imm):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, imm),
            (Opcode.BZ, 4),
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7


class TestPureNeuralBNZ:
    """Branch-if-not-zero conditional."""

    @pytest.mark.xfail(reason="pure_neural BNZ taken-path not yet supported (PC never redirects at step 1)")
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

    @pytest.mark.xfail(reason="pure_neural BNZ fall-through not yet supported (hangs at BNZ at step 1)")
    def test_bnz_not_taken(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 0),
            (Opcode.BNZ, 4),
            (Opcode.IMM, 7),
            Opcode.EXIT,
        ]) == 7


class TestPureNeuralLoop:
    """Tiny loop combining PSH/SUB/BNZ to a backward target."""

    @pytest.mark.xfail(reason="pure_neural backward branch in loop not yet supported (BNZ never redirects PC)")
    def test_countdown_loop(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 3),
            Opcode.PSH,
            (Opcode.IMM, 1),
            Opcode.SUB,
            (Opcode.BNZ, 1),
            Opcode.EXIT,
        ], max_steps=15) == 0
