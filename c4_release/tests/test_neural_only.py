import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.embedding import Opcode


class TestNeuralOnlyALU:
    """Test ALU ops work with pure neural computation (no Python handler overrides).

    Uses neural_only_runner which removes all _func_call_handlers and _syscall_handlers,
    so the model's register predictions are used directly.
    """

    def _run_neural(self, runner, bytecode, max_steps=20):
        return runner.run(bytecode, b'', max_steps=max_steps)

    def test_add_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 30

    def test_sub_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 50), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 30

    def test_and_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0x0F), Opcode.AND,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 0x0F

    def test_or_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 0xF0), Opcode.PSH,
            (Opcode.IMM, 0x0F), Opcode.OR,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 0xFF

    def test_xor_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0x0F), Opcode.XOR,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 0xF0

    def test_mul_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 6), Opcode.PSH,
            (Opcode.IMM, 7), Opcode.MUL,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 42

    def test_add_carry_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 200), Opcode.PSH,
            (Opcode.IMM, 100), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 300

    def test_shl_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 1), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHL,
            Opcode.EXIT
        ])
        _, result = self._run_neural(neural_only_runner, bc)
        assert result == 256
