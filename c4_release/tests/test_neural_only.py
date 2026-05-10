import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.embedding import Opcode


def _run(runner, bytecode, max_steps=20):
    return runner.run(bytecode, b'', max_steps=max_steps)


class TestNeuralOnlyALU:

    def test_add_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 30

    def test_sub_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 50), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 30

    def test_and_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0x0F), Opcode.AND,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 0x0F

    def test_or_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 0xF0), Opcode.PSH,
            (Opcode.IMM, 0x0F), Opcode.OR,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 0xFF

    def test_xor_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0x0F), Opcode.XOR,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 0xF0

    def test_mul_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 6), Opcode.PSH,
            (Opcode.IMM, 7), Opcode.MUL,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 42

    def test_add_carry_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 200), Opcode.PSH,
            (Opcode.IMM, 100), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 300

    def test_shl_neural(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 1), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHL,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 256


class TestNeuralOnlyADD:

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0),
        (3, 4, 7),
        (10, 20, 30),
        (100, 200, 300),
        (200, 100, 300),
        (255, 1, 256),
        (1, 255, 256),
        (250, 250, 500),
        (127, 128, 255),
    ])
    def test_add_variants(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected


class TestNeuralOnlySUB:

    @pytest.mark.parametrize("a,b,expected", [
        (50, 20, 30),
        (100, 50, 50),
        (200, 100, 100),
        (42, 42, 0),
        (100, 0, 100),
    ])
    def test_sub_variants(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected

    @pytest.mark.parametrize("a,b,expected", [
        (256, 1, 255),  # Cross-byte borrow: 0x100 - 0x01 = 0xFF
    ])
    def test_sub_borrow(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected

    @pytest.mark.xfail(reason="Multi-byte SUB borrow propagation not yet implemented")
    @pytest.mark.parametrize("a,b,expected", [
        (300, 300, 0),  # Both operands > 255, requires multi-byte borrow
    ])
    def test_sub_multibyte_borrow(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected


class TestNeuralOnlyBitwise:

    @pytest.mark.parametrize("a,b,expected", [
        (0xFF, 0x0F, 0x0F),
        (0xAA, 0x55, 0x00),
        (0xF0, 0x0F, 0x00),
        (0xFF, 0xFF, 0xFF),
        (0x00, 0xFF, 0x00),
    ])
    def test_and_variants(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.AND,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected

    @pytest.mark.parametrize("a,b,expected", [
        (0xF0, 0x0F, 0xFF),
        (0xAA, 0x55, 0xFF),
        (0x00, 0x00, 0x00),
        (0xF0, 0xF0, 0xF0),
    ])
    def test_or_variants(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.OR,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected

    @pytest.mark.parametrize("a,b,expected", [
        (0xFF, 0x0F, 0xF0),
        (0xAA, 0x55, 0xFF),
        (0xFF, 0xFF, 0x00),
        (0xF0, 0x0F, 0xFF),
        (0x00, 0x00, 0x00),
    ])
    def test_xor_variants(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.XOR,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected


class TestNeuralOnlyMUL:

    @pytest.mark.parametrize("a,b,expected", [
        (6, 7, 42),
        (0, 100, 0),
        (1, 255, 255),
        (10, 10, 100),
        (3, 0, 0),
        (1, 1, 1),
        (5, 5, 25),
    ])
    def test_mul_variants(self, neural_only_runner, make_bytecode, a, b, expected):
        bc = make_bytecode([
            (Opcode.IMM, a), Opcode.PSH,
            (Opcode.IMM, b), Opcode.MUL,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected


class TestNeuralOnlySHL:

    @pytest.mark.parametrize("val,shift,expected", [
        (1, 8, 256),
    ])
    def test_shl_cross_byte(self, neural_only_runner, make_bytecode, val, shift, expected):
        bc = make_bytecode([
            (Opcode.IMM, val), Opcode.PSH,
            (Opcode.IMM, shift), Opcode.SHL,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected

    @pytest.mark.parametrize("val,shift,expected", [
        (1, 0, 1),
        (1, 1, 2),
        (1, 4, 16),
        (1, 7, 128),
        (3, 4, 48),
        (15, 4, 240),
    ])
    def test_shl_within_byte(self, neural_only_runner, make_bytecode, val, shift, expected):
        bc = make_bytecode([
            (Opcode.IMM, val), Opcode.PSH,
            (Opcode.IMM, shift), Opcode.SHL,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected


class TestNeuralOnlySHR:

    @pytest.mark.parametrize("val,shift,expected", [
        (128, 7, 1),    # Within byte 0
        (16, 4, 1),     # Within byte 0
        (255, 0, 255),  # No shift
        (255, 4, 15),   # Within byte 0
        (0, 8, 0),      # Zero input
    ])
    def test_shr_within_byte(self, neural_only_runner, make_bytecode, val, shift, expected):
        bc = make_bytecode([
            (Opcode.IMM, val), Opcode.PSH,
            (Opcode.IMM, shift), Opcode.SHR,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected

    @pytest.mark.xfail(reason="SHR cross-byte not yet implemented")
    @pytest.mark.parametrize("val,shift,expected", [
        (256, 8, 1),    # Cross-byte: byte 1 → byte 0
    ])
    def test_shr_cross_byte(self, neural_only_runner, make_bytecode, val, shift, expected):
        bc = make_bytecode([
            (Opcode.IMM, val), Opcode.PSH,
            (Opcode.IMM, shift), Opcode.SHR,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == expected


class TestNeuralOnlyMultiStep:

    def test_two_adds(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 5), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 35

    def test_add_then_sub(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 100), Opcode.PSH,
            (Opcode.IMM, 50), Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 30), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 120

    def test_mul_then_sub(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 7), Opcode.PSH,
            (Opcode.IMM, 6), Opcode.MUL,
            Opcode.PSH,
            (Opcode.IMM, 2), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc)
        assert result == 40

    @pytest.mark.xfail(reason="SHL cross-byte (shift >= 8) not yet working")
    def test_shl_then_add(self, neural_only_runner, make_bytecode):
        bc = make_bytecode([
            (Opcode.IMM, 1), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHL,
            Opcode.PSH,
            (Opcode.IMM, 1), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = _run(neural_only_runner, bc, max_steps=30)
        assert result == 257
