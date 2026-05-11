"""Phase 3 gate: Multi-byte AX coherence + ALU completeness.

Builds on Phase 1 (PC + AX coherence up through 5 IMMs) and Phase 2
(SP + PSH + binary ALU on byte-sized operands). These tests stress:
  - Sustained IMM dispatch beyond 5 instructions (multi-IMM + EXIT)
  - Binary ALU producing results that overflow byte 0 into byte 1+
  - Carry propagation across the 4 bytes of AX (ADD overflow)
  - Multi-byte products (MUL where result > 255 or > 65535)

Each step the network must coherently track all 4 bytes of AX, since
IMM is 8-bit (encoded as `(imm << 8) | op`, two nibbles) and operations
that exceed 255 must populate higher bytes correctly.

Phase 3 closes when all tests in this file pass.
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


class TestPureNeuralLongIMMSequences:
    """Multi-IMM dispatch beyond Phase 1's 5-IMM ceiling."""

    def test_six_imms(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 1),
            (Opcode.IMM, 2),
            (Opcode.IMM, 3),
            (Opcode.IMM, 4),
            (Opcode.IMM, 5),
            (Opcode.IMM, 6),
            Opcode.EXIT,
        ]) == 6

    def test_ten_imms(self, pure_neural_runner):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 1),
            (Opcode.IMM, 2),
            (Opcode.IMM, 3),
            (Opcode.IMM, 4),
            (Opcode.IMM, 6),
            (Opcode.IMM, 7),
            (Opcode.IMM, 8),
            (Opcode.IMM, 9),
            (Opcode.IMM, 10),
            (Opcode.IMM, 5),
            Opcode.EXIT,
        ], max_steps=40) == 5


class TestPureNeuralAddOverflow:
    """ADD producing results > 255 — requires byte 1 carry-out."""

    @pytest.mark.xfail(
        reason="_set_layer9_alu carry handling + post_op CarryPropagationPostOp "
               "are not yet wired for multi-byte ADD results in pure_neural mode.",
        strict=False,
    )
    def test_add_overflow_300(self, pure_neural_runner):
        # 200 + 100 = 300 → byte 0 = 44 (300 & 0xFF), byte 1 = 1
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 200),
            Opcode.PSH,
            (Opcode.IMM, 100),
            Opcode.ADD,
            Opcode.EXIT,
        ]) == 300

    @pytest.mark.xfail(
        reason="_set_layer9_alu carry handling + post_op CarryPropagationPostOp "
               "are not yet wired for multi-byte ADD results in pure_neural mode.",
        strict=False,
    )
    def test_add_overflow_510(self, pure_neural_runner):
        # 255 + 255 = 510 → byte 0 = 254, byte 1 = 1
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 255),
            Opcode.PSH,
            (Opcode.IMM, 255),
            Opcode.ADD,
            Opcode.EXIT,
        ]) == 510


class TestPureNeuralMulHighByte:
    """MUL producing 16-bit results — requires bytes 1+ correctness."""

    @pytest.mark.xfail(
        reason="_set_layer11_mul_partial + _set_layer12_mul_combine are not "
               "yet wired to write byte 1+ in pure_neural mode.",
        strict=False,
    )
    def test_mul_to_high_byte_300(self, pure_neural_runner):
        # 30 * 10 = 300 → byte 0 = 44, byte 1 = 1
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 30),
            Opcode.PSH,
            (Opcode.IMM, 10),
            Opcode.MUL,
            Opcode.EXIT,
        ]) == 300

    @pytest.mark.xfail(
        reason="_set_layer11_mul_partial + _set_layer12_mul_combine are not "
               "yet wired for 16-bit MUL products in pure_neural mode.",
        strict=False,
    )
    def test_mul_two_byte_result_10000(self, pure_neural_runner):
        # 100 * 100 = 10000 → byte 0 = 16, byte 1 = 39
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 100),
            Opcode.PSH,
            (Opcode.IMM, 100),
            Opcode.MUL,
            Opcode.EXIT,
        ]) == 10000

    @pytest.mark.xfail(
        reason="_set_layer11_mul_partial + _set_layer12_mul_combine are not "
               "yet wired for 16-bit MUL products in pure_neural mode.",
        strict=False,
    )
    def test_mul_max_byte_pair(self, pure_neural_runner):
        # 255 * 255 = 65025 → spans bytes 0 and 1 fully
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 255),
            Opcode.PSH,
            (Opcode.IMM, 255),
            Opcode.MUL,
            Opcode.EXIT,
        ]) == 65025


class TestPureNeuralSubBorrow:
    """SUB borrow propagation across bytes.

    IMM is 8-bit only (encoded as 2 nibbles in `(imm << 8) | op`), so
    minuends > 255 must be constructed via ADD. These tests build a
    multi-byte value first, then subtract from it.
    """

    @pytest.mark.xfail(
        reason="Multi-byte SUB borrow propagation depends on multi-byte ADD "
               "(byte 1 carry-out) landing first; until layers 9 + 11 + 12 "
               "wire the high-byte path the SUB borrow has nothing to read.",
        strict=False,
    )
    def test_sub_borrow_into_byte1(self, pure_neural_runner):
        # Minuend 300 must be constructed via ADD since IMM is 8-bit;
        # SUB then has to read a multi-byte stack value coherently.
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 200),
            Opcode.PSH,
            (Opcode.IMM, 100),
            Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 50),
            Opcode.SUB,
            Opcode.EXIT,
        ]) == 250

    @pytest.mark.xfail(
        reason="Multi-byte SUB borrow propagation depends on multi-byte ADD/MUL "
               "landing first.",
        strict=False,
    )
    def test_sub_borrow_zero_byte0(self, pure_neural_runner):
        # 256 - 1 = 255 forces byte 0 to borrow from byte 1.
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 200),
            Opcode.PSH,
            (Opcode.IMM, 56),
            Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 1),
            Opcode.SUB,
            Opcode.EXIT,
        ]) == 255
