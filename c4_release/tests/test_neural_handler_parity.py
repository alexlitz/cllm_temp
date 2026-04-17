#!/usr/bin/env python3
"""
Neural vs Handler Parity Tests

Tests that verify neural execution matches handler execution for each opcode.
Also documents which opcodes require handlers with xfail markers.

Neural Opcodes (100% neural, no handler needed):
- Arithmetic: ADD, SUB, MUL, DIV, MOD
- Bitwise: OR, XOR, AND
- Shift: SHL, SHR
- Control: NOP, EXIT

Handler Required Opcodes (neural path broken/incomplete):
- IMM: Needed for _last_ax tracking (PRTF format string)
- JSR: Neural PC increment broken step 2+
- JMP, BZ, BNZ: FETCH only works for first step
- ENT: Wrong addresses in MEM tokens
- LEV: SP/BP restoration fails (L16 FFN outputs 0)
- PSH: Shadow memory for PRTF format string
- LI, LC, SI, SC: Memory operations use handler fallback
- MALC, FREE, etc.: External syscalls

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


# =============================================================================
# Helper Functions
# =============================================================================

def make_bytecode(ops):
    """Convert operation list to bytecode list with proper encoding.

    Each op can be:
    - int: Raw opcode
    - tuple: (opcode, immediate_value) for ops with immediates
    """
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            # Encode as opcode + (imm << 8)
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


def run_with_handlers(bytecode, max_steps=100):
    """Run bytecode with default handlers."""
    runner = AutoregressiveVMRunner()
    _, exit_code = runner.run(bytecode, b'', max_steps=max_steps)
    return exit_code


def run_neural_only(bytecode, remove_handlers=None, max_steps=100):
    """Run bytecode with specified handlers removed (neural path only)."""
    runner = AutoregressiveVMRunner()

    if remove_handlers:
        for op in remove_handlers:
            if op in runner._func_call_handlers:
                del runner._func_call_handlers[op]
            if op in runner._syscall_handlers:
                del runner._syscall_handlers[op]

    _, exit_code = runner.run(bytecode, b'', max_steps=max_steps)
    return exit_code


# =============================================================================
# 100% Neural Opcodes - ADD, SUB, MUL, DIV, MOD
# =============================================================================

class TestNeuralArithmetic:
    """Test arithmetic opcodes work 100% neurally (no handlers needed)."""

    @pytest.fixture
    def neural_runner(self):
        """Runner with arithmetic handlers removed."""
        runner = AutoregressiveVMRunner()
        for op in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD]:
            if op in runner._func_call_handlers:
                del runner._func_call_handlers[op]
        return runner

    def test_add_basic(self, neural_runner):
        """ADD: 5 + 3 = 8"""
        # IMM 5, PSH, IMM 3, ADD, EXIT
        bytecode = make_bytecode([
            (Opcode.IMM, 5),   # AX = 5
            Opcode.PSH,        # push AX
            (Opcode.IMM, 3),   # AX = 3
            Opcode.ADD,        # AX = pop + AX = 5 + 3 = 8
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 8

    def test_sub_basic(self, neural_runner):
        """SUB: 10 - 3 = 7"""
        bytecode = make_bytecode([
            (Opcode.IMM, 10),
            Opcode.PSH,
            (Opcode.IMM, 3),
            Opcode.SUB,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 7

    def test_mul_basic(self, neural_runner):
        """MUL: 6 * 7 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 6),
            Opcode.PSH,
            (Opcode.IMM, 7),
            Opcode.MUL,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_div_basic(self, neural_runner):
        """DIV: 84 / 2 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 84),
            Opcode.PSH,
            (Opcode.IMM, 2),
            Opcode.DIV,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_mod_basic(self, neural_runner):
        """MOD: 100 % 58 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 100),
            Opcode.PSH,
            (Opcode.IMM, 58),
            Opcode.MOD,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_add_with_zero(self, neural_runner):
        """ADD: 42 + 0 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 0),
            Opcode.ADD,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_mul_by_one(self, neural_runner):
        """MUL: 42 * 1 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 1),
            Opcode.MUL,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# 100% Neural Opcodes - OR, XOR, AND
# =============================================================================

class TestNeuralBitwise:
    """Test bitwise opcodes work 100% neurally (no handlers needed)."""

    @pytest.fixture
    def neural_runner(self):
        """Runner with bitwise handlers removed."""
        runner = AutoregressiveVMRunner()
        for op in [Opcode.OR, Opcode.XOR, Opcode.AND]:
            if op in runner._func_call_handlers:
                del runner._func_call_handlers[op]
        return runner

    def test_or_basic(self, neural_runner):
        """OR: 0x0F | 0xF0 = 0xFF"""
        bytecode = make_bytecode([
            (Opcode.IMM, 0x0F),
            Opcode.PSH,
            (Opcode.IMM, 0xF0),
            Opcode.OR,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 0xFF

    def test_xor_basic(self, neural_runner):
        """XOR: 0xFF ^ 0x0F = 0xF0"""
        bytecode = make_bytecode([
            (Opcode.IMM, 0xFF),
            Opcode.PSH,
            (Opcode.IMM, 0x0F),
            Opcode.XOR,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 0xF0

    def test_and_basic(self, neural_runner):
        """AND: 0xFF & 0x0F = 0x0F"""
        bytecode = make_bytecode([
            (Opcode.IMM, 0xFF),
            Opcode.PSH,
            (Opcode.IMM, 0x0F),
            Opcode.AND,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 0x0F

    def test_or_with_zero(self, neural_runner):
        """OR: 42 | 0 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 0),
            Opcode.OR,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_and_with_all_ones(self, neural_runner):
        """AND: 42 & 0xFF = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 0xFF),
            Opcode.AND,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# 100% Neural Opcodes - SHL, SHR
# =============================================================================

class TestNeuralShift:
    """Test shift opcodes work 100% neurally (no handlers needed)."""

    @pytest.fixture
    def neural_runner(self):
        """Runner with shift handlers removed."""
        runner = AutoregressiveVMRunner()
        for op in [Opcode.SHL, Opcode.SHR]:
            if op in runner._func_call_handlers:
                del runner._func_call_handlers[op]
        return runner

    def test_shl_basic(self, neural_runner):
        """SHL: 1 << 4 = 16"""
        bytecode = make_bytecode([
            (Opcode.IMM, 1),
            Opcode.PSH,
            (Opcode.IMM, 4),
            Opcode.SHL,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 16

    def test_shr_basic(self, neural_runner):
        """SHR: 64 >> 4 = 4"""
        bytecode = make_bytecode([
            (Opcode.IMM, 64),
            Opcode.PSH,
            (Opcode.IMM, 4),
            Opcode.SHR,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 4

    def test_shl_by_zero(self, neural_runner):
        """SHL: 42 << 0 = 42"""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 0),
            Opcode.SHL,
            Opcode.EXIT,
        ])
        _, exit_code = neural_runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# Handler Required Opcodes - Document with xfail
# =============================================================================

class TestHandlerRequired:
    """Document opcodes that still need handlers (neural path incomplete)."""

    @pytest.mark.xfail(reason="JSR neural PC increment broken step 2+")
    def test_jsr_without_handler(self):
        """JSR requires handler for correct PC increment."""
        runner = AutoregressiveVMRunner()
        del runner._func_call_handlers[Opcode.JSR]

        # Simple function call that should return 42
        bytecode = make_bytecode([
            (Opcode.JSR, 4),   # JSR to addr 4 (the function)
            (Opcode.IMM, 99),  # Should not execute (return comes back here)
            Opcode.EXIT,
            # Function at addr 3 (but JSR jumps to addr 4 = next instruction):
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    @pytest.mark.xfail(reason="JMP FETCH only works for first step (L5 head 3 fixed address)")
    def test_jmp_without_handler(self):
        """JMP requires handler after step 1."""
        runner = AutoregressiveVMRunner()
        del runner._func_call_handlers[Opcode.JMP]

        # Jump over an instruction
        bytecode = make_bytecode([
            (Opcode.JMP, 2),   # JMP to addr 2
            (Opcode.IMM, 99),  # Skip this
            (Opcode.IMM, 42),  # Jump target
            Opcode.EXIT,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    @pytest.mark.xfail(reason="ENT generates wrong addresses in MEM tokens")
    def test_ent_without_handler(self):
        """ENT requires handler for correct MEM section generation."""
        runner = AutoregressiveVMRunner()
        del runner._func_call_handlers[Opcode.ENT]

        # Simple function entry
        bytecode = make_bytecode([
            (Opcode.ENT, 0),   # Enter function, 0 locals
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    @pytest.mark.xfail(reason="LEV SP/BP restoration fails (L16 FFN outputs 0)")
    def test_lev_without_handler(self):
        """LEV requires handler for SP/BP restoration."""
        runner = AutoregressiveVMRunner()
        del runner._func_call_handlers[Opcode.LEV]

        # LEV without handler
        bytecode = make_bytecode([
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# Parity Tests - Compare neural vs handler execution
# =============================================================================

class TestParityComparison:
    """Compare neural vs handler execution for consistency."""

    @pytest.mark.parametrize("a,b", [
        (5, 3),
        (0, 42),
        (42, 0),
        (100, 100),
        (255, 1),
    ])
    def test_add_parity(self, a, b):
        """ADD produces same result with and without handler."""
        bytecode = make_bytecode([
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.ADD,
            Opcode.EXIT,
        ])

        # With handlers (default)
        result_handler = run_with_handlers(bytecode)

        # Without ADD handler (neural only)
        result_neural = run_neural_only(bytecode, remove_handlers=[Opcode.ADD])

        # Both should match expected value
        assert result_handler == (a + b) & 0xFF
        assert result_neural == (a + b) & 0xFF
        assert result_handler == result_neural

    @pytest.mark.parametrize("a,b", [
        (10, 3),
        (100, 50),
        (42, 0),
        (255, 255),
    ])
    def test_sub_parity(self, a, b):
        """SUB produces same result with and without handler."""
        bytecode = make_bytecode([
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.SUB,
            Opcode.EXIT,
        ])

        result_handler = run_with_handlers(bytecode)
        result_neural = run_neural_only(bytecode, remove_handlers=[Opcode.SUB])

        expected = (a - b) & 0xFF
        assert result_handler == expected
        assert result_neural == expected


# =============================================================================
# Exit/NOP Tests
# =============================================================================

class TestNeuralExitNop:
    """Test EXIT and NOP work correctly (100% neural)."""

    def test_exit_with_value(self):
        """EXIT returns correct value."""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=10)
        assert exit_code == 42

    def test_nop_does_not_change_ax(self):
        """NOP preserves AX."""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.NOP,
            Opcode.NOP,
            Opcode.NOP,
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
