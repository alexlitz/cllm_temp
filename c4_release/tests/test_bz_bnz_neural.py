#!/usr/bin/env python3
"""
BZ/BNZ Neural Behavior Tests

Tests for conditional branch instructions:
- BZ (Branch if Zero): Jump when AX == 0
- BNZ (Branch if Not Zero): Jump when AX != 0

Current Status:
- BZ/BNZ work with handlers
- Neural path limited due to L5 attention issues (same as JMP)

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


def make_bytecode(ops):
    """Convert operation list to bytecode."""
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


# =============================================================================
# BZ (Branch if Zero) with Handler
# =============================================================================

class TestBZWithHandler:
    """Test BZ works correctly with handler enabled."""

    def test_bz_zero_branches(self):
        """BZ branches when AX=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 0),    # AX = 0
            (Opcode.BZ, 3),     # Branch to 3
            (Opcode.IMM, 99),   # Skip (AX=0 so branch taken)
            (Opcode.IMM, 42),   # Target
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_bz_nonzero_falls_through(self):
        """BZ falls through when AX!=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 1),    # AX = 1 (non-zero)
            (Opcode.BZ, 3),     # Branch not taken
            (Opcode.IMM, 42),   # Fall through here
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_bz_with_arithmetic_result(self):
        """BZ branches on arithmetic result."""
        # 5 - 5 = 0, so BZ should branch
        bytecode = make_bytecode([
            (Opcode.IMM, 5),
            Opcode.PSH,
            (Opcode.IMM, 5),
            Opcode.SUB,         # 5 - 5 = 0
            (Opcode.BZ, 6),     # Branch (AX=0)
            (Opcode.IMM, 99),   # Skip
            (Opcode.IMM, 42),   # Target
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42


# =============================================================================
# BNZ (Branch if Not Zero) with Handler
# =============================================================================

class TestBNZWithHandler:
    """Test BNZ works correctly with handler enabled."""

    def test_bnz_nonzero_branches(self):
        """BNZ branches when AX!=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 1),    # AX = 1
            (Opcode.BNZ, 3),    # Branch to 3
            (Opcode.IMM, 99),   # Skip
            (Opcode.IMM, 42),   # Target
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_bnz_zero_falls_through(self):
        """BNZ falls through when AX=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 0),    # AX = 0
            (Opcode.BNZ, 3),    # Branch not taken
            (Opcode.IMM, 42),   # Fall through here
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_bnz_large_value(self):
        """BNZ branches on large value."""
        bytecode = make_bytecode([
            (Opcode.IMM, 255),  # AX = 255
            (Opcode.BNZ, 3),    # Branch (non-zero)
            (Opcode.IMM, 99),   # Skip
            (Opcode.IMM, 42),   # Target
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42


# =============================================================================
# Loop Patterns with BZ/BNZ
# =============================================================================

class TestBranchLoops:
    """Test loop patterns using BZ/BNZ."""

    def test_countdown_loop_bnz(self):
        """Countdown loop using BNZ."""
        # Count down from 3 to 0
        bytecode = make_bytecode([
            (Opcode.IMM, 3),    # 0: Counter = 3
            Opcode.PSH,         # 1: Push counter
            (Opcode.IMM, 1),    # 2: AX = 1
            Opcode.SUB,         # 3: Counter - 1
            (Opcode.BNZ, 1),    # 4: If not zero, loop
            Opcode.EXIT,        # 5: Exit with AX = 0
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 0

    def test_search_loop_bz(self):
        """Search loop using BZ."""
        # Decrement until 0, then branch to result
        bytecode = make_bytecode([
            (Opcode.IMM, 2),    # 0: AX = 2
            Opcode.PSH,         # 1: Push
            (Opcode.IMM, 1),    # 2: AX = 1
            Opcode.SUB,         # 3: AX = pop - 1
            (Opcode.BZ, 6),     # 4: If zero, branch to result
            (Opcode.JMP, 1),    # 5: Loop back
            (Opcode.IMM, 42),   # 6: Result
            Opcode.EXIT,        # 7: Exit
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# BZ/BNZ Neural Path Tests
# =============================================================================

class TestBranchNeural:
    """Test BZ/BNZ neural behavior (documents limitations)."""

    @pytest.mark.xfail(reason="BZ neural path limited due to L5 attention (same as JMP)")
    def test_bz_without_handler(self):
        """BZ requires handler for reliable execution."""
        runner = AutoregressiveVMRunner()
        if Opcode.BZ in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.BZ]

        bytecode = make_bytecode([
            (Opcode.IMM, 0),
            (Opcode.BZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    @pytest.mark.xfail(reason="BNZ neural path limited due to L5 attention (same as JMP)")
    def test_bnz_without_handler(self):
        """BNZ requires handler for reliable execution."""
        runner = AutoregressiveVMRunner()
        if Opcode.BNZ in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.BNZ]

        bytecode = make_bytecode([
            (Opcode.IMM, 1),
            (Opcode.BNZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42


# =============================================================================
# Edge Cases
# =============================================================================

class TestBranchEdgeCases:
    """Test edge cases for branch instructions."""

    def test_bz_at_boundary(self):
        """BZ at edge of bytecode."""
        bytecode = make_bytecode([
            (Opcode.IMM, 0),
            (Opcode.BZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_bnz_negative_like(self):
        """BNZ with 0xFF (255, looks like -1)."""
        # 0 - 1 = 0xFF (255, two's complement -1)
        bytecode = make_bytecode([
            (Opcode.IMM, 0),
            Opcode.PSH,
            (Opcode.IMM, 1),
            Opcode.SUB,         # 0 - 1 = 255 (or -1)
            (Opcode.BNZ, 6),    # Branch (non-zero)
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42

    def test_bz_comparison_result(self):
        """BZ on comparison result."""
        # EQ returns 1 if equal, 0 if not
        bytecode = make_bytecode([
            (Opcode.IMM, 5),
            Opcode.PSH,
            (Opcode.IMM, 10),
            Opcode.EQ,          # 5 == 10 -> 0
            (Opcode.BZ, 6),     # Branch (result is 0)
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42

    def test_bnz_comparison_result(self):
        """BNZ on comparison result."""
        # EQ returns 1 if equal
        bytecode = make_bytecode([
            (Opcode.IMM, 5),
            Opcode.PSH,
            (Opcode.IMM, 5),
            Opcode.EQ,          # 5 == 5 -> 1
            (Opcode.BNZ, 6),    # Branch (result is 1)
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42


# =============================================================================
# Combined Branch Tests
# =============================================================================

class TestCombinedBranches:
    """Test combined BZ/BNZ scenarios."""

    def test_bz_bnz_sequence(self):
        """Sequential BZ and BNZ."""
        bytecode = make_bytecode([
            (Opcode.IMM, 1),    # AX = 1
            (Opcode.BZ, 10),    # Not taken (AX != 0)
            (Opcode.BNZ, 4),    # Taken (AX != 0)
            (Opcode.IMM, 99),   # Skip
            (Opcode.IMM, 42),   # Target
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_nested_branches(self):
        """Nested conditional logic."""
        # if (5 > 3) { if (1 == 1) { return 42 } }
        bytecode = make_bytecode([
            (Opcode.IMM, 5),
            Opcode.PSH,
            (Opcode.IMM, 3),
            Opcode.GT,          # 5 > 3 -> 1
            (Opcode.BZ, 12),    # Skip if not GT

            (Opcode.IMM, 1),
            Opcode.PSH,
            (Opcode.IMM, 1),
            Opcode.EQ,          # 1 == 1 -> 1
            (Opcode.BZ, 12),    # Skip if not EQ

            (Opcode.IMM, 42),   # Result
            Opcode.EXIT,

            (Opcode.IMM, 99),   # Unreached
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
