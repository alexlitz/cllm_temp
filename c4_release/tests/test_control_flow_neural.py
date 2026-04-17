#!/usr/bin/env python3
"""
Control Flow Neural Tests.

Tests for JMP, BZ, BNZ control flow instructions.

Status (2026-04-17):
- JMP: Works in step 1, broken step 2+ (L5 head 3 fixed address)
- BZ: Branch-if-zero, requires handler
- BNZ: Branch-if-not-zero, requires handler
- JSR: Covered in test_lev_comprehensive.py

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


def make_bytecode(ops):
    """Convert operation list to bytecode list."""
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


class TestJMPBasic:
    """Test JMP (unconditional jump) instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_jmp_step1_works(self, runner):
        """JMP works correctly in the first step."""
        # JMP to instruction 4, skip instructions 1-3
        bytecode = make_bytecode([
            (Opcode.JMP, 4),       # [0] JMP to instruction 4
            (Opcode.IMM, 99),      # [1] should be skipped
            Opcode.EXIT,           # [2] should be skipped
            Opcode.NOP,            # [3] padding
            (Opcode.IMM, 42),      # [4] target
            Opcode.EXIT,           # [5] exit with 42
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 42, f"JMP should skip to IMM 42, got {ax}"

    @pytest.mark.xfail(reason="JMP FETCH broken step 2+ (L5 head 3 fixed address)")
    def test_jmp_step2_broken(self, runner):
        """Document: JMP broken after step 1."""
        # Remove JMP handler to test neural path
        if Opcode.JMP in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.JMP]

        # Simple loop that requires multiple JMP
        bytecode = make_bytecode([
            (Opcode.IMM, 0),       # [0] AX = 0
            (Opcode.IMM, 42),      # [1] AX = 42
            (Opcode.JMP, 4),       # [2] JMP to EXIT
            Opcode.NOP,            # [3] padding
            Opcode.EXIT,           # [4] exit
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 42


class TestBZBasic:
    """Test BZ (branch-if-zero) instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_bz_zero_branches(self, runner):
        """BZ branches when AX=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 0),       # [0] AX = 0
            (Opcode.BZ, 4),        # [1] BZ to instruction 4 (should branch)
            (Opcode.IMM, 99),      # [2] should be skipped
            Opcode.EXIT,           # [3] should be skipped
            (Opcode.IMM, 42),      # [4] target
            Opcode.EXIT,           # [5] exit with 42
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 42, f"BZ should branch when AX=0, got {ax}"

    def test_bz_nonzero_continues(self, runner):
        """BZ does not branch when AX!=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 1),       # [0] AX = 1
            (Opcode.BZ, 4),        # [1] BZ to instruction 4 (should NOT branch)
            (Opcode.IMM, 42),      # [2] should execute
            Opcode.EXIT,           # [3] exit with 42
            (Opcode.IMM, 99),      # [4] BZ target (should be skipped)
            Opcode.EXIT,           # [5] exit (should be skipped)
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 42, f"BZ should NOT branch when AX!=0, got {ax}"

    @pytest.mark.xfail(reason="BZ neural path requires handler")
    def test_bz_neural_only(self):
        """Document: BZ requires handler."""
        runner = AutoregressiveVMRunner()
        if Opcode.BZ in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.BZ]

        bytecode = make_bytecode([
            (Opcode.IMM, 0),       # AX = 0
            (Opcode.BZ, 3),        # BZ to EXIT
            (Opcode.IMM, 99),      # should be skipped
            Opcode.EXIT,
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 0  # Should preserve AX=0


class TestBNZBasic:
    """Test BNZ (branch-if-not-zero) instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_bnz_nonzero_branches(self, runner):
        """BNZ branches when AX!=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 1),       # [0] AX = 1
            (Opcode.BNZ, 4),       # [1] BNZ to instruction 4 (should branch)
            (Opcode.IMM, 99),      # [2] should be skipped
            Opcode.EXIT,           # [3] should be skipped
            (Opcode.IMM, 42),      # [4] target
            Opcode.EXIT,           # [5] exit with 42
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 42, f"BNZ should branch when AX!=0, got {ax}"

    def test_bnz_zero_continues(self, runner):
        """BNZ does not branch when AX=0."""
        bytecode = make_bytecode([
            (Opcode.IMM, 0),       # [0] AX = 0
            (Opcode.BNZ, 4),       # [1] BNZ to instruction 4 (should NOT branch)
            (Opcode.IMM, 42),      # [2] should execute
            Opcode.EXIT,           # [3] exit with 42
            (Opcode.IMM, 99),      # [4] BNZ target (should be skipped)
            Opcode.EXIT,           # [5] exit (should be skipped)
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 42, f"BNZ should NOT branch when AX=0, got {ax}"

    @pytest.mark.xfail(reason="BNZ neural path requires handler")
    def test_bnz_neural_only(self):
        """Document: BNZ requires handler."""
        runner = AutoregressiveVMRunner()
        if Opcode.BNZ in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.BNZ]

        bytecode = make_bytecode([
            (Opcode.IMM, 1),       # AX = 1
            (Opcode.BNZ, 3),       # BNZ to EXIT
            (Opcode.IMM, 99),      # should be skipped
            Opcode.EXIT,
        ])
        _, ax = runner.run(bytecode, max_steps=10)
        assert ax == 1  # Should preserve AX=1


class TestJSRStatus:
    """Document JSR neural status (covered more in test_lev_comprehensive.py)."""

    @pytest.mark.xfail(reason="JSR PC increment broken step 2+")
    def test_jsr_neural_broken(self):
        """Document: JSR neural path broken."""
        runner = AutoregressiveVMRunner()
        if Opcode.JSR in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.JSR]

        from neural_vm.constants import idx_to_pc
        func_pc = idx_to_pc(4)
        bytecode = make_bytecode([
            (Opcode.JSR, func_pc >> 8),  # Call function
            Opcode.EXIT,
            0, 0,
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 42


class TestLoopPatterns:
    """Test common loop patterns with control flow."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_simple_while_loop(self, runner):
        """Simple while loop: sum 0..4 = 10."""
        from src.compiler import compile_c

        source = """
        int main() {
            int sum;
            int i;
            sum = 0;
            i = 0;
            while (i < 5) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=500)
        assert result == 10, f"Expected sum=10, got {result}"

    def test_if_else(self, runner):
        """If-else: condition true."""
        from src.compiler import compile_c

        source = """
        int main() {
            int x;
            x = 10;
            if (x > 5) {
                return 42;
            } else {
                return 99;
            }
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=100)
        assert result == 42, f"Expected 42 (condition true), got {result}"

    def test_if_else_false(self, runner):
        """If-else: condition false."""
        from src.compiler import compile_c

        source = """
        int main() {
            int x;
            x = 3;
            if (x > 5) {
                return 42;
            } else {
                return 99;
            }
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=100)
        assert result == 99, f"Expected 99 (condition false), got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
