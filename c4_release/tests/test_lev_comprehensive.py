#!/usr/bin/env python3
"""
Comprehensive LEV instruction tests.

Tests the LEV (leave/return) instruction for function returns, verifying:
- AX preservation through function calls
- SP/BP restoration
- PC return address handling

Status (2026-04-17):
- L15 attention: returns correct return_addr
- L16 FFN: AX passthrough working after OUTPUT_HI fix
- SP/BP: Still using handler for restoration

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.constants import idx_to_pc


class TestLEVBasic:
    """Basic LEV tests - function returns with AX preservation."""

    @pytest.fixture
    def runner(self):
        """Create a fresh runner for each test."""
        return AutoregressiveVMRunner()

    def test_basic_jsr_lev(self, runner):
        """JSR to function, set AX=42, LEV returns with AX=42."""
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),  # idx 0: call function
            Opcode.EXIT,                  # idx 1: exit after return
            0, 0, 0,                      # idx 2-4: padding
            Opcode.ENT | (0 << 8),        # idx 5: function entry
            Opcode.IMM | (42 << 8),       # idx 6: set AX=42
            Opcode.LEV,                   # idx 7: return
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 42, f"Expected AX=42, got {ax}"

    def test_lev_16_byte_stack(self, runner):
        """LEV with ENT allocating 16 bytes of stack, AX=99."""
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (16 << 8),       # Allocate 16 bytes
            Opcode.IMM | (99 << 8),
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 99, f"Expected AX=99, got {ax}"

    def test_sequential_imm_before_lev(self, runner):
        """Multiple IMM before LEV, last one (200) should be returned."""
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (0 << 8),
            Opcode.IMM | (100 << 8),      # AX = 100
            Opcode.IMM | (200 << 8),      # AX = 200 (overwrites)
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 200, f"Expected AX=200, got {ax}"

    def test_return_continues(self, runner):
        """After function return, execution continues at caller, AX=77."""
        ent_pc = idx_to_pc(6)
        bytecode = [
            Opcode.IMM | (1 << 8),        # idx 0: AX = 1
            Opcode.JSR | (ent_pc << 8),   # idx 1: call function
            Opcode.EXIT,                  # idx 2: exit (AX should be from function)
            0, 0, 0,                      # idx 3-5: padding
            Opcode.ENT | (0 << 8),        # idx 6: function
            Opcode.IMM | (77 << 8),       # idx 7: AX = 77
            Opcode.LEV,                   # idx 8: return
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 77, f"Expected AX=77, got {ax}"

    def test_lea_then_lev(self, runner):
        """LEA before LEV (common pattern), AX=55."""
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (8 << 8),        # Allocate 8 bytes
            Opcode.LEA | (0 << 8),        # LEA 0 (address of local)
            Opcode.IMM | (55 << 8),       # AX = 55
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 55, f"Expected AX=55, got {ax}"


class TestLEVNeural:
    """Tests for neural LEV behavior (document what works/doesn't)."""

    def test_ax_preserved_neurally(self):
        """Verify AX preservation works (fixed 2026-04-17)."""
        # This was broken before the L16 FFN OUTPUT_HI fix
        runner = AutoregressiveVMRunner()
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (0 << 8),
            Opcode.IMM | (42 << 8),
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        # AX should be 42, not 0 or 58 (which were the broken values)
        assert ax == 42, f"AX preservation broken: expected 42, got {ax}"

    @pytest.mark.xfail(reason="L16 FFN SP/BP outputs 0 - handler still needed")
    def test_sp_bp_neural(self):
        """Document: SP/BP restoration still requires handler."""
        runner = AutoregressiveVMRunner()
        # Remove LEV handler
        if Opcode.LEV in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.LEV]

        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (0 << 8),
            Opcode.IMM | (42 << 8),
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 42


class TestLEVEdgeCases:
    """Edge cases for LEV instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_lev_with_zero_ax(self, runner):
        """LEV with AX=0 (edge case)."""
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (0 << 8),
            Opcode.IMM | (0 << 8),        # AX = 0
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 0, f"Expected AX=0, got {ax}"

    def test_lev_with_max_byte(self, runner):
        """LEV with AX=255 (max byte value)."""
        ent_pc = idx_to_pc(5)
        bytecode = [
            Opcode.JSR | (ent_pc << 8),
            Opcode.EXIT,
            0, 0, 0,
            Opcode.ENT | (0 << 8),
            Opcode.IMM | (255 << 8),      # AX = 255
            Opcode.LEV,
        ]
        _, ax = runner.run(bytecode, max_steps=15)
        assert ax == 255, f"Expected AX=255, got {ax}"

    def test_nested_function_calls(self, runner):
        """Nested function calls: main -> func1 -> func2."""
        # This is a more complex test
        func2_pc = idx_to_pc(10)
        func1_pc = idx_to_pc(6)
        bytecode = [
            Opcode.JSR | (func1_pc << 8),  # idx 0: call func1
            Opcode.EXIT,                    # idx 1: exit
            0, 0, 0, 0,                     # idx 2-5: padding
            Opcode.ENT | (0 << 8),          # idx 6: func1 entry
            Opcode.JSR | (func2_pc << 8),   # idx 7: call func2
            Opcode.LEV,                     # idx 8: return from func1
            0,                              # idx 9: padding
            Opcode.ENT | (0 << 8),          # idx 10: func2 entry
            Opcode.IMM | (123 << 8),        # idx 11: AX = 123
            Opcode.LEV,                     # idx 12: return from func2
        ]
        _, ax = runner.run(bytecode, max_steps=25)
        assert ax == 123, f"Expected AX=123 from nested call, got {ax}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
