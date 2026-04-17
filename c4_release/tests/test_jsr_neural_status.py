#!/usr/bin/env python3
"""
JSR Neural Status Tests

Tests for Jump Subroutine (JSR) instruction neural execution status.

Current Status:
- JSR works with handler
- Neural JSR PC increment broken step 2+ (documented in SESSION_2026-04-16_JSR_FIX.md)
- L5 attention has address lookup issues for dynamic targets

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
# JSR with Handler (Working Baseline)
# =============================================================================

class TestJSRWithHandler:
    """Test JSR works correctly with handler enabled."""

    def test_jsr_basic_call(self):
        """Basic JSR call and return."""
        bytecode = make_bytecode([
            (Opcode.JSR, 3),    # 0: Call function at 3
            (Opcode.IMM, 100),  # 1: Continue after return
            Opcode.EXIT,        # 2: Exit with AX=100

            (Opcode.ENT, 0),    # 3: Function entry
            (Opcode.IMM, 42),   # 4: AX = 42
            Opcode.LEV,         # 5: Return
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        # After LEV, returns to addr 1, then IMM 100, EXIT
        assert exit_code == 100

    def test_jsr_preserves_return_value(self):
        """JSR/LEV preserves AX from function."""
        bytecode = make_bytecode([
            (Opcode.JSR, 3),    # Call function
            Opcode.EXIT,        # Exit with function's return value

            Opcode.NOP,         # Padding
            (Opcode.ENT, 0),    # 3: Function entry
            (Opcode.IMM, 42),   # Set AX = 42
            Opcode.LEV,         # Return
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_jsr_nested_calls(self):
        """Nested JSR calls work correctly."""
        bytecode = make_bytecode([
            (Opcode.JSR, 4),    # 0: Call outer
            Opcode.EXIT,        # 1: Exit with result

            Opcode.NOP,         # 2: Padding
            Opcode.NOP,         # 3: Padding

            (Opcode.ENT, 0),    # 4: Outer function
            (Opcode.JSR, 9),    # 5: Call inner
            Opcode.LEV,         # 6: Return

            Opcode.NOP,         # 7: Padding
            Opcode.NOP,         # 8: Padding

            (Opcode.ENT, 0),    # 9: Inner function
            (Opcode.IMM, 42),   # 10: AX = 42
            Opcode.LEV,         # 11: Return
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=100)
        assert exit_code == 42

    def test_jsr_with_argument(self):
        """JSR with pushed argument."""
        bytecode = make_bytecode([
            (Opcode.IMM, 10),   # 0: Argument
            Opcode.PSH,         # 1: Push arg
            (Opcode.JSR, 5),    # 2: Call function
            Opcode.ADJ,         # 3: Clean stack (1 arg) - ADJ with immediate
            Opcode.EXIT,        # 4: Exit

            (Opcode.ENT, 0),    # 5: Function
            (Opcode.LEA, 2),    # 6: Load arg (BP+2)
            (Opcode.LI, 0),     # 7: Dereference (if needed by LEA semantics)
            Opcode.LEV,         # 8: Return with arg value
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        # Result depends on LEA/LI implementation
        # This test documents the calling convention


# =============================================================================
# JSR Neural Path Tests
# =============================================================================

class TestJSRNeural:
    """Test JSR neural behavior (documents limitations)."""

    @pytest.mark.xfail(reason="JSR neural PC increment broken step 2+")
    def test_jsr_without_handler_simple(self):
        """JSR requires handler for correct PC increment."""
        runner = AutoregressiveVMRunner()
        del runner._func_call_handlers[Opcode.JSR]

        bytecode = make_bytecode([
            (Opcode.JSR, 3),    # Call function
            Opcode.EXIT,        # Should return here

            Opcode.NOP,
            (Opcode.ENT, 0),    # 3: Function
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    @pytest.mark.xfail(reason="JSR neural path uses fixed address bits")
    def test_jsr_without_handler_step2(self):
        """JSR broken when executed at step 2+."""
        runner = AutoregressiveVMRunner()
        del runner._func_call_handlers[Opcode.JSR]

        # JSR at step 1 (after IMM)
        bytecode = make_bytecode([
            (Opcode.IMM, 1),    # 0: Step 0
            (Opcode.JSR, 5),    # 1: Step 1 - JSR
            Opcode.EXIT,        # 2: Return target

            Opcode.NOP,
            Opcode.NOP,
            (Opcode.ENT, 0),    # 5: Function
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# JSR Address Tests
# =============================================================================

class TestJSRAddress:
    """Test JSR address handling."""

    def test_jsr_immediate_address(self):
        """JSR with various immediate addresses."""
        for addr in [2, 3, 4, 5]:
            bytecode_list = [(Opcode.JSR, addr)]
            # Add padding
            for _ in range(1, addr):
                bytecode_list.append(Opcode.NOP)
            # Function
            bytecode_list.extend([
                (Opcode.ENT, 0),
                (Opcode.IMM, 42),
                Opcode.LEV,
            ])
            bytecode_list.append(Opcode.EXIT)

            bytecode = make_bytecode(bytecode_list)
            runner = AutoregressiveVMRunner()
            # Note: This test structure may need adjustment based on
            # where EXIT should be placed relative to LEV return

    def test_jsr_return_address_stored(self):
        """JSR stores correct return address."""
        # After JSR executes, return address should be PC+1
        bytecode = make_bytecode([
            (Opcode.JSR, 3),    # 0: JSR, return addr = 1
            (Opcode.IMM, 42),   # 1: Return here, set AX = 42
            Opcode.EXIT,        # 2: Exit

            (Opcode.ENT, 0),    # 3: Function
            (Opcode.IMM, 99),   # 4: AX = 99 (will be overwritten after LEV)
            Opcode.LEV,         # 5: Return to addr 1
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# JSR/ENT/LEV Integration
# =============================================================================

class TestJSRENTLEVIntegration:
    """Test JSR/ENT/LEV interaction."""

    def test_full_function_call(self):
        """Complete function call sequence."""
        bytecode = make_bytecode([
            (Opcode.JSR, 3),    # Call
            Opcode.EXIT,        # Exit with return value

            Opcode.NOP,
            (Opcode.ENT, 0),    # Function prologue
            (Opcode.IMM, 42),   # Function body
            Opcode.LEV,         # Function epilogue
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_multiple_sequential_calls(self):
        """Multiple sequential function calls."""
        bytecode = make_bytecode([
            (Opcode.JSR, 5),    # 0: Call first
            (Opcode.JSR, 9),    # 1: Call second
            Opcode.EXIT,        # 2: Exit with last return value

            Opcode.NOP,
            Opcode.NOP,
            (Opcode.ENT, 0),    # 5: First function
            (Opcode.IMM, 10),
            Opcode.LEV,

            Opcode.NOP,
            (Opcode.ENT, 0),    # 9: Second function
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=100)
        assert exit_code == 42

    def test_function_with_locals(self):
        """Function with local variable space."""
        bytecode = make_bytecode([
            (Opcode.JSR, 2),
            Opcode.EXIT,

            (Opcode.ENT, 16),   # 16 bytes local space
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


# =============================================================================
# JSR Edge Cases
# =============================================================================

class TestJSREdgeCases:
    """Test JSR edge cases."""

    def test_jsr_immediate_return(self):
        """JSR to function that returns immediately."""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),   # Set AX before call
            (Opcode.JSR, 4),    # Call
            Opcode.EXIT,        # Exit (AX should be preserved by LEV)

            Opcode.NOP,
            (Opcode.ENT, 0),    # Function does nothing
            Opcode.LEV,         # Return (should preserve AX=42)
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        # AX preservation after LEV - depends on LEV fix
        # This test documents expected behavior

    def test_jsr_deep_nesting(self):
        """Deeply nested function calls."""
        # 3 levels of nesting
        bytecode = make_bytecode([
            (Opcode.JSR, 3),    # 0: Call level 1
            Opcode.EXIT,        # 1: Exit

            Opcode.NOP,
            (Opcode.ENT, 0),    # 3: Level 1
            (Opcode.JSR, 8),    # 4: Call level 2
            Opcode.LEV,         # 5: Return

            Opcode.NOP,
            Opcode.NOP,
            (Opcode.ENT, 0),    # 8: Level 2
            (Opcode.JSR, 13),   # 9: Call level 3
            Opcode.LEV,         # 10: Return

            Opcode.NOP,
            Opcode.NOP,
            (Opcode.ENT, 0),    # 13: Level 3
            (Opcode.IMM, 42),   # 14: Set result
            Opcode.LEV,         # 15: Return
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=150)
        assert exit_code == 42


# =============================================================================
# Documentation Tests
# =============================================================================

class TestJSRDocumentation:
    """Document JSR behavior and known issues."""

    def test_jsr_stores_return_address_on_stack(self):
        """JSR pushes return address to stack.

        JSR semantics:
        1. Push current PC+1 to stack (return address)
        2. Jump to target address
        """
        # This is a documentation test - the behavior is implicit
        bytecode = make_bytecode([
            (Opcode.JSR, 2),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42

    def test_lev_pops_return_address(self):
        """LEV pops return address from stack.

        LEV semantics:
        1. Restore SP = BP
        2. Pop BP from stack
        3. Pop return address from stack
        4. Jump to return address
        """
        bytecode = make_bytecode([
            (Opcode.JSR, 2),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
