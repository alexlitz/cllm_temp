#!/usr/bin/env python3
"""
JMP Neural Behavior Tests

Tests for unconditional jump (JMP) instruction neural execution.

Current Status:
- JMP works in first step (step 0)
- JMP broken after step 1 (L5 head 3 has fixed address lookup)
- Handler required for reliable multi-step execution

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
# JMP with Handler (Working Baseline)
# =============================================================================

class TestJMPWithHandler:
    """Test JMP works correctly with handler enabled (baseline)."""

    def test_jmp_forward_basic(self):
        """JMP forward skips instructions."""
        bytecode = make_bytecode([
            (Opcode.JMP, 2),    # 0: JMP to addr 2
            (Opcode.IMM, 99),   # 1: Skip this
            (Opcode.IMM, 42),   # 2: Jump target
            Opcode.EXIT,        # 3: Exit
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_jmp_skip_multiple(self):
        """JMP forward skips multiple instructions."""
        bytecode = make_bytecode([
            (Opcode.JMP, 4),    # 0: JMP to addr 4
            (Opcode.IMM, 1),    # 1: Skip
            (Opcode.IMM, 2),    # 2: Skip
            (Opcode.IMM, 3),    # 3: Skip
            (Opcode.IMM, 42),   # 4: Jump target
            Opcode.EXIT,        # 5: Exit
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_jmp_chain(self):
        """Multiple JMPs work correctly."""
        bytecode = make_bytecode([
            (Opcode.JMP, 2),    # 0: JMP to 2
            (Opcode.IMM, 99),   # 1: Skip
            (Opcode.JMP, 4),    # 2: JMP to 4
            (Opcode.IMM, 88),   # 3: Skip
            (Opcode.IMM, 42),   # 4: Final target
            Opcode.EXIT,        # 5: Exit
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42

    def test_jmp_backward_loop(self):
        """JMP backward creates a loop (with exit condition)."""
        # Loop: decrement counter until 0, then exit
        # Uses BNZ for loop control
        bytecode = make_bytecode([
            (Opcode.IMM, 3),    # 0: Counter = 3
            Opcode.PSH,         # 1: Push counter
            (Opcode.IMM, 1),    # 2: AX = 1
            Opcode.SUB,         # 3: Counter - 1
            (Opcode.BNZ, 1),    # 4: If not zero, jump to PSH
            Opcode.EXIT,        # 5: Exit with AX=0
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=50)
        assert exit_code == 0


# =============================================================================
# JMP Neural Path Tests (xfail for broken cases)
# =============================================================================

class TestJMPNeural:
    """Test JMP neural behavior (documents limitations)."""

    def test_jmp_step1_works(self):
        """JMP works in first step."""
        # Simple JMP at step 0, then IMM, EXIT
        bytecode = make_bytecode([
            (Opcode.JMP, 2),    # 0: JMP to 2 (first step)
            (Opcode.IMM, 99),   # 1: Skip
            (Opcode.IMM, 42),   # 2: Target
            Opcode.EXIT,        # 3: Exit
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    @pytest.mark.xfail(reason="L5 head 3 has fixed address lookup - JMP broken step 2+")
    def test_jmp_step2_broken(self):
        """JMP broken after step 1 (neural path limitation)."""
        runner = AutoregressiveVMRunner()
        # Remove JMP handler to test neural path
        if Opcode.JMP in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.JMP]

        # JMP at step 1 (after IMM)
        bytecode = make_bytecode([
            (Opcode.IMM, 1),    # 0: Step 0
            (Opcode.JMP, 4),    # 1: Step 1 - JMP should go to 4
            (Opcode.IMM, 99),   # 2: Skip
            (Opcode.IMM, 88),   # 3: Skip
            (Opcode.IMM, 42),   # 4: Target
            Opcode.EXIT,        # 5: Exit
        ])
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42

    @pytest.mark.xfail(reason="Neural JMP uses fixed address bits, not dynamic")
    def test_jmp_variable_target(self):
        """JMP to computed address broken neurally."""
        runner = AutoregressiveVMRunner()
        if Opcode.JMP in runner._func_call_handlers:
            del runner._func_call_handlers[Opcode.JMP]

        # Different jump targets - neural path may hardcode first seen target
        for target in [2, 3, 4]:
            bytecode_list = [(Opcode.JMP, target)]
            # Add padding instructions
            for i in range(1, target):
                bytecode_list.append((Opcode.IMM, 99))
            bytecode_list.append((Opcode.IMM, 42))
            bytecode_list.append(Opcode.EXIT)

            bytecode = make_bytecode(bytecode_list)
            _, exit_code = runner.run(bytecode, b'', max_steps=20)
            assert exit_code == 42, f"JMP {target} failed"


# =============================================================================
# JMP Address Conversion Tests
# =============================================================================

class TestJMPAddressConversion:
    """Test JMP address handling."""

    def test_jmp_to_immediate(self):
        """JMP directly to a bytecode address."""
        bytecode = make_bytecode([
            (Opcode.JMP, 3),
            (Opcode.IMM, 1),
            (Opcode.IMM, 2),
            (Opcode.IMM, 42),  # Address 3
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_jmp_large_offset(self):
        """JMP with larger offset."""
        bytecode = make_bytecode([
            (Opcode.JMP, 10),
        ])
        # Add 9 NOPs
        for _ in range(9):
            bytecode.append(Opcode.NOP)
        bytecode.extend(make_bytecode([
            (Opcode.IMM, 42),  # Address 10
            Opcode.EXIT,
        ]))

        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42


# =============================================================================
# JMP Interaction with Other Ops
# =============================================================================

class TestJMPInteractions:
    """Test JMP interactions with other instructions."""

    def test_jmp_preserves_ax(self):
        """JMP doesn't modify AX."""
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            (Opcode.JMP, 3),
            (Opcode.IMM, 99),  # Skip
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=20)
        assert exit_code == 42

    def test_jmp_preserves_stack(self):
        """JMP doesn't modify stack."""
        bytecode = make_bytecode([
            (Opcode.IMM, 10),
            Opcode.PSH,
            (Opcode.IMM, 32),
            (Opcode.JMP, 5),
            (Opcode.IMM, 99),  # Skip
            Opcode.ADD,        # 10 + 32 = 42
            Opcode.EXIT,
        ])
        runner = AutoregressiveVMRunner()
        _, exit_code = runner.run(bytecode, b'', max_steps=30)
        assert exit_code == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
