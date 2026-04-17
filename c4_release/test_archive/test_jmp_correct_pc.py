#!/usr/bin/env python3
"""Test JMP with correct PC value (with PC_OFFSET)."""

import sys
sys.path.insert(0, '.')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

def test_jmp():
    runner = AutoregressiveVMRunner()

    # Remove JMP handler to test neural path
    if Opcode.JMP in runner._func_call_handlers:
        del runner._func_call_handlers[Opcode.JMP]
        print("Removed JMP handler - testing pure neural path")

    # Calculate correct PC values with PC_OFFSET
    # Instruction 0: PC = 0*8+2 = 2
    # Instruction 1: PC = 1*8+2 = 10
    # Instruction 2: PC = 2*8+2 = 18
    # Instruction 3: PC = 3*8+2 = 26

    target_idx = 2  # Jump to instruction 2 (IMM 42)
    target_pc = target_idx * INSTR_WIDTH + PC_OFFSET  # = 18

    print(f"PC_OFFSET = {PC_OFFSET}, INSTR_WIDTH = {INSTR_WIDTH}")
    print(f"Target instruction index: {target_idx}")
    print(f"Target PC: {target_pc} (0x{target_pc:02x})")
    print()

    bytecode = [
        Opcode.JMP | (target_pc << 8),    # JMP to PC=18 (instruction 2)
        Opcode.IMM | (99 << 8),            # index 1: should be skipped
        Opcode.IMM | (42 << 8),            # index 2: target - load 42
        Opcode.EXIT                        # index 3: exit with 42
    ]

    print("=== Testing JMP 18 (correct PC) ===")
    _, result = runner.run(bytecode, b'', max_steps=10)
    print(f"Result: {result} (expected 42)")

    if result == 42:
        print("SUCCESS - JMP neural path works with correct PC!")
    else:
        print("FAILED - still broken")

    # Compare with wrong PC (16)
    print("\n=== Testing JMP 16 (wrong PC, missing PC_OFFSET) ===")
    runner2 = AutoregressiveVMRunner()
    if Opcode.JMP in runner2._func_call_handlers:
        del runner2._func_call_handlers[Opcode.JMP]

    bytecode_wrong = [
        Opcode.JMP | (16 << 8),    # JMP to 16 (wrong - should be 18)
        Opcode.IMM | (99 << 8),
        Opcode.IMM | (42 << 8),
        Opcode.EXIT
    ]
    _, result2 = runner2.run(bytecode_wrong, b'', max_steps=10)
    print(f"Result: {result2} (expected 42 if neural path auto-corrects, 0 otherwise)")

if __name__ == "__main__":
    test_jmp()
