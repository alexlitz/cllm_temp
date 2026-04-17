#!/usr/bin/env python3
"""Test JSR with correct target PC value."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Calculate correct PC for instruction 3:
# PC = instruction_index * 8 + 2 = 3 * 8 + 2 = 26
TARGET_PC = 26

bytecode = [
    Opcode.JSR | (TARGET_PC << 8),  # JSR to PC=26 (instruction 3)
    Opcode.EXIT,                     # Instruction 1: Should not execute
    Opcode.NOP,                      # Instruction 2: Padding
    Opcode.IMM | (42 << 8),         # Instruction 3 at PC=26: AX = 42
    Opcode.EXIT,                     # Instruction 4: EXIT with code 42
]

print(f"Testing JSR with target PC={TARGET_PC}")
print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    pc = i * 8 + 2
    print(f"  Instruction {i} (PC={pc:2d}): opcode={op:2d}, imm={imm:3d}")

print("\nExpected execution:")
print("  1. JSR 26: Jump to PC=26")
print("  2. IMM 42: Set AX=42")
print("  3. EXIT: Exit with code 42")

runner = AutoregressiveVMRunner()
result = runner.run(bytecode, b"", [], "")

print(f"\nResult: {result}")
print(f"Expected: ('', 42)")

if result == ('', 42):
    print("\n✓ SUCCESS! JSR is working neurally!")
else:
    exit_code = result[1] if isinstance(result, tuple) else result
    print(f"\n✗ FAILED. Exit code: {exit_code}")
