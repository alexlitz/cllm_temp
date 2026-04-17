#!/usr/bin/env python3
"""Simple test for neural JSR - matches the diagnostic pattern."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token

# With INSTR_WIDTH=8 and PC_OFFSET=2:
# - Instruction 0: bytes 0-7, PC at byte 2
# - Instruction 1: bytes 8-15, PC at byte 10
# - Instruction 2: bytes 16-23, PC at byte 18
# - Instruction 3: bytes 24-31, PC at byte 26
#
# JSR with target 26 jumps to instruction 3 (PC=26)

print("="*60)
print("Simple Neural JSR Test")
print("="*60)
print()
print("Bytecode layout:")
print("  Instr 0 (PC= 2): JSR 26  - Jump to instruction 3")
print("  Instr 1 (PC=10): EXIT    - Never reached")
print("  Instr 2 (PC=18): NOP     - Padding")
print("  Instr 3 (PC=26): IMM 42  - Target of JSR")
print("  Instr 4 (PC=34): EXIT    - Exit with AX=42")
print()

bytecode = [
    Opcode.JSR | (26 << 8),  # Instr 0: Jump to PC=26 (instruction 3)
    Opcode.EXIT,              # Instr 1: Never reached
    Opcode.NOP,               # Instr 2: Padding
    Opcode.IMM | (42 << 8),   # Instr 3: Target - load 42
    Opcode.EXIT,              # Instr 4: Exit with 42
]

print("Creating runner (no JSR handler)...")
runner = AutoregressiveVMRunner()
print(f"Function handlers: {list(runner._func_call_handlers.keys())}")
print()

# Run
print("Running...")
result = runner.run(bytecode, b"", max_steps=10)

if isinstance(result, tuple):
    output, exit_code = result
else:
    output, exit_code = "", result

print(f"\nResult: exit_code={exit_code}, output='{output}'")

if exit_code == 42:
    print("\n✓ SUCCESS! Neural JSR works!")
else:
    print(f"\n✗ FAILED - Expected exit_code=42, got {exit_code}")
