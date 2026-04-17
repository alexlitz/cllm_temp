#!/usr/bin/env python3
"""Final JSR test after adding OP_JSR to L5 heads 6 and 7."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# JSR 25; EXIT ... (padding) ... IMM 42; EXIT
bytecode = [
    Opcode.JSR | (25 << 8),  # JSR to address 25
    Opcode.EXIT,              # Should not reach here
    *([Opcode.NOP] * 9),     # Padding to address 25
    Opcode.IMM | (42 << 8),   # Address 25: AX = 42
    Opcode.EXIT,              # EXIT with code 42
]

print("Testing JSR with neural implementation...")
print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    print(f"  [{i*8:3d}] {op:3d} {imm:5d}")

runner = AutoregressiveVMRunner()
result = runner.run(bytecode, b"", [], "")

print(f"\nResult: {result}")
print(f"Expected: ('', 42)")
print(f"JSR works: {result == ('', 42)}")

if result == ('', 42):
    print("\n✓ SUCCESS! JSR is working neurally!")
else:
    print(f"\n✗ FAILED. Got exit code {result[1] if isinstance(result, tuple) else result}")
