#!/usr/bin/env python3
"""Simple test: Does JSR work now?"""

from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# JSR 25; EXIT
bytecode = [
    Opcode.JSR | (25 << 8),  # JSR to address 25
    Opcode.EXIT,              # Should not reach here
    # Padding to address 25
    *([Opcode.NOP] * 9),
    # Address 25: return 42
    Opcode.IMM | (42 << 8),   # AX = 42
    Opcode.EXIT,               # EXIT with code 42
]

runner = AutoregressiveVMRunner()
result = runner.run(bytecode, b"", [], "")

print(f"Exit code: {result}")
print(f"Expected: 42")
print(f"JSR works: {result == 42}")
