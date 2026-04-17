#!/usr/bin/env python3
"""
Debug ADJ implementation - check if it's actually being called.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

# Compile simple program with local variable
code = "int main() { int a; a = 42; return a; }"
print(f"Compiling: {code}")
bytecode, data = compile_c(code)

print(f"\nBytecode ({len(bytecode)} instructions):")
for i, instr in enumerate(bytecode[:20]):
    op = instr & 0xFF
    imm = instr >> 8
    op_name = "?"
    for name in dir(Opcode):
        if not name.startswith('_') and getattr(Opcode, name) == op:
            op_name = name
            break
    print(f"  {i:3d}: {op_name:8s} {imm:8d} (0x{imm:06x})")
    if op == Opcode.EXIT:
        break

# Look for ADJ instructions
adj_count = sum(1 for instr in bytecode if (instr & 0xFF) == Opcode.ADJ)
print(f"\nADJ instructions found: {adj_count}")

if adj_count == 0:
    print("✗ No ADJ instructions - test won't exercise ADJ neural implementation")
else:
    print("✓ ADJ instructions present - should test neural implementation")
