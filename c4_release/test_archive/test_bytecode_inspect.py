#!/usr/bin/env python3
"""Inspect bytecode for simple program."""

from src.compiler import compile_c
from neural_vm.embedding import Opcode

code = '''
int main() {
    return 42;
}
'''

print("Code:")
print(code)
print("\nCompiling...")

bytecode, data = compile_c(code)

print(f"\nBytecode ({len(bytecode)} instructions):")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF

    # Find opcode name
    op_name = None
    for name in dir(Opcode):
        if not name.startswith('_'):
            val = getattr(Opcode, name)
            if isinstance(val, int) and val == op:
                op_name = name
                break

    if op_name is None:
        op_name = f"UNKNOWN_{op}"

    if imm:
        print(f"  {i:3d}: {op_name:10s} imm={imm} (0x{imm:x})")
    else:
        print(f"  {i:3d}: {op_name:10s}")

print(f"\nData section: {len(data)} bytes")
if data:
    print(f"  {data[:64]}...")
