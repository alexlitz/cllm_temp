"""Check what address printf uses."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c

c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)
print(f"Code: {len(code)} instructions")
print(f"Data: {len(data)} bytes = {bytes(data).hex()}")

print("\nDisassembling...")
for i, instr in enumerate(code):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    imm_signed = imm if imm < 0x800000 else imm - 0x1000000
    print(f"  {i}: op={op:3d} (0x{op:02x}) imm={imm_signed:8d} (0x{imm:06x})")

print("\nExpected data_base in C4: 0x10000")
print(f"Format string should be at: 0x10000")
print(f"Format string bytes: {bytes(data).hex()} = ", end="")
for b in data[:3]:
    if 32 <= b < 127:
        print(f"'{chr(b)}'", end=" ")
    else:
        print(f"0x{b:02x}", end=" ")
print()
