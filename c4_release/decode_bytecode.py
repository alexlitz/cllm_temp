"""Decode the compiled bytecode for '10 + 32'"""

from src.compiler import compile_c
from neural_vm.embedding import Opcode

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print(f"C code: {code}\n")
print(f"Compiled bytecode ({len(bytecode)} values):\n")

# C4 bytecode format: each instruction is a 5-byte value
# byte 0 = opcode, bytes 1-4 = immediate value
for i, instr in enumerate(bytecode):
    opcode = instr & 0xFF
    immediate = instr >> 8

    # Find opcode name
    opname = "UNK"
    for name, value in vars(Opcode).items():
        if not name.startswith('_') and value == opcode:
            opname = name
            break

    if immediate != 0:
        print(f"  {i}: {opname:6s} {immediate}")
    else:
        print(f"  {i}: {opname}")

print()
