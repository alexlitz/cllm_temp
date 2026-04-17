"""Check what bytecode is generated for 10 + 32."""

from src.compiler import compile_c
from neural_vm.embedding import Opcode
import sys

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print(f"Bytecode for: {code}", file=sys.stderr)
print("=" * 70, file=sys.stderr)

opcode_names = {v: k for k, v in vars(Opcode).items() if k.isupper() and isinstance(v, int)}

for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    op_name = opcode_names.get(op, f"UNKNOWN_{op}")

    print(f"{i}: {op_name:10s} (op={op:3d}, imm={imm:8d} = 0x{imm:06x})", file=sys.stderr)

print("=" * 70, file=sys.stderr)
print("\nExpected sequence for expression evaluation:", file=sys.stderr)
print("  1. Push first operand to stack", file=sys.stderr)
print("  2. Evaluate second operand (into AX)", file=sys.stderr)
print("  3. Binary op uses stack top + AX", file=sys.stderr)
print("=" * 70, file=sys.stderr)
