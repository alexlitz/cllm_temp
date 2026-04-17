#!/usr/bin/env python3
"""Check initial PC value and context structure."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token

# Simple program
bytecode = [Opcode.JSR | (25 << 8)]

runner = AutoregressiveVMRunner()
context = runner._build_context(bytecode, b"", [], "")

print("Context tokens:")
for i, tok in enumerate(context):
    if tok == Token.CODE_START:
        print(f"  {i}: CODE_START ({tok})")
    elif tok == Token.CODE_END:
        print(f"  {i}: CODE_END ({tok})")
    elif tok < 256:
        print(f"  {i}: {tok} (byte)")
    else:
        print(f"  {i}: {tok} (special)")

print(f"\nTotal context length: {len(context)}")
print(f"\nFirst 15 tokens: {context[:15]}")

# Check what the addressing is:
print("\nBytecode structure analysis:")
print("  JSR = 3")
print("  Immediate = 25")
print("Expected context after CODE_START:")
print("  Address 0: 3 (opcode)")
print("  Address 1: 25 (imm byte 0)")
print("  Address 2: 0 (imm byte 1)")
print("  Address 3: 0 (imm byte 2)")
print("  Address 4: 0 (imm byte 3)")
print("  Address 5-7: 0 (padding)")

print("\nActual:")
code_start_idx = context.index(Token.CODE_START)
for i in range(8):
    addr = i
    tok_idx = code_start_idx + 1 + i
    if tok_idx < len(context):
        print(f"  Address {addr}: {context[tok_idx]}")
