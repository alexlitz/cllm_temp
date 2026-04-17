#\!/usr/bin/env python3
"""Check what's at address 2 for IMM vs JSR programs."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

# IMM program
bytecode_imm = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
context_imm = runner._build_context(bytecode_imm, b"", [], "")

# JSR program  
bytecode_jsr = [Opcode.JSR | (25 << 8)]
context_jsr = runner._build_context(bytecode_jsr, b"", [], "")

print("IMM program (IMM 42; EXIT):")
for i in range(8):
    print(f"  Address {i}: {context_imm[1+i]}")

print("\nJSR program (JSR 25):")
for i in range(8):
    print(f"  Address {i}: {context_jsr[1+i]}")

print("\nBoth programs have byte 0 at address 2 (immediate byte 1).")
print("L5 head 7 fetches from address 2 and relays OP_* flags.")
print("Byte 0 has OP_LEA=1.0 in embedding (not OP_IMM or OP_JSR).")
print("\nSo L5 head 7 is broken for BOTH programs\!")
print("Yet IMM works... there must be another path.")
