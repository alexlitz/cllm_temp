#!/usr/bin/env python3
"""Understand why IMM works despite L5 fetching from wrong address."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token

print("=" * 80)
print("IMM MYSTERY INVESTIGATION")
print("=" * 80)

print("\n### HYPOTHESIS ###")
print("L5 head 2/7 fetch from PC_OFFSET=2, but:")
print("  - For JSR: address 2 has 0 (immediate byte 1)")
print("  - For IMM: address 2 has 0 (immediate byte 1)")
print("Yet IMM works and JSR doesn't! Why?")

# Test 1: IMM bytecode structure
print("\n### IMM BYTECODE STRUCTURE ###")
bytecode_imm = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner_imm = AutoregressiveVMRunner()
context_imm = runner_imm._build_context(bytecode_imm, b"", [], "")

print(f"IMM opcode: {Opcode.IMM} (0x{Opcode.IMM:02x})")
print(f"IMM | (42 << 8) = {Opcode.IMM | (42 << 8)}")

# Break down the instruction
instr = Opcode.IMM | (42 << 8)
byte0 = instr & 0xFF  # Opcode
byte1 = (instr >> 8) & 0xFF  # Immediate byte 0
byte2 = (instr >> 16) & 0xFF  # Immediate byte 1
byte3 = (instr >> 24) & 0xFF  # Immediate byte 2

print(f"\nInstruction breakdown:")
print(f"  Byte 0 (opcode): {byte0} (0x{byte0:02x})")
print(f"  Byte 1 (imm0):   {byte1} (0x{byte1:02x})")
print(f"  Byte 2 (imm1):   {byte2} (0x{byte2:02x})")
print(f"  Byte 3 (imm2):   {byte3} (0x{byte3:02x})")

code_start_idx = context_imm.index(Token.CODE_START)
print(f"\nContext after CODE_START:")
for i in range(8):
    tok_idx = code_start_idx + 1 + i
    if tok_idx < len(context_imm):
        print(f"  Address {i}: {context_imm[tok_idx]}")

# Test 2: JSR bytecode structure
print("\n### JSR BYTECODE STRUCTURE ###")
bytecode_jsr = [Opcode.JSR | (25 << 8)]
runner_jsr = AutoregressiveVMRunner()
context_jsr = runner_jsr._build_context(bytecode_jsr, b"", [], "")

print(f"JSR opcode: {Opcode.JSR} (0x{Opcode.JSR:02x})")
print(f"JSR | (25 << 8) = {Opcode.JSR | (25 << 8)}")

instr = Opcode.JSR | (25 << 8)
byte0 = instr & 0xFF
byte1 = (instr >> 8) & 0xFF
byte2 = (instr >> 16) & 0xFF
byte3 = (instr >> 24) & 0xFF

print(f"\nInstruction breakdown:")
print(f"  Byte 0 (opcode): {byte0} (0x{byte0:02x})")
print(f"  Byte 1 (imm0):   {byte1} (0x{byte1:02x})")
print(f"  Byte 2 (imm1):   {byte2} (0x{byte2:02x})")
print(f"  Byte 3 (imm2):   {byte3} (0x{byte3:02x})")

code_start_idx = context_jsr.index(Token.CODE_START)
print(f"\nContext after CODE_START:")
for i in range(8):
    tok_idx = code_start_idx + 1 + i
    if tok_idx < len(context_jsr):
        print(f"  Address {i}: {context_jsr[tok_idx]}")

print("\n### ANALYSIS ###")
print("Both programs have 0 at address 2 (immediate byte 1).")
print("So the issue isn't WHAT is fetched from address 2.")
print("\nPossible explanations:")
print("1. IMM uses a DIFFERENT code path than L5 head 2/7 opcode fetch")
print("2. IMM's opcode nibbles (lo=1, hi=0) happen to match some other mechanism")
print("3. The model learned to work around the bug for common opcodes")
print("4. There's a fallback mechanism that works for IMM but not JSR")

print("\n### CHECK OPCODE NIBBLES ###")
print(f"IMM opcode 1 (0x{Opcode.IMM:02x}): lo={Opcode.IMM & 0xF}, hi={(Opcode.IMM >> 4) & 0xF}")
print(f"JSR opcode 3 (0x{Opcode.JSR:02x}): lo={Opcode.JSR & 0xF}, hi={(Opcode.JSR >> 4) & 0xF}")

print("\nIf L5 fetches byte 0 from address 2:")
print(f"  IMM would get: lo={0 & 0xF}, hi={(0 >> 4) & 0xF}")
print(f"  JSR would get: lo={0 & 0xF}, hi={(0 >> 4) & 0xF}")
print("Both would get the same nibbles (0, 0) - neither would work!")

print("\n### CONCLUSION ###")
print("The mystery deepens. Both should fail with the current addressing,")
print("yet IMM works. This suggests L5 heads 2/7 are NOT the primary path")
print("for IMM opcode detection on the first step.")
print("\nNext: Check if there's an ALTERNATIVE mechanism for first-step IMM.")
