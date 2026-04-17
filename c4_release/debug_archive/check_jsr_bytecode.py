#\!/usr/bin/env python3
"""Check JSR bytecode formatting."""

from neural_vm.embedding import Opcode

# Create JSR 25 instruction
jsr_instr = Opcode.JSR | (25 << 8)
print(f"JSR 25 instruction value: {jsr_instr} (0x{jsr_instr:08x})")

# Extract bytes
byte0 = jsr_instr & 0xFF
byte1 = (jsr_instr >> 8) & 0xFF
byte2 = (jsr_instr >> 16) & 0xFF
byte3 = (jsr_instr >> 24) & 0xFF

print(f"Byte 0 (opcode): {byte0}")
print(f"Byte 1 (imm0): {byte1}")
print(f"Byte 2 (imm1): {byte2}")
print(f"Byte 3 (imm2): {byte3}")

print(f"\nExpected:")
print(f"  Opcode: 3 (JSR)")
print(f"  Immediate: 25")
print(f"  So byte 1 should be 25")
