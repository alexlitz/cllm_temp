"""
Verify bytecode memory layout.
"""

from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET, opcode_address, immediate_address

value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

print("Bytecode (32-bit words):")
print(f"  [0] 0x{bytecode[0]:08x} (IMM | (0x{value:02x} << 8))")
print(f"  [1] 0x{bytecode[1]:08x} (EXIT)")

print("\nConstants:")
print(f"  INSTR_WIDTH = {INSTR_WIDTH}")
print(f"  PC_OFFSET = {PC_OFFSET}")

print("\nInstruction layout (assuming 8-byte slots):")
# Instruction 0: IMM
print(f"  Instruction 0 (IMM):")
print(f"    PC value: {0 * INSTR_WIDTH + PC_OFFSET}")
print(f"    Opcode at byte: {opcode_address(PC_OFFSET)}")
print(f"    Immediate at byte: {immediate_address(PC_OFFSET)}")

# Instruction 1: EXIT  
print(f"  Instruction 1 (EXIT):")
print(f"    PC value: {1 * INSTR_WIDTH + PC_OFFSET}")
print(f"    Opcode at byte: {opcode_address(1 * INSTR_WIDTH + PC_OFFSET)}")
print(f"    Immediate at byte: {immediate_address(1 * INSTR_WIDTH + PC_OFFSET)}")

# Convert to byte array
print("\nByte-by-byte layout:")
for idx, word in enumerate(bytecode):
    base_addr = idx * INSTR_WIDTH
    for i in range(4):  # 4 bytes per word
        byte_val = (word >> (i * 8)) & 0xFF
        addr = base_addr + i
        print(f"  [0x{addr:02x}]: 0x{byte_val:02x}", end="")
        if addr == opcode_address(PC_OFFSET):
            print(" ← IMM opcode", end="")
        elif addr == immediate_address(PC_OFFSET):
            print(" ← IMM immediate", end="")
        elif addr == opcode_address(1 * INSTR_WIDTH + PC_OFFSET):
            print(" ← EXIT opcode", end="")
        print()

print("\nWhat should be fetched:")
print(f"  At PC=2 (step 0): fetch byte at 0x{opcode_address(2):02x} = IMM opcode")
print(f"  At PC=10 (step 1): fetch byte at 0x{opcode_address(10):02x} = EXIT opcode")
