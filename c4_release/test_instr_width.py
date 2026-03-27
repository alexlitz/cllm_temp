"""Test INSTR_WIDTH is imported correctly."""
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET

print(f"INSTR_WIDTH = {INSTR_WIDTH}")
print(f"PC_OFFSET = {PC_OFFSET}")

# Test the calculation
print(f"\nFor PC increment:")
print(f"  Old nibble 2 + INSTR_WIDTH: {(2 + INSTR_WIDTH) % 16} (should be 10)")
print(f"  Old nibble 15 + INSTR_WIDTH: {(15 + INSTR_WIDTH) % 16} (should be 7 with carry)")

print(f"\nCarry threshold: {16 - INSTR_WIDTH} (should be 8)")
