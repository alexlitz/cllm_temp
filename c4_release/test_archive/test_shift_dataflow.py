"""Test to understand SHIFT data flow in GenericE format."""

import torch
from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.shift import build_shl_layers
from neural_vm.alu.ops.common import GenericE

def int_to_chunks(val, config):
    chunks = []
    mask = config.chunk_max
    for _ in range(config.num_positions):
        chunks.append(val & mask)
        val >>= config.chunk_bits
    return chunks

# Build SHL
config = NIBBLE
shl = build_shl_layers(config, opcode=23)
ge = GenericE(config)

# Test: shift 0x7A (122) left by 2
# Expected: 0x1E8 & 0xFF = 0xE8
a = 0x7A
shift_amount = 2
expected = (a << shift_amount) & 0xFF

print(f"Test: 0x{a:02X} << {shift_amount} = 0x{expected:02X}")

# Create GenericE format input
N = config.num_positions  # 8 for NIBBLE
x = torch.zeros(1, N, 160)

# Encode operand A in NIB_A slots
a_chunks = int_to_chunks(a, config)
print(f"A chunks: {a_chunks}")
for i in range(N):
    x[0, i, ge.NIB_A] = float(a_chunks[i])
    x[0, i, ge.OP_START + 23] = 1.0  # Opcode

# Encode shift amount in NIB_B slots  
sa = shift_amount
for i in range(N):
    x[0, i, ge.NIB_B] = float(sa % config.base)
    sa //= config.base
    if sa == 0:
        break

print(f"Input NIB_A: {[x[0, i, ge.NIB_A].item() for i in range(N)]}")
print(f"Input NIB_B: {[x[0, i, ge.NIB_B].item() for i in range(N)]}")

# Run layers
with torch.no_grad():
    for layer in shl:
        x = layer(x)

# Extract result
result_chunks = [round(x[0, i, ge.RESULT].item()) for i in range(N)]
result = sum(int(c) << (config.chunk_bits * i) for i, c in enumerate(result_chunks)) & 0xFF

print(f"\nResult chunks: {result_chunks}")
print(f"Result: 0x{result:02X} (expected 0x{expected:02X})")
print(f"{'✓ PASS' if result == expected else '✗ FAIL'}")
