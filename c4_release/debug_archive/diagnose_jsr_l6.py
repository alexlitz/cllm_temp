#!/usr/bin/env python3
"""Debug L6 FFN PC override for JSR."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

bytecode = [Opcode.JSR | (26 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")
device = next(model.parameters()).device

print("="*60)
print("DEBUGGING L6 FFN PC OVERRIDE")
print("="*60)

# Generate PC marker
next_token = model.generate_next(context)
context.append(next_token)
print(f"Generated PC marker: {next_token}")

# Get embeddings and run through layers
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

# Run through L5
for layer_idx in range(6):
    x = model.blocks[layer_idx](x)

pc_pos = len(context) - 1
print(f"\nAfter L5 (position {pc_pos}):")
temp0_l5 = x[0, pc_pos, BD.TEMP + 0].item()
print(f"  TEMP[0]: {temp0_l5:.2f}")

# Check FETCH values in detail
print(f"  FETCH_LO (nibbles):", end=" ")
for k in range(4):  # First 4 nibbles = first 2 bytes
    val = 0
    for j in range(16):
        if x[0, pc_pos, BD.FETCH_LO + k*16 + j].item() > 0.5:
            val = j
    print(f"[{k}]={val}", end=" ")
print()

# Now run through L6
x = model.blocks[6](x)

print(f"\nAfter L6 (position {pc_pos}):")
temp0_l6 = x[0, pc_pos, BD.TEMP + 0].item()
print(f"  TEMP[0]: {temp0_l6:.2f}")

# Check OUTPUT values (this is what becomes the PC bytes)
print(f"  OUTPUT_LO (nibbles):", end=" ")
output_nibbles = []
for k in range(4):
    val = 0
    for j in range(16):
        if x[0, pc_pos, BD.OUTPUT_LO + k*16 + j].item() > 0.5:
            val = j
    output_nibbles.append(val)
    print(f"[{k}]={val}", end=" ")
print()

# Compute OUTPUT PC value
output_byte0 = output_nibbles[0] + output_nibbles[1] * 16
output_byte1 = output_nibbles[2] + output_nibbles[3] * 16 if len(output_nibbles) > 2 else 0
output_pc = output_byte0 + output_byte1 * 256
print(f"  OUTPUT PC value: {output_pc}")
print(f"  Expected: 26 if JSR override worked, 10 if normal PC+8")

# Also check FETCH at L6 output
print(f"\n  FETCH_LO after L6 (nibbles):", end=" ")
for k in range(4):
    val = 0
    for j in range(16):
        if x[0, pc_pos, BD.FETCH_LO + k*16 + j].item() > 0.5:
            val = j
    print(f"[{k}]={val}", end=" ")
print()

print(f"\n{'='*60}")
print("DIAGNOSIS:")
if temp0_l5 >= 4.0:
    print("  ✓ TEMP[0] >= 4.0 after L5 - JSR condition met")
else:
    print("  ✗ TEMP[0] < 4.0 after L5 - JSR condition NOT met")

if output_pc == 26:
    print("  ✓ L6 FFN override worked - PC = 26")
elif output_pc == 10:
    print("  ✗ L6 FFN override FAILED - PC = 10 (normal increment)")
else:
    print(f"  ? Unexpected OUTPUT PC: {output_pc}")
print("="*60)
