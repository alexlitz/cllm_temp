#\!/usr/bin/env python3
"""Comprehensive JSR debug - trace full execution."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

print("="*80)
print("COMPREHENSIVE JSR DEBUG")
print("="*80)

# JSR 25; code at byte 25: IMM 42; EXIT
bytecode = [
    Opcode.JSR | (25 << 8),
    Opcode.NOP,  # padding
    Opcode.NOP,
    Opcode.NOP,
    Opcode.NOP,
    Opcode.IMM | (42 << 8),  # Function at byte 25
    Opcode.EXIT
]

runner = AutoregressiveVMRunner()
model = runner.model

# Build context
context = runner._build_context(bytecode, b"", [], "")
print(f"\nContext bytes 0-30:")
for i in range(min(30, len(context))):
    print(f"  [{i}]: {context[i]}")

# Generate tokens
print(f"\nGenerating step 1 (35 tokens)...")
for i in range(35):
    context.append(model.generate_next(context))

print(f"Context length after step 1: {len(context)}")

# Analyze
device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

pc_idx = len(context) - 35  # PC marker at start of step 1

print(f"\n### After embedding (PC marker at {pc_idx}):")
print(f"  OP_JSR: {x[0, pc_idx, BD.OP_JSR].item():.3f}")

# Through L5
for i in range(5):
    x = model.blocks[i](x)

print(f"\n### After L5 (head 7 should relay OP_JSR from CODE to PC):")
print(f"  PC OP_JSR: {x[0, pc_idx, BD.OP_JSR].item():.3f}")
print(f"  PC FETCH_LO[0-7] (first 8 nibbles):")
fetch_nibbles = [x[0, pc_idx, BD.FETCH_LO + k].item() for k in range(8)]
print(f"    {[f'{n:.1f}' for n in fetch_nibbles]}")

# Through L6
x = model.blocks[5](x)
ax_idx = pc_idx + 5

print(f"\n### After L6 (head 5 should relay OP_JSR from PC to AX):")
print(f"  PC OP_JSR: {x[0, pc_idx, BD.OP_JSR].item():.3f}")
print(f"  AX OP_JSR: {x[0, ax_idx, BD.OP_JSR].item():.3f}")
print(f"  PC TEMP[0] (IS_JSR from head 3): {x[0, pc_idx, BD.TEMP + 0].item():.3f}")

# Check PC OUTPUT
pc_byte0_nibbles = [x[0, pc_idx + k, BD.OUTPUT_LO + (k % 5)].item() for k in range(1, 5)]
print(f"\n### PC output bytes 0-3:")
for k in range(4):
    byte_val = sum([x[0, pc_idx + 1 + k, BD.OUTPUT_LO + n].item() * (2**n) 
                    for n in range(8) if x[0, pc_idx + 1 + k, BD.OUTPUT_LO + n].item() > 0.5])
    print(f"  Byte {k}: {int(byte_val)}")

print(f"\n### Expected:")
print(f"  - After L5: PC OP_JSR ≈ 1.0 (from L5 head 7)")
print(f"  - After L6: AX OP_JSR ≈ 5.0 (from L6 head 5)")
print(f"  - After L6: PC TEMP[0] ≈ 5.0 (from L6 head 3)")
print(f"  - PC output byte 0 = 25 (if L6 FFN JSR override works)")

# Check HAS_SE
print(f"\n### Checking HAS_SE at PC marker:")
x_check = torch.tensor([context], dtype=torch.long, device=device)
x_check = model.embed(x_check)
print(f"  After embedding: HAS_SE = {x_check[0, pc_idx, BD.HAS_SE].item():.3f}")
