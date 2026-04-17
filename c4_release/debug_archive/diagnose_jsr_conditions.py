#!/usr/bin/env python3
"""Check if JSR decode conditions are met after L5."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

bytecode = [Opcode.JSR | (26 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")

# Manually run through layers to inspect after L5
device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

print("="*60)
print("AFTER EMBEDDING")
print("="*60)
print(f"Context length: {len(context)}")
print(f"Context: {context}")

# Run through blocks
for i in range(6):  # Through layer 5 (blocks 0-5)
    x = model.blocks[i](x)

print("\n" + "="*60)
print("AFTER LAYER 5 (blocks[5])")
print("="*60)

# Check all positions for potential PC marker
print("\nSearching for PC marker indicators...")
for pos in range(x.shape[1]):
    mark_pc = x[0, pos, BD.MARK_PC].item()
    has_se = x[0, pos, BD.HAS_SE].item()
    temp0 = x[0, pos, BD.TEMP + 0].item()

    # Check OPCODE_BYTE at this position
    opcode_lo = sum([k for k in range(16) if x[0, pos, BD.OPCODE_BYTE_LO + k].item() > 0.5])
    opcode_hi = sum([k for k in range(16) if x[0, pos, BD.OPCODE_BYTE_HI + k].item() > 0.5])

    # Check FETCH at this position
    fetch_b0 = sum([k for k in range(16) if x[0, pos, BD.FETCH_LO + k].item() > 0.5])

    if mark_pc > 0.5 or temp0 > 0.5 or opcode_lo > 0 or fetch_b0 > 0:
        print(f"\nPosition {pos}: token={context[pos] if pos < len(context) else '(generated)'}")
        print(f"  MARK_PC: {mark_pc:.2f}")
        print(f"  HAS_SE: {has_se:.2f}")
        print(f"  TEMP[0]: {temp0:.2f}")
        print(f"  OPCODE_BYTE: lo={opcode_lo}, hi={opcode_hi}")
        print(f"  FETCH byte 0: {fetch_b0}")

        if opcode_lo == 3 and opcode_hi == 0:
            print(f"  → JSR opcode detected!")
        if temp0 > 4.0:
            print(f"  → TEMP[0] > 4.0, JSR should activate!")

print("\n" + "="*60)
print("EXPECTED for JSR to work:")
print("  - PC marker should have MARK_PC ≈ 1.0, HAS_SE ≈ 0.0")
print("  - OPCODE_BYTE should be 3,0 (JSR)")
print("  - TEMP[0] should be ≈ 5.0 (from L5 FFN decode)")
print("  - FETCH should have 26 (jump target)")
print("="*60)
