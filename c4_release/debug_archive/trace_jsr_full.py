#!/usr/bin/env python3
"""Full trace of JSR execution through all layers."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

bytecode = [Opcode.JSR | (25 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")
device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

print("="*60)
print("INITIAL EMBEDDING (after embed)")
print("="*60)
print(f"Context length: {len(context)}")
print(f"Context tokens: {context}")

# For now, just analyze the CODE section (positions 0-10)
print("\nCODE section:")
for i in range(min(10, len(context))):
    token = context[i]
    op_jsr = x[0, i, BD.OP_JSR].item()
    if op_jsr > 0.5:
        print(f"  Position {i}: token={token}, OP_JSR={op_jsr:.1f}")

# PC marker is not in initial context - it's generated autoregressively
print("\nNote: PC marker will appear after first generation step")
print("="*60)
print("Skipping layer-by-layer trace (would need autoregressive generation)")
print("="*60)
return  # Exit early

pc_idx = 36  # PC marker should be here (5 tokens per section * 7 + 1)
print(f"PC marker at position {pc_idx}:")
print(f"  MARK_PC: {x[0, pc_idx, BD.MARK_PC].item():.1f}")
print(f"  HAS_SE: {x[0, pc_idx, BD.HAS_SE].item():.1f}")
print(f"  OP_JSR: {x[0, pc_idx, BD.OP_JSR].item():.1f}")
for k in range(4):
    lo = sum([j for j in range(16) if x[0, pc_idx, BD.FETCH_LO + k*16 + j].item() > 0.5])
    print(f"  FETCH byte {k}: {lo}")

# Run through layers
for i, block in enumerate(model.blocks):
    x = block(x)

    if i == 5:  # After Layer 5
        print(f"\n{'='*60}")
        print(f"AFTER LAYER {i+1}")
        print(f"{'='*60}")
        print(f"PC marker at position {pc_idx}:")
        print(f"  OP_JSR: {x[0, pc_idx, BD.OP_JSR].item():.3f}")
        for k in range(4):
            lo = sum([j for j in range(16) if x[0, pc_idx, BD.FETCH_LO + k*16 + j].item() > 0.5])
            print(f"  FETCH byte {k}: {lo}")

    elif i == 6:  # After Layer 6
        ax_idx = pc_idx + 5  # AX marker is 5 positions after PC
        print(f"\n{'='*60}")
        print(f"AFTER LAYER {i+1}")
        print(f"{'='*60}")
        print(f"AX marker at position {ax_idx}:")
        print(f"  OP_JSR: {x[0, ax_idx, BD.OP_JSR].item():.3f}")
        print(f"PC marker at position {pc_idx}:")
        print(f"  TEMP[0] (IS_JSR): {x[0, pc_idx, BD.TEMP + 0].item():.3f}")
        for k in range(4):
            lo = sum([j for j in range(16) if x[0, pc_idx, BD.FETCH_LO + k*16 + j].item() > 0.5])
            print(f"  FETCH byte {k}: {lo}")
        for k in range(4):
            lo = sum([j for j in range(16) if x[0, pc_idx, BD.OUTPUT_LO + k*16 + j].item() > 0.5])
            print(f"  OUTPUT byte {k}: {lo}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Expected behavior:")
print("  After L5: PC should have OP_JSR ≈ 1.0, FETCH should have 25")
print("  After L6: AX should have OP_JSR ≈ 5.0, PC TEMP[0] ≈ 5.0, PC OUTPUT should be 25")
