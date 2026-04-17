#!/usr/bin/env python3
"""Debug JSR relay - check TEMP[0] at PC marker in generated output."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD

# Create model
model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

# JSR 16
context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]
token_ids = torch.tensor([context], dtype=torch.long)

# Generate first token (REG_PC marker)
logits = model.forward(token_ids)
pred0 = logits[0, -1, :].argmax(-1).item()
print(f"First output token: {pred0} (REG_PC={Token.REG_PC})")

# Extend context and generate PC byte 0
context2 = context + [pred0]
token_ids2 = torch.tensor([context2], dtype=torch.long)

# Run through layers to inspect
x = model.embed(token_ids2)
print(f"\nAfter embedding:")
print(f"  Last position (REG_PC marker): MARK_PC = {x[0, -1, BD.MARK_PC].item():.4f}")
print(f"  Last position: MARK_AX = {x[0, -1, BD.MARK_AX].item():.4f}")

# Find AX marker position in this sequence
for i in range(len(context2)):
    if x[0, i, BD.MARK_AX].item() > 0.5:
        print(f"\n  AX marker at position {i}")
        print(f"    OP_JSR = {x[0, i, BD.OP_JSR].item():.4f}")
        break

# Run layers 0-5
for layer_idx in range(6):
    x = model.blocks[layer_idx](x, kv_cache=None)

print(f"\nAfter Layer 5 (before L6 attention):")
print(f"  PC marker (pos -1): OP_JSR = {x[0, -1, BD.OP_JSR].item():.4f}")
print(f"  PC marker: TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")

# Check AX marker
for i in range(len(context2)):
    if token_ids2[0, i].item() == Token.CODE_START + 1 or \
       (i > 0 and x[0, i, BD.MARK_AX].item() > 0.5):
        if x[0, i, BD.OP_JSR].item() > 1.0:
            print(f"  AX marker (pos {i}): OP_JSR = {x[0, i, BD.OP_JSR].item():.4f}")
            break

# Run Layer 6
x = model.blocks[6](x, kv_cache=None)

print(f"\nAfter Layer 6:")
print(f"  PC marker (pos -1): TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  PC marker: OUTPUT_LO[0-3] = {[x[0, -1, BD.OUTPUT_LO + k].item() for k in range(4)]}")
print(f"\nExpected for PC=16: OUTPUT_LO = [0, 1, 0, 0, ...]")
print(f"Actual PC byte 0: {x[0, -1, BD.OUTPUT_LO + 0].item() + x[0, -1, BD.OUTPUT_LO + 1].item() * 2 + x[0, -1, BD.OUTPUT_LO + 2].item() * 4 + x[0, -1, BD.OUTPUT_LO + 3].item() * 8:.0f}")
