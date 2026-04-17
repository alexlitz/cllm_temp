#!/usr/bin/env python3
"""Check FETCH after Layer 5."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END, Token.REG_PC]
token_ids = torch.tensor([context], dtype=torch.long)

# Run through Layer 5
x = model.embed(token_ids)
for i in range(6):  # Layers 0-5
    x = model.blocks[i](x, kv_cache=None)

print(f"After Layer 5, at PC marker (pos -1):")
print(f"  TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  FETCH_LO non-zero:")
for k in range(16):
    val = x[0, -1, BD.FETCH_LO + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")
print(f"  FETCH_HI non-zero:")
for k in range(16):
    val = x[0, -1, BD.FETCH_HI + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

# Check position 1-2 where JSR instruction is
print(f"\nAt JSR opcode position (pos 1):")
print(f"  Token: {context[1]}")
for k in range(16):
    val = x[0, 1, BD.FETCH_LO + k].item()
    if abs(val) > 0.1:
        print(f"  FETCH_LO[{k}] = {val:.4f}")
for k in range(16):
    val = x[0, 1, BD.FETCH_HI + k].item()
    if abs(val) > 0.1:
        print(f"  FETCH_HI[{k}] = {val:.4f}")
