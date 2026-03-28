#!/usr/bin/env python3
"""Debug JSR relay - check if TEMP[0] has IS_JSR flag at PC marker."""
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

# Run through model layer by layer
x = model.embed(token_ids)

# Run through layers 0-5
for i in range(6):
    x = model.blocks[i](x, kv_cache=None)

print(f"After Layer 5 (before Layer 6 attention):")
print(f"  At AX marker (pos {len(context)-1}): OP_JSR = {x[0, -1, BD.OP_JSR].item():.4f}")
print(f"  At AX marker: TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")

# Run Layer 6 attention only
attn_out = model.blocks[6].attn(model.blocks[6].norm1(x))
x_after_attn = x + attn_out

print(f"\nAfter Layer 6 attention:")
print(f"  At PC marker (pos {len(context)-1}): TEMP[0] = {x_after_attn[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  At PC marker: OP_JSR = {x_after_attn[0, -1, BD.OP_JSR].item():.4f}")
print(f"  At PC marker: MARK_PC = {x_after_attn[0, -1, BD.MARK_PC].item():.4f}")
print(f"  At PC marker: MARK_AX = {x_after_attn[0, -1, BD.MARK_AX].item():.4f}")

# Run Layer 6 FFN
x_after_l6 = model.blocks[6](x, kv_cache=None)
print(f"\nAfter Layer 6 FFN:")
print(f"  At PC marker: OUTPUT_LO[0-3] = {[x_after_l6[0, -1, BD.OUTPUT_LO + k].item() for k in range(4)]}")
print(f"  Expected PC=16 → OUTPUT_LO = [0, 1, 0, 0, ...]")
