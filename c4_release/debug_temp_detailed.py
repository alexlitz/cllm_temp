#!/usr/bin/env python3
"""Detailed TEMP[0] debugging through Layer 6."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

# JSR 16
context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END, Token.REG_PC]
token_ids = torch.tensor([context], dtype=torch.long)

# Run through layers 0-5
x = model.embed(token_ids)
for i in range(6):
    x = model.blocks[i](x, kv_cache=None)

print(f"BEFORE Layer 6 (input): TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.6f}")

# Manually run Layer 6 attention
block6 = model.blocks[6]
attn_out = block6.attn(x, kv_cache=None)

print(f"After attention (WITH residual): TEMP[0] = {attn_out[0, -1, BD.TEMP + 0].item():.6f}")

# Manually run Layer 6 FFN
ffn_out = block6.ffn(attn_out)

print(f"After FFN (WITH residual): TEMP[0] = {ffn_out[0, -1, BD.TEMP + 0].item():.6f}")

# Compare with full block call
x_reset = model.embed(token_ids)
for i in range(6):
    x_reset = model.blocks[i](x_reset, kv_cache=None)

x_full = model.blocks[6](x_reset, kv_cache=None)
print(f"Full block call: TEMP[0] = {x_full[0, -1, BD.TEMP + 0].item():.6f}")

print(f"\nMatch: {torch.allclose(ffn_out, x_full, atol=1e-5)}")
