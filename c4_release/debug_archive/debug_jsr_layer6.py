#!/usr/bin/env python3
"""Debug Layer 6 attention and FFN effects on TEMP[0]."""
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

print(f"After Layer 5:")
print(f"  PC marker TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  PC marker OP_JSR = {x[0, -1, BD.OP_JSR].item():.4f}")

# Layer 6 block
block6 = model.blocks[6]

# Attention only
x_norm = block6.norm_attn(x)
attn_out = block6.attn(x_norm, kv_cache=None)
x_after_attn = x + attn_out

print(f"\nAfter Layer 6 attention:")
print(f"  PC marker TEMP[0] (input) = {x[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  PC marker TEMP[0] (attn output) = {attn_out[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  PC marker TEMP[0] (residual) = {x_after_attn[0, -1, BD.TEMP + 0].item():.4f}")

# FFN only
x_norm2 = block6.norm_ffn(x_after_attn)
ffn_out = block6.ffn(x_norm2)
x_after_ffn = x_after_attn + ffn_out

print(f"\nAfter Layer 6 FFN:")
print(f"  PC marker TEMP[0] (input) = {x_after_attn[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  PC marker TEMP[0] (ffn output) = {ffn_out[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  PC marker TEMP[0] (residual) = {x_after_ffn[0, -1, BD.TEMP + 0].item():.4f}")

print(f"\nFinal OUTPUT_LO:")
for k in range(8):
    print(f"  [{k}] = {x_after_ffn[0, -1, BD.OUTPUT_LO + k].item():.4f}")
