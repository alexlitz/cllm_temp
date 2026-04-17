#!/usr/bin/env python3
"""Simple JSR debug - just check PC byte output."""
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

# Full forward pass
x = model.embed(token_ids)
for i in range(len(model.blocks)):
    x_before = x[0, -1, BD.TEMP + 0].item() if i == 6 else None
    x = model.blocks[i](x, kv_cache=None)
    if i == 5:
        print(f"After Layer 5: TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")
    if i == 6:
        print(f"After Layer 6: TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")

logits = model.head(x)
pred = logits[0, -1, :].argmax(-1).item()

print(f"\nPredicted PC byte 0: {pred} (expected 16)")
print(f"Match: {'✓' if pred == 16 else '✗'}")

# Check OUTPUT_LO to see what the model computed
print(f"\nOUTPUT_LO nibbles:")
for k in range(8):
    val = x[0, -1, BD.OUTPUT_LO + k].item()
    print(f"  [{k}] = {val:.4f}")
