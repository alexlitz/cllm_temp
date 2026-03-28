#!/usr/bin/env python3
"""Debug JSR - check OUTPUT after Layer 6."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

# JSR 16 (target = 16 = 0x10 = 0b00010000)
# Expected OUTPUT_LO: nibble encoding of 16
# 16 in nibbles: LO nibble = 0 (0000), HI nibble = 1 (0001)
# So OUTPUT_LO should be one-hot at position 0 for LO, position 1 for HI

context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END, Token.REG_PC]
token_ids = torch.tensor([context], dtype=torch.long)

# Run full model
x = model.embed(token_ids)
for i in range(len(model.blocks)):
    x = model.blocks[i](x, kv_cache=None)

print(f"After all layers:")
print(f"  TEMP[0] = {x[0, -1, BD.TEMP + 0].item():.4f}")
print(f"  MARK_PC = {x[0, -1, BD.MARK_PC].item():.4f}")
print(f"  FETCH_LO nibbles:")
for k in range(16):
    val = x[0, -1, BD.FETCH_LO + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  FETCH_HI nibbles:")
for k in range(16):
    val = x[0, -1, BD.FETCH_HI + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  OUTPUT_LO nibbles:")
for k in range(16):
    val = x[0, -1, BD.OUTPUT_LO + k].item()
    print(f"    [{k}] = {val:.4f}")

print(f"\n  OUTPUT_HI nibbles:")
for k in range(16):
    val = x[0, -1, BD.OUTPUT_HI + k].item()
    print(f"    [{k}] = {val:.4f}")

# Expected for target=16: OUTPUT_LO one-hot at 0, OUTPUT_HI one-hot at 1
# Actual PC byte = sum of OUTPUT_LO[k] * 2^k (for nibbles forming byte)
pc_byte = sum(max(0, min(1, x[0, -1, BD.OUTPUT_LO + k].item())) * (2**k) for k in range(8))
print(f"\nDecoded PC byte 0: {pc_byte:.0f} (expected 16)")
