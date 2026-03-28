#!/usr/bin/env python3
"""Check what immediate-related dimensions are available at PC marker."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

# JSR 16: immediate = 16 = 0x10
# Expected: LO nibble = 0, HI nibble = 1
context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END, Token.REG_PC]
token_ids = torch.tensor([context], dtype=torch.long)

# Check after Layer 5
x = model.embed(token_ids)
for i in range(6):
    x = model.blocks[i](x, kv_cache=None)

print(f"At PC marker after Layer 5:")
print(f"\n  OPCODE_BYTE_LO:")
for k in range(16):
    val = x[0, -1, BD.OPCODE_BYTE_LO + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  OPCODE_BYTE_HI:")
for k in range(16):
    val = x[0, -1, BD.OPCODE_BYTE_HI + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  CLEAN_EMBED_LO:")
for k in range(16):
    val = x[0, -1, BD.CLEAN_EMBED_LO + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  CLEAN_EMBED_HI:")
for k in range(16):
    val = x[0, -1, BD.CLEAN_EMBED_HI + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  EMBED_LO:")
for k in range(16):
    val = x[0, -1, BD.EMBED_LO + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

print(f"\n  EMBED_HI:")
for k in range(16):
    val = x[0, -1, BD.EMBED_HI + k].item()
    if abs(val) > 0.1:
        print(f"    [{k}] = {val:.4f}")

# Target=16 means we need EMBED_LO=0 (nibble 0) and EMBED_HI=1 (nibble 1)
print(f"\n  Expected for target=16: LO nibble=0, HI nibble=1")
