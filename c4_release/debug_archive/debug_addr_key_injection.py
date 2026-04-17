#!/usr/bin/env python3
"""Debug ADDR_KEY injection."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import Token

# Check Token values
print(f'Token.CODE_START = {Token.CODE_START}')
print(f'Token.CODE_END = {Token.CODE_END}')

# Simulated context
context = [Token.CODE_START, 1, 42, 0, 0, 0, 0, 0, 0, Token.CODE_END]
print(f'\\nContext: {context}')

# Simulate injection logic
token_ids = torch.tensor([context], dtype=torch.long)
B, S = token_ids.shape
print(f'B={B}, S={S}')

for b in range(B):
    cs_pos = None
    for i in range(S):
        tok = token_ids[b, i].item()
        if tok == Token.CODE_START:
            cs_pos = i
            print(f'Found CODE_START at position {i}')
        elif tok == Token.CODE_END:
            print(f'Found CODE_END at position {i}')
            break
        elif cs_pos is not None and tok < 256:
            addr = i - cs_pos - 1
            print(f'Position {i} (token {tok}): addr = {i} - {cs_pos} - 1 = {addr}')
