#!/usr/bin/env python3
"""Debug NOP - trace OUTPUT_HI source."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode, data=b""):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

bytecode = [Opcode.NOP]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

# Build context up to AX marker
current_ctx = context + draft_tokens[:6]  # Through AX marker

token_ids = torch.tensor([current_ctx], dtype=torch.long)

print(f"Context: {current_ctx}")
print(f"Last token (AX marker): {current_ctx[-1]}")

with torch.no_grad():
    x = model.embed(token_ids)

    # Track OUTPUT_HI after each layer
    print("\nOUTPUT_HI[8] and OUTPUT_HI[13] at AX marker after each layer:")
    for i, block in enumerate(model.blocks):
        x = block(x)
        o8 = x[0, -1, BD.OUTPUT_HI + 8].item()
        o13 = x[0, -1, BD.OUTPUT_HI + 13].item()
        if abs(o8) > 0.1 or abs(o13) > 0.1:
            print(f"  L{i}: OUTPUT_HI[8]={o8:.2f}, OUTPUT_HI[13]={o13:.2f}")

    # Check all OUTPUT_LO/HI after all layers
    print("\nFinal OUTPUT values (non-zero):")
    print("OUTPUT_LO:")
    for k in range(16):
        val = x[0, -1, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.1:
            print(f"  [{k}] = {val:.2f}")
    print("OUTPUT_HI:")
    for k in range(16):
        val = x[0, -1, BD.OUTPUT_HI + k].item()
        if abs(val) > 0.1:
            print(f"  [{k}] = {val:.2f}")

# Check what OUTPUT_LO/HI should be for NOP AX byte 0 = 0
# AX = 0 = 0x00 = lo nibble 0, hi nibble 0
print(f"\nExpected: OUTPUT_LO[0] and OUTPUT_HI[0] should be high")
print(f"For NOP, AX stays at 0, so byte 0 = 0")
