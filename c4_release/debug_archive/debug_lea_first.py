#!/usr/bin/env python3
"""Debug LEA 0 on first step - check ALU and OUTPUT."""
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

bytecode = [Opcode.LEA | (0 << 8)]  # LEA 0
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

# Context up to AX marker
current_ctx = context + draft_tokens[:6]

token_ids = torch.tensor([current_ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(6):
        x = model.blocks[i](x)

    print("After L5:")
    print(f"  OP_LEA at AX marker: {x[0, -1, BD.OP_LEA].item():.2f}")
    print(f"  MARK_AX at AX marker: {x[0, -1, BD.MARK_AX].item():.2f}")
    print(f"  HAS_SE at AX marker: {x[0, -1, BD.HAS_SE].item():.2f}")

    x = model.blocks[6](x)

    print("\nAfter L6 at AX marker:")
    print("  ALU_LO (non-zero):")
    for k in range(16):
        v = x[0, -1, BD.ALU_LO + k].item()
        if abs(v) > 0.1:
            print(f"    [{k}] = {v:.2f}")
    print("  ALU_HI (non-zero):")
    for k in range(16):
        v = x[0, -1, BD.ALU_HI + k].item()
        if abs(v) > 0.1:
            print(f"    [{k}] = {v:.2f}")

    # Run through remaining layers
    for i in range(7, 17):
        x = model.blocks[i](x)

    print("\nAfter all layers at AX marker:")
    print("  OUTPUT_LO (non-zero):")
    for k in range(16):
        v = x[0, -1, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.1:
            print(f"    [{k}] = {v:.2f}")
    print("  OUTPUT_HI (non-zero):")
    for k in range(16):
        v = x[0, -1, BD.OUTPUT_HI + k].item()
        if abs(v) > 0.1:
            print(f"    [{k}] = {v:.2f}")
