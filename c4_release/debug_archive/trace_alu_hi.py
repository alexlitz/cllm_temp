#!/usr/bin/env python3
"""Trace when ALU_HI[0] gets huge value at PC marker for LEA 0."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("Initializing model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# LEA 0
bytecode = [Opcode.LEA | (0 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
pos_9 = 9  # PC marker position
target_dim = BD.ALU_HI  # dim 376

print(f"Tracing ALU_HI[0] (dim {target_dim}) at pos 9 for LEA 0:")
print()

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    val = x[0, pos_9, target_dim].item()
    print(f"After embed: ALU_HI[0] = {val:.2f}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        val = x[0, pos_9, target_dim].item()
        if abs(val) > 100:
            print(f"After L{i}: ALU_HI[0] = {val:.2f} ***")
        elif abs(val) > 1:
            print(f"After L{i}: ALU_HI[0] = {val:.2f} *")
        else:
            print(f"After L{i}: ALU_HI[0] = {val:.2f}")

# Now trace for LEA 8
print()
print("=" * 60)
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)

print(f"Tracing ALU_HI[0] (dim {target_dim}) at pos 9 for LEA 8:")
print()

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    val = x[0, pos_9, target_dim].item()
    print(f"After embed: ALU_HI[0] = {val:.2f}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        val = x[0, pos_9, target_dim].item()
        if abs(val) > 100:
            print(f"After L{i}: ALU_HI[0] = {val:.2f} ***")
        elif abs(val) > 1:
            print(f"After L{i}: ALU_HI[0] = {val:.2f} *")
        else:
            print(f"After L{i}: ALU_HI[0] = {val:.2f}")
