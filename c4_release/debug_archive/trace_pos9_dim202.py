#!/usr/bin/env python3
"""Trace when dim 202 gets huge value at pos 9 for LEA 0."""

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
print(f"Context: {context}")
print(f"Opcode.LEA = {Opcode.LEA}")

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
target_dim = BD.OUTPUT_HI + 12  # dim 202

# Position 9 = context position 9 = first draft token
pos_9 = 9
print(f"\nPosition 9 token: {(context + draft_tokens)[9]} (should be 257=REG_PC)")
print(f"Tracing dim {target_dim} at pos {pos_9}")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    val = x[0, pos_9, target_dim].item()
    print(f"\nAfter embed: dim {target_dim} = {val:.2f}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        val = x[0, pos_9, target_dim].item()
        if abs(val) > 100:
            print(f"After L{i}: dim {target_dim} = {val:.2f} ***")
        elif abs(val) > 1:
            print(f"After L{i}: dim {target_dim} = {val:.2f} *")
        else:
            print(f"After L{i}: dim {target_dim} = {val:.2f}")

# Also check what's special about this - is it a computation or relay?
print(f"\nDim {target_dim} = OUTPUT_HI[12]")
print(f"What's at this dim in embeddings?")
print(f"  CLEAN_EMBED_HI = {BD.CLEAN_EMBED_HI} (so CLEAN_EMBED_HI[12] = {BD.CLEAN_EMBED_HI + 12})")
print(f"  If CLEAN_EMBED_HI + 12 overlaps with something...")
