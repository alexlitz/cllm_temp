#!/usr/bin/env python3
"""Trace L6 attention vs FFN for TEMP[0] at pos 9."""

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
pos_9 = 9
target_dim = BD.TEMP  # dim 480

print(f"Checking L6 for LEA 0:")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    # Run through L0-L5
    for i in range(6):
        x = model.blocks[i](x)

    print(f"\nBefore L6: TEMP[0] at pos {pos_9} = {x[0, pos_9, target_dim].item():.2f}")

    # L6 attention
    block6 = model.blocks[6]
    attn_out = block6.attn(x)
    x_after_attn = x + attn_out
    print(f"After L6 attn: TEMP[0] at pos {pos_9} = {x_after_attn[0, pos_9, target_dim].item():.2f}")
    print(f"  L6 attn added: {attn_out[0, pos_9, target_dim].item():.2f}")

    # L6 FFN
    ffn_out = block6.ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out
    print(f"After L6 ffn: TEMP[0] at pos {pos_9} = {x_after_ffn[0, pos_9, target_dim].item():.2f}")
    print(f"  L6 ffn added: {ffn_out[0, pos_9, target_dim].item():.2f}")

# Check TEMP range at different positions in embedding
print(f"\n\nChecking TEMP[0] values in embedding for code bytes:")
with torch.no_grad():
    emb = model.embed(ctx_tensor)
    for p in range(1, 9):  # code bytes and markers
        tok = (context + draft_tokens)[p]
        val = emb[0, p, target_dim].item()
        print(f"  pos {p}: token {tok}, TEMP[0] = {val:.2f}")
