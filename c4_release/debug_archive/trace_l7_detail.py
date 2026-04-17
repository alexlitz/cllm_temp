#!/usr/bin/env python3
"""Trace L7 attention vs FFN for ALU_HI[0] at pos 9."""

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

print(f"Checking L7 components for ALU_HI[0] (dim {target_dim}) at pos 9 for LEA 0:")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    # Run through L0-L6
    for i in range(7):
        x = model.blocks[i](x)

    print(f"\nAfter L6: ALU_HI[0] = {x[0, pos_9, target_dim].item():.2f}")

    # L7 breakdown
    block7 = model.blocks[7]

    # L7 attention
    attn_out = block7.attn(x)
    x_after_attn = x + attn_out
    print(f"After L7 attn: ALU_HI[0] = {x_after_attn[0, pos_9, target_dim].item():.2f}")
    print(f"  L7 attn added: {attn_out[0, pos_9, target_dim].item():.2f}")

    # L7 FFN
    ffn_out = block7.ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out
    print(f"After L7 ffn: ALU_HI[0] = {x_after_ffn[0, pos_9, target_dim].item():.2f}")
    print(f"  L7 ffn added: {ffn_out[0, pos_9, target_dim].item():.2f}")

# Now check which L7 attention heads write to dim 376
print(f"\n--- L7 attention W_o analysis for dim {target_dim} ---")
attn7 = model.blocks[7].attn
NH = attn7.num_heads
HD = 512 // NH  # 64

for h in range(NH):
    base = h * HD
    wo_sum = 0
    for k in range(HD):
        wo_sum += abs(attn7.W_o[target_dim, base + k].item())
    if wo_sum > 0.01:
        print(f"  Head {h}: total |W_o| to dim {target_dim} = {wo_sum:.4f}")
