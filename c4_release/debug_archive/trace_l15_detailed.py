#!/usr/bin/env python3
"""Trace L15 attention vs FFN contribution to OUTPUT_HI[12]."""

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
ctx_len = len(context)
pos_6 = ctx_len + 6  # AX byte 0 position
target_dim = BD.OUTPUT_HI + 12  # dim 202

print(f"Tracing OUTPUT_HI[12] (dim {target_dim}) at pos {pos_6}")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    # Run through first 14 layers
    for i in range(15):
        x = model.blocks[i](x)

    print(f"\nAfter L14: OUTPUT_HI[12] = {x[0, pos_6, target_dim].item():.2f}")

    # L15 block breakdown
    block15 = model.blocks[15]

    # L15 attention
    x_before_attn = x.clone()
    attn_out = block15.attn(x)
    x_after_attn = x + attn_out  # residual
    print(f"After L15 attn: OUTPUT_HI[12] = {x_after_attn[0, pos_6, target_dim].item():.2f}")
    print(f"  L15 attn added: {attn_out[0, pos_6, target_dim].item():.2f}")

    # L15 FFN
    x_before_ffn = x_after_attn.clone()
    ffn_out = block15.ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out  # residual
    print(f"After L15 ffn: OUTPUT_HI[12] = {x_after_ffn[0, pos_6, target_dim].item():.2f}")
    print(f"  L15 ffn added: {ffn_out[0, pos_6, target_dim].item():.2f}")

    # L16
    x_after_l16 = model.blocks[16](x_after_ffn)
    print(f"\nAfter L16: OUTPUT_HI[12] = {x_after_l16[0, pos_6, target_dim].item():.2f}")

    # Check L15 attention W_o for dim 202
    print(f"\n--- L15 attention W_o analysis for dim {target_dim} ---")
    attn15 = block15.attn
    NH = attn15.num_heads
    HD = 512 // NH  # 64

    for h in range(NH):
        base = h * HD
        wo_sum = 0
        for k in range(HD):
            wo_sum += abs(attn15.W_o[target_dim, base + k].item())
        if wo_sum > 0.01:
            print(f"  Head {h}: total |W_o| to dim {target_dim} = {wo_sum:.4f}")
            # Show specific non-zero entries
            for k in range(HD):
                w = attn15.W_o[target_dim, base + k].item()
                if abs(w) > 0.001:
                    print(f"    W_o[{target_dim}, {base}+{k}] = {w:.4f}")
