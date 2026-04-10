#!/usr/bin/env python3
"""Debug blocks[10] attention vs FFN for OUTPUT_HI[1]."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

bytecode = [Opcode.JMP | (16 << 8)]

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4): tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(3): tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()

ctx = context + tokens[:2]  # PC marker + PC byte 0
pc_byte0_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x)

    print(f"Before blocks[10]:")
    print(f"  OUTPUT_HI[0] = {x[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item():.4f}")
    print(f"  OUTPUT_HI[1] = {x[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item():.4f}")
    print(f"  HAS_SE = {x[0, pc_byte0_pos, BD.HAS_SE].item():.4f}")
    print(f"  IS_BYTE = {x[0, pc_byte0_pos, BD.IS_BYTE].item():.4f}")

    block10 = model.blocks[10]
    x_pre = x.clone()

    # Attention (includes residual)
    x_after_attn = block10.attn(x_pre)
    attn_delta0 = x_after_attn[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item() - x_pre[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item()
    attn_delta1 = x_after_attn[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item() - x_pre[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item()

    print(f"\nAfter blocks[10].attn:")
    print(f"  OUTPUT_HI[0] delta = {attn_delta0:.4f}")
    print(f"  OUTPUT_HI[1] delta = {attn_delta1:.4f}")

    # FFN (includes residual)
    x_after_ffn = block10.ffn(x_after_attn)
    ffn_delta0 = x_after_ffn[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item() - x_after_attn[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item()
    ffn_delta1 = x_after_ffn[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item() - x_after_attn[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item()

    print(f"\nAfter blocks[10].ffn:")
    print(f"  OUTPUT_HI[0] delta = {ffn_delta0:.4f}")
    print(f"  OUTPUT_HI[1] delta = {ffn_delta1:.4f}")

    # post_ops
    x_after_post = x_after_ffn.clone()
    for op in block10.post_ops:
        x_after_post = op(x_after_post)

    post_delta0 = x_after_post[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item() - x_after_ffn[0, pc_byte0_pos, BD.OUTPUT_HI + 0].item()
    post_delta1 = x_after_post[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item() - x_after_ffn[0, pc_byte0_pos, BD.OUTPUT_HI + 1].item()

    print(f"\nAfter blocks[10].post_ops:")
    print(f"  OUTPUT_HI[0] delta = {post_delta0:.4f}")
    print(f"  OUTPUT_HI[1] delta = {post_delta1:.4f}")
