#\!/usr/bin/env python3
"""Trace which layer sets BYTE_INDEX_0 at PC byte 1."""
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

ctx = context + tokens[:2]  # up to PC byte 1
pc_byte1_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    print(f"After embed: BYTE_INDEX_0 = {x[0,pc_byte1_pos,BD.BYTE_INDEX_0].item():.4f}")

    for i in range(16):
        x_before = x.clone()
        x = model.blocks[i](x)
        
        delta = x[0,pc_byte1_pos,BD.BYTE_INDEX_0].item() - x_before[0,pc_byte1_pos,BD.BYTE_INDEX_0].item()
        if abs(delta) > 0.01:
            print(f"blocks[{i:2d}]: BYTE_INDEX_0 += {delta:.4f} (total={x[0,pc_byte1_pos,BD.BYTE_INDEX_0].item():.4f})")

    print(f"\nFinal BYTE_INDEX_0 at PC byte 1: {x[0,pc_byte1_pos,BD.BYTE_INDEX_0].item():.4f}")
