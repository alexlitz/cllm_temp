#!/usr/bin/env python3
"""Debug where OP_AND gets set at PC byte 0 position."""
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

ctx = context + tokens[:2]
pc_byte0_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    print(f"After embed:")
    print(f"  OP_AND: {x[0, pc_byte0_pos, BD.OP_AND].item():.4f}")
    print(f"  OP_JMP: {x[0, pc_byte0_pos, BD.OP_JMP].item():.4f}")

    for i in range(16):
        x_before = x.clone()
        x = model.blocks[i](x)

        delta_and = x[0, pc_byte0_pos, BD.OP_AND].item() - x_before[0, pc_byte0_pos, BD.OP_AND].item()
        delta_jmp = x[0, pc_byte0_pos, BD.OP_JMP].item() - x_before[0, pc_byte0_pos, BD.OP_JMP].item()

        if abs(delta_and) > 0.01 or abs(delta_jmp) > 0.01:
            print(f"blocks[{i:2d}]: OP_AND += {delta_and:.4f}, OP_JMP += {delta_jmp:.4f}")
            print(f"          totals: OP_AND = {x[0, pc_byte0_pos, BD.OP_AND].item():.4f}, "
                  f"OP_JMP = {x[0, pc_byte0_pos, BD.OP_JMP].item():.4f}")

    print(f"\nFinal:")
    print(f"  OP_AND: {x[0, pc_byte0_pos, BD.OP_AND].item():.4f}")
    print(f"  OP_JMP: {x[0, pc_byte0_pos, BD.OP_JMP].item():.4f}")
