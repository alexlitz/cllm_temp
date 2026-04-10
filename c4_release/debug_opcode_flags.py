#!/usr/bin/env python3
"""Debug opcode flags at PC byte 0 position."""
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
    for i in range(10):
        x = model.blocks[i](x)
    x = model.blocks[10].attn(x)

    print(f"At position {pc_byte0_pos} (PC byte 0):")
    print(f"\nOpcode flags:")
    print(f"  OP_JMP: {x[0, pc_byte0_pos, BD.OP_JMP].item():.4f}")
    print(f"  OP_AND: {x[0, pc_byte0_pos, BD.OP_AND].item():.4f}")
    print(f"  OP_ADD: {x[0, pc_byte0_pos, BD.OP_ADD].item():.4f}")
    print(f"  OP_MUL: {x[0, pc_byte0_pos, BD.OP_MUL].item():.4f}")

    print(f"\nALU staging:")
    print(f"  ALU_HI[1]: {x[0, pc_byte0_pos, BD.ALU_HI + 1].item():.4f}")
    print(f"  AX_CARRY_HI[1]: {x[0, pc_byte0_pos, BD.AX_CARRY_HI + 1].item():.4f}")

    print(f"\nMarkers:")
    print(f"  MARK_AX: {x[0, pc_byte0_pos, BD.MARK_AX].item():.4f}")
    print(f"  MARK_PC: {x[0, pc_byte0_pos, BD.MARK_PC].item():.4f}")
    print(f"  IS_BYTE: {x[0, pc_byte0_pos, BD.IS_BYTE].item():.4f}")
