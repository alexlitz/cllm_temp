#!/usr/bin/env python3
"""Check opcode flags at PC marker for LEA 16."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.LEA | (16 << 8)]

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

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x)

    print(f"At PC marker (pos {pc_marker_pos}):")
    print(f"  OP_LEA = {x[0, pc_marker_pos, BD.OP_LEA].item():.4f}")
    print(f"  OP_AND = {x[0, pc_marker_pos, BD.OP_AND].item():.4f}")
    print(f"  OP_OR = {x[0, pc_marker_pos, BD.OP_OR].item():.4f}")
    print(f"  OP_XOR = {x[0, pc_marker_pos, BD.OP_XOR].item():.4f}")
    print(f"  OP_MUL = {x[0, pc_marker_pos, BD.OP_MUL].item():.4f}")

    # Check what Opcode.AND is
    print(f"\n  Opcode.AND = {Opcode.AND} (hex: 0x{Opcode.AND:02X})")
    print(f"  Token at PC marker: {ctx[pc_marker_pos]} (257 = PC marker)")

    # Check the immediate byte 0 value
    imm_byte0_pos = 2  # pos 1 is opcode, pos 2 is immediate byte 0
    print(f"\n  Immediate byte 0 at pos {imm_byte0_pos}: {ctx[imm_byte0_pos]} (16)")
    print(f"  Does 16 == Opcode.AND? {16 == Opcode.AND}")
