#!/usr/bin/env python3
"""Check CARRY flag at AX marker for LEA 8."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.LEA | (8 << 8)]

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

ctx = context + step1_tokens[:6]  # Up to AX marker
ax_marker_pos = len(ctx) - 1

print(f"Expected: no carry (0 + 8 = 8 < 16)")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    print(f"\nTracing CARRY through layers:")
    for i in range(10):
        x_before = x.clone()
        x = model.blocks[i](x)

        carry_val = x[0, ax_marker_pos, BD.CARRY].item()
        carry_delta = carry_val - x_before[0, ax_marker_pos, BD.CARRY].item()
        if abs(carry_delta) > 0.1:
            print(f"blocks[{i}]: CARRY = {carry_val:.4f} (delta = {carry_delta:.4f})")

    print(f"\nBefore blocks[9], at AX marker:")
    print(f"  CARRY = {x[0, ax_marker_pos, BD.CARRY].item():.4f}")
    print(f"  MARK_AX = {x[0, ax_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"  ALU_HI[0] = {x[0, ax_marker_pos, BD.ALU_HI + 0].item():.4f}")
    print(f"  FETCH_HI[0] = {x[0, ax_marker_pos, BD.FETCH_HI + 0].item():.4f}")

    # Check what L9 does
    x_before_l9 = x.clone()
    x = model.blocks[9](x)

    print(f"\nAfter blocks[9]:")
    print(f"  OUTPUT_HI[0] delta = {(x[0, ax_marker_pos, BD.OUTPUT_HI + 0] - x_before_l9[0, ax_marker_pos, BD.OUTPUT_HI + 0]).item():.4f}")
    print(f"  OUTPUT_HI[1] delta = {(x[0, ax_marker_pos, BD.OUTPUT_HI + 1] - x_before_l9[0, ax_marker_pos, BD.OUTPUT_HI + 1]).item():.4f}")
