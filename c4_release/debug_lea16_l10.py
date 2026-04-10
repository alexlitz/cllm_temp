#!/usr/bin/env python3
"""Debug L10 OUTPUT_LO[0] for LEA 16 at PC marker."""
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

S = 100

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

    print(f"At PC marker (pos {pc_marker_pos}) before blocks[10]:")
    print(f"  MARK_AX = {x[0, pc_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"  OP_LEA = {x[0, pc_marker_pos, BD.OP_LEA].item():.4f}")
    print(f"  OUTPUT_LO[0] = {x[0, pc_marker_pos, BD.OUTPUT_LO + 0].item():.4f}")
    print(f"  OUTPUT_LO[10] = {x[0, pc_marker_pos, BD.OUTPUT_LO + 10].item():.4f}")

    # Run blocks[10] attention
    x_before_attn = x.clone()
    x_after_attn = model.blocks[10].attn(x)

    delta_attn = x_after_attn - x_before_attn
    print(f"\nAfter blocks[10].attn:")
    for k in range(16):
        d = delta_attn[0, pc_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(d) > 0.5:
            print(f"  OUTPUT_LO[{k}] delta = {d:.4f}")

    # Run blocks[10] FFN
    x_after_ffn = model.blocks[10](x)

    delta_ffn = x_after_ffn - x_after_attn
    print(f"\nAfter blocks[10].ffn:")
    for k in range(16):
        d = delta_ffn[0, pc_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(d) > 0.5:
            print(f"  OUTPUT_LO[{k}] delta = {d:.4f}")

    # Check ALU_LO and AX_CARRY_LO values
    print(f"\nALU_LO values:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"  ALU_LO[{k}] = {v:.4f}")

    print(f"\nAX_CARRY_LO values:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item()
        if abs(v) > 0.5:
            print(f"  AX_CARRY_LO[{k}] = {v:.4f}")
