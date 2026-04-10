#!/usr/bin/env python3
"""Debug which part of blocks[8] writes OUTPUT_LO[8] at PC marker for LEA."""
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

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

print(f"Context: {ctx}")
print(f"PC marker pos: {pc_marker_pos}")
print(f"Expected PC: {draft.pc} (byte 0 = {draft.pc & 0xFF})")
print(f"Immediate: 8")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    # Run through blocks[0..7]
    for i in range(8):
        x = model.blocks[i](x)

    print(f"\nBefore blocks[8]:")
    print(f"  OUTPUT_LO values at PC marker:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.5:
            print(f"    OUTPUT_LO[{k}] = {v:.4f}")

    # Run blocks[8] attention
    x_before_attn = x.clone()
    x_after_attn = model.blocks[8].attn(x)

    print(f"\nAfter blocks[8].attn:")
    delta_attn = x_after_attn - x_before_attn
    for k in range(16):
        d = delta_attn[0, pc_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(d) > 0.5:
            print(f"  OUTPUT_LO[{k}] delta = {d:.4f}")

    # Run blocks[8] FFN
    x_before_ffn = x_after_attn.clone()
    x_after_ffn = model.blocks[8](x)

    print(f"\nAfter blocks[8].ffn (full block):")
    delta_ffn = x_after_ffn - x_after_attn
    for k in range(16):
        d = delta_ffn[0, pc_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(d) > 0.5:
            print(f"  OUTPUT_LO[{k}] delta = {d:.4f}")

    print(f"\nFinal values after blocks[8]:")
    for k in range(16):
        v = x_after_ffn[0, pc_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.5:
            print(f"  OUTPUT_LO[{k}] = {v:.4f}")

    # Also check what's at the opcode position in code prefix
    opcode_pos = 1  # CODE_START is 0, opcode is 1
    print(f"\nAt opcode position {opcode_pos}:")
    print(f"  OP_LEA: {x[0, opcode_pos, BD.OP_LEA].item():.4f}")
    print(f"  FETCH_LO values:")
    for k in range(16):
        v = x[0, opcode_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")
