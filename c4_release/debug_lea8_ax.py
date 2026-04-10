#!/usr/bin/env python3
"""Debug LEA 8 AX byte 0 prediction."""
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

print(f"Draft state: PC={draft.pc}, AX={draft.ax} (0x{draft.ax:08X}), BP={draft.bp}")
print(f"Step 1 tokens: {step1_tokens[:15]}")
print(f"Token 6 (AX byte 0) expected: {step1_tokens[6]}")

# Predict at AX marker position (token 5 in step1_tokens)
ctx = context + step1_tokens[:6]  # Up to and including AX marker
ax_marker_pos = len(ctx) - 1

print(f"\nContext ends with: {ctx[-10:]}")
print(f"AX marker pos: {ax_marker_pos}, token = {ctx[ax_marker_pos]}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)
    pred = logits[0, -1, :].argmax().item()
    exp = step1_tokens[6]

    print(f"\nPredicting AX byte 0:")
    print(f"  Expected: {exp}")
    print(f"  Got: {pred}")
    print(f"  Match: {pred == exp}")

    top5 = torch.topk(logits[0, -1, :], 5)
    print(f"  Top 5: {[(idx.item(), f'{val.item():.2f}') for idx, val in zip(top5.indices, top5.values)]}")

    # Check intermediate values at AX marker
    x = model.embed(token_ids)
    for i in range(8):
        x = model.blocks[i](x)

    print(f"\nBefore blocks[8] at AX marker:")
    print(f"  MARK_AX: {x[0, ax_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"  OP_LEA: {x[0, ax_marker_pos, BD.OP_LEA].item():.4f}")

    print(f"\n  ALU_LO values (for BP lo nibble = 0):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"\n  FETCH_LO values (for immediate lo = 8):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")
