#!/usr/bin/env python3
"""Debug LEA 8 AX byte 2 prediction."""
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

print(f"Draft state: AX={draft.ax} (0x{draft.ax:08X})")
print(f"  AX byte 0: {draft.ax & 0xFF}")
print(f"  AX byte 1: {(draft.ax >> 8) & 0xFF}")
print(f"  AX byte 2: {(draft.ax >> 16) & 0xFF}")
print(f"  AX byte 3: {(draft.ax >> 24) & 0xFF}")
print(f"\nStep 1 tokens: {step1_tokens[:15]}")
print(f"  Token 8 (AX byte 2): expected = {step1_tokens[8]}")

# Predict at AX byte 1 position (after AX marker + byte 0)
# step1_tokens: [257=PC_MARK, 10, 0, 0, 0, 258=AX_MARK, 8, 0, 1, 0, ...]
#                0           1   2  3  4  5            6  7  8  9
ctx = context + step1_tokens[:8]  # Up to and including AX byte 1
byte2_pos = len(ctx) - 1

print(f"\nContext ends with: {ctx[-10:]}")
print(f"Predicting at pos {len(ctx)} after last token {ctx[-1]}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)
    pred = logits[0, -1, :].argmax().item()
    exp = step1_tokens[8]

    print(f"\nPredicting AX byte 2:")
    print(f"  Expected: {exp}")
    print(f"  Got: {pred}")
    print(f"  Match: {pred == exp}")

    top5 = torch.topk(logits[0, -1, :], 5)
    print(f"  Top 5: {[(idx.item(), f'{val.item():.2f}') for idx, val in zip(top5.indices, top5.values)]}")
