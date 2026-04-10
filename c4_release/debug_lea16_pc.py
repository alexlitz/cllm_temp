#!/usr/bin/env python3
"""Debug LEA 16 PC byte 0 prediction."""
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

print(f"Bytecode: LEA 16")
print(f"Context: {context}")
print(f"Draft state: PC={draft.pc}, AX={draft.ax}")
print(f"Step 1 tokens: {step1_tokens[:10]}")
print(f"Token 1 (PC byte 0) expected: {step1_tokens[1]}")

# Predict at PC marker position
ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

print(f"\nPredicting at PC marker (pos {pc_marker_pos}):")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)
    pred = logits[0, -1, :].argmax().item()
    exp = step1_tokens[1]

    print(f"  Expected: {exp}")
    print(f"  Got: {pred}")
    print(f"  Match: {pred == exp}")

    top5 = torch.topk(logits[0, -1, :], 5)
    print(f"  Top 5: {[(idx.item(), f'{val.item():.2f}') for idx, val in zip(top5.indices, top5.values)]}")

    # Check OUTPUT_LO at PC marker
    x = model.embed(token_ids)
    for i in range(16):
        x = model.blocks[i](x)

    print(f"\nFinal OUTPUT_LO at PC marker:")
    output_lo = [(k, x[0, pc_marker_pos, BD.OUTPUT_LO + k].item()) for k in range(16)]
    output_lo.sort(key=lambda x: -x[1])
    for k, v in output_lo[:5]:
        print(f"  OUTPUT_LO[{k}] = {v:.4f}")

    print(f"\nFinal OUTPUT_HI at PC marker:")
    output_hi = [(k, x[0, pc_marker_pos, BD.OUTPUT_HI + k].item()) for k in range(16)]
    output_hi.sort(key=lambda x: -x[1])
    for k, v in output_hi[:5]:
        print(f"  OUTPUT_HI[{k}] = {v:.4f}")
