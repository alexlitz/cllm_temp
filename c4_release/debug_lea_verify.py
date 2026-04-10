#!/usr/bin/env python3
"""Verify LEA fix - blocks[8] should no longer write OUTPUT_LO[8] at PC marker."""
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

print(f"Expected: PC={draft.pc}, AX={draft.ax}")
print(f"Expected PC byte 0: {step1_tokens[1]} (lo nibble: {step1_tokens[1] & 0xF})")

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    # Full forward pass
    logits = model(token_ids)
    pred = logits[0, -1, :].argmax().item()
    exp = step1_tokens[1]

    print(f"\nPredicting PC byte 0 after PC marker:")
    print(f"  Expected: {exp}")
    print(f"  Got: {pred}")
    print(f"  Match: {pred == exp}")

    # Show top 5 predictions
    top5 = torch.topk(logits[0, -1, :], 5)
    print(f"  Top 5: {[(idx.item(), f'{val.item():.2f}') for idx, val in zip(top5.indices, top5.values)]}")

    # Also check OUTPUT_LO values at PC marker
    x = model.embed(token_ids)
    for i in range(16):
        x = model.blocks[i](x)

    print(f"\nFinal OUTPUT_LO values at PC marker:")
    output_lo_vals = [(k, x[0, pc_marker_pos, BD.OUTPUT_LO + k].item()) for k in range(16)]
    output_lo_vals.sort(key=lambda x: -x[1])
    for k, v in output_lo_vals[:5]:
        print(f"  OUTPUT_LO[{k}] = {v:.4f}")
