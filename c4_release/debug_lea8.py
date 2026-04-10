#\!/usr/bin/env python3
"""Debug LEA 8 PC prediction."""
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

print(f"Step 1 expected: {step1_tokens[:10]}")
print(f"Draft PC: {draft.pc}, Draft AX: {draft.ax}")
print(f"Expected PC byte 0: {step1_tokens[1]} (PC = {draft.pc})")

with torch.no_grad():
    # Test PC byte 0 prediction (position after PC marker)
    ctx = context + step1_tokens[:1]  # Just PC marker
    ids = torch.tensor([ctx], dtype=torch.long)
    logits = model(ids)
    pred = logits[0, -1, :].argmax().item()
    exp = step1_tokens[1]
    print(f"\nPredicting PC byte 0:")
    print(f"  Expected: {exp}")
    print(f"  Got: {pred}")
    print(f"  Match: {pred == exp}")

    # Check top 5
    top5 = torch.topk(logits[0, -1, :], 5)
    print(f"  Top 5: {[(idx.item(), val.item()) for idx, val in zip(top5.indices, top5.values)]}")
