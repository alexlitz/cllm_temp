#!/usr/bin/env python3
"""Debug NOP - why does it predict AX=0xDE?"""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

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

bytecode = [Opcode.NOP]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

print(f"NOP opcode value: {Opcode.NOP}")
print(f"Context: {context}")
print(f"Draft step 1 tokens: {draft_tokens[:15]}...")

# Predict tokens one by one
predicted = []
current_ctx = context[:]

print("\nPredicting step 1 tokens:")
with torch.no_grad():
    for i in range(39):
        token_ids = torch.tensor([current_ctx], dtype=torch.long)
        logits = model(token_ids)
        next_logits = logits[0, -1, :]
        pred = next_logits.argmax().item()
        predicted.append(pred)
        current_ctx.append(pred)

        expected = draft_tokens[i] if i < len(draft_tokens) else "?"
        match = "✓" if pred == expected else "✗"
        if i < 15 or pred != expected:
            print(f"  [{i}] predicted={pred}, expected={expected} {match}")

# Show AX bytes
ax_bytes = predicted[6:10]
predicted_ax = ax_bytes[0] | (ax_bytes[1] << 8) | (ax_bytes[2] << 16) | (ax_bytes[3] << 24)
print(f"\nAX bytes: {ax_bytes}")
print(f"Predicted AX: 0x{predicted_ax:08X}")
print(f"Expected AX: 0x{draft.ax:08X}")
