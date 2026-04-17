#!/usr/bin/env python3
"""Debug LEA 0 byte 2 - should be 1 (from BP = 0x10000)."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
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

bytecode = [Opcode.LEA | (0 << 8)]  # LEA 0
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

print(f"LEA 0 draft tokens: {draft_tokens[:15]}")
print(f"Expected AX: 0x{draft.ax:08X}")
print(f"AX bytes: [{draft_tokens[6]}, {draft_tokens[7]}, {draft_tokens[8]}, {draft_tokens[9]}]")

# Context up to AX byte 2 position
# Step tokens: PC_marker(0), 4 PC bytes(1-4), AX_marker(5), 4 AX bytes(6-9)
# So byte 2 is at position 8 in step tokens
current_ctx = context + draft_tokens[:8]  # Up through AX byte 1

token_ids = torch.tensor([current_ctx], dtype=torch.long)

with torch.no_grad():
    # Run full model to get logits for byte 2
    logits = model(token_ids)
    byte2_logits = logits[0, -1, :]

    # Top predictions
    top_k = torch.topk(byte2_logits, 5)
    print(f"\nTop 5 predictions for AX byte 2:")
    for val, idx in zip(top_k.values, top_k.indices):
        print(f"  token {idx.item()}: {val.item():.2f}")

    print(f"\nExpected: token 1 (byte value 0x01)")
    print(f"Logit for token 1: {byte2_logits[1].item():.2f}")
    print(f"Logit for token 0: {byte2_logits[0].item():.2f}")

    # Check OUTPUT at byte 1 position (predicting byte 2)
    x = model.embed(token_ids)
    for block in model.blocks:
        x = block(x)

    print(f"\nOUTPUT at last position (predicting byte 2):")
    print("OUTPUT_LO (non-zero):")
    for k in range(16):
        v = x[0, -1, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.5:
            print(f"  [{k}] = {v:.2f}")
    print("OUTPUT_HI (non-zero):")
    for k in range(16):
        v = x[0, -1, BD.OUTPUT_HI + k].item()
        if abs(v) > 0.5:
            print(f"  [{k}] = {v:.2f}")
