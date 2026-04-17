#!/usr/bin/env python3
"""Debug NOP AX byte 0 logits."""
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

bytecode = [Opcode.NOP]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

# Build context up to AX marker
current_ctx = context + draft_tokens[:6]  # Through AX marker
print(f"Context up to AX marker: {current_ctx}")
print(f"Predicting AX byte 0...")

token_ids = torch.tensor([current_ctx], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)
    ax_logits = logits[0, -1, :]

# Top 10 predictions
top_k = torch.topk(ax_logits, 10)
print(f"\nTop 10 logits for AX byte 0:")
for i, (val, idx) in enumerate(zip(top_k.values, top_k.indices)):
    print(f"  {i+1}. token {idx.item()}: {val.item():.2f}")

# Check logit for correct answer (0)
print(f"\nLogit for token 0: {ax_logits[0].item():.2f}")
print(f"Logit for token 222: {ax_logits[222].item():.2f}")

# Check OUTPUT_LO/HI at last position
x = model.embed(token_ids)
for layer in model.blocks:
    x = layer(x)

print(f"\nOUTPUT_LO at last position (AX marker):")
for k in range(16):
    val = x[0, -1, BD.OUTPUT_LO + k].item()
    if abs(val) > 0.5:
        print(f"  OUTPUT_LO[{k}] = {val:.2f}")

print(f"\nOUTPUT_HI at last position (AX marker):")
for k in range(16):
    val = x[0, -1, BD.OUTPUT_HI + k].item()
    if abs(val) > 0.5:
        print(f"  OUTPUT_HI[{k}] = {val:.2f}")

# What does model.out_proj look like?
print(f"\nout_proj weights for token 222:")
out_proj = model.out_proj
if hasattr(out_proj, 'weight'):
    w = out_proj.weight[222, :]
    # Find which dims have high weights
    high_dims = (w.abs() > 0.1).nonzero().flatten()
    print(f"  Dims with |weight| > 0.1: {high_dims.tolist()[:20]}...")
