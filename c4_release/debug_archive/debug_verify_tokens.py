#!/usr/bin/env python3
"""Debug which tokens are being rejected."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

# Simple test: IMM 42, EXIT
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Build context
context = [Token.CODE_START]
for instr in bytecode:
    opcode = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFFFF
    context.append(opcode)
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
    context.extend([0, 0, 0])
context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])

# Generate draft tokens
draft = DraftVM(bytecode)
draft.step()  # Step 0: IMM 42
draft_tokens = draft.draft_tokens()

print(f'DraftVM state: PC={draft.pc}, AX={draft.ax}')
print(f'Draft tokens (first 10): {draft_tokens[0:10]}')

# Forward pass
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

full_context = context + draft_tokens
device = model.embed.weight.device
token_ids = torch.tensor([full_context], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model.forward(token_ids)  # [1, S, vocab]

    ctx_len = len(context)

    # Check predictions for each draft token
    print(f'\nContext length: {ctx_len}')
    print(f'Draft token predictions (autoregressive: pos K predicts token K+1):')
    for i in range(min(10, len(draft_tokens))):
        pos = ctx_len + i - 1  # Position K predicts token K+1
        if pos < 0:
            continue
        pred = logits[0, pos, :].argmax().item()
        expected = draft_tokens[i]
        match = "✓" if pred == expected else "✗"
        print(f'  Position {pos} predicts token {i}: pred={pred}, expected={expected} {match}')
        if pred != expected:
            # Show top 5 predictions
            top_logits, top_indices = logits[0, pos, :256].topk(5)
            print(f'    Top 5: {[(top_indices[j].item(), f"{top_logits[j].item():.2f}") for j in range(5)]}')
