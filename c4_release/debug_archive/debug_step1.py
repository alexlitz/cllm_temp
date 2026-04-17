#!/usr/bin/env python3
"""Debug step 1 (PSH instruction)."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

# Program: IMM 5, PSH
bytecode = [Opcode.IMM | (5 << 8), Opcode.PSH]

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

# Execute step 0 (IMM 5)
draft = DraftVM(bytecode)
draft.step()
draft_tokens_0 = draft.draft_tokens()
print(f'After step 0 (IMM 5): PC={draft.pc}, AX={draft.ax}, SP={draft.sp}')
context = context + draft_tokens_0

# Execute step 1 (PSH)
draft.step()
draft_tokens_1 = draft.draft_tokens()
print(f'After step 1 (PSH):   PC={draft.pc}, AX={draft.ax}, SP={draft.sp} (SP should be -8)')
print(f'Draft tokens: {draft_tokens_1[:15]}')

# Verify with neural VM
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Concatenate draft tokens to context for forward pass
full_context = context + draft_tokens_1
device = model.embed.weight.device
x = torch.tensor([full_context], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model.forward(x)

    print(f'\nToken predictions (autoregressive offset):')
    ctx_len = len(context)
    for i in range(min(15, len(draft_tokens_1))):
        pos = ctx_len + i - 1
        pred = logits[0, pos, :].argmax().item()
        expected = draft_tokens_1[i]
        match = '✓' if pred == expected else '✗'

        # Decode token type
        if expected == Token.REG_PC:
            token_name = 'REG_PC'
        elif expected == Token.REG_AX:
            token_name = 'REG_AX'
        elif expected == Token.REG_SP:
            token_name = 'REG_SP'
        elif expected == Token.REG_BP:
            token_name = 'REG_BP'
        elif expected == Token.STACK0:
            token_name = 'STACK0'
        elif expected == Token.MEM:
            token_name = 'MEM'
        elif expected == Token.STEP_END:
            token_name = 'STEP_END'
        else:
            token_name = f'byte {expected}'

        print(f'  Token {i:2d} (pos {pos}): pred={pred:3d}, expected={expected:3d} ({token_name:10s}) {match}')
        if not match:
            break
