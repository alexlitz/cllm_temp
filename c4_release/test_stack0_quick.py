#!/usr/bin/env python3
"""Quick test of STACK0 prediction for IMM after PSH."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        op, imm = instr & 0xFF, instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])
    return context

# Test: IMM 1, PSH, IMM 0, MUL, EXIT
bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)
vm.step(); draft_step0 = vm.draft_tokens()
vm.step(); draft_step1 = vm.draft_tokens()
vm.step(); draft_step2 = vm.draft_tokens()

context = build_context(bytecode)
model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()

# Test step 2 (IMM after PSH) - STACK0 byte 0 should be 1
full_context = context + draft_step0 + draft_step1 + draft_step2[:21]
token_ids = torch.tensor([full_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)
pred = logits[0, -1, :].argmax().item()
exp = draft_step2[21]
print(f'Step 2 STACK0 byte 0: Expected {exp}, Predicted {pred}, Match: {pred == exp}')

# Test all 4 STACK0 bytes in step 2
errors = []
for i in range(4):
    full_context = context + draft_step0 + draft_step1 + draft_step2[:21+i]
    token_ids = torch.tensor([full_context], dtype=torch.long)
    with torch.no_grad():
        logits = model(token_ids)
    pred = logits[0, -1, :].argmax().item()
    exp = draft_step2[21+i]
    if pred != exp:
        errors.append(f'byte {i}: exp={exp}, pred={pred}')
    print(f'  STACK0 byte {i}: Expected {exp}, Predicted {pred}, Match: {pred == exp}')

if not errors:
    print('\nAll STACK0 bytes correct!')
else:
    print(f'\nErrors: {errors}')
