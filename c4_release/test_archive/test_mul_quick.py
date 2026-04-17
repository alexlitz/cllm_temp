#!/usr/bin/env python3
"""Quick test of MUL program: IMM 1, PSH, IMM 0, MUL, EXIT."""
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

# Get all draft steps
steps = []
for i in range(5):
    vm.step()
    steps.append(vm.draft_tokens())

context = build_context(bytecode)
model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()

print("Testing MUL program step by step:")
errors = []
for step_idx in range(min(3, len(steps))):  # Test first 3 steps
    draft = steps[step_idx]
    full_context = context + sum(steps[:step_idx], [])

    step_errors = []
    for token_idx in range(35):
        test_context = full_context + draft[:token_idx]
        token_ids = torch.tensor([test_context], dtype=torch.long)
        with torch.no_grad():
            logits = model(token_ids)
        pred = logits[0, -1, :].argmax().item()
        exp = draft[token_idx]
        if pred != exp:
            step_errors.append(f"token {token_idx}: exp={exp}, pred={pred}")

    if step_errors:
        print(f"  Step {step_idx}: FAILED - {len(step_errors)} errors")
        for e in step_errors[:3]:
            print(f"    {e}")
        if len(step_errors) > 3:
            print(f"    ... and {len(step_errors) - 3} more")
        errors.extend(step_errors)
    else:
        print(f"  Step {step_idx}: PASS")

if not errors:
    print("\nAll steps correct!")
else:
    print(f"\nTotal errors: {len(errors)}")
