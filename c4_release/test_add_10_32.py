#!/usr/bin/env python3
"""Test ADD 10+32 prediction."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# ADD 10+32 test: IMM 10, PSH, IMM 32, ADD, EXIT
BYTECODE = [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (32 << 8), Opcode.ADD, Opcode.EXIT]

def build_context(bytecode, data=b''):
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

def main():
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    context = build_context(BYTECODE)
    draft = DraftVM(BYTECODE)

    # Run through all steps
    all_pass = True
    for step_num in range(1, 10):
        try:
            draft.step()
        except SystemExit:
            print(f"Program exited at step {step_num}")
            break

        step_tokens = draft.draft_tokens()
        print(f"\nStep {step_num}: expected tokens = {step_tokens[:7]}... (AX={draft.ax})")

        errors = []
        with torch.no_grad():
            for i, expected in enumerate(step_tokens):
                full_context = context + step_tokens[:i]
                token_ids = torch.tensor([full_context], dtype=torch.long)
                logits = model(token_ids)
                pred = logits[0, -1, :].argmax().item()
                if pred != expected:
                    status = f'FAIL (got {pred})'
                    errors.append((i, expected, pred))
                    if i < 10:  # Show first 10 tokens
                        print(f'  T{i}: expected {expected}, got {pred}')
                else:
                    if i < 7:
                        print(f'  T{i}: {expected} OK')

        if errors:
            print(f'  Step {step_num}: {len(step_tokens)-len(errors)}/{len(step_tokens)} correct')
            all_pass = False
        else:
            print(f'  Step {step_num}: PASS')

        context.extend(step_tokens)

    if all_pass:
        print("\nALL STEPS PASSED!")
    else:
        print("\nSOME STEPS FAILED")

if __name__ == "__main__":
    main()
