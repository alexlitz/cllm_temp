#!/usr/bin/env python3
"""Quick test for ADD 10+32 Step 4 only."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

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

    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    draft.step()
    step4_tokens = draft.draft_tokens()

    print(f'Step 4 expected AX: {draft.ax} (10 + 32 = 42)')

    errors = []
    with torch.no_grad():
        for i, expected in enumerate(step4_tokens):
            full_context = context + step4_tokens[:i]
            token_ids = torch.tensor([full_context], dtype=torch.long)
            logits = model(token_ids)
            pred = logits[0, -1, :].argmax().item()
            if pred != expected:
                print(f'S4T{i}: expected {expected}, got {pred} FAIL')
                errors.append(i)

    if errors:
        print(f'\nRESULT: {len(step4_tokens)-len(errors)}/{len(step4_tokens)} correct')
    else:
        print(f'\nPASS: {len(step4_tokens)}/{len(step4_tokens)} correct')

if __name__ == "__main__":
    main()
