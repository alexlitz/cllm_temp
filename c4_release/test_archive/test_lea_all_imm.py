#!/usr/bin/env python3
"""Test LEA with various immediate values."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("Initializing model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()
print("Model ready\n")

test_cases = [
    (0, "LEA 0 (AX = BP = 0x10000)"),
    (8, "LEA 8 (AX = BP + 8 = 0x10008)"),
    (256, "LEA 256 (AX = BP + 256 = 0x10100)"),
    (65535, "LEA 65535 (AX = BP + 65535 = 0x1FFFF)"),
]

total_match = 0
total_mismatch = 0

for imm, desc in test_cases:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)

    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    ctx_len = len(context)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    matches = 0
    mismatches = []
    for i in range(35):
        pos = ctx_len - 1 + i
        pred = logits[0, pos, :].argmax().item()
        expected = draft_tokens[i]
        if pred == expected:
            matches += 1
        else:
            mismatches.append((i, expected, pred))

    total_match += matches
    total_mismatch += len(mismatches)

    status = "PASS" if len(mismatches) == 0 else "FAIL"
    print(f"{desc}: {matches}/35 matches [{status}]")
    if mismatches:
        for idx, exp, pred in mismatches:
            print(f"    Position {idx}: expected {exp}, got {pred}")

print()
print(f"Total: {total_match}/{total_match + total_mismatch} matches ({100*total_match/(total_match+total_mismatch):.1f}%)")
