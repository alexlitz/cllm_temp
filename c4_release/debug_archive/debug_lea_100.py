#!/usr/bin/env python3
"""Debug LEA to find remaining mismatches."""

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

# Test LEA 8 (the one used in test_all_opcodes_fast.py)
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"LEA 8 (imm=8, expected AX = 0x10008)")
print(f"Draft AX: 0x{draft_vm.ax:08X}")
print(f"Draft tokens ({len(draft_tokens)}): {draft_tokens}")
print()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
print(f"Context length: {ctx_len}")
print(f"Checking all 35 draft token predictions:\n")

mismatches = []
for i in range(35):
    pos = ctx_len - 1 + i  # Position predicting draft_tokens[i]
    pred = logits[0, pos, :].argmax().item()
    expected = draft_tokens[i]

    # Decode token type
    if expected < 256:
        token_type = f"byte {expected}"
    elif expected == Token.REG_PC:
        token_type = "REG_PC"
    elif expected == Token.REG_AX:
        token_type = "REG_AX"
    elif expected == Token.REG_SP:
        token_type = "REG_SP"
    elif expected == Token.REG_BP:
        token_type = "REG_BP"
    elif expected == Token.STACK0:
        token_type = "STACK0"
    elif expected == Token.STEP_END:
        token_type = "STEP_END"
    else:
        token_type = f"token {expected}"

    if pred == expected:
        status = "OK"
    else:
        status = f"FAIL (pred={pred})"
        mismatches.append((i, expected, pred, token_type))

    print(f"  [{i:2d}] {token_type:15s} expected={expected:3d}, pred={pred:3d} {status}")

print(f"\nTotal mismatches: {len(mismatches)}")
for idx, exp, pred, ttype in mismatches:
    print(f"  Position {idx}: {ttype} - expected {exp}, got {pred}")
