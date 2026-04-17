#!/usr/bin/env python3
"""Trace when OUTPUT_HI[12] gets set for LEA 0."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
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

# LEA 0
bytecode = [Opcode.LEA | (0 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)

pos_6 = ctx_len + 6  # AX byte 0 position

print(f"Tracing OUTPUT_HI[12] at AX byte 0 (pos {pos_6}) for LEA 0:")
print()

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    output_hi_12 = x[0, pos_6, BD.OUTPUT_HI + 12].item()
    print(f"After embed: OUTPUT_HI[12] = {output_hi_12:.1f}")

    for i, block in enumerate(model.blocks):
        x_before = x.clone()
        x = block(x)
        output_hi_12 = x[0, pos_6, BD.OUTPUT_HI + 12].item()
        if abs(output_hi_12) > 1.0:
            print(f"After L{i}: OUTPUT_HI[12] = {output_hi_12:.1f} ***")
        else:
            print(f"After L{i}: OUTPUT_HI[12] = {output_hi_12:.1f}")

print()
print(f"OUTPUT_HI[12] dim = {BD.OUTPUT_HI + 12}")
print(f"What else is at dim {BD.OUTPUT_HI + 12}?")
