#!/usr/bin/env python3
"""Trace TEMP[0] at pos 9 for LEA 0."""

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

target_dim = BD.TEMP  # dim 480

for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    pos_9 = 9

    print(f"\n{label}: Tracing TEMP[0] (dim {target_dim}) at pos {pos_9}")

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb

        val = x[0, pos_9, target_dim].item()
        print(f"  After embed: {val:.2f}")

        for i, block in enumerate(model.blocks):
            x = block(x)
            val = x[0, pos_9, target_dim].item()
            if abs(val) > 10:
                print(f"  After L{i}: {val:.2f} ***")
            elif abs(val) > 1:
                print(f"  After L{i}: {val:.2f} *")
            else:
                print(f"  After L{i}: {val:.2f}")
