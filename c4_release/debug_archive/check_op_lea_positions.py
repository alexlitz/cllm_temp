#!/usr/bin/env python3
"""Check OP_LEA at various positions for LEA 0 vs LEA 8."""

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

print(f"Opcode.LEA = {Opcode.LEA}")
print()

for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    print(f"\n{label}:")
    print(f"  Context: {context}")

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb

        print(f"\n  After embed, OP_LEA at each position:")
        for p in range(15):
            tok = (context + draft_tokens)[p]
            val = x[0, p, BD.OP_LEA].item()
            if abs(val) > 0.1:
                print(f"    pos {p}: token {tok}, OP_LEA = {val:.2f} ***")
            else:
                print(f"    pos {p}: token {tok}, OP_LEA = {val:.2f}")

        # Also check after L6 (before L7)
        for i in range(7):
            x = model.blocks[i](x)

        print(f"\n  After L6, OP_LEA at key positions:")
        for p in [9, 14, 15]:  # PC marker, AX marker, AX byte 0
            val = x[0, p, BD.OP_LEA].item()
            name = {9: "PC marker", 14: "AX marker", 15: "AX byte 0"}.get(p, f"pos {p}")
            print(f"    {name} (pos {p}): OP_LEA = {val:.2f}")
