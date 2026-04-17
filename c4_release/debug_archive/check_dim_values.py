#!/usr/bin/env python3
"""Check the dim values that cause huge activation."""

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

print(f"Key dims:")
print(f"  OP_LEA = {BD.OP_LEA} (dim 262)")
print(f"  MARK_AX = {BD.MARK_AX} (dim 1)")
print(f"  ALU_HI = {BD.ALU_HI} (starts at 376)")
print(f"  DIV_STAGING = {BD.DIV_STAGING} (starts at 432)")
print()

for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    pos_9 = 9  # PC marker position

    print(f"\n{label} at pos 9 (PC marker):")

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb

        for i in range(9):
            x = model.blocks[i](x)

        # Check the dims that unit 524 reads
        print(f"  After L8:")
        print(f"    dim 1 (MARK_AX) = {x[0, pos_9, 1].item():.2f}")
        print(f"    dim 376 (ALU_HI[0]) = {x[0, pos_9, 376].item():.2f}")
        print(f"    dim 444 (DIV_STAGING[12]) = {x[0, pos_9, 444].item():.2f}")
        print(f"    dim 262 (OP_LEA) = {x[0, pos_9, 262].item():.2f}")

        # What does unit 524 actually compute?
        # up = 100 * x[1] + 100 * x[376] + 100 * x[444] - 5800
        up = 100 * x[0, pos_9, 1].item() + 100 * x[0, pos_9, 376].item() + 100 * x[0, pos_9, 444].item() - 5800
        gate = x[0, pos_9, 262].item()  # OP_LEA
        print(f"    Computed up = {up:.2f}")
        print(f"    Computed gate = {gate:.2f}")
