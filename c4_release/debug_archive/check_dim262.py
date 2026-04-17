#!/usr/bin/env python3
"""Check dim 262 at position 9 for LEA 0."""

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

print(f"Dim mapping:")
print(f"  OPCODE_FLAGS = {BD.OPCODE_FLAGS} (34 dims: {BD.OPCODE_FLAGS}-{BD.OPCODE_FLAGS+33})")
print(f"  Dim 262 = OPCODE_FLAGS[{262 - BD.OPCODE_FLAGS}] = OPCODE_FLAGS[0]")
print()

for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    pos_9 = 9  # PC marker position

    print(f"{label}:")

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb

        print(f"  After embed, dim 262 at pos 9: {x[0, pos_9, 262].item():.2f}")

        for i in range(9):
            x = model.blocks[i](x)

        print(f"  After L8, dim 262 at pos 9: {x[0, pos_9, 262].item():.2f}")

        # Also check dims 1, 376, 444 (the W_up dims)
        print(f"  After L8, dim 1 (MARK_AX) at pos 9: {x[0, pos_9, 1].item():.2f}")
        print(f"  After L8, dim 376 (ALU_HI[0]) at pos 9: {x[0, pos_9, 376].item():.2f}")
        print(f"  After L8, dim 444 at pos 9: {x[0, pos_9, 444].item():.2f}")
        print()

        # What is dim 444?
        if 444 >= BD.MUL_ACCUM and 444 < BD.MUL_ACCUM + 16:
            print(f"  Dim 444 = MUL_ACCUM[{444 - BD.MUL_ACCUM}]")
        elif 444 >= BD.DIV_STAGING and 444 < BD.DIV_STAGING + 16:
            print(f"  Dim 444 = DIV_STAGING[{444 - BD.DIV_STAGING}]")
        else:
            print(f"  Dim 444 = unknown (MUL_ACCUM={BD.MUL_ACCUM}, DIV_STAGING={BD.DIV_STAGING})")
