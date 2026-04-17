#!/usr/bin/env python3
"""Direct trace of L7 attention output to ALU_HI[0]."""

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
pos_9 = 9  # PC marker position
target_dim = BD.ALU_HI  # dim 376

print(f"Direct L7 trace for LEA 0:")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    # Run through L0-L6
    for i in range(7):
        x = model.blocks[i](x)

    print(f"\nBefore L7: ALU_HI[0] at pos {pos_9} = {x[0, pos_9, target_dim].item():.2f}")

    # Get L7 block
    block7 = model.blocks[7]

    # Check what the attention module returns
    attn_out = block7.attn(x)
    print(f"L7 attn output to dim {target_dim} at pos {pos_9}: {attn_out[0, pos_9, target_dim].item():.2f}")

    # Check full range of OUTPUT dims at pos 9 in attn output
    print(f"\nL7 attn output to OUTPUT_HI at pos {pos_9}:")
    output_hi = [attn_out[0, pos_9, BD.OUTPUT_HI + k].item() for k in range(16)]
    for k, v in enumerate(output_hi):
        if abs(v) > 0.1:
            print(f"  OUTPUT_HI[{k}]: {v:.2f} ***")

    # Check ALU range
    print(f"\nL7 attn output to ALU_HI at pos {pos_9}:")
    alu_hi = [attn_out[0, pos_9, BD.ALU_HI + k].item() for k in range(16)]
    for k, v in enumerate(alu_hi):
        if abs(v) > 0.1:
            print(f"  ALU_HI[{k}]: {v:.2f} ***")

    # What does the input x look like at various positions for V dims?
    print(f"\nInput x at pos 9, checking V-related dims:")
    print(f"  CLEAN_EMBED_LO[0] = {x[0, pos_9, BD.CLEAN_EMBED_LO].item():.2f}")
    print(f"  CLEAN_EMBED_HI[0] = {x[0, pos_9, BD.CLEAN_EMBED_HI].item():.2f}")
    print(f"  OUTPUT_LO[0] = {x[0, pos_9, BD.OUTPUT_LO].item():.2f}")
    print(f"  OUTPUT_HI[0] = {x[0, pos_9, BD.OUTPUT_HI].item():.2f}")

    # Check if CLEAN_EMBED_HI has changed (my fix moved it to 404)
    print(f"\n  BD.CLEAN_EMBED_HI = {BD.CLEAN_EMBED_HI}")
    print(f"  Input x[pos_9, 404] = {x[0, pos_9, 404].item():.2f}")
    print(f"  Input x[pos_9, 400] = {x[0, pos_9, 400].item():.2f}")
