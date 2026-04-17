#!/usr/bin/env python3
"""Trace where CMP[4] comes from at SP marker and SP byte positions."""

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

# LEA 8
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)

# SP marker at draft index 10
sp_marker_pos = ctx_len + 10
# SP byte 1 at draft index 11
sp_byte1_pos = ctx_len + 11
# SP byte 2 at draft index 12
sp_byte2_pos = ctx_len + 12

print(f"\nTracing CMP[4] at SP positions:")
print(f"  SP marker: pos {sp_marker_pos}")
print(f"  SP byte 1: pos {sp_byte1_pos}")
print(f"  SP byte 2: pos {sp_byte2_pos}")
print()

# Forward pass with layer-by-layer inspection
with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    for i, block in enumerate(model.blocks):
        x = block(x)
        cmp4_marker = x[0, sp_marker_pos, BD.CMP + 4].item()
        cmp4_byte1 = x[0, sp_byte1_pos, BD.CMP + 4].item()
        cmp4_byte2 = x[0, sp_byte2_pos, BD.CMP + 4].item()

        marker_flag = "***" if abs(cmp4_marker) > 0.1 else ""
        byte1_flag = "***" if abs(cmp4_byte1) > 0.1 else ""
        byte2_flag = "***" if abs(cmp4_byte2) > 0.1 else ""

        print(f"After L{i:2d}: marker={cmp4_marker:5.2f}{marker_flag:3s}  byte1={cmp4_byte1:5.2f}{byte1_flag:3s}  byte2={cmp4_byte2:5.2f}{byte2_flag:3s}")

# Also check OP_JSR and OP_LEA at AX marker (where opcodes are decoded)
print()
print("=" * 60)
print("Checking OP flags at PC and AX markers after L5 (opcode decode):")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb
    pc_pos = ctx_len + 0  # PC marker
    ax_pos = ctx_len + 5  # AX marker

    for i, block in enumerate(model.blocks):
        x = block(x)
        if i == 5:
            print(f"\n  At PC marker (pos {pc_pos}):")
            print(f"    OP_JSR = {x[0, pc_pos, BD.OP_JSR].item():.2f}")
            print(f"    OP_LEA = {x[0, pc_pos, BD.OP_LEA].item():.2f}")
            print(f"\n  At AX marker (pos {ax_pos}):")
            print(f"    OP_JSR = {x[0, ax_pos, BD.OP_JSR].item():.2f}")
            print(f"    OP_LEA = {x[0, ax_pos, BD.OP_LEA].item():.2f}")
            print(f"\n  At SP marker (pos {sp_marker_pos}):")
            print(f"    OP_JSR = {x[0, sp_marker_pos, BD.OP_JSR].item():.2f}")
            print(f"    OP_LEA = {x[0, sp_marker_pos, BD.OP_LEA].item():.2f}")
            print(f"    CMP[4] = {x[0, sp_marker_pos, BD.CMP + 4].item():.2f}")
            break
