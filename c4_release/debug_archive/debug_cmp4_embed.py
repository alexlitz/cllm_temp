#!/usr/bin/env python3
"""Check CMP[4] in embedding directly."""

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

print(f"\nCMP+4 dimension = {BD.CMP + 4}")

# LEA 8
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)

print(f"\nToken values at key positions:")
print(f"  Pos {ctx_len + 10}: token {(context + draft_tokens)[ctx_len + 10]} (SP marker)")
print(f"  Pos {ctx_len + 11}: token {(context + draft_tokens)[ctx_len + 11]} (SP byte 0)")
print(f"  Pos {ctx_len + 12}: token {(context + draft_tokens)[ctx_len + 12]} (SP byte 1)")

# Check raw embedding
with torch.no_grad():
    emb = model.embed(ctx_tensor)

    print(f"\nAfter embed() but before first layer:")
    for pos, name in [(ctx_len + 10, "SP marker"), (ctx_len + 11, "SP byte 0"), (ctx_len + 12, "SP byte 1")]:
        cmp4 = emb[0, pos, BD.CMP + 4].item()
        print(f"  {name}: CMP[4] = {cmp4:.2f}")

# Check raw token embedding table
print(f"\nRaw embedding table for token 0 (byte 0x00):")
token_0_emb = model.embed.tok_embed.weight[0]
print(f"  CMP[4] = {token_0_emb[BD.CMP + 4].item():.2f}")

# Check all CMP dims for token 0
print(f"\n  CMP[0..7] for token 0:")
for k in range(8):
    val = token_0_emb[BD.CMP + k].item()
    if abs(val) > 0.01:
        print(f"    CMP[{k}] = {val:.2f} ***")
    else:
        print(f"    CMP[{k}] = {val:.2f}")
