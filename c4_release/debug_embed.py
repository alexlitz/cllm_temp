#!/usr/bin/env python3
"""Debug embedding flags."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        op, imm = instr & 0xFF, instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])
    return context

context = build_context(bytecode)

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

vm = DraftVM(bytecode)
all_tokens = context[:]
vm.step()
step1_tokens = vm.draft_tokens()
all_tokens.extend(step1_tokens)
vm.step()
step2_tokens = vm.draft_tokens()

# Test at STACK0 marker position
test_context = all_tokens + step2_tokens[:26]
token_ids = torch.tensor([test_context], dtype=torch.long)

# Get raw embeddings
with torch.no_grad():
    emb = model.embed(token_ids)

ax_marker_pos = len(all_tokens) + 10
print(f"=== Token {ax_marker_pos} = {test_context[ax_marker_pos]} (AX marker = 259) ===")
print(f"MARK_AX in embedding: {emb[0, ax_marker_pos, BD.MARK_AX].item():.3f}")
print(f"IS_MARK in embedding: {emb[0, ax_marker_pos, BD.IS_MARK].item():.3f}")

# Check embedding weights for token 259
print(f"\n=== Embedding matrix check ===")
print(f"embed.weight[259, MARK_AX]: {model.embed.weight[259, BD.MARK_AX].item():.3f}")
print(f"embed.weight[259, IS_MARK]: {model.embed.weight[259, BD.IS_MARK].item():.3f}")

# Compare with the Token constants
print(f"\n=== Token values ===")
print(f"Token.REG_AX = {Token.REG_AX}")
print(f"Token.REG_SP = {Token.REG_SP}")
print(f"Token.REG_BP = {Token.REG_BP}")
print(f"step2_tokens[10] = {step2_tokens[10]}")

# Check OPCODE_BYTE in embedding
print(f"\n=== OPCODE_BYTE in embedding for AX marker ===")
lo = emb[0, ax_marker_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].tolist()
hi = emb[0, ax_marker_pos, BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].tolist()
print(f"OPCODE_BYTE_LO: {[f'{v:.2f}' for v in lo]}")
print(f"OPCODE_BYTE_HI: {[f'{v:.2f}' for v in hi]}")

# Check at PC marker position where opcode is fetched
pc_marker_pos = len(all_tokens) + 5
print(f"\n=== Token {pc_marker_pos} = {test_context[pc_marker_pos]} (PC marker = 258) ===")
lo = emb[0, pc_marker_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].tolist()
hi = emb[0, pc_marker_pos, BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].tolist()
print(f"OPCODE_BYTE_LO: {[f'{v:.2f}' for v in lo]}")
print(f"OPCODE_BYTE_HI: {[f'{v:.2f}' for v in hi]}")
