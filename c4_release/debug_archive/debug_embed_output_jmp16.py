#!/usr/bin/env python3
"""Debug EMBED_LO vs OUTPUT_LO at PC byte 0 position."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

bytecode = [Opcode.JMP | (16 << 8)]

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4): tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(3): tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()

ctx = context + tokens[:2]  # PC marker + PC byte 0
pc_byte0_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    # Run full model
    logits = model(token_ids)

    # Get hidden state before head
    x = model.embed(token_ids)
    for i in range(16):
        x = model.blocks[i](x)

    print(f"At position {pc_byte0_pos} (PC byte 0, predicting byte 1):")
    print(f"  Current token: {ctx[pc_byte0_pos]} = byte value 16 = 0x10")
    print(f"  Next expected: {tokens[2]} = byte value 0 = 0x00")

    print(f"\nEMBED_LO (current token's nibble):")
    for k in range(16):
        v = x[0, pc_byte0_pos, BD.EMBED_LO + k].item()
        if abs(v) > 0.1:
            print(f"  EMBED_LO[{k}] = {v:.4f}")

    print(f"\nEMBED_HI (current token's high nibble):")
    for k in range(16):
        v = x[0, pc_byte0_pos, BD.EMBED_HI + k].item()
        if abs(v) > 0.1:
            print(f"  EMBED_HI[{k}] = {v:.4f}")

    print(f"\nOUTPUT_LO (should predict next byte's low nibble = 0):")
    for k in range(16):
        v = x[0, pc_byte0_pos, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.1:
            print(f"  OUTPUT_LO[{k}] = {v:.4f}")

    print(f"\nOUTPUT_HI (should predict next byte's high nibble = 0):")
    for k in range(16):
        v = x[0, pc_byte0_pos, BD.OUTPUT_HI + k].item()
        if abs(v) > 0.1:
            print(f"  OUTPUT_HI[{k}] = {v:.4f}")

    # Check model head weights
    head = model.head
    print(f"\nModel head weight shape: {head.weight.shape}")

    # Logit contributions
    h = x[0, pc_byte0_pos, :]
    logit_0 = (head.weight[0] @ h).item()
    logit_16 = (head.weight[16] @ h).item()

    print(f"\nLogit[0] = {logit_0:.4f}")
    print(f"Logit[16] = {logit_16:.4f}")
    print(f"Difference = {logit_16 - logit_0:.4f}")

    # Find what's causing logit[16] > logit[0]
    w0 = head.weight[0]
    w16 = head.weight[16]
    diff = w16 - w0  # Positive values favor 16, negative favor 0

    # Contribution from each dimension range
    contrib = (diff * h)
    embed_lo_contrib = contrib[BD.EMBED_LO:BD.EMBED_LO+16].sum().item()
    embed_hi_contrib = contrib[BD.EMBED_HI:BD.EMBED_HI+16].sum().item()
    output_lo_contrib = contrib[BD.OUTPUT_LO:BD.OUTPUT_LO+16].sum().item()
    output_hi_contrib = contrib[BD.OUTPUT_HI:BD.OUTPUT_HI+16].sum().item()

    print(f"\nContributions to (logit[16] - logit[0]):")
    print(f"  EMBED_LO range: {embed_lo_contrib:.4f}")
    print(f"  EMBED_HI range: {embed_hi_contrib:.4f}")
    print(f"  OUTPUT_LO range: {output_lo_contrib:.4f}")
    print(f"  OUTPUT_HI range: {output_hi_contrib:.4f}")

    # Total from key dimensions
    print(f"\nTop 10 dimensional contributors:")
    sorted_contrib = contrib.abs().sort(descending=True)
    for i in range(10):
        dim = sorted_contrib.indices[i].item()
        c = contrib[dim].item()
        print(f"  dim {dim}: {c:.4f} (h={h[dim].item():.4f})")
