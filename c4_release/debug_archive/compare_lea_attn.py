#!/usr/bin/env python3
"""Compare L15 attention for LEA 0 vs LEA 8."""

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

target_dim = BD.OUTPUT_HI + 12  # dim 202

for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    ctx_len = len(context)
    pos_6 = ctx_len + 6  # AX byte 0 position

    print(f"\n{'='*60}")
    print(f"{label}: checking OUTPUT_HI[12] at pos {pos_6}")

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb

        # Run through first 14 layers
        for i in range(15):
            x = model.blocks[i](x)

        # Check what's at dim 202 in the context before attention
        print(f"\nBefore L15, dim {target_dim} at each position:")
        for p in range(min(pos_6 + 5, x.size(1))):
            val = x[0, p, target_dim].item()
            if abs(val) > 0.1:
                print(f"  pos {p}: {val:.2f} ***")

        # L15 attention
        block15 = model.blocks[15]
        attn_out = block15.attn(x)
        print(f"\nL15 attn output to dim {target_dim} at pos {pos_6}: {attn_out[0, pos_6, target_dim].item():.2f}")

        # Try to identify which positions are being attended to
        # by looking at what values exist at dim 202 in the V space

        # Check ADDR_KEY values at position 6
        addr_key = [x[0, pos_6, BD.ADDR_KEY + k].item() for k in range(48)]
        nonzero_addr = [(k, addr_key[k]) for k in range(48) if abs(addr_key[k]) > 0.1]
        print(f"ADDR_KEY at pos {pos_6}: {nonzero_addr[:5]}...")

        # Check context tokens around position
        tokens = ctx_tensor[0].tolist()
        print(f"\nContext around pos {pos_6}:")
        for p in range(max(0, pos_6-3), min(len(tokens), pos_6+5)):
            tok = tokens[p]
            marker = " <-- target" if p == pos_6 else ""
            print(f"  pos {p}: token {tok}{marker}")
