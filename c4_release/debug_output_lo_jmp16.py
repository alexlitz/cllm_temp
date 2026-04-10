#!/usr/bin/env python3
"""Debug OUTPUT_LO at PC byte 0 position to understand why byte 1 prediction fails."""
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

# Context up to PC byte 0
ctx = context + tokens[:2]  # PC marker + PC byte 0

print(f"Context tokens: {ctx}")
print(f"  Position 12: {ctx[12]} (PC marker = 257)")
print(f"  Position 13: {ctx[13]} (PC byte 0 = 16)")
print(f"Draft expects next token: {tokens[2]} (PC byte 1 = 0)")

pc_byte0_pos = len(ctx) - 1  # Position where we predict byte 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    print(f"\nAt position {pc_byte0_pos} (PC byte 0), predicting byte 1:")
    print(f"  BYTE_INDEX_0: {x[0, pc_byte0_pos, BD.BYTE_INDEX_0].item():.4f}")
    print(f"  BYTE_INDEX_1: {x[0, pc_byte0_pos, BD.BYTE_INDEX_1].item():.4f}")

    # Trace OUTPUT_LO through layers
    for i in range(16):
        x = model.blocks[i](x)

        # Check OUTPUT_LO changes
        output_lo = [x[0, pc_byte0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        active_lo = [(k, v) for k, v in enumerate(output_lo) if abs(v) > 0.5]
        if active_lo:
            print(f"blocks[{i:2d}]: OUTPUT_LO = {active_lo}")

    print(f"\nFinal OUTPUT_LO at pos {pc_byte0_pos}:")
    for k in range(16):
        v = x[0, pc_byte0_pos, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.1:
            print(f"  OUTPUT_LO[{k}] = {v:.4f}")

    # Check prediction
    logits = model.head(x)
    top5 = torch.topk(logits[0, pc_byte0_pos, :], 5)
    print(f"\nTop 5 predictions at pos {pc_byte0_pos}:")
    for v, idx in zip(top5.values, top5.indices):
        print(f"  {idx.item():3d}: {v.item():.4f}")
