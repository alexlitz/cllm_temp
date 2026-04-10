#\!/usr/bin/env python3
"""Debug JMP 16 PC byte 1 leakage."""
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
        for _ in range(3): tokens.append(0)  # padding
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

# Context up to PC byte 1 position
ctx = context + tokens[:2]  # PC marker + PC byte 0
pc_byte1_pos = len(ctx) - 1

print(f"PC = 16 = 0x10, bytes = [16, 0, 0, 0]")
print(f"Expected tokens: {tokens[:5]}")
print(f"Context length: {len(ctx)}, PC byte 1 position: {pc_byte1_pos}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    print(f"\nAfter embed: OUTPUT_LO = {[(k,x[0,pc_byte1_pos,BD.OUTPUT_LO+k].item()) for k in range(16) if abs(x[0,pc_byte1_pos,BD.OUTPUT_LO+k].item())>0.1]}")

    for i in range(16):
        x_before = x.clone()
        x = model.blocks[i](x)
        
        deltas = []
        for k in range(16):
            delta = x[0,pc_byte1_pos,BD.OUTPUT_LO+k].item() - x_before[0,pc_byte1_pos,BD.OUTPUT_LO+k].item()
            if abs(delta) > 0.1:
                deltas.append((k, delta))
        
        if deltas:
            print(f"blocks[{i:2d}]: OUTPUT_LO deltas = {deltas}")

    print(f"\nFinal OUTPUT_LO at PC byte 1:")
    for k in range(16):
        v = x[0,pc_byte1_pos,BD.OUTPUT_LO+k].item()
        if abs(v) > 0.5:
            print(f"  OUTPUT_LO[{k}] = {v:.4f}")

    # Check what dimensions are active at this position
    print(f"\nKey dimensions at PC byte 1 position:")
    print(f"  IS_BYTE: {x[0,pc_byte1_pos,BD.IS_BYTE].item():.4f}")
    print(f"  MARK_PC: {x[0,pc_byte1_pos,BD.MARK_PC].item():.4f}")
    print(f"  HAS_SE: {x[0,pc_byte1_pos,BD.HAS_SE].item():.4f}")
    print(f"  BYTE_INDEX_0: {x[0,pc_byte1_pos,BD.BYTE_INDEX_0].item():.4f}")
    print(f"  BYTE_INDEX_1: {x[0,pc_byte1_pos,BD.BYTE_INDEX_1].item():.4f}")
