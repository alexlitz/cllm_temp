#!/usr/bin/env python3
"""Debug threshold heads at PC byte 1 position."""
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

# Context up to PC byte 1 position
ctx = context + tokens[:2]  # PC marker + PC byte 0
pc_byte1_pos = len(ctx) - 1

print(f"Context: {ctx}")
print(f"PC byte 1 position: {pc_byte1_pos}")
print(f"Token at PC byte 1 pos: {ctx[pc_byte1_pos]}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    # After Layer 0 (to get threshold heads computed)
    x = model.blocks[0](x)

    print(f"\nAfter Layer 1 (blocks[0]):")
    print(f"At PC byte 1 position ({pc_byte1_pos}):")
    print(f"  IS_BYTE: {x[0, pc_byte1_pos, BD.IS_BYTE].item():.4f}")

    # Threshold heads for PC marker (index 0)
    PC_I = 0
    BP_I = 3

    print(f"\n  Threshold heads for PC marker (idx {PC_I}):")
    print(f"    L1H0[PC]: {x[0, pc_byte1_pos, BD.L1H0 + PC_I].item():.4f}  (d≤0.5)")
    print(f"    L1H1[PC]: {x[0, pc_byte1_pos, BD.L1H1 + PC_I].item():.4f}  (d≤1.5)")
    print(f"    L1H2[PC]: {x[0, pc_byte1_pos, BD.L1H2 + PC_I].item():.4f}  (d≤2.5)")
    print(f"    H0[PC]:   {x[0, pc_byte1_pos, BD.H0 + PC_I].item():.4f}  (d≤3.5)")
    print(f"    H1[PC]:   {x[0, pc_byte1_pos, BD.H1 + PC_I].item():.4f}  (d≤4.5)")

    print(f"\n  Threshold heads for BP marker (idx {BP_I}):")
    print(f"    L1H4[BP]: {x[0, pc_byte1_pos, BD.L1H4 + BP_I].item():.4f}  (d≤6.5)")
    print(f"    H1[BP]:   {x[0, pc_byte1_pos, BD.H1 + BP_I].item():.4f}  (d≤4.5)")
    print(f"    H2[BP]:   {x[0, pc_byte1_pos, BD.H2 + BP_I].item():.4f}  (d≤7.5)")

    print(f"\n  Current BYTE_INDEX values:")
    print(f"    BYTE_INDEX_0: {x[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")
    print(f"    BYTE_INDEX_1: {x[0, pc_byte1_pos, BD.BYTE_INDEX_1].item():.4f}")

    # After Layer 1 (blocks[1])
    x = model.blocks[1](x)

    print(f"\nAfter Layer 2 (blocks[1]):")
    print(f"  BYTE_INDEX_0: {x[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")
    print(f"  BYTE_INDEX_1: {x[0, pc_byte1_pos, BD.BYTE_INDEX_1].item():.4f}")

    # Check what the Layer 2 FFN BYTE_INDEX_0 unit computes
    # Unit computes: silu(S*(L1H4[BP] + IS_BYTE) - 1.5*S) * (1 - H1[BP])
    is_byte = x[0, pc_byte1_pos, BD.IS_BYTE].item()
    l1h4_bp = x[0, pc_byte1_pos, BD.L1H4 + BP_I].item()
    h1_bp = x[0, pc_byte1_pos, BD.H1 + BP_I].item()

    print(f"\n  L2 FFN BYTE_INDEX_0 unit inputs (post blocks[0]):")
    print(f"    L1H4[BP] + IS_BYTE - 1.5 = {l1h4_bp + is_byte - 1.5:.4f}")
    print(f"    gate = 1 - H1[BP] = {1 - h1_bp:.4f}")
