#!/usr/bin/env python3
"""Check dims used by L6 FFN unit 1368."""

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

ffn6 = model.blocks[6].ffn

# Check unit 1368
unit = 1368
print(f"L6 FFN unit {unit} details:")
print(f"  b_up = {ffn6.b_up[unit].item():.2f}")
print(f"  b_gate = {ffn6.b_gate[unit].item():.2f}")

# Non-zero W_up entries
W_up = ffn6.W_up[unit, :]
up_nonzero = [(d, W_up[d].item()) for d in range(512) if abs(W_up[d].item()) > 0.01]
print(f"  W_up nonzero: {up_nonzero}")

# Non-zero W_gate entries
W_gate = ffn6.W_gate[unit, :]
gate_nonzero = [(d, W_gate[d].item()) for d in range(512) if abs(W_gate[d].item()) > 0.01]
print(f"  W_gate nonzero: {gate_nonzero}")

# Check what dims 0, 480, 432 are
print(f"\n  Dim 0 = MARK_PC = {BD.MARK_PC}")
print(f"  Dim 480 = TEMP[0] or OUTPUT_BYTE_LO[0]")
print(f"    TEMP = {BD.TEMP}")
print(f"    OUTPUT_BYTE_LO = {BD.OUTPUT_BYTE_LO if hasattr(BD, 'OUTPUT_BYTE_LO') else 'N/A'}")
print(f"  Dim 432 = FETCH_HI or DIV_STAGING")
print(f"    FETCH_HI = {BD.FETCH_HI}")
print(f"    DIV_STAGING = {BD.DIV_STAGING}")

# Check values at pos 9 for LEA 0 vs LEA 8
for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    pos_9 = 9

    print(f"\n{label} at pos {pos_9}:")

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb
        for i in range(6):
            x = model.blocks[i](x)

        # After L6 attention
        attn_out = model.blocks[6].attn(x)
        x_after_attn = x + attn_out

        x_at_pos = x_after_attn[0, pos_9, :]
        print(f"  dim 0 (MARK_PC) = {x_at_pos[0].item():.2f}")
        print(f"  dim 480 (TEMP[0]) = {x_at_pos[480].item():.2f}")
        print(f"  dim 432 (FETCH_HI[0]) = {x_at_pos[432].item():.2f}")

        # Compute up and gate
        up = (W_up * x_at_pos).sum().item() + ffn6.b_up[unit].item()
        gate = (W_gate * x_at_pos).sum().item() + ffn6.b_gate[unit].item()
        print(f"  up = {up:.2f}, gate = {gate:.2f}")
