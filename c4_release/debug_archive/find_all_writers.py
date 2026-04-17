#!/usr/bin/env python3
"""Find ALL L15 FFN units that could affect OUTPUT_HI[12]."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    from neural_vm.vm_step import Token
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

# Check L15 FFN W_down for writes to OUTPUT_HI[12]
ffn15 = model.blocks[15].ffn
target_dim = BD.OUTPUT_HI + 12  # dim 202

print(f"\nAll L15 FFN units with non-zero W_down to dim {target_dim}:")
units_writing = []
for unit in range(4096):
    weight = ffn15.W_down[target_dim, unit].item()
    if abs(weight) > 1e-6:
        units_writing.append((unit, weight))
        print(f"  Unit {unit}: weight = {weight:.6f}")

print(f"\nTotal: {len(units_writing)} units write to OUTPUT_HI[12]")

# Now run forward pass and check which units are activating
bytecode = [Opcode.LEA | (0 << 8)]  # LEA 0
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
pos_6 = ctx_len + 6  # AX byte 0 position

print(f"\nChecking activations at pos {pos_6} for LEA 0:")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb
    for i in range(15):  # Run through first 14 layers
        x = model.blocks[i](x)

    # Now at L15 input, check FFN activation for the units
    x_at_pos = x[0, pos_6, :].unsqueeze(0).unsqueeze(0)  # [1, 1, d]

    # Compute FFN activations manually
    W_up = ffn15.W_up  # [4096, 512]
    W_gate = ffn15.W_gate  # [4096, 512]
    b_up = ffn15.b_up  # [4096]
    b_gate = ffn15.b_gate  # [4096]
    W_down = ffn15.W_down  # [512, 4096]

    x_flat = x[0, pos_6, :]  # [512]
    up = torch.matmul(W_up, x_flat) + b_up  # [4096]
    gate = torch.matmul(W_gate, x_flat) + b_gate  # [4096]
    silu_up = up * torch.sigmoid(up)  # silu
    hidden = silu_up * gate  # [4096]
    out = torch.matmul(W_down, hidden)  # [512]

    print(f"\nOutput to dim {target_dim}: {out[target_dim].item():.2f}")

    # Check which units contribute most
    for unit, weight in units_writing:
        contribution = weight * hidden[unit].item()
        if abs(contribution) > 1.0:
            print(f"  Unit {unit}: hidden={hidden[unit].item():.1f}, weight={weight:.4f}, contribution={contribution:.1f}")
            print(f"    up={up[unit].item():.1f}, gate={gate[unit].item():.1f}, silu_up={silu_up[unit].item():.1f}")
