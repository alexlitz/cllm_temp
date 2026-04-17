#!/usr/bin/env python3
"""Check which L9 FFN units write to dim 202."""

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

ffn9 = model.blocks[9].ffn
target_dim = BD.OUTPUT_HI + 12  # dim 202

print(f"Checking L9 FFN for writes to dim {target_dim}:")

# Find all units writing to this dim
units_writing = []
for unit in range(4096):
    weight = ffn9.W_down[target_dim, unit].item()
    if abs(weight) > 1e-6:
        units_writing.append((unit, weight))
        print(f"  Unit {unit}: weight = {weight:.6f}")

print(f"\nTotal: {len(units_writing)} units")

# Now trace which units activate for LEA 0
bytecode = [Opcode.LEA | (0 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
pos_9 = 9  # PC marker

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb
    for i in range(9):  # Run through L0-L8
        x = model.blocks[i](x)

    # Now check L9 FFN activation at position 9
    x_at_pos = x[0, pos_9, :]  # [512]

    W_up = ffn9.W_up  # [4096, 512]
    W_gate = ffn9.W_gate  # [4096, 512]
    b_up = ffn9.b_up  # [4096]
    b_gate = ffn9.b_gate  # [4096]

    up = torch.matmul(W_up, x_at_pos) + b_up  # [4096]
    gate = torch.matmul(W_gate, x_at_pos) + b_gate  # [4096]
    silu_up = up * torch.sigmoid(up)  # silu
    hidden = silu_up * gate  # [4096]

    print(f"\nActivations at pos {pos_9} for LEA 0:")
    for unit, weight in units_writing:
        h = hidden[unit].item()
        contribution = weight * h
        if abs(contribution) > 100:
            print(f"  Unit {unit}: hidden={h:.1f}, weight={weight:.4f}, contribution={contribution:.1f} ***")
        elif abs(contribution) > 1:
            print(f"  Unit {unit}: hidden={h:.1f}, weight={weight:.4f}, contribution={contribution:.1f}")

    # Check W_up/W_gate for the high-contributing units
    print(f"\nChecking W_up/W_gate for high-contributing units:")
    for unit, weight in units_writing:
        h = hidden[unit].item()
        if abs(h * weight) > 100:
            print(f"\n  Unit {unit}:")
            print(f"    b_up = {b_up[unit].item():.2f}, b_gate = {b_gate[unit].item():.2f}")
            print(f"    up = {up[unit].item():.2f}, gate = {gate[unit].item():.2f}")

            # Show what dims this unit reads
            w_up = W_up[unit, :]
            w_gate = W_gate[unit, :]
            up_nonzero = [(d, w_up[d].item()) for d in range(512) if abs(w_up[d].item()) > 0.01]
            gate_nonzero = [(d, w_gate[d].item()) for d in range(512) if abs(w_gate[d].item()) > 0.01]

            print(f"    W_up reads: {up_nonzero[:10]}...")
            print(f"    W_gate reads: {gate_nonzero[:5]}...")
