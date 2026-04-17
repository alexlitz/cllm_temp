#!/usr/bin/env python3
"""Find which L6 FFN unit writes to OUTPUT_HI[0] at pos 9."""

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
target_dim = BD.OUTPUT_HI  # dim 190

print(f"Finding L6 FFN units writing to dim {target_dim}...")

# Find units writing to OUTPUT_HI[0]
units_writing = []
for unit in range(4096):
    weight = ffn6.W_down[target_dim, unit].item()
    if abs(weight) > 1e-6:
        units_writing.append((unit, weight))

print(f"Found {len(units_writing)} units")

# Now check activations for LEA 0
bytecode = [Opcode.LEA | (0 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
pos_9 = 9

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb
    for i in range(6):
        x = model.blocks[i](x)

    # After L6 attention
    attn_out = model.blocks[6].attn(x)
    x_after_attn = x + attn_out

    # Compute FFN activations at pos 9
    x_at_pos = x_after_attn[0, pos_9, :]

    W_up = ffn6.W_up
    W_gate = ffn6.W_gate
    b_up = ffn6.b_up
    b_gate = ffn6.b_gate

    up = torch.matmul(W_up, x_at_pos) + b_up
    gate = torch.matmul(W_gate, x_at_pos) + b_gate
    silu_up = up * torch.sigmoid(up)
    hidden = silu_up * gate

    print(f"\nHigh-contributing units for LEA 0 at pos {pos_9}:")
    for unit, weight in units_writing:
        h = hidden[unit].item()
        contribution = weight * h
        if abs(contribution) > 100:
            print(f"  Unit {unit}: hidden={h:.1f}, weight={weight:.4f}, contribution={contribution:.1f} ***")
            print(f"    up={up[unit].item():.1f}, gate={gate[unit].item():.1f}")

            # Show W_up and W_gate details
            w_up = W_up[unit, :]
            w_gate = W_gate[unit, :]
            up_nonzero = [(d, w_up[d].item()) for d in range(512) if abs(w_up[d].item()) > 0.01]
            gate_nonzero = [(d, w_gate[d].item()) for d in range(512) if abs(w_gate[d].item()) > 0.01]
            print(f"    W_up reads: {up_nonzero[:10]}...")
            print(f"    W_gate reads: {gate_nonzero[:5]}...")
            print()
