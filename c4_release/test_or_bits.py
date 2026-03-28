#!/usr/bin/env python3
"""Test OR operation on individual bits."""

import torch
from neural_vm.bit_primitives import BitPrimitives

# Create dummy weight tensors
hidden_dim = 100
input_dim = 20

W_up = torch.zeros(hidden_dim, input_dim)
b_up = torch.zeros(hidden_dim)
W_gate = torch.zeros(hidden_dim, input_dim)
b_gate = torch.zeros(hidden_dim)
W_down = torch.zeros(input_dim, hidden_dim)

bit_ops = BitPrimitives(scale=100.0)

# Test OR on bit 0
# a=1 (slot 0), b=1 (slot 1) → result should be 1 (slot 10)
units_used = bit_ops.emit_or(W_up, b_up, W_gate, b_gate, W_down,
                             0, 0, 1, 10)

print(f"Units used for OR: {units_used}")
print()

# Create input: a=1, b=1
input_vec = torch.zeros(input_dim)
input_vec[0] = 1.0  # bit_a = 1
input_vec[1] = 1.0  # bit_b = 1

# Forward pass
with torch.no_grad():
    up = W_up @ input_vec + b_up
    gate = W_gate @ input_vec + b_gate

    print("Up activations for first 3 units:")
    print(f"  Unit 0: {up[0].item():.2f}")
    print(f"  Unit 1: {up[1].item():.2f}")
    print(f"  Unit 2: {up[2].item():.2f}")
    print()

    print("Gate activations for first 3 units:")
    print(f"  Unit 0: {gate[0].item():.2f}")
    print(f"  Unit 1: {gate[1].item():.2f}")
    print(f"  Unit 2: {gate[2].item():.2f}")
    print()

    # SwiGLU
    hidden = torch.nn.functional.silu(up) * gate

    print("Hidden activations for first 3 units:")
    print(f"  Unit 0: {hidden[0].item():.2f}")
    print(f"  Unit 1: {hidden[1].item():.2f}")
    print(f"  Unit 2: {hidden[2].item():.2f}")
    print()

    # Output
    output = W_down @ hidden

    print(f"Output at slot 10: {output[10].item():.4f} (expected 1.0)")
    print()

    # Check W_down weights
    print("W_down[10, :3]:", W_down[10, :3])
