#!/usr/bin/env python3
"""Debug the graph compiler forward pass."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import GraphWeightCompiler
from neural_vm.embedding import E

# Simple ADD test: 5 + 10 = 15
dim = E.DIM
hidden_dim = 512
scale = E.SCALE

print("=" * 70)
print("DEBUG: Graph Compiler Forward Pass")
print("=" * 70)

compiler = GraphWeightCompiler(dim, hidden_dim, scale)
compiler.const(5.0, "a")
compiler.const(10.0, "b")
compiler.add("a", "b", "result")

weights = compiler.compile()

print("\nWeight shapes:")
print(f"  W_up:   {weights['W_up'].shape}")
print(f"  b_up:   {weights['b_up'].shape}")
print(f"  W_gate: {weights['W_gate'].shape}")
print(f"  b_gate: {weights['b_gate'].shape}")
print(f"  W_down: {weights['W_down'].shape}")
print(f"  b_down: {weights['b_down'].shape}")

# Check non-zero entries
print("\nNon-zero counts:")
print(f"  W_up:   {(weights['W_up'] != 0).sum().item()}")
print(f"  b_up:   {(weights['b_up'] != 0).sum().item()}")
print(f"  W_gate: {(weights['W_gate'] != 0).sum().item()}")
print(f"  b_gate: {(weights['b_gate'] != 0).sum().item()}")
print(f"  W_down: {(weights['W_down'] != 0).sum().item()}")
print(f"  b_down: {(weights['b_down'] != 0).sum().item()}")

# Show b_up values (where constants are stored)
print("\nFirst 10 b_up values:")
print(f"  {weights['b_up'][:10]}")

# Show b_gate values (should be 1.0 for units 0-5)
print("\nFirst 10 b_gate values:")
print(f"  {weights['b_gate'][:10]}")

# Show W_gate structure (computation)
print("\nW_gate non-zero entries:")
nz_gate = torch.nonzero(weights['W_gate'], as_tuple=False)
for idx in nz_gate[:20]:  # First 20
    unit, dim_idx = idx[0].item(), idx[1].item()
    value = weights['W_gate'][unit, dim_idx].item()
    print(f"  W_gate[{unit}, {dim_idx}] = {value:.4f}")

# Show W_down structure (output)
print("\nW_down non-zero entries:")
nz_down = torch.nonzero(weights['W_down'], as_tuple=False)
for idx in nz_down[:20]:  # First 20
    dim_idx, unit = idx[0].item(), idx[1].item()
    value = weights['W_down'][dim_idx, unit].item()
    print(f"  W_down[{dim_idx}, {unit}] = {value:.4f}")

# Manual forward pass with detailed trace
print("\n" + "=" * 70)
print("MANUAL FORWARD PASS")
print("=" * 70)

x = torch.zeros(1, 1, dim)
print(f"\nInput x: shape={x.shape}, all zeros")

# Step 1: W_up @ x + b_up
up = F.linear(x, weights['W_up'], weights['b_up'])
print(f"\nAfter W_up @ x + b_up:")
print(f"  Shape: {up.shape}")
print(f"  First 10 values: {up[0, 0, :10]}")
print(f"  Range: [{up.min():.4f}, {up.max():.4f}]")

# Step 2: silu(up)
up_activated = F.silu(up)
print(f"\nAfter silu(up):")
print(f"  First 10 values: {up_activated[0, 0, :10]}")
print(f"  Range: [{up_activated.min():.4f}, {up_activated.max():.4f}]")

# Step 3: W_gate @ x + b_gate
gate = F.linear(x, weights['W_gate'], weights['b_gate'])
print(f"\nAfter W_gate @ x + b_gate:")
print(f"  Shape: {gate.shape}")
print(f"  First 10 values: {gate[0, 0, :10]}")
print(f"  Range: [{gate.min():.4f}, {gate.max():.4f}]")

# Step 4: hidden = silu(up) * gate
hidden = up_activated * gate
print(f"\nAfter silu(up) * gate:")
print(f"  First 10 values: {hidden[0, 0, :10]}")
print(f"  Range: [{hidden.min():.4f}, {hidden.max():.4f}]")

# Step 5: W_down @ hidden + b_down
output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
print(f"\nAfter W_down @ hidden + b_down:")
print(f"  Shape: {output_delta.shape}")
print(f"  First 10 values: {output_delta[0, 0, :10]}")
print(f"  Range: [{output_delta.min():.4f}, {output_delta.max():.4f}]")

# Step 6: x + output_delta (residual)
output = x + output_delta
print(f"\nAfter residual (x + delta):")
print(f"  First 10 values: {output[0, 0, :10]}")
print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")

# Check result register (dim[0] according to allocation)
result_dim = 0
result_value = output[0, 0, result_dim].item()

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print(f"Expected: 5.0 + 10.0 = 15.0")
print(f"Got:      result = {result_value:.4f} (at dim[{result_dim}])")
print(f"Match:    {abs(result_value - 15.0) < 0.1}")
