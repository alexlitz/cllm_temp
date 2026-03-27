#!/usr/bin/env python3
"""Detailed debug of MUL operation - check unit 0."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter
from neural_vm.embedding import E

def onehot_encode(value, base=16):
    """Convert integer to one-hot vector."""
    vec = torch.zeros(base)
    if 0 <= value < base:
        vec[value] = 1.0
    return vec

# Test MUL(3, 4) = 12
base = 16
dim = E.DIM
hidden_dim = 2 * base
scale = E.SCALE

print(f"Testing MUL(3, 4) = 12")
print()

# Create graph
graph = ComputationGraph()

a_node_id = len(graph.nodes)
graph.nodes[a_node_id] = IRNode(
    id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
    params={'value': 0}, physical_reg=0
)

b_node_id = len(graph.nodes)
graph.nodes[b_node_id] = IRNode(
    id=b_node_id, op=OpType.CONST, inputs=[], output_reg="b",
    params={'value': 0}, physical_reg=base
)

mul_node_id = len(graph.nodes)
mul_node = IRNode(
    id=mul_node_id,
    op=OpType.MUL,
    inputs=[a_node_id, b_node_id],
    output_reg="result",
    params={'base': base},
    physical_reg=2*base,
    gate=None
)
graph.nodes[mul_node_id] = mul_node

# Compile
emitter = WeightEmitter(dim, hidden_dim, scale)
emitter.emit_mul(mul_node, graph)

weights = {
    'W_up': emitter.W_up,
    'b_up': emitter.b_up,
    'W_gate': emitter.W_gate,
    'b_gate': emitter.b_gate,
    'W_down': emitter.W_down,
    'b_down': emitter.b_down,
}

# Check weights for unit 0 (k=0)
print("Weights for unit 0 (k=0):")
print("Non-zero W_up[0, :]:")
for i in range(dim):
    if weights['W_up'][0, i] != 0:
        print(f"  W_up[0, {i}] = {weights['W_up'][0, i].item()}")

print("\nNon-zero W_gate[0, :]:")
for i in range(dim):
    if weights['W_gate'][0, i] != 0:
        print(f"  W_gate[0, {i}] = {weights['W_gate'][0, i].item()}")

print(f"\nb_up[0] = {weights['b_up'][0].item()}")
print(f"b_gate[0] = {weights['b_gate'][0].item()}")

# Create input
a_val = 3
b_val = 4
x = torch.zeros(1, 1, dim)
x[0, 0, 0:base] = onehot_encode(a_val, base)
x[0, 0, base:2*base] = onehot_encode(b_val, base)

print(f"\nInput: a={a_val}, b={b_val}")
print(f"  x[3] = {x[0, 0, 3].item()}")
print(f"  x[20] = {x[0, 0, 20].item()}")

# Forward pass
up = F.linear(x, weights['W_up'], weights['b_up'])
gate = F.linear(x, weights['W_gate'], weights['b_gate'])
hidden = F.silu(up) * gate
output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
output = x + output_delta

# Check unit 0
print(f"\nUnit 0 activations:")
print(f"  up[0] = {up[0, 0, 0].item():.4f}")
print(f"  gate[0] = {gate[0, 0, 0].item():.4f}")
print(f"  silu(up[0]) = {F.silu(up[0, 0, 0]).item():.4f}")
print(f"  hidden[0] = {hidden[0, 0, 0].item():.4f}")

# Check unit 1
print(f"\nUnit 1 activations:")
print(f"  up[1] = {up[0, 0, 1].item():.4f}")
print(f"  gate[1] = {gate[0, 0, 1].item():.4f}")
print(f"  silu(up[1]) = {F.silu(up[0, 0, 1]).item():.4f}")
print(f"  hidden[1] = {hidden[0, 0, 1].item():.4f}")

# Check output position 32 (k=0)
print(f"\nOutput position 32 (k=0):")
print(f"  W_down[32, 0] = {weights['W_down'][32, 0].item():.6f}")
print(f"  W_down[32, 1] = {weights['W_down'][32, 1].item():.6f}")
print(f"  contribution from unit 0: {weights['W_down'][32, 0].item() * hidden[0, 0, 0].item():.4f}")
print(f"  contribution from unit 1: {weights['W_down'][32, 1].item() * hidden[0, 0, 1].item():.4f}")
print(f"  x[32] (initial) = {x[0, 0, 32].item():.4f}")
print(f"  output_delta[32] = {output_delta[0, 0, 32].item():.4f}")
print(f"  output[32] (final) = {output[0, 0, 32].item():.4f}")
