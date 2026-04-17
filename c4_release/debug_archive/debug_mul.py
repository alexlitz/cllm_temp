#!/usr/bin/env python3
"""Debug MUL operation."""

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

print(f"Testing MUL(3, 4) = 12 with base={base}, scale={scale}")
print(f"dim={dim}, hidden_dim={hidden_dim}")
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

# Check weight setup for output 12
k = 12
unit_pos = 2 * k
unit_neg = 2 * k + 1

print(f"Checking weights for output k={k}:")
print(f"  unit_pos={unit_pos}, unit_neg={unit_neg}")
print()

# Check which input pairs activate these units
print("Non-zero W_up entries for unit_pos:")
for i in range(dim):
    if weights['W_up'][unit_pos, i] != 0:
        print(f"  W_up[{unit_pos}, {i}] = {weights['W_up'][unit_pos, i].item()}")

print("\nNon-zero W_down entries for output position 44 (32 + 12):")
for i in range(hidden_dim):
    if weights['W_down'][44, i] != 0:
        print(f"  W_down[44, {i}] = {weights['W_down'][44, i].item()}")

# Create input
a_val = 3
b_val = 4
x = torch.zeros(1, 1, dim)
x[0, 0, 0:base] = onehot_encode(a_val, base)
x[0, 0, base:2*base] = onehot_encode(b_val, base)

print(f"\nInput encoding:")
print(f"  a={a_val} → x[0,0,{a_val}] = {x[0, 0, a_val].item()}")
print(f"  b={b_val} → x[0,0,{base + b_val}] = {x[0, 0, base + b_val].item()}")

# Forward pass
up = F.linear(x, weights['W_up'], weights['b_up'])
gate = F.linear(x, weights['W_gate'], weights['b_gate'])
hidden = F.silu(up) * gate
output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
output = x + output_delta

# Check activations for unit_pos
print(f"\nActivations for unit {unit_pos}:")
print(f"  up[{unit_pos}] = {up[0, 0, unit_pos].item():.2f}")
print(f"  gate[{unit_pos}] = {gate[0, 0, unit_pos].item():.2f}")
print(f"  hidden[{unit_pos}] = {hidden[0, 0, unit_pos].item():.2f}")

print(f"\nOutput:")
print(f"  output_delta[44] = {output_delta[0, 0, 44].item():.4f}")
print(f"  output[44] = {output[0, 0, 44].item():.4f}")

# Check all output positions
print(f"\nAll output positions (32-47):")
for i in range(base):
    val = output[0, 0, 32 + i].item()
    if abs(val) > 0.01:
        print(f"  output[{32 + i}] (k={i}) = {val:.4f}")

# Decode output
result_vec = output[0, 0, 2*base:3*base]
result = torch.argmax(result_vec).item()
print(f"\nDecoded result: {result} (expected 12)")
