#!/usr/bin/env python3
"""Debug CMP_EQ to understand why it returns 0.5 for unequal values."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter
from neural_vm.embedding import E

dim = E.DIM
hidden_dim = 512
scale = E.SCALE

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
    params={'value': 0}, physical_reg=1
)

cmp_node_id = len(graph.nodes)
graph.nodes[cmp_node_id] = IRNode(
    id=cmp_node_id,
    op=OpType.CMP_EQ,
    inputs=[a_node_id, b_node_id],
    output_reg="result",
    params={},
    physical_reg=2,
    gate=None
)

# Compile
emitter = WeightEmitter(dim, hidden_dim, scale)
emitter.emit_cmp_eq(graph.nodes[cmp_node_id], graph)

weights = {
    'W_up': emitter.W_up,
    'b_up': emitter.b_up,
    'W_gate': emitter.W_gate,
    'b_gate': emitter.b_gate,
    'W_down': emitter.W_down,
    'b_down': emitter.b_down,
}

print("=" * 70)
print("DEBUG: CMP_EQ Pattern")
print("=" * 70)

print("\nWeight structure:")
print(f"First 4 b_up values: {weights['b_up'][:4]}")
print(f"First 4 b_gate values: {weights['b_gate'][:4]}")

print("\nW_up non-zero entries:")
for unit in range(4):
    nz = torch.nonzero(weights['W_up'][unit], as_tuple=False)
    if len(nz) > 0:
        for idx in nz:
            dim_idx = idx[0].item()
            value = weights['W_up'][unit, dim_idx].item()
            print(f"  W_up[{unit}, {dim_idx}] = {value:.1f}")

print("\nW_down non-zero entries:")
for dim_idx in range(3):
    nz = torch.nonzero(weights['W_down'][dim_idx], as_tuple=False)
    if len(nz) > 0:
        for idx in nz:
            unit = idx[0].item()
            value = weights['W_down'][dim_idx, unit].item()
            print(f"  W_down[{dim_idx}, {unit}] = {value:.4f}")

# Test with a=5, b=10 (unequal)
print("\n" + "=" * 70)
print("TEST: a=5, b=10 (unequal, should give 0)")
print("=" * 70)

a_val, b_val = 5.0, 10.0
x = torch.zeros(1, 1, dim)
x[0, 0, 0] = a_val
x[0, 0, 1] = b_val

# Forward pass with detailed trace
up = F.linear(x, weights['W_up'], weights['b_up'])
print(f"\nAfter W_up @ x + b_up:")
for unit in range(4):
    print(f"  up[{unit}] = {up[0, 0, unit].item():.2f}")
    print(f"    Calculation: {weights['W_up'][unit, 0].item():.1f}*{a_val} + {weights['W_up'][unit, 1].item():.1f}*{b_val} + {weights['b_up'][unit].item():.1f}")
    expected_up = weights['W_up'][unit, 0].item()*a_val + weights['W_up'][unit, 1].item()*b_val + weights['b_up'][unit].item()
    print(f"    = {expected_up:.2f}")

up_activated = F.silu(up)
print(f"\nAfter silu(up):")
for unit in range(4):
    print(f"  silu(up[{unit}]) = {up_activated[0, 0, unit].item():.2f}")

gate = F.linear(x, weights['W_gate'], weights['b_gate'])
print(f"\nAfter W_gate @ x + b_gate:")
for unit in range(4):
    print(f"  gate[{unit}] = {gate[0, 0, unit].item():.2f}")

hidden = up_activated * gate
print(f"\nAfter silu(up) * gate:")
for unit in range(4):
    print(f"  hidden[{unit}] = {hidden[0, 0, unit].item():.2f}")

output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
print(f"\nAfter W_down @ hidden:")
print(f"  output_delta[2] = {output_delta[0, 0, 2].item():.4f}")
print(f"    Calculation:")
for unit in range(4):
    contrib = hidden[0, 0, unit].item() * weights['W_down'][2, unit].item()
    print(f"      Unit {unit}: {hidden[0, 0, unit].item():.2f} * {weights['W_down'][2, unit].item():.4f} = {contrib:.4f}")

output = x + output_delta
result = output[0, 0, 2].item()

print(f"\nFinal result: {result:.4f}")
print(f"Expected: 0.0")
print(f"Error: {abs(result - 0.0):.4f}")

# Analyze the pattern
print("\n" + "=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)

S = scale
a, b = a_val, b_val

print(f"\nTheoretical step function values:")
print(f"  Unit 0: step(a-b >= 0) = step({a}-{b} >= 0) = step({a-b} >= 0) = {1 if a-b >= 0 else 0}")
print(f"  Unit 1: step(b-a >= 0) = step({b}-{a} >= 0) = step({b-a} >= 0) = {1 if b-a >= 0 else 0}")
print(f"  Unit 2: step(a-b >= 1) = step({a}-{b} >= 1) = step({a-b} >= 1) = {1 if a-b >= 1 else 0}")
print(f"  Unit 3: step(b-a >= 1) = step({b}-{a} >= 1) = step({b-a} >= 1) = {1 if b-a >= 1 else 0}")

print(f"\nExpected output: ({1 if a-b >= 0 else 0} + {1 if b-a >= 0 else 0} - {1 if a-b >= 1 else 0} - {1 if b-a >= 1 else 0}) * 0.5")
expected_sum = (1 if a-b >= 0 else 0) + (1 if b-a >= 0 else 0) - (1 if a-b >= 1 else 0) - (1 if b-a >= 1 else 0)
print(f"           = {expected_sum} * 0.5 = {expected_sum * 0.5}")

print(f"\nActual output: {result:.4f}")
