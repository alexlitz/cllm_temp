#!/usr/bin/env python3
"""
Test Graph Compiler with VM-Style Pre-loaded Inputs

This test demonstrates that the compiled operations work correctly when
inputs are pre-loaded into the token (like the existing VM architecture).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import GraphWeightCompiler
from neural_vm.embedding import E

def test_add_vm_style():
    """Test ADD with inputs pre-loaded in token (VM style)."""
    print("=" * 70)
    print("TEST: ADD with Pre-loaded Inputs (VM Style)")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    # Create compiler and compile just ADD (no CONST)
    compiler = GraphWeightCompiler(dim, hidden_dim, scale)

    # Manually create nodes without using the high-level API
    # This simulates compiling just the ADD operation
    from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph

    graph = ComputationGraph()

    # Virtual registers map directly to input dimensions
    # Assume a is at dim[0], b is at dim[1], result goes to dim[2]

    # Create "input" nodes that represent dimensions (no actual operation)
    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id,
        op=OpType.CONST,  # Placeholder
        inputs=[],
        output_reg="a",
        params={'value': 0},  # Value will be in input
        physical_reg=0  # dim[0]
    )

    b_node_id = len(graph.nodes)
    graph.nodes[b_node_id] = IRNode(
        id=b_node_id,
        op=OpType.CONST,  # Placeholder
        inputs=[],
        output_reg="b",
        params={'value': 0},
        physical_reg=1  # dim[1]
    )

    # ADD operation
    add_node_id = len(graph.nodes)
    graph.nodes[add_node_id] = IRNode(
        id=add_node_id,
        op=OpType.ADD,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={},
        physical_reg=2,  # dim[2]
        gate=None
    )

    # Compile just the ADD operation (skip CONST nodes)
    from neural_vm.graph_weight_compiler import WeightEmitter
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_add(graph.nodes[add_node_id], graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nWeight structure:")
    print(f"  Non-zero W_up: {(weights['W_up'] != 0).sum().item()}")
    print(f"  Non-zero b_up: {(weights['b_up'] != 0).sum().item()}")
    print(f"  Non-zero W_gate: {(weights['W_gate'] != 0).sum().item()}")
    print(f"  Non-zero b_gate: {(weights['b_gate'] != 0).sum().item()}")
    print(f"  Non-zero W_down: {(weights['W_down'] != 0).sum().item()}")

    # Create input token with a=5, b=10 PRE-LOADED
    x = torch.zeros(1, 1, dim)
    x[0, 0, 0] = 5.0   # a at dim[0]
    x[0, 0, 1] = 10.0  # b at dim[1]

    print(f"\nInput token:")
    print(f"  dim[0] (a) = {x[0, 0, 0].item()}")
    print(f"  dim[1] (b) = {x[0, 0, 1].item()}")

    # Forward pass
    up = F.linear(x, weights['W_up'], weights['b_up'])
    gate = F.linear(x, weights['W_gate'], weights['b_gate'])
    hidden = F.silu(up) * gate
    output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
    output = x + output_delta

    result = output[0, 0, 2].item()  # Read from dim[2]

    print(f"\nResult:")
    print(f"  dim[2] (result) = {result:.4f}")
    print(f"  Expected: 15.0")
    print(f"  Match: {abs(result - 15.0) < 0.1}")

    success = abs(result - 15.0) < 0.1
    print(f"\n{'✓' if success else '✗'} Test {'PASSED' if success else 'FAILED'}")
    return success


def test_cmp_ge_vm_style():
    """Test CMP_GE with inputs pre-loaded in token (VM style)."""
    print("\n" + "=" * 70)
    print("TEST: CMP_GE with Pre-loaded Inputs (VM Style)")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    # Compile just CMP_GE operation
    from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter

    graph = ComputationGraph()

    # Input nodes
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

    # CMP_GE operation
    cmp_node_id = len(graph.nodes)
    graph.nodes[cmp_node_id] = IRNode(
        id=cmp_node_id,
        op=OpType.CMP_GE,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={},
        physical_reg=2,
        gate=None
    )

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_cmp_ge(graph.nodes[cmp_node_id], graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting various inputs:")

    test_cases = [
        (10.0, 16.0, 0.0, "10 >= 16"),
        (16.0, 16.0, 1.0, "16 >= 16"),
        (20.0, 16.0, 1.0, "20 >= 16"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        # Create input with values pre-loaded
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = a_val  # a at dim[0]
        x[0, 0, 1] = b_val  # b at dim[1]

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 2].item()  # Read from dim[2]

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected:.4f}) {'✓' if match else '✗'}")

    print(f"\n{'✓' if all_pass else '✗'} Test {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def run_all_tests():
    """Run all VM-style tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Graph Compiler VM-Style Tests" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []
    results.append(("ADD with pre-loaded inputs", test_add_vm_style()))
    results.append(("CMP_GE with pre-loaded inputs", test_cmp_ge_vm_style()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\nResults: {passing}/{total} tests passing")

    if passing == total:
        print("\n✅ All operations work correctly with pre-loaded inputs!")
        print("   The compiler weights are correct.")
        print("   Issue: Architecture mismatch (operations can't chain in single pass)")
    else:
        print(f"\n⚠️  {total - passing} test(s) failed")

    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
