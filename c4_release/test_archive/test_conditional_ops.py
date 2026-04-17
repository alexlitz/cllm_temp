#!/usr/bin/env python3
"""
Test Conditional Operations: SELECT, IF_THEN

Tests newly implemented conditional operations that enable control flow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter
from neural_vm.embedding import E


def test_select():
    """Test SELECT operation: out = cond ? a : b."""
    print("=" * 70)
    print("TEST: SELECT Operation (Ternary Conditional)")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    # Create graph
    graph = ComputationGraph()

    # Input nodes
    cond_node_id = len(graph.nodes)
    graph.nodes[cond_node_id] = IRNode(
        id=cond_node_id, op=OpType.CONST, inputs=[], output_reg="cond",
        params={'value': 0}, physical_reg=0
    )

    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
        params={'value': 0}, physical_reg=1
    )

    b_node_id = len(graph.nodes)
    graph.nodes[b_node_id] = IRNode(
        id=b_node_id, op=OpType.CONST, inputs=[], output_reg="b",
        params={'value': 0}, physical_reg=2
    )

    # SELECT operation
    select_node_id = len(graph.nodes)
    select_node = IRNode(
        id=select_node_id,
        op=OpType.SELECT,
        inputs=[cond_node_id, a_node_id, b_node_id],
        output_reg="result",
        params={},
        physical_reg=3,
        gate=None
    )
    graph.nodes[select_node_id] = select_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_select(select_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting SELECT(cond, a, b) = cond ? a : b:")

    test_cases = [
        (1, 42, 99, 42, "SELECT(true, 42, 99) = 42"),
        (0, 42, 99, 99, "SELECT(false, 42, 99) = 99"),
        (1, -10, 20, -10, "SELECT(true, -10, 20) = -10"),
        (0, -10, 20, 20, "SELECT(false, -10, 20) = 20"),
        (1, 0, 100, 0, "SELECT(true, 0, 100) = 0"),
        (0, 0, 100, 100, "SELECT(false, 0, 100) = 100"),
        (1, 100, 0, 100, "SELECT(true, 100, 0) = 100"),
    ]

    all_pass = True
    for cond_val, a_val, b_val, expected, desc in test_cases:
        # Create input
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = cond_val
        x[0, 0, 1] = a_val
        x[0, 0, 2] = b_val

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 3].item()

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_if_then():
    """Test IF_THEN operation: out = cond ? a : out (conditional update)."""
    print("\n" + "=" * 70)
    print("TEST: IF_THEN Operation (Conditional Update)")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    # Create graph
    graph = ComputationGraph()

    cond_node_id = len(graph.nodes)
    graph.nodes[cond_node_id] = IRNode(
        id=cond_node_id, op=OpType.CONST, inputs=[], output_reg="cond",
        params={'value': 0}, physical_reg=0
    )

    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
        params={'value': 0}, physical_reg=1
    )

    # IF_THEN operation
    if_then_node_id = len(graph.nodes)
    if_then_node = IRNode(
        id=if_then_node_id,
        op=OpType.IF_THEN,
        inputs=[cond_node_id, a_node_id],
        output_reg="result",
        params={},
        physical_reg=2,
        gate=None
    )
    graph.nodes[if_then_node_id] = if_then_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_if_then(if_then_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting IF_THEN(cond, a) with initial value:")

    test_cases = [
        (1, 42, 0, 42, "IF_THEN(true, 42) with init=0 → 42"),
        (0, 42, 0, 0, "IF_THEN(false, 42) with init=0 → 0 (unchanged)"),
        (1, 100, 50, 100, "IF_THEN(true, 100) with init=50 → 100"),
        (0, 100, 50, 50, "IF_THEN(false, 100) with init=50 → 50 (unchanged)"),
        (1, -10, 20, -10, "IF_THEN(true, -10) with init=20 → -10"),
        (0, -10, 20, 20, "IF_THEN(false, -10) with init=20 → 20 (unchanged)"),
    ]

    all_pass = True
    for cond_val, a_val, initial_val, expected, desc in test_cases:
        # Create input with initial value in result register
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = cond_val
        x[0, 0, 1] = a_val
        x[0, 0, 2] = initial_val  # Initial value in result register

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 2].item()

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def run_all_tests():
    """Run all conditional operation tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 16 + "Conditional Operations Tests" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []
    results.append(("SELECT Operation", test_select()))
    results.append(("IF_THEN Operation", test_if_then()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\nResults: {passing}/{total} test groups passing")

    if passing == total:
        print("\n✅ All conditional operations working correctly!")
        print("   SELECT, IF_THEN enable control flow in the compiler")
    else:
        print(f"\n⚠️  {total - passing} test group(s) failed")

    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
