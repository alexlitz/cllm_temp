#!/usr/bin/env python3
"""
Test Register Operations: NOT, MOVE, CLEAR

Tests the remaining implemented-but-untested operations to verify they work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter
from neural_vm.embedding import E


def test_not():
    """Test NOT operation."""
    print("=" * 70)
    print("TEST: NOT Operation")
    print("=" * 70)

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

    not_node_id = len(graph.nodes)
    not_node = IRNode(
        id=not_node_id,
        op=OpType.NOT,
        inputs=[a_node_id],
        output_reg="result",
        params={},
        physical_reg=1,
        gate=None
    )
    graph.nodes[not_node_id] = not_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_logical_not(not_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting NOT truth table:")

    test_cases = [
        (0.0, 1.0, "NOT 0 = 1"),
        (1.0, 0.0, "NOT 1 = 0"),
        (0.5, 0.0, "NOT 0.5 = 0 (any positive is true)"),
        (2.0, 0.0, "NOT 2 = 0 (any positive is true)"),
    ]

    all_pass = True
    for a_val, expected, desc in test_cases:
        # Create input
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = a_val

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 1].item()

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected:.1f}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_move():
    """Test MOVE operation."""
    print("\n" + "=" * 70)
    print("TEST: MOVE Operation")
    print("=" * 70)

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

    move_node_id = len(graph.nodes)
    move_node = IRNode(
        id=move_node_id,
        op=OpType.MOVE,
        inputs=[a_node_id],
        output_reg="result",
        params={},
        physical_reg=1,
        gate=None
    )
    graph.nodes[move_node_id] = move_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_move(move_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting MOVE (copy register):")

    test_cases = [
        (0.0, 0.0, "MOVE 0 → result"),
        (1.0, 1.0, "MOVE 1 → result"),
        (42.0, 42.0, "MOVE 42 → result"),
        (-10.0, -10.0, "MOVE -10 → result"),
        (3.14159, 3.14159, "MOVE π → result"),
    ]

    all_pass = True
    for a_val, expected, desc in test_cases:
        # Create input
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = a_val

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 1].item()

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected:.2f}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_clear():
    """Test CLEAR operation."""
    print("\n" + "=" * 70)
    print("TEST: CLEAR Operation")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    # Create graph
    graph = ComputationGraph()

    # CLEAR doesn't need inputs, but let's create a dummy input
    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
        params={'value': 0}, physical_reg=0
    )

    clear_node_id = len(graph.nodes)
    clear_node = IRNode(
        id=clear_node_id,
        op=OpType.CLEAR,
        inputs=[],  # CLEAR takes no inputs
        output_reg="result",
        params={},
        physical_reg=1,
        gate=None
    )
    graph.nodes[clear_node_id] = clear_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_clear(clear_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting CLEAR (set register to 0):")

    # Test with different initial values - should all clear to 0
    test_cases = [
        (0.0, "CLEAR when already 0"),
        (42.0, "CLEAR when set to 42"),
        (-10.0, "CLEAR when set to -10"),
        (100.0, "CLEAR when set to 100"),
    ]

    all_pass = True
    for initial_val, desc in test_cases:
        # Create input with initial value in target register
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = 999.0  # Some value in input register (shouldn't matter)
        x[0, 0, 1] = initial_val  # Initial value in target register

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 1].item()

        # CLEAR should set to 0 regardless of initial value
        match = abs(result) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected 0.0) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_const():
    """Test CONST operation (used for testing infrastructure)."""
    print("\n" + "=" * 70)
    print("TEST: CONST Operation")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    print("\nTesting CONST (load immediate value):")

    test_cases = [
        (0.0, "CONST 0"),
        (1.0, "CONST 1"),
        (42.0, "CONST 42"),
        (-10.0, "CONST -10"),
        (3.14159, "CONST π"),
    ]

    all_pass = True
    for const_val, desc in test_cases:
        # Create graph
        graph = ComputationGraph()

        const_node_id = len(graph.nodes)
        const_node = IRNode(
            id=const_node_id,
            op=OpType.CONST,
            inputs=[],
            output_reg="result",
            params={'value': const_val},
            physical_reg=0,
            gate=None
        )
        graph.nodes[const_node_id] = const_node

        # Compile
        emitter = WeightEmitter(dim, hidden_dim, scale)
        emitter.emit_const(const_node, graph)

        weights = {
            'W_up': emitter.W_up,
            'b_up': emitter.b_up,
            'W_gate': emitter.W_gate,
            'b_gate': emitter.b_gate,
            'W_down': emitter.W_down,
            'b_down': emitter.b_down,
        }

        # Create input (empty)
        x = torch.zeros(1, 1, dim)

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 0].item()

        match = abs(result - const_val) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {const_val:.2f}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def run_all_tests():
    """Run all register operation tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "Register Operations Tests" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []
    results.append(("NOT Operation", test_not()))
    results.append(("MOVE Operation", test_move()))
    results.append(("CLEAR Operation", test_clear()))
    results.append(("CONST Operation", test_const()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\nResults: {passing}/{total} test groups passing")

    if passing == total:
        print("\n✅ All register operations working correctly!")
        print("   NOT, MOVE, CLEAR, CONST all verified")
    else:
        print(f"\n⚠️  {total - passing} test group(s) failed")

    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
