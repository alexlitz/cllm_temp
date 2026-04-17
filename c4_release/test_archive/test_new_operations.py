#!/usr/bin/env python3
"""
Test Newly Implemented Operations (CMP_LE, CMP_GT, CMP_NE, XOR)

Tests all comparison operations and XOR with VM-style pre-loaded inputs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter
from neural_vm.embedding import E


def test_comparison_ops():
    """Test all comparison operations: ==, !=, <, <=, >, >="""
    print("=" * 70)
    print("TEST: All Comparison Operations")
    print("=" * 70)

    dim = E.DIM
    hidden_dim = 512
    scale = E.SCALE

    # Test cases: (a, b, expected results for each op)
    test_cases = [
        (5.0, 10.0, {"==": 0, "!=": 1, "<": 1, "<=": 1, ">": 0, ">=": 0}),
        (10.0, 10.0, {"==": 1, "!=": 0, "<": 0, "<=": 1, ">": 0, ">=": 1}),
        (15.0, 10.0, {"==": 0, "!=": 1, "<": 0, "<=": 0, ">": 1, ">=": 1}),
    ]

    # Operations to test
    ops = [
        (OpType.CMP_EQ, "=="),
        (OpType.CMP_NE, "!="),
        (OpType.CMP_LT, "<"),
        (OpType.CMP_LE, "<="),
        (OpType.CMP_GT, ">"),
        (OpType.CMP_GE, ">="),
    ]

    all_pass = True

    for op_type, op_sym in ops:
        print(f"\nTesting {op_sym} operation:")

        # Create graph with placeholder input nodes
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

        # Add comparison operation
        cmp_node_id = len(graph.nodes)
        graph.nodes[cmp_node_id] = IRNode(
            id=cmp_node_id,
            op=op_type,
            inputs=[a_node_id, b_node_id],
            output_reg="result",
            params={},
            physical_reg=2,
            gate=None
        )

        # Compile
        emitter = WeightEmitter(dim, hidden_dim, scale)

        # Dispatch to appropriate emitter
        if op_type == OpType.CMP_EQ:
            emitter.emit_cmp_eq(graph.nodes[cmp_node_id], graph)
        elif op_type == OpType.CMP_NE:
            emitter.emit_cmp_ne(graph.nodes[cmp_node_id], graph)
        elif op_type == OpType.CMP_LT:
            emitter.emit_cmp_lt(graph.nodes[cmp_node_id], graph)
        elif op_type == OpType.CMP_LE:
            emitter.emit_cmp_le(graph.nodes[cmp_node_id], graph)
        elif op_type == OpType.CMP_GT:
            emitter.emit_cmp_gt(graph.nodes[cmp_node_id], graph)
        elif op_type == OpType.CMP_GE:
            emitter.emit_cmp_ge(graph.nodes[cmp_node_id], graph)

        weights = {
            'W_up': emitter.W_up,
            'b_up': emitter.b_up,
            'W_gate': emitter.W_gate,
            'b_gate': emitter.b_gate,
            'W_down': emitter.W_down,
            'b_down': emitter.b_down,
        }

        # Test each case
        op_pass = True
        for a_val, b_val, expected_dict in test_cases:
            expected = expected_dict[op_sym]

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
            op_pass = op_pass and match
            all_pass = all_pass and match

            status = "✓" if match else "✗"
            print(f"  {a_val:.0f} {op_sym} {b_val:.0f} = {result:.4f} (expected {expected}) {status}")

        print(f"  {'✓ PASSED' if op_pass else '✗ FAILED'}")

    print(f"\n{'✓' if all_pass else '✗'} All comparison operations {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def test_xor():
    """Test XOR operation."""
    print("\n" + "=" * 70)
    print("TEST: XOR Operation")
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

    b_node_id = len(graph.nodes)
    graph.nodes[b_node_id] = IRNode(
        id=b_node_id, op=OpType.CONST, inputs=[], output_reg="b",
        params={'value': 0}, physical_reg=1
    )

    xor_node_id = len(graph.nodes)
    graph.nodes[xor_node_id] = IRNode(
        id=xor_node_id,
        op=OpType.XOR,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={},
        physical_reg=2,
        gate=None
    )

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_logical_xor(graph.nodes[xor_node_id], graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting XOR truth table:")

    test_cases = [
        (0.0, 0.0, 0.0, "0 XOR 0 = 0"),
        (0.0, 1.0, 1.0, "0 XOR 1 = 1"),
        (1.0, 0.0, 1.0, "1 XOR 0 = 1"),
        (1.0, 1.0, 0.0, "1 XOR 1 = 0"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        # Create input
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = a_val
        x[0, 0, 1] = b_val

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 2].item()

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected:.1f}) {'✓' if match else '✗'}")

    print(f"\n{'✓' if all_pass else '✗'} XOR operation {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def test_sub():
    """Test SUB operation."""
    print("\n" + "=" * 70)
    print("TEST: SUB Operation")
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

    b_node_id = len(graph.nodes)
    graph.nodes[b_node_id] = IRNode(
        id=b_node_id, op=OpType.CONST, inputs=[], output_reg="b",
        params={'value': 0}, physical_reg=1
    )

    sub_node_id = len(graph.nodes)
    graph.nodes[sub_node_id] = IRNode(
        id=sub_node_id,
        op=OpType.SUB,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={},
        physical_reg=2,
        gate=None
    )

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_sub(graph.nodes[sub_node_id], graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print("\nTesting SUB operation:")

    test_cases = [
        (10.0, 3.0, 7.0, "10 - 3 = 7"),
        (5.0, 10.0, -5.0, "5 - 10 = -5"),
        (20.0, 20.0, 0.0, "20 - 20 = 0"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        # Create input
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0] = a_val
        x[0, 0, 1] = b_val

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result = output[0, 0, 2].item()

        match = abs(result - expected) < 0.1
        all_pass = all_pass and match

        print(f"  {desc}: {result:.4f} (expected {expected:.1f}) {'✓' if match else '✗'}")

    print(f"\n{'✓' if all_pass else '✗'} SUB operation {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def run_all_tests():
    """Run all tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "New Operations Tests" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []
    results.append(("All Comparison Operations", test_comparison_ops()))
    results.append(("XOR Operation", test_xor()))
    results.append(("SUB Operation", test_sub()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\nResults: {passing}/{total} test groups passing")

    if passing == total:
        print("\n✅ All new operations implemented correctly!")
        print("   Implemented: CMP_LE, CMP_GT, CMP_NE, XOR, SUB")
        print("   Total operations: 13/28 primitives (46%)")
    else:
        print(f"\n⚠️  {total - passing} test group(s) failed")

    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
