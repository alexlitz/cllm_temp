#!/usr/bin/env python3
"""
Test One-Hot Operations: MUL, DIV, MOD

Tests operations on one-hot encoded nibbles (values 0-15).
"""

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


def onehot_decode(vec):
    """Convert one-hot vector to integer."""
    return torch.argmax(vec).item()


def test_mul():
    """Test MUL operation: out = (a * b) mod base."""
    print("=" * 70)
    print("TEST: MUL Operation (One-Hot Multiplication)")
    print("=" * 70)

    base = 16
    dim = E.DIM  # Should be >= 3*base = 48
    hidden_dim = 2 * base * base  # 512 units for MUL (one cancel-pair per i,j pair)
    scale = E.SCALE

    # Create graph
    graph = ComputationGraph()

    # Input nodes (one-hot vectors)
    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
        params={'value': 0}, physical_reg=0  # a occupies regs 0-15
    )

    b_node_id = len(graph.nodes)
    graph.nodes[b_node_id] = IRNode(
        id=b_node_id, op=OpType.CONST, inputs=[], output_reg="b",
        params={'value': 0}, physical_reg=base  # b occupies regs 16-31
    )

    # MUL operation
    mul_node_id = len(graph.nodes)
    mul_node = IRNode(
        id=mul_node_id,
        op=OpType.MUL,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,  # result occupies regs 32-47
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

    print(f"\nTesting MUL(a, b) = (a * b) mod {base}:")

    test_cases = [
        (3, 4, 12, "MUL(3, 4) = 12"),
        (5, 3, 15, "MUL(5, 3) = 15"),
        (7, 7, 1, "MUL(7, 7) = 49 mod 16 = 1"),
        (15, 1, 15, "MUL(15, 1) = 15"),
        (8, 2, 0, "MUL(8, 2) = 16 mod 16 = 0"),
        (0, 10, 0, "MUL(0, 10) = 0"),
        (1, 1, 1, "MUL(1, 1) = 1"),
        (4, 4, 0, "MUL(4, 4) = 16 mod 16 = 0"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        # Create input with one-hot encoding
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(b_val, base)

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        # Decode output
        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_div():
    """Test DIV operation: out = a // b (integer division)."""
    print("\n" + "=" * 70)
    print("TEST: DIV Operation (One-Hot Division)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * (base - 1)  # 480 units for DIV (exclude div by 0)
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
        params={'value': 0}, physical_reg=base
    )

    div_node_id = len(graph.nodes)
    div_node = IRNode(
        id=div_node_id,
        op=OpType.DIV,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[div_node_id] = div_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_div(div_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting DIV(a, b) = a // b:")

    test_cases = [
        (12, 3, 4, "DIV(12, 3) = 4"),
        (15, 4, 3, "DIV(15, 4) = 3"),
        (7, 2, 3, "DIV(7, 2) = 3"),
        (0, 5, 0, "DIV(0, 5) = 0"),
        (10, 1, 10, "DIV(10, 1) = 10"),
        (15, 15, 1, "DIV(15, 15) = 1"),
        (8, 3, 2, "DIV(8, 3) = 2"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        if b_val == 0:  # Skip division by zero
            continue

        # Create input with one-hot encoding
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(b_val, base)

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        # Decode output
        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_mod():
    """Test MOD operation: out = a mod b."""
    print("\n" + "=" * 70)
    print("TEST: MOD Operation (One-Hot Modulo)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * (base - 1)  # 480 units for MOD (exclude mod by 0)
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
        params={'value': 0}, physical_reg=base
    )

    mod_node_id = len(graph.nodes)
    mod_node = IRNode(
        id=mod_node_id,
        op=OpType.MOD,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[mod_node_id] = mod_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_mod(mod_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting MOD(a, b) = a mod b:")

    test_cases = [
        (12, 5, 2, "MOD(12, 5) = 2"),
        (15, 4, 3, "MOD(15, 4) = 3"),
        (7, 3, 1, "MOD(7, 3) = 1"),
        (0, 5, 0, "MOD(0, 5) = 0"),
        (10, 10, 0, "MOD(10, 10) = 0"),
        (13, 7, 6, "MOD(13, 7) = 6"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        if b_val == 0:  # Skip modulo by zero
            continue

        # Create input with one-hot encoding
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(b_val, base)

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        # Decode output
        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def run_all_tests():
    """Run all one-hot operation tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "One-Hot Operations Tests" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []
    results.append(("MUL Operation", test_mul()))
    results.append(("DIV Operation", test_div()))
    results.append(("MOD Operation", test_mod()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\nResults: {passing}/{total} test groups passing")

    if passing == total:
        print("\n✅ All one-hot operations working correctly!")
        print("   MUL, DIV, MOD enable arithmetic on encoded values")
    else:
        print(f"\n⚠️  {total - passing} test group(s) failed")

    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
