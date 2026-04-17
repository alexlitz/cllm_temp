#!/usr/bin/env python3
"""
Test Bitwise Operations: BIT_AND, BIT_OR, BIT_XOR, SHL, SHR

Tests bitwise operations on one-hot encoded nibbles (4-bit values, 0-15).
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


def test_bit_and():
    """Test BIT_AND operation: out = a & b."""
    print("=" * 70)
    print("TEST: BIT_AND Operation (Bitwise AND)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * base  # 512 units (one cancel-pair per input pair)
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

    and_node_id = len(graph.nodes)
    and_node = IRNode(
        id=and_node_id,
        op=OpType.BIT_AND,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[and_node_id] = and_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_bit_and(and_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting BIT_AND(a, b) = a & b:")

    test_cases = [
        (15, 15, 15, "BIT_AND(0b1111, 0b1111) = 0b1111 (15)"),
        (15, 0, 0, "BIT_AND(0b1111, 0b0000) = 0b0000 (0)"),
        (12, 10, 8, "BIT_AND(0b1100, 0b1010) = 0b1000 (8)"),
        (7, 3, 3, "BIT_AND(0b0111, 0b0011) = 0b0011 (3)"),
        (5, 6, 4, "BIT_AND(0b0101, 0b0110) = 0b0100 (4)"),
        (0, 15, 0, "BIT_AND(0b0000, 0b1111) = 0b0000 (0)"),
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


def test_bit_or():
    """Test BIT_OR operation: out = a | b."""
    print("\n" + "=" * 70)
    print("TEST: BIT_OR Operation (Bitwise OR)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * base
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

    or_node_id = len(graph.nodes)
    or_node = IRNode(
        id=or_node_id,
        op=OpType.BIT_OR,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[or_node_id] = or_node

    # Compile
    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_bit_or(or_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting BIT_OR(a, b) = a | b:")

    test_cases = [
        (15, 15, 15, "BIT_OR(0b1111, 0b1111) = 0b1111 (15)"),
        (15, 0, 15, "BIT_OR(0b1111, 0b0000) = 0b1111 (15)"),
        (12, 10, 14, "BIT_OR(0b1100, 0b1010) = 0b1110 (14)"),
        (7, 3, 7, "BIT_OR(0b0111, 0b0011) = 0b0111 (7)"),
        (5, 6, 7, "BIT_OR(0b0101, 0b0110) = 0b0111 (7)"),
        (0, 0, 0, "BIT_OR(0b0000, 0b0000) = 0b0000 (0)"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(b_val, base)

        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_bit_xor():
    """Test BIT_XOR operation: out = a ^ b."""
    print("\n" + "=" * 70)
    print("TEST: BIT_XOR Operation (Bitwise XOR)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * base
    scale = E.SCALE

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

    xor_node_id = len(graph.nodes)
    xor_node = IRNode(
        id=xor_node_id,
        op=OpType.BIT_XOR,
        inputs=[a_node_id, b_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[xor_node_id] = xor_node

    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_bit_xor(xor_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting BIT_XOR(a, b) = a ^ b:")

    test_cases = [
        (15, 15, 0, "BIT_XOR(0b1111, 0b1111) = 0b0000 (0)"),
        (15, 0, 15, "BIT_XOR(0b1111, 0b0000) = 0b1111 (15)"),
        (12, 10, 6, "BIT_XOR(0b1100, 0b1010) = 0b0110 (6)"),
        (7, 3, 4, "BIT_XOR(0b0111, 0b0011) = 0b0100 (4)"),
        (5, 6, 3, "BIT_XOR(0b0101, 0b0110) = 0b0011 (3)"),
        (0, 0, 0, "BIT_XOR(0b0000, 0b0000) = 0b0000 (0)"),
    ]

    all_pass = True
    for a_val, b_val, expected, desc in test_cases:
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(b_val, base)

        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_shl():
    """Test SHL operation: out = a << n."""
    print("\n" + "=" * 70)
    print("TEST: SHL Operation (Shift Left)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * base
    scale = E.SCALE

    graph = ComputationGraph()

    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
        params={'value': 0}, physical_reg=0
    )

    n_node_id = len(graph.nodes)
    graph.nodes[n_node_id] = IRNode(
        id=n_node_id, op=OpType.CONST, inputs=[], output_reg="n",
        params={'value': 0}, physical_reg=base
    )

    shl_node_id = len(graph.nodes)
    shl_node = IRNode(
        id=shl_node_id,
        op=OpType.SHL,
        inputs=[a_node_id, n_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[shl_node_id] = shl_node

    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_shl(shl_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting SHL(a, n) = (a << n) & 0xF:")

    test_cases = [
        (1, 0, 1, "SHL(0b0001, 0) = 0b0001 (1)"),
        (1, 1, 2, "SHL(0b0001, 1) = 0b0010 (2)"),
        (1, 2, 4, "SHL(0b0001, 2) = 0b0100 (4)"),
        (1, 3, 8, "SHL(0b0001, 3) = 0b1000 (8)"),
        (1, 4, 0, "SHL(0b0001, 4) = 0b0000 (0, overflow)"),
        (3, 2, 12, "SHL(0b0011, 2) = 0b1100 (12)"),
        (7, 1, 14, "SHL(0b0111, 1) = 0b1110 (14)"),
        (15, 1, 14, "SHL(0b1111, 1) = 0b1110 (14, overflow)"),
    ]

    all_pass = True
    for a_val, n_val, expected, desc in test_cases:
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(n_val, base)

        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_shr():
    """Test SHR operation: out = a >> n."""
    print("\n" + "=" * 70)
    print("TEST: SHR Operation (Shift Right)")
    print("=" * 70)

    base = 16
    dim = E.DIM
    hidden_dim = 2 * base * base
    scale = E.SCALE

    graph = ComputationGraph()

    a_node_id = len(graph.nodes)
    graph.nodes[a_node_id] = IRNode(
        id=a_node_id, op=OpType.CONST, inputs=[], output_reg="a",
        params={'value': 0}, physical_reg=0
    )

    n_node_id = len(graph.nodes)
    graph.nodes[n_node_id] = IRNode(
        id=n_node_id, op=OpType.CONST, inputs=[], output_reg="n",
        params={'value': 0}, physical_reg=base
    )

    shr_node_id = len(graph.nodes)
    shr_node = IRNode(
        id=shr_node_id,
        op=OpType.SHR,
        inputs=[a_node_id, n_node_id],
        output_reg="result",
        params={'base': base},
        physical_reg=2*base,
        gate=None
    )
    graph.nodes[shr_node_id] = shr_node

    emitter = WeightEmitter(dim, hidden_dim, scale)
    emitter.emit_shr(shr_node, graph)

    weights = {
        'W_up': emitter.W_up,
        'b_up': emitter.b_up,
        'W_gate': emitter.W_gate,
        'b_gate': emitter.b_gate,
        'W_down': emitter.W_down,
        'b_down': emitter.b_down,
    }

    print(f"\nTesting SHR(a, n) = a >> n:")

    test_cases = [
        (8, 0, 8, "SHR(0b1000, 0) = 0b1000 (8)"),
        (8, 1, 4, "SHR(0b1000, 1) = 0b0100 (4)"),
        (8, 2, 2, "SHR(0b1000, 2) = 0b0010 (2)"),
        (8, 3, 1, "SHR(0b1000, 3) = 0b0001 (1)"),
        (8, 4, 0, "SHR(0b1000, 4) = 0b0000 (0)"),
        (15, 1, 7, "SHR(0b1111, 1) = 0b0111 (7)"),
        (15, 2, 3, "SHR(0b1111, 2) = 0b0011 (3)"),
        (12, 2, 3, "SHR(0b1100, 2) = 0b0011 (3)"),
    ]

    all_pass = True
    for a_val, n_val, expected, desc in test_cases:
        x = torch.zeros(1, 1, dim)
        x[0, 0, 0:base] = onehot_encode(a_val, base)
        x[0, 0, base:2*base] = onehot_encode(n_val, base)

        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        result_vec = output[0, 0, 2*base:3*base]
        result = onehot_decode(result_vec)

        match = (result == expected)
        all_pass = all_pass and match

        print(f"  {desc}: {result} (expected {expected}) {'✓' if match else '✗'}")

    print(f"\n{'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def run_all_tests():
    """Run all bitwise operation tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 22 + "Bitwise Operations Tests" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []
    results.append(("BIT_AND Operation", test_bit_and()))
    results.append(("BIT_OR Operation", test_bit_or()))
    results.append(("BIT_XOR Operation", test_bit_xor()))
    results.append(("SHL Operation", test_shl()))
    results.append(("SHR Operation", test_shr()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\nResults: {passing}/{total} test groups passing")

    if passing == total:
        print("\n✅ All bitwise operations working correctly!")
        print("   BIT_AND, BIT_OR, BIT_XOR, SHL, SHR enable bit manipulation")
    else:
        print(f"\n⚠️  {total - passing} test group(s) failed")

    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
