#!/usr/bin/env python3
"""
Test Graph Weight Compiler - Fixed Architecture

Key insight: FFN is a single-pass transformation.
- Inputs must be in the embedding (x)
- FFN transforms inputs to outputs
- CONST operations produce outputs via bias (no inputs needed)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import GraphWeightCompiler


def swiglu_forward(x, W_up, b_up, W_gate, b_gate, W_down, b_down):
    """SwiGLU forward pass."""
    up = F.linear(x, W_up, b_up)
    gate = F.linear(x, W_gate, b_gate)
    activated = F.silu(up) * torch.sigmoid(gate)
    out = F.linear(activated, W_down, b_down)
    return out


def test_add_with_input_embeddings():
    """Test ADD: c = a + b, where a and b are in input embedding."""
    print("=" * 70)
    print("TEST 1: ADD with Input Embeddings")
    print("=" * 70)

    dim = 512
    hidden_dim = 4096
    S = 5.0

    # Compile: result = input_a + input_b
    # (no CONST, just read from input)
    compiler = GraphWeightCompiler(dim, hidden_dim, S)

    # We need input nodes that represent existing embeddings
    # Let me create a simpler test: directly add two input dimensions

    # Actually, the issue is that our compiler API assumes we're creating
    # values with operations. Let me think about this differently...

    # For pure FFN, inputs should already exist in the embedding.
    # Let's just test the weight patterns directly.

    print("Testing weight emission patterns directly...")
    print()

    # Test 1: Simple ADD weight pattern
    from neural_vm.graph_weight_compiler import (
        ComputationGraph, OpType, WeightEmitter, RegisterAllocator
    )

    graph = ComputationGraph()

    # Create "input" nodes that represent embedding dimensions
    a_id = graph.add_node(OpType.CONST, [], "a", params={'value': 0.0})  # Placeholder
    b_id = graph.add_node(OpType.CONST, [], "b", params={'value': 0.0})  # Placeholder
    c_id = graph.add_node(OpType.ADD, [a_id, b_id], "c")

    # Allocate registers
    allocator = RegisterAllocator(dim)
    allocation = allocator.allocate(graph)

    print(f"Register allocation:")
    for vreg, preg in sorted(allocation.items()):
        print(f"  {vreg} → dim[{preg}]")
    print()

    # Manually set up correct input (skip CONST, treat as inputs)
    a_reg = graph.nodes[a_id].physical_reg
    b_reg = graph.nodes[b_id].physical_reg
    c_reg = graph.nodes[c_id].physical_reg

    # Create input with a=5, b=10
    x = torch.zeros(1, 1, dim)
    x[0, 0, a_reg] = 5.0
    x[0, 0, b_reg] = 10.0

    # Emit weights (only for ADD, skip CONST)
    emitter = WeightEmitter(dim, hidden_dim, S)

    # Manually emit ADD
    add_node = graph.nodes[c_id]
    emitter.emit_add(add_node, graph)

    weights = emitter.get_weights()

    # Forward pass
    out = swiglu_forward(x, weights['W_up'], weights['b_up'],
                        weights['W_gate'], weights['b_gate'],
                        weights['W_down'], weights['b_down'])

    result = out[0, 0, c_reg].item()

    print(f"Input: a = 5.0, b = 10.0")
    print(f"Expected: c = 15.0")
    print(f"Got:      c = {result:.4f}")

    passed = abs(result - 15.0) < 0.5

    print(f"\n{'✓' if passed else '✗'} Test {'PASSED' if passed else 'FAILED'}")
    print()
    return passed


def test_comparison_with_inputs():
    """Test CMP_GE with inputs."""
    print("=" * 70)
    print("TEST 2: Comparison with Input Embeddings")
    print("=" * 70)

    dim = 512
    hidden_dim = 4096
    S = 5.0

    from neural_vm.graph_weight_compiler import (
        ComputationGraph, OpType, WeightEmitter, RegisterAllocator
    )

    graph = ComputationGraph()

    # Input nodes
    a_id = graph.add_node(OpType.CONST, [], "a", params={'value': 0.0})
    b_id = graph.add_node(OpType.CONST, [], "b", params={'value': 0.0})
    result_id = graph.add_node(OpType.CMP_GE, [a_id, b_id], "result")

    # Allocate
    allocator = RegisterAllocator(dim)
    allocation = allocator.allocate(graph)

    a_reg = graph.nodes[a_id].physical_reg
    b_reg = graph.nodes[b_id].physical_reg
    result_reg = graph.nodes[result_id].physical_reg

    # Emit weights (only CMP_GE)
    emitter = WeightEmitter(dim, hidden_dim, S)
    cmp_node = graph.nodes[result_id]
    emitter.emit_cmp_ge(cmp_node, graph)
    weights = emitter.get_weights()

    # Test cases
    test_cases = [
        (10.0, 16.0, 0.0),
        (16.0, 16.0, 1.0),
        (20.0, 16.0, 1.0),
    ]

    all_passed = True
    for a_val, b_val, expected in test_cases:
        x = torch.zeros(1, 1, dim)
        x[0, 0, a_reg] = a_val
        x[0, 0, b_reg] = b_val

        out = swiglu_forward(x, weights['W_up'], weights['b_up'],
                            weights['W_gate'], weights['b_gate'],
                            weights['W_down'], weights['b_down'])

        result = out[0, 0, result_reg].item()

        passed = abs(result - expected) < 0.3

        print(f"Test: {a_val} >= {b_val}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result:.4f}")
        print(f"  {'✓' if passed else '✗'} {'PASS' if passed else 'FAIL'}")
        print()

        all_passed = all_passed and passed

    print(f"{'✓' if all_passed else '✗'} Test {'PASSED' if all_passed else 'FAILED'}")
    print()
    return all_passed


def test_sparsity():
    """Test that weights are highly sparse."""
    print("=" * 70)
    print("TEST 3: Weight Sparsity")
    print("=" * 70)

    dim = 512
    hidden_dim = 4096
    S = 5.0

    compiler = GraphWeightCompiler(dim, hidden_dim, S)

    # Build a computation (even though we won't use outputs correctly)
    compiler.const(1.0, "dummy")

    weights = compiler.compile()

    total_up = hidden_dim * dim
    total_down = dim * hidden_dim

    nonzero_up = (weights['W_up'] != 0).sum().item()
    nonzero_down = (weights['W_down'] != 0).sum().item()

    sparsity_up = 100.0 * (1.0 - nonzero_up / total_up)
    sparsity_down = 100.0 * (1.0 - nonzero_down / total_down)

    print(f"Weight matrices:")
    print(f"  W_up:   {nonzero_up:,}/{total_up:,} non-zero ({sparsity_up:.4f}% sparse)")
    print(f"  W_down: {nonzero_down:,}/{total_down:,} non-zero ({sparsity_down:.4f}% sparse)")

    highly_sparse = sparsity_up > 99.9 and sparsity_down > 99.9

    print(f"\n{'✓' if highly_sparse else '✗'} Weights are {'highly sparse' if highly_sparse else 'dense'}")
    print()
    return highly_sparse


def test_register_reuse():
    """Test that register allocation reuses dimensions when possible."""
    print("=" * 70)
    print("TEST 4: Register Allocation and Reuse")
    print("=" * 70)

    dim = 512
    hidden_dim = 4096
    S = 5.0

    from neural_vm.graph_weight_compiler import (
        ComputationGraph, OpType, RegisterAllocator
    )

    graph = ComputationGraph()

    # Create: a, b, c=a+b, d=a+c (c reuses b's register since b is dead)
    a_id = graph.add_node(OpType.CONST, [], "a", params={'value': 1.0})
    b_id = graph.add_node(OpType.CONST, [], "b", params={'value': 2.0})
    c_id = graph.add_node(OpType.ADD, [a_id, b_id], "c")
    d_id = graph.add_node(OpType.ADD, [a_id, c_id], "d")

    allocator = RegisterAllocator(dim)
    allocation = allocator.allocate(graph)

    print("Virtual registers:")
    for vreg, preg in sorted(allocation.items()):
        print(f"  {vreg} → dim[{preg}]")

    unique_regs = len(set(allocation.values()))
    total_regs = len(allocation)

    print(f"\nAllocation efficiency:")
    print(f"  Virtual registers: {total_regs}")
    print(f"  Physical registers: {unique_regs}")
    print(f"  Reuse factor: {total_regs / unique_regs:.2f}x")

    # Should reuse registers (not need 4 separate dimensions)
    efficient = unique_regs < total_regs

    print(f"\n{'✓' if efficient else '✗'} Register allocation {'efficient' if efficient else 'inefficient'}")
    print()
    return efficient


def run_all_tests():
    """Run all tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Graph Weight Compiler - Corrected Tests" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    tests = [
        ("ADD with Input Embeddings", test_add_with_input_embeddings),
        ("Comparison with Inputs", test_comparison_with_inputs),
        ("Weight Sparsity", test_sparsity),
        ("Register Allocation", test_register_reuse),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            print(f"✗ {name} failed with exception:")
            traceback.print_exc()
            print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passing = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if error:
            print(f"    Error: {error}")

    print()
    print(f"Results: {passing}/{total} tests passing")
    print("=" * 70)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
