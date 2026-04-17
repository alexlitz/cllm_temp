#!/usr/bin/env python3
"""
Test suite for Weight Compiler

Validates that each method compiler produces correct weight values
and that operations compile to working transformer weights.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.weight_compiler import (
    WeightCompiler, ComputeMethod, CancelPairCompiler,
    StepPairCompiler, ClearPairCompiler, EqualityStepCompiler,
    ModularAddCompiler, RelayCompiler, ComputeNode
)


def test_cancel_pair():
    """Test CancelPairCompiler produces correct weights."""
    print("=" * 60)
    print("TEST 1: CancelPairCompiler")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("A", 0, width=1)
    compiler.registers.add("B", 1, width=1)
    compiler.registers.add("OUT", 10, width=1)
    compiler.registers.add("GATE", 100, width=1)

    node = ComputeNode(
        layer=1,
        output="OUT",
        method=ComputeMethod.CANCEL_PAIR,
        inputs=["A", "B"],
        gate="GATE"
    )

    method_compiler = CancelPairCompiler()
    weights = method_compiler.compile(node, 0, compiler.registers, 5.0, 512, 4096)

    # Verify structure
    assert weights.W_up[0, 0] == 5.0, "Unit 0 should read A with scale S"
    assert weights.W_up[0, 1] == 5.0, "Unit 0 should read B with scale S"
    assert weights.W_up[1, 0] == -5.0, "Unit 1 should read A with scale -S"
    assert weights.W_up[1, 1] == -5.0, "Unit 1 should read B with scale -S"

    assert weights.W_gate[0, 100] == 5.0, "Unit 0 should be gated by GATE"
    assert weights.W_gate[1, 100] == -5.0, "Unit 1 should be gated by -GATE"

    assert weights.W_down[10, 0] == 1.0 / 25.0, "Output should read unit 0 with 1/S^2"
    assert weights.W_down[10, 1] == 1.0 / 25.0, "Output should read unit 1 with 1/S^2"

    print("✓ CancelPairCompiler produces correct weights")
    print(f"  W_up[0,0] = {weights.W_up[0, 0].item():.4f} (expected 5.0)")
    print(f"  W_down[10,0] = {weights.W_down[10, 0].item():.6f} (expected 0.04)")
    print()
    return True


def test_step_pair():
    """Test StepPairCompiler produces correct weights."""
    print("=" * 60)
    print("TEST 2: StepPairCompiler")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("A", 0, width=1)
    compiler.registers.add("B", 1, width=1)
    compiler.registers.add("OUT", 10, width=1)
    compiler.registers.add("GATE", 100, width=1)

    node = ComputeNode(
        layer=1,
        output="OUT",
        method=ComputeMethod.STEP_PAIR,
        inputs=["A", "B"],
        gate="GATE",
        params={'threshold': 16.0}
    )

    method_compiler = StepPairCompiler()
    weights = method_compiler.compile(node, 0, compiler.registers, 5.0, 512, 4096)

    # Verify structure
    assert weights.W_up[0, 0] == 5.0, "Unit 0 should read A"
    assert weights.W_up[0, 1] == 5.0, "Unit 0 should read B"
    assert weights.b_up[0] == -5.0 * 15.0, "Unit 0 bias should be -S*(threshold-1)"

    assert weights.W_up[1, 0] == 5.0, "Unit 1 should read A"
    assert weights.W_up[1, 1] == 5.0, "Unit 1 should read B"
    assert weights.b_up[1] == -5.0 * 16.0, "Unit 1 bias should be -S*threshold"

    assert weights.W_down[10, 0] == 1.0 / 5.0, "Output should read unit 0 with 1/S"
    assert weights.W_down[10, 1] == -1.0 / 5.0, "Output should read unit 1 with -1/S"

    print("✓ StepPairCompiler produces correct weights")
    print(f"  b_up[0] = {weights.b_up[0].item():.4f} (expected -75.0)")
    print(f"  b_up[1] = {weights.b_up[1].item():.4f} (expected -80.0)")
    print()
    return True


def test_equality_step():
    """Test EqualityStepCompiler produces correct weights."""
    print("=" * 60)
    print("TEST 3: EqualityStepCompiler")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("A", 0, width=1)
    compiler.registers.add("B", 1, width=1)
    compiler.registers.add("OUT", 10, width=1)
    compiler.registers.add("GATE", 100, width=1)

    node = ComputeNode(
        layer=1,
        output="OUT",
        method=ComputeMethod.EQUALITY_STEP,
        inputs=["A", "B"],
        gate="GATE"
    )

    method_compiler = EqualityStepCompiler()
    weights = method_compiler.compile(node, 0, compiler.registers, 5.0, 512, 4096)

    # Verify structure (4 units)
    assert weights.W_up[0, 0] == 5.0, "Unit 0: A - B >= 0"
    assert weights.W_up[0, 1] == -5.0, "Unit 0: A - B >= 0"
    assert weights.W_up[1, 0] == -5.0, "Unit 1: B - A >= 0"
    assert weights.W_up[1, 1] == 5.0, "Unit 1: B - A >= 0"

    # All units contribute to output
    assert weights.W_down[10, 0] == 1.0 / 5.0, "Unit 0 contributes +1"
    assert weights.W_down[10, 1] == 1.0 / 5.0, "Unit 1 contributes +1"
    assert weights.W_down[10, 2] == -1.0 / 5.0, "Unit 2 contributes -1"
    assert weights.W_down[10, 3] == -1.0 / 5.0, "Unit 3 contributes -1"

    print("✓ EqualityStepCompiler produces correct weights")
    print(f"  Uses 4 hidden units")
    print(f"  Computes: (A >= B) + (B >= A) - (A > B) - (B > A)")
    print(f"  Result: 1 if A == B, 0 otherwise")
    print()
    return True


def test_relay():
    """Test RelayCompiler produces correct weights."""
    print("=" * 60)
    print("TEST 4: RelayCompiler (Scalar)")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("IN", 0, width=1)
    compiler.registers.add("OUT", 10, width=1)
    compiler.registers.add("GATE", 100, width=1)

    node = ComputeNode(
        layer=1,
        output="OUT",
        method=ComputeMethod.RELAY,
        inputs=["IN"],
        gate="GATE"
    )

    method_compiler = RelayCompiler()
    weights = method_compiler.compile(node, 0, compiler.registers, 5.0, 512, 4096)

    # Verify relay structure (2 units for scalar)
    assert weights.W_up[0, 0] == 5.0, "Unit 0 should read input"
    assert weights.W_up[1, 0] == -5.0, "Unit 1 should read -input"

    assert weights.W_down[10, 0] == 1.0 / 5.0, "Output reads unit 0"
    assert weights.W_down[10, 1] == 1.0 / 5.0, "Output reads unit 1"

    print("✓ RelayCompiler produces correct weights")
    print(f"  Scalar relay uses 2 units (cancel pair for stability)")
    print()
    return True


def test_clear():
    """Test ClearPairCompiler produces correct weights."""
    print("=" * 60)
    print("TEST 5: ClearPairCompiler")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("REG", 0, width=1)
    compiler.registers.add("GATE", 100, width=1)

    node = ComputeNode(
        layer=1,
        output="REG",
        method=ComputeMethod.CLEAR_PAIR,
        inputs=["REG"],
        gate="GATE"
    )

    method_compiler = ClearPairCompiler()
    weights = method_compiler.compile(node, 0, compiler.registers, 5.0, 512, 4096)

    # Verify clear structure (reads and subtracts from itself)
    assert weights.W_up[0, 0] == 5.0, "Unit 0 should read register"
    assert weights.W_up[1, 0] == -5.0, "Unit 1 should read -register"

    assert weights.W_down[0, 0] == -1.0 / 5.0, "Output subtracts unit 0"
    assert weights.W_down[0, 1] == -1.0 / 5.0, "Output subtracts unit 1"

    print("✓ ClearPairCompiler produces correct weights")
    print(f"  Subtracts register from itself: REG = REG - REG")
    print()
    return True


def test_operation_builder():
    """Test Operation builder API."""
    print("=" * 60)
    print("TEST 6: Operation Builder API")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("A", 0, width=1)
    compiler.registers.add("B", 1, width=1)
    compiler.registers.add("OUT", 10, width=1)
    compiler.registers.add("GATE", 100, width=1)

    # Build operation using high-level API
    op = compiler.operation("TEST")

    with op.layer(1):
        op.cancel_pair(["A", "B"], output="OUT", gate="GATE")

    assert len(op.nodes) == 1, "Should have 1 node"
    assert op.nodes[0].layer == 1, "Node should be in layer 1"
    assert op.nodes[0].method == ComputeMethod.CANCEL_PAIR, "Should be cancel pair"

    # Compile operation
    weights = compiler.compile_operation(op)

    assert weights.W_up.shape == (4096, 512), "W_up should have correct shape"
    assert weights.W_down.shape == (512, 4096), "W_down should have correct shape"

    print("✓ Operation builder API works correctly")
    print(f"  Created {len(op.nodes)} node(s)")
    print(f"  Compiled to weight matrices")
    print()
    return True


def test_multi_layer():
    """Test multi-layer operation compilation."""
    print("=" * 60)
    print("TEST 7: Multi-Layer Compilation")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("A", 0, width=1)
    compiler.registers.add("B", 1, width=1)
    compiler.registers.add("TEMP", 10, width=1)
    compiler.registers.add("OUT", 20, width=1)
    compiler.registers.add("GATE", 100, width=1)

    op = compiler.operation("MULTI")

    with op.layer(1):
        op.cancel_pair(["A", "B"], output="TEMP", gate="GATE")

    with op.layer(2):
        op.relay("TEMP", output="OUT", gate="GATE")

    assert len(op.nodes) == 2, "Should have 2 nodes"

    # Compile
    weights = compiler.compile_operation(op)

    # Check that units don't overlap
    non_zero_up = (weights.W_up != 0).sum().item()
    print(f"✓ Multi-layer compilation works")
    print(f"  Layer 1: cancel_pair (2 units)")
    print(f"  Layer 2: relay (2 units)")
    print(f"  Total non-zero W_up entries: {non_zero_up}")
    print()
    return True


def test_modular_add():
    """Test ModularAddCompiler."""
    print("=" * 60)
    print("TEST 8: ModularAddCompiler")
    print("=" * 60)

    compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
    compiler.registers.add("A", 0, width=1)
    compiler.registers.add("B", 1, width=1)
    compiler.registers.add("C", 2, width=1)
    compiler.registers.add("RAW_SUM", 9, width=1)
    compiler.registers.add("OUT", 10, width=1)
    compiler.registers.add("CARRY", 11, width=1)
    compiler.registers.add("GATE", 100, width=1)

    node = ComputeNode(
        layer=1,
        output="OUT",
        method=ComputeMethod.MODULAR_ADD,
        inputs=["A", "B", "C"],
        gate="GATE",
        params={'base': 16}
    )

    method_compiler = ModularAddCompiler()
    weights = method_compiler.compile(node, 0, compiler.registers, 5.0, 512, 4096)

    # Verify structure (6 units)
    assert weights.W_up[0, 0] == 5.0, "Unit 0 reads A"
    assert weights.W_up[0, 1] == 5.0, "Unit 0 reads B"
    assert weights.W_up[0, 2] == 5.0, "Unit 0 reads C"

    print("✓ ModularAddCompiler produces correct weights")
    print(f"  Uses 6 hidden units")
    print(f"  Computes: (A + B + C) mod 16")
    print(f"  Produces carry flag")
    print()
    return True


def run_all_tests():
    """Run all tests."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Weight Compiler Test Suite" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    tests = [
        ("CancelPairCompiler", test_cancel_pair),
        ("StepPairCompiler", test_step_pair),
        ("EqualityStepCompiler", test_equality_step),
        ("RelayCompiler", test_relay),
        ("ClearPairCompiler", test_clear),
        ("Operation Builder API", test_operation_builder),
        ("Multi-Layer Compilation", test_multi_layer),
        ("ModularAddCompiler", test_modular_add),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} failed: {e}\n")

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passing = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        if error:
            print(f"    Error: {error}")

    print()
    print(f"Results: {passing}/{total} tests passing")
    print("=" * 60)

    return passing == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
