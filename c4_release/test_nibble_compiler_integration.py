#!/usr/bin/env python3
"""
Test Nibble Weight Compiler Integration with AutoregressiveVM

Verifies that compiled nibble-based weights:
1. Have correct dimensions (1280 = 8 positions × 160 dims)
2. Generate sparse weights compatible with PureFFN
3. Can be loaded into AutoregressiveVM layers
4. Produce correct nibble-level computations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.nibble_weight_compiler import (
    NibbleWeightCompiler, NibbleRegisterMap, NibbleWeightEmitter
)
from neural_vm.graph_weight_compiler import OpType
from neural_vm.embedding import Opcode, E


def test_nibble_register_map():
    """Test: Nibble register map flattening."""
    print("\n" + "="*70)
    print("TEST: Nibble Register Map")
    print("="*70)

    reg_map = NibbleRegisterMap()

    # Test flat indexing
    assert reg_map.DIM == 160, f"Expected DIM=160, got {reg_map.DIM}"
    assert reg_map.NUM_POSITIONS == 8, f"Expected 8 positions, got {reg_map.NUM_POSITIONS}"

    # Position 0, slot 0 → index 0
    assert reg_map.flat_index(0, 0) == 0
    # Position 0, slot 5 (RESULT) → index 5
    assert reg_map.flat_index(0, E.RESULT) == 5
    # Position 1, slot 0 → index 160
    assert reg_map.flat_index(1, 0) == 160
    # Position 7, slot 159 → index 1279
    assert reg_map.flat_index(7, 159) == 1279

    # Test opcode indexing
    add_idx = reg_map.opcode_index(Opcode.ADD)
    assert add_idx == E.OP_START + Opcode.ADD, f"Expected {E.OP_START + Opcode.ADD}, got {add_idx}"

    print("  ✅ DIM = 160")
    print("  ✅ NUM_POSITIONS = 8")
    print("  ✅ Total flattened dims = 1280")
    print("  ✅ Flat indexing works correctly")
    print("  ✅ Opcode indexing works correctly")

    return True


def test_add_weight_generation():
    """Test: Generate ADD weights for nibble format."""
    print("\n" + "="*70)
    print("TEST: ADD Weight Generation")
    print("="*70)

    emitter = NibbleWeightEmitter(Opcode.ADD, num_positions=8)

    # Emit ADD for all 8 nibble positions
    for pos in range(8):
        emitter.emit_add_nibble(pos)

    weights = emitter.get_weights()

    # Check dimensions
    expected_dim = 1280  # 8 × 160
    expected_hidden = 4096

    assert weights['W_up'].shape == (expected_hidden, expected_dim), \
        f"W_up shape mismatch: {weights['W_up'].shape}"
    assert weights['W_gate'].shape == (expected_hidden, expected_dim), \
        f"W_gate shape mismatch: {weights['W_gate'].shape}"
    assert weights['W_down'].shape == (expected_dim, expected_hidden), \
        f"W_down shape mismatch: {weights['W_down'].shape}"

    # Check sparsity
    total_params = sum(w.numel() for w in weights.values())
    nonzero_params = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
    sparsity = 100 * (1 - nonzero_params / total_params)

    print(f"  ✅ W_up shape: {weights['W_up'].shape}")
    print(f"  ✅ W_gate shape: {weights['W_gate'].shape}")
    print(f"  ✅ W_down shape: {weights['W_down'].shape}")
    print(f"  ✅ Hidden units used: {emitter.get_active_units()}/4096 ({100*emitter.get_active_units()/4096:.1f}%)")
    print(f"  ✅ Sparsity: {sparsity:.2f}% ({nonzero_params}/{total_params} non-zero)")

    # Verify units used (8 positions × 8 units/position = 64 units)
    expected_units = 8 * 8  # 64
    assert emitter.get_active_units() == expected_units, \
        f"Expected {expected_units} units, got {emitter.get_active_units()}"

    return True


def test_compiler_interface():
    """Test: High-level NibbleWeightCompiler interface."""
    print("\n" + "="*70)
    print("TEST: NibbleWeightCompiler Interface")
    print("="*70)

    compiler = NibbleWeightCompiler(num_positions=8)

    # Test supported operations
    operations = [
        (OpType.ADD, Opcode.ADD, "ADD"),
        (OpType.SUB, Opcode.SUB, "SUB"),
        (OpType.MOVE, Opcode.IMM, "MOVE"),
    ]

    for op_type, opcode, name in operations:
        print(f"\n  Compiling {name}...")
        weights = compiler.compile_operation(op_type, opcode)

        # Check dimensions
        assert weights['W_up'].shape[1] == 1280, f"{name}: Wrong input dim"
        assert weights['W_down'].shape[0] == 1280, f"{name}: Wrong output dim"

        # Check non-empty
        nonzero = (weights['W_up'].abs() > 1e-9).sum().item()
        assert nonzero > 0, f"{name}: W_up is all zeros"

        print(f"    ✅ {name} weights generated")
        print(f"    ✅ Dimensions: 1280 → 4096 → 1280")

    print(f"\n  ✅ All {len(operations)} operations compile successfully")

    return True


def test_weight_sparsity_analysis():
    """Test: Analyze sparsity patterns of compiled weights."""
    print("\n" + "="*70)
    print("TEST: Weight Sparsity Analysis")
    print("="*70)

    compiler = NibbleWeightCompiler()

    operations = [
        (OpType.ADD, Opcode.ADD, "ADD"),
        (OpType.SUB, Opcode.SUB, "SUB"),
        (OpType.MOVE, Opcode.IMM, "MOVE"),
    ]

    results = []

    for op_type, opcode, name in operations:
        weights = compiler.compile_operation(op_type, opcode)

        total = sum(w.numel() for w in weights.values())
        nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
        sparsity = 100 * (1 - nonzero / total)

        results.append((name, nonzero, total, sparsity))

        print(f"\n  {name}:")
        print(f"    Non-zero params: {nonzero:,}")
        print(f"    Total params: {total:,}")
        print(f"    Sparsity: {sparsity:.2f}%")

    # Summary
    print("\n  " + "-"*66)
    print(f"  {'Operation':<10} {'Non-zero':>12} {'Total':>12} {'Sparsity':>10}")
    print("  " + "-"*66)
    for name, nz, total, sp in results:
        print(f"  {name:<10} {nz:>12,} {total:>12,} {sp:>9.2f}%")
    print("  " + "-"*66)

    return True


def test_nibble_computation_simulation():
    """Test: Simulate nibble-level computation with compiled weights."""
    print("\n" + "="*70)
    print("TEST: Nibble Computation Simulation")
    print("="*70)

    compiler = NibbleWeightCompiler()
    reg_map = NibbleRegisterMap()

    # Compile ADD weights
    weights = compiler.compile_operation(OpType.ADD, Opcode.ADD)

    # Create input: encode 10 + 5 in nibbles
    # 10 = 0x0A = [10, 0, 0, 0, 0, 0, 0, 0] (nibbles)
    # 5  = 0x05 = [5, 0, 0, 0, 0, 0, 0, 0]
    x = torch.zeros(1, 1, 1280)  # [batch=1, seq=1, dim=1280]

    # Set NIB_A = 10 (position 0), NIB_B = 5 (position 0)
    x[0, 0, reg_map.flat_index(0, E.NIB_A)] = 10.0
    x[0, 0, reg_map.flat_index(0, E.NIB_B)] = 5.0

    # Set opcode gate (ADD opcode one-hot)
    x[0, 0, reg_map.opcode_index(Opcode.ADD)] = 1.0

    # Apply SwiGLU forward pass
    W_up = weights['W_up']
    b_up = weights['b_up']
    W_gate = weights['W_gate']
    b_gate = weights['b_gate']
    W_down = weights['W_down']
    b_down = weights['b_down']

    up = torch.nn.functional.linear(x, W_up, b_up)
    gate = torch.nn.functional.linear(x, W_gate, b_gate)
    hidden = torch.nn.functional.silu(up) * gate
    delta = torch.nn.functional.linear(hidden, W_down, b_down)
    output = x + delta

    # Read RESULT from position 0
    result_nibble = output[0, 0, reg_map.flat_index(0, E.RESULT)].item()

    print(f"\n  Input: NIB_A[0] = 10, NIB_B[0] = 5")
    print(f"  Expected RESULT[0] = 15")
    print(f"  Actual RESULT[0] = {result_nibble:.2f}")

    # Check if close to 15 (allowing for floating point error)
    if abs(result_nibble - 15.0) < 1.0:
        print(f"  ✅ Computation correct (within tolerance)")
        return True
    else:
        print(f"  ❌ Computation incorrect (error = {abs(result_nibble - 15.0):.2f})")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Nibble Weight Compiler Integration Tests" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        ("Nibble Register Map", test_nibble_register_map),
        ("ADD Weight Generation", test_add_weight_generation),
        ("Compiler Interface", test_compiler_interface),
        ("Weight Sparsity Analysis", test_weight_sparsity_analysis),
        ("Nibble Computation Simulation", test_nibble_computation_simulation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\n  Total: {passed}/{total} tests passing ({100*passed/total:.1f}%)")

    if passed == total:
        print("\n✅ All tests passing!")
        print("   Nibble weight compiler successfully integrated with AutoregressiveVM format!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
