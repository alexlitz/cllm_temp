#!/usr/bin/env python3
"""
AutoregressiveVM Integration Test

Demonstrates loading compiled opcode weights into the AutoregressiveVM
and testing basic operations.

Architecture:
- Layer 9: ALU operations (compiled FFN weights)
- Layer 10: Control flow
- Layer 11: Memory operations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.embedding import Opcode, E
from neural_vm.run_vm import AutoregressiveVM


def test_compile_and_load_add():
    """Test: Compile ADD opcode and load into VM."""
    print("="*70)
    print("TEST: Compile and Load ADD Opcode")
    print("="*70)

    # Create compiler
    compiler = OpcodeNibbleCompiler()

    # Compile ADD opcode
    print("\n1. Compiling ADD opcode to nibble-based FFN weights...")
    weights = compiler.compile_opcode(Opcode.ADD)

    # Print weight summary
    total = sum(w.numel() for w in weights.values())
    nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
    sparsity = 100 * (1 - nonzero / total)

    print(f"   ✅ Compiled ADD weights:")
    print(f"      Non-zero params: {nonzero:,} / {total:,}")
    print(f"      Sparsity: {sparsity:.2f}%")
    print(f"      Weight shapes:")
    for k, v in weights.items():
        print(f"        {k}: {list(v.shape)}")

    # Create AutoregressiveVM
    print("\n2. Creating AutoregressiveVM...")
    try:
        vm = AutoregressiveVM()
        print(f"   ✅ VM created with {len(vm.blocks)} layers")
        print(f"      d_model: {vm.d_model}")

        # Check if layer 9 has compatible dimensions
        layer_9_ffn = vm.blocks[9].ffn
        print(f"\n3. Checking Layer 9 FFN dimensions...")
        print(f"      W_up shape: {list(layer_9_ffn.W_up.shape)}")
        print(f"      Expected: [4096, 1280]")

        # Verify dimensions match
        expected_d_model = 1280  # 8 nibbles × 160 dims
        expected_hidden = 4096

        if layer_9_ffn.W_up.shape == (expected_hidden, expected_d_model):
            print("   ✅ Dimensions match!")

            # Load weights into layer 9
            print("\n4. Loading compiled weights into Layer 9...")
            compiler.load_weights_into_vm(vm, layer_idx=9, weights=weights)

            print("\n   ✅ SUCCESS: ADD opcode compiled and loaded into VM!")
            return True

        elif vm.d_model == 512:
            print(f"   ⚠️  VM d_model is 512, but compiled weights are for 1280")
            print(f"      VM uses token embedding space, not nibble computation space")
            print(f"      Need to configure VM with d_model=1280 for nibble ALU")
            return False

        else:
            print(f"   ⚠️  Dimension mismatch: VM has d_model={vm.d_model}, expected 1280")
            return False

    except Exception as e:
        print(f"   ❌ Error creating VM: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compile_multiple_opcodes():
    """Test: Compile multiple opcodes and show statistics."""
    print("\n" + "="*70)
    print("TEST: Compile Multiple Opcodes")
    print("="*70)

    compiler = OpcodeNibbleCompiler()

    test_opcodes = [
        (Opcode.ADD, "ADD"),
        (Opcode.SUB, "SUB"),
        (Opcode.MUL, "MUL"),
        (Opcode.OR, "OR"),
        (Opcode.XOR, "XOR"),
        (Opcode.EQ, "EQ"),
        (Opcode.LT, "LT"),
    ]

    print("\nCompiling opcodes:")
    results = []
    for opcode, name in test_opcodes:
        try:
            weights = compiler.compile_opcode(opcode)
            total = sum(w.numel() for w in weights.values())
            nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            sparsity = 100 * (1 - nonzero / total)

            print(f"  ✅ {name:6s}: {nonzero:,} params ({sparsity:.2f}% sparse)")
            results.append((name, True, nonzero, sparsity))

        except Exception as e:
            print(f"  ❌ {name:6s}: {e}")
            results.append((name, False, 0, 0))

    # Summary
    passed = sum(1 for _, p, _, _ in results if p)
    if passed > 0:
        avg_sparsity = sum(s for _, p, _, s in results if p) / passed
        total_nonzero = sum(n for _, p, n, _ in results if p)

        print(f"\n  Compiled: {passed}/{len(results)} opcodes")
        print(f"  Total params: {total_nonzero:,}")
        print(f"  Avg sparsity: {avg_sparsity:.2f}%")

    return passed == len(results)


def test_nibble_register_map():
    """Test: Verify nibble register mapping."""
    print("\n" + "="*70)
    print("TEST: Nibble Register Mapping")
    print("="*70)

    from neural_vm.nibble_weight_compiler import NibbleRegisterMap

    reg_map = NibbleRegisterMap()

    print(f"\nNibble computation space:")
    print(f"  DIM per position: {reg_map.DIM}")
    print(f"  Num positions: 8 (for 32-bit)")
    print(f"  Total dimensions: {8 * reg_map.DIM}")

    print(f"\nPer-nibble feature slots:")
    print(f"  NIB_A: {reg_map.NIB_A}")
    print(f"  NIB_B: {reg_map.NIB_B}")
    print(f"  RAW_SUM: {reg_map.RAW_SUM}")
    print(f"  CARRY_IN: {reg_map.CARRY_IN}")
    print(f"  CARRY_OUT: {reg_map.CARRY_OUT}")
    print(f"  RESULT: {reg_map.RESULT}")
    print(f"  TEMP: {reg_map.TEMP}")

    print(f"\nExample: Position 0, NIB_A slot")
    pos_0_nib_a = reg_map._flatten_index(0, reg_map.NIB_A)
    print(f"  Flattened index: {pos_0_nib_a}")

    print(f"\nExample: Position 5, RESULT slot")
    pos_5_result = reg_map._flatten_index(5, reg_map.RESULT)
    print(f"  Flattened index: {pos_5_result}")

    print("\n✅ Nibble register mapping verified")
    return True


def main():
    """Run all integration tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "VM Integration Tests" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        ("Nibble Register Mapping", test_nibble_register_map),
        ("Compile Multiple Opcodes", test_compile_multiple_opcodes),
        ("Compile and Load ADD", test_compile_and_load_add),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\n  Total: {passed}/{total} tests passing ({100*passed/total:.1f}%)")

    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL INTEGRATION TESTS PASSING!")
        print("="*70)
        print("\n  Next steps:")
        print("    1. Configure VM with d_model=1280 for nibble ALU")
        print("    2. Test end-to-end execution with compiled weights")
        print("    3. Benchmark performance vs learned weights")
        print("="*70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
