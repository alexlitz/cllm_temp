#!/usr/bin/env python3
"""
Simple integration test for score-based eviction with actual VM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer_vm import C4TransformerVM, C4Config
from src.compiler import compile_c


def test_simple_program():
    """Test simple program with score-based eviction."""
    print("=" * 60)
    print("TEST: Simple Program with Score-Based Eviction")
    print("=" * 60)

    # Very simple program: just return a value
    source = """
    int main() {
        return 42;
    }
    """

    print("\nCompiling program...")
    bytecode, data = compile_c(source)
    print(f"Bytecode: {len(bytecode)} instructions")

    # Test with score-based eviction
    print("\n--- Testing with Score-Based Eviction ---")
    config = C4Config(use_score_based_eviction=True)
    vm = C4TransformerVM(config)
    vm.reset()
    vm.load_bytecode(bytecode, data)

    print("Running VM...")
    result = vm.run(max_steps=1000)
    expected = 42

    print(f"Result: {result}")
    print(f"Expected: {expected}")

    if result == expected:
        print("\n✅ PASSED - Score-based eviction works!")
        return True
    else:
        print(f"\n❌ FAILED - Got {result}, expected {expected}")
        return False


def test_simple_arithmetic():
    """Test simple arithmetic with score-based eviction."""
    print("\n" + "=" * 60)
    print("TEST: Simple Arithmetic with Score-Based Eviction")
    print("=" * 60)

    source = """
    int main() {
        int x;
        x = 10 + 5;
        return x;
    }
    """

    print("\nCompiling program...")
    bytecode, data = compile_c(source)

    config = C4Config(use_score_based_eviction=True)
    vm = C4TransformerVM(config)
    vm.reset()
    vm.load_bytecode(bytecode, data)

    print("Running VM...")
    result = vm.run(max_steps=1000)
    expected = 15

    print(f"Result: {result}")
    print(f"Expected: {expected}")

    if result == expected:
        print("\n✅ PASSED - Arithmetic works!")
        return True
    else:
        print(f"\n❌ FAILED - Got {result}, expected {expected}")
        return False


def main():
    try:
        results = []

        # Test 1: Simple return
        results.append(("Simple Return", test_simple_program()))

        # Test 2: Simple arithmetic
        results.append(("Simple Arithmetic", test_simple_arithmetic()))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:25s} {status}")

        all_passed = all(passed for _, passed in results)

        if all_passed:
            print("\n✅ ALL TESTS PASSED!")
            print("\nScore-based eviction is working correctly with the neural VM.")
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
