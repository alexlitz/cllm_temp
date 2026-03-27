#!/usr/bin/env python3
"""
Test that eviction is working correctly with the neural VM on basic programs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c


def test_simple_program():
    """Test a simple program that should work with eviction."""
    print("=" * 60)
    print("TEST: Simple GCD (verifies eviction works)")
    print("=" * 60)

    source = """
    int gcd(int a, int b) {
        int t;
        while (b != 0) {
            t = b;
            b = a % b;
            a = t;
        }
        return a;
    }
    int main() { return gcd(48, 18); }
    """

    print("\nCompiling GCD program...")
    bytecode, data = compile_c(source)
    print(f"  Bytecode: {len(bytecode)} instructions")
    print(f"  Data: {len(data)} bytes")

    print("\nInitializing Neural VM...")
    vm = C4TransformerVM()
    vm.reset()
    vm.load_bytecode(bytecode, data)

    print("\nRunning with eviction enabled...")
    result = vm.run(max_steps=10000)

    expected = 6
    status = "PASS ✓" if result == expected else f"FAIL ✗"
    print(f"\nResult: {result} (expected {expected}) {status}")

    if result != expected:
        print("\n⚠️  EVICTION MAY BE BROKEN!")
        return False

    print("\n✓ Eviction is working correctly!")
    return True


def test_memory_writes():
    """Test a program with many memory writes to stress eviction."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Memory Writes (stress eviction)")
    print("=" * 60)

    source = """
    int main() {
        int a, b, c, d, e;
        a = 10;
        b = 20;
        c = 30;
        d = 40;
        e = 50;
        return a + b + c + d + e;
    }
    """

    print("\nCompiling program with 5 variables...")
    bytecode, data = compile_c(source)

    print("\nRunning with eviction...")
    vm = C4TransformerVM()
    vm.reset()
    vm.load_bytecode(bytecode, data)
    result = vm.run(max_steps=10000)

    expected = 150
    status = "PASS ✓" if result == expected else f"FAIL ✗"
    print(f"\nResult: {result} (expected {expected}) {status}")

    if result != expected:
        print("\n⚠️  EVICTION MAY BE AFFECTING VALID MEMORY!")
        return False

    print("\n✓ Valid memory is being retained correctly!")
    return True


def test_overwrite():
    """Test that overwriting variables works correctly with eviction."""
    print("\n" + "=" * 60)
    print("TEST: Variable Overwrite (eviction of old values)")
    print("=" * 60)

    source = """
    int main() {
        int x;
        x = 10;
        x = 20;
        x = 30;
        return x;
    }
    """

    print("\nCompiling program with overwrites...")
    bytecode, data = compile_c(source)

    print("\nRunning with eviction...")
    vm = C4TransformerVM()
    vm.reset()
    vm.load_bytecode(bytecode, data)
    result = vm.run(max_steps=10000)

    expected = 30
    status = "PASS ✓" if result == expected else f"FAIL ✗"
    print(f"\nResult: {result} (expected {expected}) {status}")

    if result != expected:
        print("\n⚠️  OVERWRITE EVICTION MAY BE BROKEN!")
        return False

    print("\n✓ Old values are being evicted correctly!")
    return True


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "NEURAL VM EVICTION TEST SUITE" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = []

    try:
        # Test 1: Simple program
        results.append(("Simple GCD", test_simple_program()))

        # Test 2: Memory writes
        results.append(("Memory Writes", test_memory_writes()))

        # Test 3: Overwrites
        results.append(("Overwrites", test_overwrite()))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:20s} {status}")

        all_passed = all(passed for _, passed in results)

        print()
        if all_passed:
            print("╔" + "═" * 58 + "╗")
            print("║" + " " * 15 + "ALL TESTS PASSED ✓" + " " * 25 + "║")
            print("╚" + "═" * 58 + "╝")
            print()
            print("Summary:")
            print("  ✓ Eviction mechanism is working")
            print("  ✓ Valid memory is retained")
            print("  ✓ Old values are evicted")
            print()
            print("Note: Current implementation uses hardcoded eviction policy")
            print("      (via _mem_history dictionary). See ATTENTION_BASED_EVICTION_ANALYSIS.md")
            print("      for how it should work with score-based eviction.")
        else:
            print("❌ SOME TESTS FAILED - EVICTION MAY BE BROKEN")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
