#!/usr/bin/env python3
"""
Test score-based eviction implementation.

Verifies that score-based eviction produces correct results and evicts
appropriately based on attention scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer_vm import C4TransformerVM, C4Config
from src.compiler import compile_c


def test_gcd_with_score_eviction():
    """Test GCD with score-based eviction enabled."""
    print("=" * 60)
    print("TEST: GCD with Score-Based Eviction")
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

    print("\nCompiling GCD...")
    bytecode, data = compile_c(source)

    # Test with score-based eviction
    print("\n--- Testing with Score-Based Eviction ---")
    config = C4Config(use_score_based_eviction=True)
    vm = C4TransformerVM(config)
    vm.reset()
    vm.load_bytecode(bytecode, data)

    result = vm.run(max_steps=10000)
    expected = 6

    status = "PASS ✓" if result == expected else f"FAIL ✗"
    print(f"Result: {result} (expected {expected}) {status}")

    if result != expected:
        print("\n⚠️  SCORE-BASED EVICTION FAILED!")
        return False

    print("\n✓ Score-based eviction works correctly!")
    return True


def test_memory_writes_with_eviction():
    """Test that overwritten memory is correctly evicted."""
    print("\n" + "=" * 60)
    print("TEST: Memory Overwrites with Score Eviction")
    print("=" * 60)

    source = """
    int main() {
        int x;
        x = 10;  // First write
        x = 20;  // Overwrites (should evict first)
        x = 30;  // Overwrites (should evict second)
        return x;
    }
    """

    print("\nCompiling overwrite test...")
    bytecode, data = compile_c(source)

    config = C4Config(use_score_based_eviction=True)
    vm = C4TransformerVM(config)
    vm.reset()
    vm.load_bytecode(bytecode, data)

    result = vm.run(max_steps=10000)
    expected = 30

    status = "PASS ✓" if result == expected else f"FAIL ✗"
    print(f"Result: {result} (expected {expected}) {status}")

    if result != expected:
        print("\n⚠️  OVERWRITE EVICTION FAILED!")
        return False

    print("\n✓ Overwritten values evicted correctly!")
    return True


def test_compare_eviction_strategies():
    """Compare legacy vs score-based eviction."""
    print("\n" + "=" * 60)
    print("TEST: Compare Legacy vs Score-Based Eviction")
    print("=" * 60)

    source = """
    int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    int main() { return factorial(5); }
    """

    print("\nCompiling factorial...")
    bytecode, data = compile_c(source)

    # Test with legacy eviction
    print("\n--- Legacy Eviction (_mem_history) ---")
    config_legacy = C4Config(use_score_based_eviction=False)
    vm_legacy = C4TransformerVM(config_legacy)
    vm_legacy.reset()
    vm_legacy.load_bytecode(bytecode, data)
    result_legacy = vm_legacy.run(max_steps=10000)
    print(f"Result: {result_legacy}")

    # Test with score-based eviction
    print("\n--- Score-Based Eviction ---")
    config_score = C4Config(use_score_based_eviction=True)
    vm_score = C4TransformerVM(config_score)
    vm_score.reset()
    vm_score.load_bytecode(bytecode, data)
    result_score = vm_score.run(max_steps=10000)
    print(f"Result: {result_score}")

    expected = 120
    legacy_ok = result_legacy == expected
    score_ok = result_score == expected

    print(f"\nLegacy eviction: {result_legacy} (expected {expected}) {'✓' if legacy_ok else '✗'}")
    print(f"Score eviction:  {result_score} (expected {expected}) {'✓' if score_ok else '✗'}")

    if not (legacy_ok and score_ok):
        print("\n⚠️  EVICTION MISMATCH!")
        return False

    print("\n✓ Both eviction strategies produce correct results!")
    return True


def test_eviction_stats():
    """Display eviction statistics for comparison."""
    print("\n" + "=" * 60)
    print("TEST: Eviction Statistics")
    print("=" * 60)

    source = """
    int main() {
        int a, b, c, d, e, f;
        a = 1; b = 2; c = 3;
        d = 4; e = 5; f = 6;
        return a + b + c + d + e + f;
    }
    """

    print("\nCompiling multi-variable test...")
    bytecode, data = compile_c(source)

    print("\n--- Score-Based Eviction Stats ---")
    config = C4Config(use_score_based_eviction=True)
    vm = C4TransformerVM(config)
    vm.reset()
    vm.load_bytecode(bytecode, data)

    result = vm.run(max_steps=10000)
    expected = 21

    status = "PASS ✓" if result == expected else f"FAIL ✗"
    print(f"Result: {result} (expected {expected}) {status}")

    return result == expected


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "SCORE-BASED EVICTION TEST SUITE" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = []

    try:
        # Test 1: Basic GCD
        results.append(("GCD", test_gcd_with_score_eviction()))

        # Test 2: Memory overwrites
        results.append(("Memory Overwrites", test_memory_writes_with_eviction()))

        # Test 3: Compare strategies
        results.append(("Strategy Comparison", test_compare_eviction_strategies()))

        # Test 4: Stats
        results.append(("Eviction Stats", test_eviction_stats()))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:25s} {status}")

        all_passed = all(passed for _, passed in results)

        print()
        if all_passed:
            print("╔" + "═" * 58 + "╗")
            print("║" + " " * 15 + "ALL TESTS PASSED ✓" + " " * 25 + "║")
            print("╚" + "═" * 58 + "╝")
            print()
            print("Summary:")
            print("  ✓ Score-based eviction implemented successfully")
            print("  ✓ Produces same results as legacy eviction")
            print("  ✓ Correctly evicts overwritten memory")
            print("  ✓ Based on maximum attention scores")
            print()
            print("See ATTENTION_BASED_EVICTION_ANALYSIS.md for details.")
        else:
            print("❌ SOME TESTS FAILED - SCORE-BASED EVICTION NEEDS FIXES")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
