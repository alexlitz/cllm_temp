#!/usr/bin/env python3
"""
Comprehensive test of score-based eviction vs legacy eviction.

Tests both strategies on multiple programs to verify:
1. Same results
2. Correct eviction behavior
3. Performance characteristics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from src.transformer_vm import C4TransformerVM, C4Config
from src.compiler import compile_c


def test_program(name: str, source: str, expected: int, max_steps: int = 10000):
    """Test a program with both eviction strategies."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    # Compile once
    bytecode, data = compile_c(source)
    print(f"Bytecode: {len(bytecode)} instructions")

    # Test with legacy eviction
    print("\n--- Legacy Eviction ---")
    config_legacy = C4Config(use_score_based_eviction=False)
    vm_legacy = C4TransformerVM(config_legacy)
    vm_legacy.reset()
    vm_legacy.load_bytecode(bytecode, data)

    start = time.time()
    result_legacy = vm_legacy.run(max_steps=max_steps)
    time_legacy = time.time() - start

    print(f"Result: {result_legacy}")
    print(f"Time: {time_legacy:.3f}s")

    # Test with score-based eviction
    print("\n--- Score-Based Eviction ---")
    config_score = C4Config(use_score_based_eviction=True)
    vm_score = C4TransformerVM(config_score)
    vm_score.reset()
    vm_score.load_bytecode(bytecode, data)

    start = time.time()
    result_score = vm_score.run(max_steps=max_steps)
    time_score = time.time() - start

    print(f"Result: {result_score}")
    print(f"Time: {time_score:.3f}s")

    # Compare results
    print(f"\n--- Comparison ---")
    legacy_ok = (result_legacy == expected)
    score_ok = (result_score == expected)
    match = (result_legacy == result_score)

    print(f"Expected: {expected}")
    print(f"Legacy: {result_legacy} {'✓' if legacy_ok else '✗'}")
    print(f"Score:  {result_score} {'✓' if score_ok else '✗'}")
    print(f"Match:  {'✓' if match else '✗ MISMATCH!'}")
    print(f"Overhead: {((time_score / time_legacy - 1) * 100):.1f}%")

    if not match:
        print("\n⚠️  EVICTION STRATEGIES PRODUCED DIFFERENT RESULTS!")
        return False

    if not (legacy_ok and score_ok):
        print(f"\n⚠️  INCORRECT RESULT (expected {expected})!")
        return False

    print(f"\n✅ PASSED - Both strategies work correctly")
    return True


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "SCORE-BASED EVICTION - COMPREHENSIVE TESTS" + " " * 9 + "║")
    print("╚" + "═" * 58 + "╝")

    results = []

    # Test 1: Simple return
    print("\n\n")
    results.append(test_program(
        "Simple Return",
        """
        int main() {
            return 42;
        }
        """,
        expected=42,
        max_steps=1000
    ))

    # Test 2: Arithmetic
    print("\n\n")
    results.append(test_program(
        "Arithmetic",
        """
        int main() {
            int x;
            x = 10 + 5;
            return x;
        }
        """,
        expected=15,
        max_steps=1000
    ))

    # Test 3: Variable overwrites
    print("\n\n")
    results.append(test_program(
        "Variable Overwrites",
        """
        int main() {
            int x;
            x = 10;
            x = 20;
            x = 30;
            return x;
        }
        """,
        expected=30,
        max_steps=1000
    ))

    # Test 4: Multiple variables
    print("\n\n")
    results.append(test_program(
        "Multiple Variables",
        """
        int main() {
            int a, b, c;
            a = 5;
            b = 10;
            c = a + b;
            return c;
        }
        """,
        expected=15,
        max_steps=1000
    ))

    # Test 5: Factorial (recursive)
    print("\n\n")
    results.append(test_program(
        "Factorial(5)",
        """
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        int main() {
            return factorial(5);
        }
        """,
        expected=120,
        max_steps=10000
    ))

    # Test 6: GCD (iterative with memory)
    print("\n\n")
    results.append(test_program(
        "GCD(48, 18)",
        """
        int gcd(int a, int b) {
            int t;
            while (b != 0) {
                t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
        int main() {
            return gcd(48, 18);
        }
        """,
        expected=6,
        max_steps=10000
    ))

    # Test 7: Fibonacci (recursive)
    print("\n\n")
    results.append(test_program(
        "Fibonacci(8)",
        """
        int fib(int n) {
            if (n < 2) return n;
            return fib(n-1) + fib(n-2);
        }
        int main() {
            return fib(8);
        }
        """,
        expected=21,
        max_steps=50000
    ))

    # Test 8: Array sum
    print("\n\n")
    results.append(test_program(
        "Array Sum",
        """
        int main() {
            int arr[5];
            int i, sum;

            arr[0] = 1;
            arr[1] = 2;
            arr[2] = 3;
            arr[3] = 4;
            arr[4] = 5;

            sum = 0;
            i = 0;
            while (i < 5) {
                sum = sum + arr[i];
                i = i + 1;
            }

            return sum;
        }
        """,
        expected=15,
        max_steps=10000
    ))

    # Test 9: Nested loops
    print("\n\n")
    results.append(test_program(
        "Nested Loops",
        """
        int main() {
            int i, j, sum;
            sum = 0;

            i = 0;
            while (i < 5) {
                j = 0;
                while (j < 3) {
                    sum = sum + 1;
                    j = j + 1;
                }
                i = i + 1;
            }

            return sum;
        }
        """,
        expected=15,
        max_steps=10000
    ))

    # Test 10: Prime check
    print("\n\n")
    results.append(test_program(
        "Is Prime(17)",
        """
        int is_prime(int n) {
            int i;
            if (n < 2) return 0;
            i = 2;
            while (i * i <= n) {
                if (n % i == 0) return 0;
                i = i + 1;
            }
            return 1;
        }
        int main() {
            return is_prime(17);
        }
        """,
        expected=1,
        max_steps=10000
    ))

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for i, (passed) in enumerate(results, 1):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Test {i:2d}: {status}")

    all_passed = all(results)
    total = len(results)
    passed_count = sum(results)

    print()
    print(f"Results: {passed_count}/{total} tests passed")

    if all_passed:
        print("\n╔" + "═" * 58 + "╗")
        print("║" + " " * 10 + "ALL COMPREHENSIVE TESTS PASSED ✓" + " " * 16 + "║")
        print("╚" + "═" * 58 + "╝")
        print()
        print("Summary:")
        print("  ✓ Score-based eviction implemented successfully")
        print("  ✓ Produces identical results to legacy eviction")
        print("  ✓ Works correctly on 10 diverse programs")
        print("  ✓ Handles recursion, loops, arrays, memory overwrites")
        print()
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
