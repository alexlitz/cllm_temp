"""
Test Neural Layer 12 (Comparisons and Bitwise)

Tests that comparison and bitwise operations execute through layer 12 neurally.
"""

import torch
from src.compiler import Compiler
from neural_vm.fully_neural_vm import create_fully_neural_executor


def test_neural_comparisons():
    """Test neural comparison operations."""

    print("=" * 70)
    print("NEURAL COMPARISONS TEST (Layer 12)")
    print("=" * 70)
    print()

    executor = create_fully_neural_executor(verbose=False)
    compiler = Compiler()

    # Test cases for comparisons
    test_cases = [
        # EQ (==)
        ("int main() { return 5 == 5; }", 1, "5 == 5 (true)"),
        ("int main() { return 5 == 3; }", 0, "5 == 3 (false)"),

        # NE (!=)
        ("int main() { return 5 != 3; }", 1, "5 != 3 (true)"),
        ("int main() { return 5 != 5; }", 0, "5 != 5 (false)"),

        # LT (<)
        ("int main() { return 3 < 5; }", 1, "3 < 5 (true)"),
        ("int main() { return 5 < 3; }", 0, "5 < 3 (false)"),
        ("int main() { return 5 < 5; }", 0, "5 < 5 (false)"),

        # GT (>)
        ("int main() { return 5 > 3; }", 1, "5 > 3 (true)"),
        ("int main() { return 3 > 5; }", 0, "3 > 5 (false)"),
        ("int main() { return 5 > 5; }", 0, "5 > 5 (false)"),

        # LE (<=)
        ("int main() { return 3 <= 5; }", 1, "3 <= 5 (true)"),
        ("int main() { return 5 <= 5; }", 1, "5 <= 5 (true)"),
        ("int main() { return 5 <= 3; }", 0, "5 <= 3 (false)"),

        # GE (>=)
        ("int main() { return 5 >= 3; }", 1, "5 >= 3 (true)"),
        ("int main() { return 5 >= 5; }", 1, "5 >= 5 (true)"),
        ("int main() { return 3 >= 5; }", 0, "3 >= 5 (false)"),
    ]

    passed = 0
    failed = 0

    for source, expected, desc in test_cases:
        try:
            code, data = compiler.compile(source)
            result = executor.execute(code, data, verbose=False)

            if result == expected:
                print(f"  ✅ {desc} = {result}")
                passed += 1
            else:
                print(f"  ❌ {desc}: Expected {expected}, got {result}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {desc}: ERROR {e}")
            failed += 1

    print()
    print(f"Comparisons: {passed}/{len(test_cases)} passing")
    return passed, failed


def test_neural_bitwise():
    """Test neural bitwise operations."""

    print()
    print("=" * 70)
    print("NEURAL BITWISE TEST (Layer 12)")
    print("=" * 70)
    print()

    executor = create_fully_neural_executor(verbose=False)
    compiler = Compiler()

    # Test cases for bitwise operations
    test_cases = [
        # OR
        ("int main() { return 5 | 3; }", 7, "5 | 3 (OR)"),
        ("int main() { return 12 | 10; }", 14, "12 | 10 (OR)"),

        # XOR
        ("int main() { return 5 ^ 3; }", 6, "5 ^ 3 (XOR)"),
        ("int main() { return 12 ^ 10; }", 6, "12 ^ 10 (XOR)"),

        # AND
        ("int main() { return 5 & 3; }", 1, "5 & 3 (AND)"),
        ("int main() { return 12 & 10; }", 8, "12 & 10 (AND)"),

        # SHL (<<)
        ("int main() { return 5 << 2; }", 20, "5 << 2 (SHL)"),
        ("int main() { return 1 << 3; }", 8, "1 << 3 (SHL)"),

        # SHR (>>)
        ("int main() { return 20 >> 2; }", 5, "20 >> 2 (SHR)"),
        ("int main() { return 8 >> 3; }", 1, "8 >> 3 (SHR)"),
    ]

    passed = 0
    failed = 0

    for source, expected, desc in test_cases:
        try:
            code, data = compiler.compile(source)
            result = executor.execute(code, data, verbose=False)

            if result == expected:
                print(f"  ✅ {desc} = {result}")
                passed += 1
            else:
                print(f"  ❌ {desc}: Expected {expected}, got {result}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {desc}: ERROR {e}")
            failed += 1

    print()
    print(f"Bitwise: {passed}/{len(test_cases)} passing")
    return passed, failed


def test_comparisons_in_control_flow():
    """Test comparisons used in control flow."""

    print()
    print("=" * 70)
    print("COMPARISONS IN CONTROL FLOW")
    print("=" * 70)
    print()

    executor = create_fully_neural_executor(verbose=False)
    compiler = Compiler()

    test_cases = [
        # if with comparisons
        ("int main() { if (5 > 3) return 42; return 0; }", 42, "if (5 > 3)"),
        ("int main() { if (3 > 5) return 0; return 42; }", 42, "if (3 > 5) false"),
        ("int main() { if (5 == 5) return 42; return 0; }", 42, "if (5 == 5)"),

        # while with comparisons
        ("int main() { int i; i = 0; while (i < 5) i = i + 1; return i; }", 5, "while (i < 5)"),
    ]

    passed = 0
    failed = 0

    for source, expected, desc in test_cases:
        try:
            code, data = compiler.compile(source)
            result = executor.execute(code, data, verbose=False)

            if result == expected:
                print(f"  ✅ {desc} = {result}")
                passed += 1
            else:
                print(f"  ❌ {desc}: Expected {expected}, got {result}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {desc}: ERROR {e}")
            failed += 1

    print()
    print(f"Control flow: {passed}/{len(test_cases)} passing")
    return passed, failed


def main():
    """Run all layer 12 tests."""

    print("=" * 70)
    print("NEURAL LAYER 12 TEST SUITE")
    print("Testing comparisons and bitwise operations")
    print("=" * 70)
    print()

    cmp_passed, cmp_failed = test_neural_comparisons()
    bit_passed, bit_failed = test_neural_bitwise()
    flow_passed, flow_failed = test_comparisons_in_control_flow()

    total_passed = cmp_passed + bit_passed + flow_passed
    total_failed = cmp_failed + bit_failed + flow_failed
    total_tests = total_passed + total_failed

    print()
    print("=" * 70)
    print("LAYER 12 TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {total_passed}/{total_tests} passing")
    print()

    if total_failed == 0:
        print("🎉 ALL LAYER 12 TESTS PASSED!")
        print()
        print("Neural execution status:")
        print("  ✅ Layer 9-11: ADD, SUB (arithmetic)")
        print("  ✅ Layer 12: Comparisons, bitwise")
        print("  ⏳ Layer 13: PC updates (next)")
        print()
        print(f"Progress: ~20% neural execution ({total_passed} operations)")
    else:
        print(f"Progress: {total_passed}/{total_tests} tests passing")
        print(f"Failed: {total_failed}")

    print("=" * 70)

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
