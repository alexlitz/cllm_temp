"""
Test Autoregressive Nibble VM

Tests the autoregressive execution framework.
Currently hybrid: neural arithmetic (ADD, SUB), Python control flow.
"""

import torch
from src.compiler import Compiler
from neural_vm.autoregressive_nibble_vm import create_autoregressive_executor


def test_autoregressive_vm():
    """Test autoregressive VM execution."""

    print("=" * 70)
    print("AUTOREGRESSIVE NIBBLE VM TEST")
    print("=" * 70)
    print()

    # Create executor
    executor = create_autoregressive_executor(verbose=True)
    print()

    # Create compiler
    compiler = Compiler()

    # Test programs
    test_cases = [
        ("int main() { return 100 + 200; }", 300, "100 + 200"),
        ("int main() { return 500 - 200; }", 300, "500 - 200"),
        ("int main() { return 2 + 3; }", 5, "2 + 3"),
        ("int main() { return 10 - 3; }", 7, "10 - 3"),
        ("int main() { return 42; }", 42, "constant"),
        ("int main() { int x; x = 100; return x + 50; }", 150, "variable + constant"),
        ("int main() { int x; x = 200; return x - 100; }", 100, "variable - constant"),
    ]

    print("=" * 70)
    print("EXECUTING PROGRAMS")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for source, expected, desc in test_cases:
        print(f"Test: {desc}")
        print(f"Expected: {expected}")

        try:
            # Compile
            code, data = compiler.compile(source)

            # Execute autoregressively
            result = executor.execute(code, data, verbose=False)

            # Check
            if result == expected:
                print(f"  ✅ PASS: {result}")
                passed += 1
            else:
                print(f"  ❌ FAIL: Got {result}, expected {expected}")
                failed += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    print()

    if passed == len(test_cases):
        print("🎉 ALL TESTS PASSED!")
        print()
        print("Autoregressive framework working correctly!")
        print("Neural: ADD, SUB (layers 9-11)")
        print("Next: Neural PC updates (layer 13)")
    else:
        print(f"✅ {passed} tests passed, ❌ {failed} failed")

    print("=" * 70)

    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_autoregressive_vm()
    exit(0 if success else 1)
