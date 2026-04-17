"""
Test Fully Neural VM

Tests VM with full 16-layer forward pass per cycle.
Currently: Neural arithmetic (layers 9-11), Python control flow.
Goal: Gradually make all layers neural.
"""

import torch
from src.compiler import Compiler
from neural_vm.fully_neural_vm import create_fully_neural_executor


def test_fully_neural_vm():
    """Test fully neural VM with forward passes through all layers."""

    print("=" * 70)
    print("FULLY NEURAL VM TEST")
    print("=" * 70)
    print()

    # Create executor
    executor = create_fully_neural_executor(verbose=True)
    print()

    # Create compiler
    compiler = Compiler()

    # Test programs including control flow
    test_cases = [
        # Basic arithmetic (neural)
        ("int main() { return 100 + 200; }", 300, "100 + 200"),
        ("int main() { return 500 - 200; }", 300, "500 - 200"),
        ("int main() { return 2 + 3; }", 5, "2 + 3"),

        # Variables (tests memory)
        ("int main() { int x; x = 100; return x + 50; }", 150, "variable + constant"),
        ("int main() { int x; x = 200; return x - 100; }", 100, "variable - constant"),

        # Control flow (tests PC updates)
        ("int main() { if (1) return 42; return 0; }", 42, "if true"),
        ("int main() { if (0) return 0; return 42; }", 42, "if false"),

        # Simple loops (tests BNZ)
        ("int main() { int i; i = 5; while (i > 0) i = i - 1; return i; }", 0, "while loop countdown"),
    ]

    print("=" * 70)
    print("EXECUTING PROGRAMS")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for source, expected, desc in test_cases:
        print(f"Test: {desc}")
        print(f"Source: {source[:60]}...")
        print(f"Expected: {expected}")

        try:
            # Compile
            code, data = compiler.compile(source)

            # Execute
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
        print("Fully neural VM working!")
        print("✅ Neural: ADD, SUB (layers 9-11)")
        print("⏳ Python: PC updates, memory, fetch")
    else:
        print(f"Progress: {passed}/{len(test_cases)} passing")
        print()
        print("Current neural operations:")
        print("  ✅ ADD, SUB with carry/borrow propagation")
        print()
        print("Next steps:")
        print("  ⏳ Neural PC updates (layer 13)")
        print("  ⏳ Neural memory (layers 14-15)")
        print("  ⏳ Neural fetch (layers 0-8)")

    print("=" * 70)

    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_fully_neural_vm()
    exit(0 if success else 1)
