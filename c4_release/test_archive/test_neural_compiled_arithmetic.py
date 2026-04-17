"""
Test Neural Compiled Arithmetic

Compiles C programs and executes them through the neural VM.
Tests that ADD and SUB operations work correctly in real compiled programs.
"""

import torch
from src.compiler import Compiler
from neural_vm.nibble_bytecode_executor import create_executor


def test_neural_compiled_arithmetic():
    """Test compiled arithmetic programs with neural execution."""

    print("=" * 70)
    print("NEURAL COMPILED ARITHMETIC TEST")
    print("=" * 70)
    print()

    # Create executor (loads VM and weights)
    executor = create_executor(verbose=True)
    print()

    # Create compiler
    compiler = Compiler()

    # Test programs - only ADD and SUB work neurally for now
    test_cases = [
        ("int main() { return 100 + 200; }", 300, "100 + 200"),
        ("int main() { return 500 - 200; }", 300, "500 - 200"),
        ("int main() { return 2 + 3; }", 5, "2 + 3"),
        ("int main() { return 10 - 3; }", 7, "10 - 3"),
        ("int main() { return 42; }", 42, "constant return"),
        ("int main() { int x; x = 100; return x + 50; }", 150, "variable + constant"),
        ("int main() { int x; x = 200; return x - 100; }", 100, "variable - constant"),
    ]

    print("=" * 70)
    print("COMPILING AND EXECUTING PROGRAMS")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for source, expected, desc in test_cases:
        print(f"Test: {desc}")
        print(f"Source: {source}")
        print(f"Expected: {expected}")

        try:
            # Compile
            code, data = compiler.compile(source)
            print(f"  ✅ Compiled ({len(code)} instructions, {len(data)} bytes data)")

            # Execute
            result = executor.execute(code, data, verbose=False)

            # Check result
            if result == expected:
                print(f"  ✅ PASS: Got {result}")
                passed += 1
            else:
                print(f"  ❌ FAIL: Expected {expected}, got {result}")
                failed += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
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
        print("Neural arithmetic (ADD, SUB) working correctly in compiled programs!")
    elif passed > 0:
        print(f"✅ {passed} tests passed")
        print(f"❌ {failed} tests failed")
        print()
        print("Note: MUL, DIV, MOD use Python fallback (not neural yet)")
    else:
        print("❌ ALL TESTS FAILED")

    print("=" * 70)

    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_neural_compiled_arithmetic()
    exit(0 if success else 1)
