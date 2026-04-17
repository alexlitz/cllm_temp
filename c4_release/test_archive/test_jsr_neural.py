"""
Test JSR neural implementation.

Tests that function calls work entirely through neural weights,
without Python handler fallbacks.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner


def test_simple_function_call():
    """Test simple function call with JSR/LEV."""
    code = """
    int helper() {
        return 42;
    }

    int main() {
        return helper();
    }
    """

    bytecode, data = compile_c(code)
    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data)

    print(f"Test: simple function call")
    print(f"Result: {result}")
    print(f"Expected: 42")

    if result == 42:
        print("✓ PASS")
        return True
    else:
        print("✗ FAIL")
        return False


def test_function_with_argument():
    """Test function call with argument passing."""
    code = """
    int double_it(int x) {
        return x + x;
    }

    int main() {
        return double_it(21);
    }
    """

    bytecode, data = compile_c(code)
    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data)

    print(f"\nTest: function with argument")
    print(f"Result: {result}")
    print(f"Expected: 42")

    if result == 42:
        print("✓ PASS")
        return True
    else:
        print("✗ FAIL")
        return False


def test_nested_function_calls():
    """Test nested function calls."""
    code = """
    int add(int a, int b) {
        return a + b;
    }

    int calculate() {
        return add(10, 32);
    }

    int main() {
        return calculate();
    }
    """

    bytecode, data = compile_c(code)
    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data)

    print(f"\nTest: nested function calls")
    print(f"Result: {result}")
    print(f"Expected: 42")

    if result == 42:
        print("✓ PASS")
        return True
    else:
        print("✗ FAIL")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing JSR Neural Implementation")
    print("=" * 60)

    tests = [
        test_simple_function_call,
        test_function_with_argument,
        test_nested_function_calls,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ FAIL - Exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if passed == len(tests):
        print("\n✓ All JSR neural tests passed!")
        exit(0)
    else:
        print(f"\n✗ {len(tests) - passed} test(s) failed")
        exit(1)
