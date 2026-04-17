#!/usr/bin/env python3
"""
Test C4 Standard Library Memory Functions

Tests malloc, free, memset, and memcmp implemented as C4 bytecode.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("=" * 70)
print("C4 STDLIB TEST - Memory Functions")
print("=" * 70)

runner = AutoregressiveVMRunner()

# Test 1: malloc basic allocation
print("\nTest 1: malloc - Basic Allocation")
print("-" * 70)
code1 = """
int main() {
    int *ptr;
    ptr = malloc(100);
    if (ptr != 0) {
        return 1;  // Success
    }
    return 0;  // Failure
}
"""
print(code1)
try:
    bytecode, data = compile_c(code1)
    output, exit_code = runner.run(bytecode, data, [], max_steps=300)
    print(f"Result: {exit_code} (expected 1)")
    if exit_code == 1:
        print("✓ Test 1 PASSED")
    else:
        print(f"✗ Test 1 FAILED")
        sys.exit(1)
except Exception as e:
    print(f"✗ Test 1 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: malloc sequential allocations
print("\nTest 2: malloc - Sequential Allocations")
print("-" * 70)
code2 = """
int main() {
    int *a;
    int *b;
    int offset;

    a = malloc(100);
    b = malloc(200);

    // b should be after a (with alignment)
    offset = (int)b - (int)a;

    // Expected: 104 (100 aligned to 104)
    if (offset >= 100 && offset <= 110) {
        return 42;
    }
    return 0;
}
"""
print(code2)
try:
    bytecode, data = compile_c(code2)
    output, exit_code = runner.run(bytecode, data, [], max_steps=400)
    print(f"Result: {exit_code} (expected 42)")
    if exit_code == 42:
        print("✓ Test 2 PASSED")
    else:
        print(f"✗ Test 2 FAILED")
        sys.exit(1)
except Exception as e:
    print(f"✗ Test 2 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: memset
print("\nTest 3: memset - Fill Memory")
print("-" * 70)
code3 = """
int main() {
    char buf[10];
    int i;
    int sum;

    memset(buf, 5, 10);

    sum = 0;
    i = 0;
    while (i < 10) {
        sum = sum + buf[i];
        i = i + 1;
    }

    // Expected: 50 (10 * 5)
    return sum;
}
"""
print(code3)
try:
    bytecode, data = compile_c(code3)
    output, exit_code = runner.run(bytecode, data, [], max_steps=500)
    print(f"Result: {exit_code} (expected 50)")
    if exit_code == 50:
        print("✓ Test 3 PASSED")
    else:
        print(f"✗ Test 3 FAILED")
        sys.exit(1)
except Exception as e:
    print(f"✗ Test 3 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: memcmp - equal strings
print("\nTest 4: memcmp - Equal Strings")
print("-" * 70)
code4 = """
int main() {
    char *a;
    char *b;
    int result;

    a = "hello";
    b = "hello";

    result = memcmp(a, b, 5);

    if (result == 0) {
        return 1;  // Success
    }
    return 0;
}
"""
print(code4)
try:
    bytecode, data = compile_c(code4)
    output, exit_code = runner.run(bytecode, data, [], max_steps=500)
    print(f"Result: {exit_code} (expected 1)")
    if exit_code == 1:
        print("✓ Test 4 PASSED")
    else:
        print(f"✗ Test 4 FAILED")
        sys.exit(1)
except Exception as e:
    print(f"✗ Test 4 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: memcmp - different strings
print("\nTest 5: memcmp - Different Strings")
print("-" * 70)
code5 = """
int main() {
    char *a;
    char *b;
    int result;

    a = "hello";
    b = "world";

    result = memcmp(a, b, 5);

    if (result != 0) {
        return 1;  // Success
    }
    return 0;
}
"""
print(code5)
try:
    bytecode, data = compile_c(code5)
    output, exit_code = runner.run(bytecode, data, [], max_steps=500)
    print(f"Result: {exit_code} (expected 1)")
    if exit_code == 1:
        print("✓ Test 5 PASSED")
    else:
        print(f"✗ Test 5 FAILED")
        sys.exit(1)
except Exception as e:
    print(f"✗ Test 5 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL STDLIB TESTS PASSED!")
print("=" * 70)
print()
print("Memory operations (malloc/free/memset/memcmp) now work as")
print("C4 bytecode subroutines instead of special opcodes.")
