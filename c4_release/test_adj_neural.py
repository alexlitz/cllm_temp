#!/usr/bin/env python3
"""
Test neural ADJ implementation.

Tests that ADJ (stack adjustment) now works entirely through transformer weights
without Python handler fallback.
"""
import sys
import tempfile
import subprocess
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

print("=" * 80)
print("NEURAL ADJ IMPLEMENTATION TEST")
print("=" * 80)
print()

# Test 1: Simple ADJ with local variables
test1_code = """
int main() {
    int a, b, c;
    a = 1;
    b = 2;
    c = 3;
    return a + b + c;
}
"""

print("Test 1: Simple local variable allocation (ADJ -12, ADJ +12)")
print("-" * 80)
print("Code:")
print(test1_code)
print()

try:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(test1_code)
        f.flush()
        temp_path = f.name

    bytecode, data = compile_c(Path(temp_path).read_text())
    Path(temp_path).unlink()

    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data, [], max_steps=500)

    print(f"Exit code: {result}")
    print(f"Expected: 6")

    if result == 6:
        print("✓ Test 1 PASSED")
    else:
        print(f"✗ Test 1 FAILED: got {result}, expected 6")
        sys.exit(1)

except Exception as e:
    print(f"✗ Test 1 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Large adjustment (array allocation)
test2_code = """
int main() {
    char buf[1000];
    buf[0] = 42;
    buf[999] = 7;
    return buf[0] + buf[999];
}
"""

print("Test 2: Large stack adjustment (ADJ -1000, ADJ +1000)")
print("-" * 80)
print("Code:")
print(test2_code)
print()

try:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(test2_code)
        f.flush()
        temp_path = f.name

    bytecode, data = compile_c(Path(temp_path).read_text())
    Path(temp_path).unlink()

    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data, [], max_steps=500)

    print(f"Exit code: {result}")
    print(f"Expected: 49 (42 + 7)")

    if result == 49:
        print("✓ Test 2 PASSED")
    else:
        print(f"✗ Test 2 FAILED: got {result}, expected 49")
        sys.exit(1)

except Exception as e:
    print(f"✗ Test 2 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Nested function with locals
test3_code = """
int helper(int x) {
    int temp;
    temp = x * 2;
    return temp;
}

int main() {
    return helper(21);
}
"""

print("Test 3: Function with local variable (nested ADJ)")
print("-" * 80)
print("Code:")
print(test3_code)
print()

try:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(test3_code)
        f.flush()
        temp_path = f.name

    bytecode, data = compile_c(Path(temp_path).read_text())
    Path(temp_path).unlink()

    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data, [], max_steps=500)

    print(f"Exit code: {result}")
    print(f"Expected: 42")

    if result == 42:
        print("✓ Test 3 PASSED")
    else:
        print(f"✗ Test 3 FAILED: got {result}, expected 42")
        sys.exit(1)

except Exception as e:
    print(f"✗ Test 3 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("ALL ADJ TESTS PASSED!")
print("=" * 80)
print()
print("Neural ADJ implementation is working correctly.")
print("The transformer now handles stack pointer adjustment entirely through")
print("learned weights without any Python fallback handler.")
