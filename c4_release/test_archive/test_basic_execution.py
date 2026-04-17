#!/usr/bin/env python3
"""Test basic program execution after JSR changes."""
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("Testing basic execution after JSR changes...")
print("=" * 60)

# Test 1: Simplest possible program (no JSR)
print("\n1. Testing: int main() { return 42; }")
try:
    bytecode, _ = compile_c('int main() { return 42; }')
    print(f"   Bytecode: {len(bytecode)} instructions")

    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, max_steps=100)
    print(f"   Result: {result}")
    print(f"   Status: {'✓ PASS' if result == ('', 42) else '✗ FAIL'}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Test 2: Simple arithmetic (no JSR)
print("\n2. Testing: int main() { return 2 + 3; }")
try:
    bytecode, _ = compile_c('int main() { return 2 + 3; }')
    print(f"   Bytecode: {len(bytecode)} instructions")

    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, max_steps=100)
    print(f"   Result: {result}")
    print(f"   Status: {'✓ PASS' if result == ('', 5) else '✗ FAIL'}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Test 3: Program with function call
print("\n3. Testing: int add(int a, int b) { return a+b; } int main() { return add(2,3); }")
try:
    code = '''
    int add(int a, int b) { return a + b; }
    int main() { return add(2, 3); }
    '''
    bytecode, _ = compile_c(code)
    print(f"   Bytecode: {len(bytecode)} instructions")

    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, max_steps=500)
    print(f"   Result: {result}")
    print(f"   Status: {'✓ PASS' if result == ('', 5) else '✗ FAIL'}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "=" * 60)
