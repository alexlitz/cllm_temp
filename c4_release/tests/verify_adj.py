#!/usr/bin/env python3
"""
Quick verification that ADJ neural implementation works.
Tests that local variable allocation/deallocation works neurally.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("ADJ Neural Implementation Verification")
print("=" * 60)

# Test 1: Single local variable
code1 = "int main() { int a; a = 42; return a; }"
print(f"\nTest 1: {code1}")
try:
    bytecode, data = compile_c(code1)
    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data, [], max_steps=200)
    if result == 42:
        print(f"✓ PASS: returned {result}")
    else:
        print(f"✗ FAIL: returned {result}, expected 42")
        sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Multiple local variables
code2 = "int main() { int x, y, z; x = 1; y = 2; z = 3; return x + y + z; }"
print(f"\nTest 2: {code2}")
try:
    bytecode, data = compile_c(code2)
    result = runner.run(bytecode, data, [], max_steps=200)
    if result == 6:
        print(f"✓ PASS: returned {result}")
    else:
        print(f"✗ FAIL: returned {result}, expected 6")
        sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All ADJ tests passed!")
print("ADJ now works entirely through neural weights.")
