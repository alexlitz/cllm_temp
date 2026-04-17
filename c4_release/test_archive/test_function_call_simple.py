#!/usr/bin/env python3
"""Test if function calls work with properly configured model."""

from neural_vm.vm_step import set_vm_weights, AutoregressiveVM
from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c
import sys

# Simple function call test
code = '''
int helper(int x) {
    return x * 2;
}

int main() {
    return helper(21);
}
'''

print("=" * 80)
print("Function Call Test with Proper Weight Configuration")
print("=" * 80)
print("\nCode:")
print(code)
print("\nExpected: 42")
print("=" * 80)

# Compile
bytecode, data = compile_c(code)
print(f"\nCompiled: {len(bytecode)} bytes of bytecode")

# Run (runner creates and configures model internally)
runner = AutoregressiveVMRunner()
print("\nRunning VM...")
result = runner.run(bytecode, data, max_steps=1000)

print(f"\nResult: {result}")
print(f"Expected: 42")

if result == 42:
    print("\n✅ SUCCESS! Function call works correctly!")
    sys.exit(0)
else:
    print(f"\n❌ FAILED! Got {result} instead of 42")
    sys.exit(1)
