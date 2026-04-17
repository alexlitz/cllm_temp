#!/usr/bin/env python3
"""Test simplest possible program - just return a constant."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

# Simplest possible program
code = 'int main() { return 42; }'

print("=" * 80)
print("Simple Return Test")
print("=" * 80)
print(f"\nCode: {code}")
print("Expected: 42")
print("\nCompiling...")

bytecode, data = compile_c(code)
print(f"✓ Compiled: {len(bytecode)} bytes")

print("Creating runner...")
runner = AutoregressiveVMRunner()
print("✓ Runner created")

print("\nRunning VM (max 100 steps)...")
result = runner.run(bytecode, data, max_steps=100)

print(f"\nResult: {result}")
print(f"Expected: 42")

if result == 42:
    print("✅ SUCCESS!")
    exit(0)
else:
    print(f"❌ FAILED - got {result}")
    exit(1)
