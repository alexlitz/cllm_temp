#!/usr/bin/env python3
"""Test multi-instruction programs to verify PC advancement works correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import inspect

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

# Test 1: IMM 42, EXIT → should return 42
print("Test 1: IMM 42, EXIT")
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT | (0 << 8)]
output, exit_code = runner.run(bytecode, b'', [], max_steps=5)
assert exit_code == 42, f"Expected 42, got {exit_code}"
print(f"  ✓ Exit code: {exit_code}")

# Test 2: IMM 10, IMM 20, ADD, EXIT → should return 30
print("\nTest 2: IMM 10, IMM 20, ADD, EXIT")
bytecode = [
    Opcode.IMM | (10 << 8),
    Opcode.IMM | (20 << 8),
    Opcode.ADD | (0 << 8),
    Opcode.EXIT | (0 << 8)
]
output, exit_code = runner.run(bytecode, b'', [], max_steps=10)
assert exit_code == 30, f"Expected 30, got {exit_code}"
print(f"  ✓ Exit code: {exit_code}")

# Test 3: IMM 100, IMM 1, SUB, EXIT → should return 99
print("\nTest 3: IMM 100, IMM 1, SUB, EXIT")
bytecode = [
    Opcode.IMM | (100 << 8),
    Opcode.IMM | (1 << 8),
    Opcode.SUB | (0 << 8),
    Opcode.EXIT | (0 << 8)
]
output, exit_code = runner.run(bytecode, b'', [], max_steps=10)
assert exit_code == 99, f"Expected 99, got {exit_code}"
print(f"  ✓ Exit code: {exit_code}")

print("\n✅ All multi-instruction tests PASSED!")
print("PC advancement is working correctly!")
