#!/usr/bin/env python3
"""Simple test: IMM 42, EXIT - should return exit code 42."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
import inspect

# Program: IMM 42, EXIT
bytecode = [1 | (42 << 8), 34 | (0 << 8)]

print("Testing: IMM 42, EXIT")
print(f"  bytecode[0] = IMM, imm=42")
print(f"  bytecode[1] = EXIT")
print(f"Expected exit code: 42")

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

print("\nRunning...")
output, exit_code = runner.run(bytecode, b'', [], max_steps=5)

print(f"\nActual exit code: {exit_code}")

if exit_code == 42:
    print("✅ SUCCESS! Exit code is correct!")
    print("The PC advancement bug is FIXED!")
else:
    print(f"❌ FAILURE! Expected 42, got {exit_code}")
    print(f"  Hex: 0x{exit_code:08x}")
