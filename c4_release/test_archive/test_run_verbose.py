#!/usr/bin/env python3
"""Test run with verbose output."""

import sys
from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

print("1. Compiling...", flush=True)
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
print(f"   Compiled: {len(bytecode)} bytes", flush=True)

print("2. Creating runner...", flush=True)
runner = AutoregressiveVMRunner()
print("   Runner created", flush=True)

print("3. Calling runner.run()...", flush=True)
sys.stdout.flush()

# Call run with a very low max_steps to see if it even starts
result = runner.run(bytecode, data, max_steps=5)

print(f"4. Run completed! Result: {result}", flush=True)
