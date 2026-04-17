#!/usr/bin/env python3
"""Debug runner execution with detailed output."""
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("Compiling test program...")
code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)
print(f"Bytecode: {len(bytecode)} instructions")

print("Creating runner...")
runner = AutoregressiveVMRunner()

print(f"Runner model type: {type(runner.model)}")
print(f"Has draft_vm: {hasattr(runner, 'draft_vm')}")

print("\nAttempting to run with max_steps=5...")
try:
    result = runner.run(bytecode, max_steps=5)
    print(f"Result after 5 steps: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
