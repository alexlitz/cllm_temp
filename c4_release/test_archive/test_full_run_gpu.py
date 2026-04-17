#!/usr/bin/env python3
"""Test full program execution with GPU."""
import sys
sys.path.insert(0, '.')
import torch
import time

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("Compiling...")
bytecode, _ = compile_c('int main() { return 42; }')

print("Creating runner...")
runner = AutoregressiveVMRunner()

# Move model to GPU
if torch.cuda.is_available():
    runner.model.cuda()
    print(f"Model on: {next(runner.model.parameters()).device}")

print("\nRunning program (max_steps=100)...")
start = time.time()
result = runner.run(bytecode, max_steps=100)
elapsed = time.time() - start

print(f"\n✅ Completed in {elapsed:.2f}s")
print(f"Result: {result}")

if result == ('', 42):
    print("\n🎉🎉🎉 PASS: Got expected result 42!")
    print("✓ PC override removal SUCCESS - JSR works without runner override!")
else:
    print(f"\n❌ FAIL: Expected ('', 42), got {result}")
