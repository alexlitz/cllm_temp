#!/usr/bin/env python3
"""Test runner with GPU."""
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
    print("Moving model to GPU...")
    runner.model.cuda()
    print(f"Model device: {next(runner.model.parameters()).device}")

print("Building context...")
context = runner._build_context(bytecode, b'', [])

print("Generating first token...")
start = time.time()
next_token = runner.model.generate_next(context)
elapsed = time.time() - start
print(f"First token={next_token}, took {elapsed:.3f}s")

print("\nRunning full program with max_steps=100...")
start = time.time()
result = runner.run(bytecode, max_steps=100)
elapsed = time.time() - start

print(f"✅ Completed in {elapsed:.2f}s")
print(f"Result: {result}")

if result == ('', 42):
    print("✅✅✅ PASS: Got expected result 42")
else:
    print(f"❌ FAIL: Expected ('', 42), got {result}")
