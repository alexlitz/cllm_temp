#!/usr/bin/env python3
"""Test runner with GPU (properly fixed)."""
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
runner.model.to(device)

print("Building context...")
context = runner._build_context(bytecode, b'', [])

print("Generating first token...")
start = time.time()
# generate_next will convert context to tensor internally, but we need to ensure it's on the right device
# Actually, let me check the generate_next method to see how it handles this
next_token = runner.model.generate_next(context)
elapsed = time.time() - start
print(f"First token={next_token}, took {elapsed:.3f}s")

print("✅ First token generated successfully!")
