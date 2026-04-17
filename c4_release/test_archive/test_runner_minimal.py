#!/usr/bin/env python3
"""Minimal runner test with explicit prints."""
import sys
sys.path.insert(0, '.')

print("A: Importing...")
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("B: Compiling...")
bytecode, _ = compile_c('int main() { return 42; }')
print(f"B: Bytecode: {len(bytecode)} instructions")

print("C: Creating runner...")
runner = AutoregressiveVMRunner()
print("C: Runner created")

print("D: Building context...")
context = runner._build_context(bytecode, b'', [])
print(f"D: Context length: {len(context)}")

print("E: Calling generate_next (first token)...")
import time
start = time.time()
next_token = runner.model.generate_next(context)
elapsed = time.time() - start
print(f"E: First token={next_token}, took {elapsed:.2f}s")

print("F: ✅ Test complete")
