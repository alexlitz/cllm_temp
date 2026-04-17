#!/usr/bin/env python3
"""Test with all fixes applied."""
import sys
sys.path.insert(0, '.')
import torch
import time

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("=" * 70)
print("Testing AutoregressiveVMRunner with fixes")
print("=" * 70)

# Test 1: Simple return
print("\nTest 1: int main() { return 42; }")
bytecode, _ = compile_c('int main() { return 42; }')
print(f"  Bytecode: {len(bytecode)} instructions")

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()
    print(f"  Using GPU: {next(runner.model.parameters()).device}")
else:
    print("  Using CPU")

start = time.time()
result = runner.run(bytecode, max_steps=100)
elapsed = time.time() - start

print(f"  Result: {result}")
print(f"  Time: {elapsed:.2f}s")
print(f"  Status: {'✅ PASS' if result == ('', 42) else '❌ FAIL'}")

# Test 2: Simple arithmetic
print("\nTest 2: int main() { return 5 + 7; }")
bytecode, _ = compile_c('int main() { return 5 + 7; }')

runner2 = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner2.model.cuda()

start = time.time()
result = runner2.run(bytecode, max_steps=100)
elapsed = time.time() - start

print(f"  Result: {result}")
print(f"  Time: {elapsed:.2f}s")
print(f"  Status: {'✅ PASS' if result == ('', 12) else '❌ FAIL'}")

print("\n" + "=" * 70)
