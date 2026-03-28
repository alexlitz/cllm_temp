#!/usr/bin/env python3
"""Simple test of AutoregressiveVMRunner."""
import sys
sys.path.insert(0, '.')

import time
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("Compiling test program...")
code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)
print(f"Bytecode: {len(bytecode)} instructions")

print("Testing speculative mode (AutoregressiveVMRunner)...")
runner = AutoregressiveVMRunner()

start = time.time()
result = runner.run(bytecode, max_steps=100)
elapsed = time.time() - start

print(f"✅ Speculative mode completed in {elapsed:.2f}s")
print(f"Result: {result}")

if result == ('', 42):
    print("✅ PASS: Got expected result 42")
else:
    print(f"❌ FAIL: Expected ('', 42), got {result}")
