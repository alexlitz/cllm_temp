#!/usr/bin/env python3
"""Test if compilation hangs."""
import sys
sys.path.insert(0, '.')
import time

print("1. Importing compiler...")
start = time.time()
from src.compiler import compile_c
print(f"   Imported in {time.time() - start:.2f}s")

print("2. Compiling program...")
start = time.time()
code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)
print(f"   Compiled in {time.time() - start:.2f}s")
print(f"   Bytecode: {len(bytecode)} instructions")

print("✅ Compilation complete")
