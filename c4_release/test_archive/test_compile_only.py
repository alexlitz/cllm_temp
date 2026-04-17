#!/usr/bin/env python3
"""Test compilation only."""

import sys
from src.compiler import compile_c

print("1. Starting compilation...", flush=True)

code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

print(f"2. Compiled: {len(bytecode)} bytes, {len(data)} data bytes", flush=True)
print(f"3. First 20 bytecode bytes: {bytecode[:20]}", flush=True)
print("✅ Compilation works")
