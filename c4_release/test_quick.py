#!/usr/bin/env python3
"""Quick test of basic and complex operations."""
import sys
sys.path.insert(0, '.')

print("Importing modules...")
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("Initializing runner...")
runner = AutoregressiveVMRunner()
runner.model.cuda()

tests = [
    # Basic
    ('int main() { return 42; }', 42),
    ('int main() { return 5 + 3; }', 8),
    ('int main() { return 5 | 3; }', 7),
    ('int main() { return 6 * 7; }', 42),
    # Complex
    ('int main() { return (5 + 3) * 2; }', 16),
    ('int main() { return (5 | 3) & 7; }', 7),
    ('int main() { int x; x = 10; return x; }', 10),
]

print("\nRunning tests:\n")
for code, expected in tests:
    try:
        bytecode, _ = compile_c(code)
        output, exit_code = runner.run(bytecode, max_steps=200)
        status = "✓" if exit_code == expected else "✗"
        print(f"{status} {code[:50]:50s} => {exit_code:4d} (expected {expected:4d})")
    except Exception as e:
        print(f"✗ {code[:50]:50s} => ERROR: {str(e)[:40]}")

print("\nDone!")
