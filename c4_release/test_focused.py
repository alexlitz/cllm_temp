#!/usr/bin/env python3
"""Focused tests for complex operations."""
import sys
sys.path.insert(0, '.')
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

# Initialize once
print("Initializing runner...")
runner = AutoregressiveVMRunner()
runner.model.cuda()
print("Runner ready!\n")

tests = [
    ('int main() { return 42; }', 42, "Basic return"),
    ('int main() { return 5 | 3; }', 7, "Bitwise OR"),
    ('int main() { return 5 & 3; }', 1, "Bitwise AND"),
    ('int main() { return 6 * 7; }', 42, "Multiplication"),
    ('int main() { return 5 + 3; }', 8, "Addition"),
    ('int main() { return (5 + 3) * 2; }', 16, "Add then multiply"),
    ('int main() { int x; x = 10; return x; }', 10, "Local variable"),
]

print("Running tests:")
for i, (code, expected, desc) in enumerate(tests, 1):
    bytecode, _ = compile_c(code)
    output, exit_code = runner.run(bytecode, max_steps=200)
    status = "✓" if exit_code == expected else "✗"
    print(f"{i}. {status} {desc:30s} => {exit_code:4d} (expected {expected:4d})")

print("\nDone!")
