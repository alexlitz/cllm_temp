#!/usr/bin/env python3
"""Test complex programs combining multiple operations."""
import sys
sys.path.insert(0, '.')
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

tests = [
    # Basic operations
    ('int main() { return 42; }', 42, "Simple return"),

    # Arithmetic combinations
    ('int main() { return (5 + 3) * 2; }', 16, "Addition then multiplication"),
    ('int main() { return 100 / 5 - 10; }', 10, "Division then subtraction"),
    ('int main() { return 7 * 6 + 2; }', 44, "Multiplication then addition"),

    # Bitwise combinations
    ('int main() { return (5 | 3) & 7; }', 7, "OR then AND"),
    ('int main() { return (12 ^ 5) | 1; }', 9, "XOR then OR"),

    # Shift combinations
    ('int main() { return (8 << 2) >> 1; }', 16, "Left shift then right shift"),

    # Mixed operations
    ('int main() { return ((3 + 2) * 4) >> 1; }', 10, "Arithmetic + shift"),
    ('int main() { return (7 & 3) + (5 | 2); }', 10, "Bitwise + arithmetic"),

    # Modulo and division
    ('int main() { return 17 % 5; }', 2, "Simple modulo"),
    ('int main() { return (20 / 3) * 3 + (20 % 3); }', 20, "Division and modulo"),

    # Local variables
    ('int main() { int x; x = 10; return x; }', 10, "Local variable"),
    ('int main() { int x, y; x = 5; y = 7; return x + y; }', 12, "Two local variables"),
    ('int main() { int x; x = 3 * 4; return x + 1; }', 13, "Local with expression"),

    # Function calls
    ('int add(int a, int b) { return a + b; } int main() { return add(10, 5); }', 15, "Simple function call"),
    ('int mul(int a, int b) { return a * b; } int main() { return mul(6, 7); }', 42, "Function with multiplication"),

    # Nested expressions
    ('int main() { return ((((5)))); }', 5, "Nested parentheses"),
    ('int main() { return 1 + 2 + 3 + 4 + 5; }', 15, "Chain addition"),
]

print("Testing complex C programs:")
print("=" * 70)

runner = AutoregressiveVMRunner()
runner.model.cuda()

passed = 0
failed = 0

for code, expected, desc in tests:
    try:
        bytecode, _ = compile_c(code)
        output, exit_code = runner.run(bytecode, max_steps=500)

        if exit_code == expected:
            print(f"✓ {desc:40s} => {exit_code:4d} (expected {expected:4d})")
            passed += 1
        else:
            print(f"✗ {desc:40s} => {exit_code:4d} (expected {expected:4d})")
            failed += 1
    except Exception as e:
        print(f"✗ {desc:40s} => ERROR: {e}")
        failed += 1

print("=" * 70)
print(f"Passed: {passed}/{len(tests)}")
print(f"Failed: {failed}/{len(tests)}")
