#!/usr/bin/env python3
"""Quick test for ADD and SUB handlers."""
import sys
sys.path.insert(0, '.')
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

def test_operation(runner, code, expected, desc):
    """Test a single operation."""
    bytecode, _ = compile_c(code)
    output, exit_code = runner.run(bytecode, max_steps=300)
    status = "✓" if exit_code == expected else "✗"
    result = f"{status} {desc:40s} => {exit_code:4d} (expected {expected:4d})"
    print(result, flush=True)
    return exit_code == expected

# Initialize
print("Initializing runner...", flush=True)
runner = AutoregressiveVMRunner()
runner.model.cuda()
print("Runner ready!\n", flush=True)

# Run tests
tests = [
    ('int main() { return 5 + 3; }', 8, "ADD: 5 + 3"),
    ('int main() { return 10 - 3; }', 7, "SUB: 10 - 3"),
    ('int main() { return 7 + 8; }', 15, "ADD: 7 + 8"),
    ('int main() { return 100 - 50; }', 50, "SUB: 100 - 50"),
    ('int main() { return (5 + 3) * 2; }', 16, "Complex: (5+3)*2"),
    ('int main() { return (10 - 3) + (4 * 2); }', 15, "Complex: (10-3)+(4*2)"),
]

print("Test Results:")
print("-" * 65, flush=True)
passed = sum(1 for code, exp, desc in tests if test_operation(runner, code, exp, desc))
print("-" * 65, flush=True)
print(f"Results: {passed}/{len(tests)} passed", flush=True)
