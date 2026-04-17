#!/usr/bin/env python3
"""Test many short programs with validation (max_steps=10)."""

import sys
import time
from src.compiler import compile_c
from src.baked_c4 import BakedC4Transformer

print('Testing multiple short programs with validation')
print('=' * 60)
print()

c4 = BakedC4Transformer(use_speculator=True)

# Short test programs that should complete quickly
test_cases = [
    ('int main() { return 0; }', 0, 'return_0'),
    ('int main() { return 1; }', 1, 'return_1'),
    ('int main() { return 42; }', 42, 'return_42'),
    ('int main() { return 100; }', 100, 'return_100'),
    ('int main() { return -1; }', -1, 'return_neg1'),
    ('int main() { int x; x = 5; return x; }', 5, 'var_assignment'),
    ('int main() { int x; x = 0; return x; }', 0, 'var_zero'),
    ('int main() { return 1 + 1; }', 2, 'add_simple'),
    ('int main() { return 0 + 0; }', 0, 'add_zero'),
    ('int main() { return 5 - 5; }', 0, 'sub_zero'),
]

print(f'Running {len(test_cases)} tests with max_steps=10')
print()

passed = 0
failed = 0
errors = 0
results = []

for i, (code, expected, name) in enumerate(test_cases):
    print(f'[{i+1:2d}] {name}...', end=' ', flush=True)

    start = time.time()
    try:
        # Compile
        bytecode, data = compile_c(code)

        # Fast VM
        c4.speculator.fast_vm.reset()
        c4.speculator.fast_vm.load(bytecode, data)
        fast_result = c4.speculator.fast_vm.run()

        # Neural VM (limited steps)
        c4.transformer_vm.reset()
        c4.transformer_vm.load_bytecode(bytecode, data)
        neural_result = c4.transformer_vm.run(max_steps=10)

        # Extract exit code
        if isinstance(neural_result, tuple):
            neural_output, neural_exit_code = neural_result
        else:
            neural_exit_code = neural_result

        elapsed = time.time() - start

        # Check results
        if fast_result == neural_exit_code == expected:
            print(f'✓ PASS (both={fast_result}) [{elapsed:.1f}s]')
            passed += 1
            results.append((name, 'PASS', fast_result, neural_exit_code))
        elif fast_result == expected:
            print(f'✗ FAIL (fast={fast_result}, neural={neural_exit_code}) [{elapsed:.1f}s]')
            failed += 1
            results.append((name, 'FAIL', fast_result, neural_exit_code))
        else:
            print(f'✗ UNEXPECTED (fast={fast_result}!={expected}) [{elapsed:.1f}s]')
            failed += 1
            results.append((name, 'UNEXPECTED', fast_result, neural_exit_code))

    except Exception as e:
        elapsed = time.time() - start
        print(f'✗ ERROR: {type(e).__name__} [{elapsed:.1f}s]')
        errors += 1
        results.append((name, 'ERROR', None, str(e)[:50]))

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Total: {len(test_cases)}')
print(f'Passed: {passed}')
print(f'Failed: {failed}')
print(f'Errors: {errors}')
print(f'Success rate: {passed/len(test_cases)*100:.1f}%')
print()

# Show which tests passed
if passed > 0:
    print('Tests that PASSED (Neural VM matches Fast VM):')
    for name, status, fast, neural in results:
        if status == 'PASS':
            print(f'  ✓ {name}: {fast}')
    print()

# Show which tests failed
if failed > 0:
    print('Tests that FAILED (Neural VM ≠ Fast VM):')
    for name, status, fast, neural in results:
        if status in ('FAIL', 'UNEXPECTED'):
            print(f'  ✗ {name}: fast={fast}, neural={neural}')
    print()

if errors > 0:
    print('Tests with ERRORS:')
    for name, status, fast, neural in results:
        if status == 'ERROR':
            print(f'  ✗ {name}: {neural}')
    print()

# Determine if neural VM has a pattern
neural_values = [n for name, status, f, n in results if status == 'PASS' and n is not None]
if neural_values:
    if all(v == 0 for v in neural_values):
        print('PATTERN: Neural VM only matches when result is 0')
    elif len(set(neural_values)) == 1:
        print(f'PATTERN: Neural VM always returns {neural_values[0]}')
    else:
        print(f'PATTERN: Neural VM matches on values: {set(neural_values)}')
