#!/usr/bin/env python3
"""Test validation with limited max_steps for faster results."""

import sys
import time
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError
from src.compiler import compile_c

print('Running validation tests with limited steps...')
print('=' * 60)
print()

c4 = BakedC4Transformer(use_speculator=True)

# Test cases
test_cases = [
    ('int main() { return 0; }', 0, 'return 0'),
    ('int main() { return 42; }', 42, 'return 42'),
    ('int main() { return 1 + 1; }', 2, 'simple add'),
]

passed = 0
failed = 0

for code, expected, desc in test_cases:
    print(f'Test: {desc}')
    print(f'  Code: {code}')
    print(f'  Expected: {expected}')
    print(f'  Validating (max_steps=1000)...', flush=True)

    start = time.time()
    try:
        # Compile
        bytecode, data = compile_c(code)

        # Run fast VM
        c4.speculator.fast_vm.reset()
        c4.speculator.fast_vm.load(bytecode, data)
        fast_result = c4.speculator.fast_vm.run()

        # Run neural VM with limited steps
        c4.transformer_vm.reset()
        c4.transformer_vm.load_bytecode(bytecode, data)
        neural_result = c4.transformer_vm.run(max_steps=1000)  # Limit steps

        elapsed = time.time() - start

        # Extract exit code from tuple if needed
        if isinstance(neural_result, tuple):
            neural_output, neural_exit_code = neural_result
        else:
            neural_exit_code = neural_result

        print(f'  Fast VM: {fast_result}')
        print(f'  Neural VM: {neural_exit_code}')
        print(f'  Time: {elapsed:.1f}s')

        if fast_result == neural_exit_code == expected:
            print(f'  ✓ PASS (both match expected)')
            passed += 1
        elif fast_result == expected:
            print(f'  ✗ FAIL (Fast VM correct, Neural VM wrong)')
            failed += 1
        else:
            print(f'  ✗ FAIL (unexpected result)')
            failed += 1

    except Exception as e:
        elapsed = time.time() - start
        print(f'  ✗ ERROR: {type(e).__name__} ({elapsed:.1f}s)')
        print(f'     {str(e)[:150]}')
        failed += 1

    print()

print('=' * 60)
print(f'Results: {passed} passed, {failed} failed out of {len(test_cases)} tests')
print()

if passed == 0:
    print('All tests failed - Neural VM is not working correctly')
    sys.exit(1)
elif failed == 0:
    print('All tests passed - Neural VM matches Fast VM!')
    sys.exit(0)
else:
    print(f'Partial success - {passed}/{len(test_cases)} tests pass')
    sys.exit(1)
