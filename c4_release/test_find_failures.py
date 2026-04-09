#!/usr/bin/env python3
"""Find failing tests in the comprehensive test suite."""

from tests.test_suite_1000 import generate_test_programs
from src.baked_c4 import BakedC4Transformer

tests = generate_test_programs()
c4 = BakedC4Transformer(use_speculator=True)

# Run all tests and find failures
failed_tests = []
for i, (source, expected, desc) in enumerate(tests):
    try:
        result = c4.run_c(source)
        if result != expected:
            failed_tests.append((i, desc, expected, result))
    except Exception as e:
        failed_tests.append((i, desc, expected, f'ERROR: {str(e)}'))

    if (i + 1) % 100 == 0:
        print(f'Tested {i+1}/{len(tests)}...', flush=True)

print(f'\n\nTotal tests: {len(tests)}')
print(f'Failed tests: {len(failed_tests)}\n')

if failed_tests:
    print('Failed tests:')
    for idx, desc, expected, result in failed_tests:
        print(f'  Test {idx}: {desc}')
        print(f'    Expected: {expected}, Got: {result}')
else:
    print('ALL TESTS PASSED!')
