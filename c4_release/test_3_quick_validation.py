#!/usr/bin/env python3
"""Run 3 quick tests with validation to show actual results."""

import sys
import time
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print('Running 3 Tests with Neural VM Validation')
print('=' * 60)
print('Expected time: ~4-5 minutes total')
print()

c4 = BakedC4Transformer(use_speculator=True)

# 3 short test cases
tests = [
    ('int main() { return 0; }', 0, 'return_0'),
    ('int main() { return 42; }', 42, 'return_42'),
    ('int main() { int x; x = 5; return x; }', 5, 'var_assign'),
]

passed = 0
failed = 0
results = []

start_time = time.time()

for i, (code, expected, name) in enumerate(tests):
    print(f'[{i+1}/3] Test: {name}')
    print(f'  Code: {code}')
    print(f'  Expected: {expected}')
    print(f'  Running Fast VM...', end=' ', flush=True)

    test_start = time.time()

    try:
        result = c4.run_c(code)
        test_time = time.time() - test_start

        print(f'Done ({test_time:.1f}s)')
        print(f'  Result: {result}')

        if result == expected:
            print(f'  ✓ PASS - Neural VM matches Fast VM')
            passed += 1
            results.append((name, 'PASS', result, expected))
        else:
            print(f'  ✗ FAIL - Neural VM returned {result}, expected {expected}')
            failed += 1
            results.append((name, 'FAIL', result, expected))

    except ValidationError as e:
        test_time = time.time() - test_start
        print(f'ValidationError ({test_time:.1f}s)')
        print(f'  ✗ FAIL - Validation mismatch')
        error_lines = str(e).split('\n')
        for line in error_lines[:5]:  # First 5 lines
            print(f'    {line}')
        failed += 1
        results.append((name, 'ValidationError', None, expected))

    except Exception as e:
        test_time = time.time() - test_start
        print(f'ERROR ({test_time:.1f}s)')
        print(f'  ✗ {type(e).__name__}: {e}')
        failed += 1
        results.append((name, 'ERROR', None, expected))

    elapsed = time.time() - start_time
    print(f'  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print()

total_time = time.time() - start_time

print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Tests run: 3')
print(f'Passed: {passed} ({passed/3*100:.0f}%)')
print(f'Failed: {failed} ({failed/3*100:.0f}%)')
print(f'Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
print(f'Avg per test: {total_time/3:.1f}s')
print()

print('DETAILED RESULTS:')
print('-' * 60)
for name, status, result, expected in results:
    if status == 'PASS':
        print(f'✓ {name}: {result} (correct)')
    elif status == 'FAIL':
        print(f'✗ {name}: got {result}, expected {expected}')
    else:
        print(f'✗ {name}: {status}')
print()

print('=' * 60)
print('INTERPRETATION')
print('=' * 60)
if passed == 3:
    print('✓ ALL TESTS PASSED')
    print('  Neural VM matches Fast VM perfectly!')
    print('  Model is working correctly.')
elif passed > 0:
    print(f'⚠ PARTIAL SUCCESS ({passed}/3 passed)')
    print(f'  Neural VM works for some cases')
    print(f'  {failed} tests failed - model needs debugging')
else:
    print('✗ ALL TESTS FAILED')
    print('  Neural VM does not match Fast VM')
    print('  Model may be broken or returning wrong values')
print()

print('Extrapolation to full 1096 test suite:')
print(f'  Expected pass rate: {passed/3*100:.0f}%')
print(f'  Expected passes: ~{int(passed/3*1096)}')
print(f'  Expected failures: ~{int(failed/3*1096)}')
print(f'  Estimated time: ~{total_time/3*1096/3600:.1f} hours')
