#!/usr/bin/env python3
"""Run actual tests with validation to get real pass/fail counts."""

import sys
import time
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError
from tests.test_suite_1000 import generate_test_programs

print('Running Tests with Neural VM Validation')
print('=' * 60)
print()
print('NOTE: Each test takes ~80-100 seconds with validation')
print('      Running 10 tests will take ~15 minutes')
print()

c4 = BakedC4Transformer(use_speculator=True)

# Get test suite
all_tests = list(generate_test_programs())
print(f'Total test suite: {len(all_tests)} tests')

# Run only first 10 tests (would take 15 minutes)
test_subset = all_tests[:10]
print(f'Running subset: {len(test_subset)} tests')
print()

passed = 0
failed = 0
errors = 0
results = []

start_time = time.time()

for i, (source, expected, desc) in enumerate(test_subset):
    print(f'[{i+1}/{len(test_subset)}] {desc}...', end=' ', flush=True)
    test_start = time.time()

    try:
        result = c4.run_c(source)
        test_time = time.time() - test_start

        if result == expected:
            print(f'✓ PASS ({test_time:.1f}s)')
            passed += 1
            results.append((desc, 'PASS', result, expected))
        else:
            print(f'✗ FAIL ({test_time:.1f}s) - got {result}, expected {expected}')
            failed += 1
            results.append((desc, 'FAIL', result, expected))

    except ValidationError as e:
        test_time = time.time() - test_start
        print(f'✗ ValidationError ({test_time:.1f}s)')
        failed += 1
        error_msg = str(e).split('\n')[0][:50]
        results.append((desc, 'ValidationError', None, expected))

    except Exception as e:
        test_time = time.time() - test_start
        print(f'✗ ERROR ({test_time:.1f}s) - {type(e).__name__}')
        errors += 1
        results.append((desc, 'ERROR', None, expected))

    # Show progress
    elapsed = time.time() - start_time
    avg_time = elapsed / (i + 1)
    remaining = avg_time * (len(test_subset) - i - 1)
    print(f'    Progress: {i+1}/{len(test_subset)} | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s')
    print()

total_time = time.time() - start_time

print('=' * 60)
print('RESULTS')
print('=' * 60)
print(f'Total tests: {len(test_subset)}')
print(f'Passed: {passed}')
print(f'Failed: {failed}')
print(f'Errors: {errors}')
print(f'Success rate: {passed/len(test_subset)*100:.1f}%')
print(f'Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
print(f'Avg per test: {total_time/len(test_subset):.1f}s')
print()

if passed > 0:
    print('Tests that PASSED (Neural VM matches Fast VM):')
    for desc, status, result, expected in results:
        if status == 'PASS':
            print(f'  ✓ {desc}: {result}')
    print()

if failed > 0:
    print('Tests that FAILED (Neural VM mismatch):')
    for desc, status, result, expected in results:
        if status in ('FAIL', 'ValidationError'):
            print(f'  ✗ {desc}: expected {expected}')
    print()

print('=' * 60)
print('EXTRAPOLATION TO FULL SUITE')
print('=' * 60)
print(f'Full suite: {len(all_tests)} tests')
print(f'Projected time: {total_time/len(test_subset)*len(all_tests)/3600:.1f} hours')
print(f'Projected pass rate: {passed/len(test_subset)*100:.1f}%')
print(f'Projected passes: {int(passed/len(test_subset)*len(all_tests))}')
print(f'Projected failures: {int(failed/len(test_subset)*len(all_tests))}')
