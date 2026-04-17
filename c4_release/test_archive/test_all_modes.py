#!/usr/bin/env python3
"""Test all three positional encoding modes."""

from neural_vm.config import VMConfig, set_config, reset_config
from tests.test_suite_1000 import generate_test_programs
from src.baked_c4 import BakedC4Transformer

# Generate a subset of tests for quicker validation
tests = generate_test_programs()[:50]  # First 50 tests

modes = ['alibi', 'rope', 'hybrid']
results = {}

for mode in modes:
    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} mode ({len(tests)} tests)")
    print(f"{'='*60}")

    # Reset and set mode
    reset_config()
    if mode == 'alibi':
        set_config(VMConfig(positional_encoding='alibi'))
    elif mode == 'rope':
        set_config(VMConfig.rope_mode())
    elif mode == 'hybrid':
        set_config(VMConfig.hybrid_mode())

    c4 = BakedC4Transformer(use_speculator=True)

    passed = 0
    failed = 0
    failed_tests = []

    for i, (source, expected, desc) in enumerate(tests):
        try:
            result = c4.run_c(source)
            if result == expected:
                passed += 1
            else:
                failed += 1
                failed_tests.append((desc, expected, result))
        except Exception as e:
            failed += 1
            failed_tests.append((desc, expected, f'ERROR: {e}'))

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(tests)}")

    results[mode] = {
        'passed': passed,
        'failed': failed,
        'failed_tests': failed_tests
    }

    print(f"\n  Results: {passed}/{len(tests)} passed")
    if failed > 0:
        print(f"  Failed tests:")
        for desc, expected, result in failed_tests[:5]:
            print(f"    - {desc}: expected {expected}, got {result}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY - All Modes")
print(f"{'='*60}\n")
print(f"{'Mode':<10} {'Passed':<10} {'Failed':<10} {'Success Rate':<15}")
print("-" * 60)
for mode in modes:
    r = results[mode]
    rate = (r['passed'] / len(tests)) * 100
    status = '✅' if r['failed'] == 0 else '❌'
    print(f"{mode:<10} {r['passed']:<10} {r['failed']:<10} {rate:>6.1f}%  {status}")

all_passed = all(results[m]['failed'] == 0 for m in modes)
print(f"\n{'='*60}")
if all_passed:
    print("✅ ALL MODES PASSING - Implementation is WORKING")
else:
    print("❌ SOME MODES FAILING - Implementation has ISSUES")
print(f"{'='*60}")
