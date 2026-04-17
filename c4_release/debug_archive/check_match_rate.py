#!/usr/bin/env python3
"""Check pass rate and neural model match rate with current configuration."""

import sys
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError
from tests.test_suite_1000 import get_quick_tests

print("=" * 70)
print("TEST SUITE PASS AND MATCH RATE ANALYSIS")
print("=" * 70)
print()

# Create transformer with validation enabled but don't raise errors
# We'll track statistics instead
print("Configuration:")
print("  use_speculator: True (Fast VM executes programs)")
print("  validate_neural: True (Validate against neural model)")
print("  validation_sample_rate: 0.1 (10% of tests validated)")
print()

c4 = BakedC4Transformer(use_speculator=True, validate_neural=True,
                        validation_sample_rate=0.1)

# Get test suite
tests = get_quick_tests()
print(f"Running {len(tests)} tests from quick test suite")
print()

# Run tests and collect statistics
passed = 0
failed = 0
validation_errors = 0
total_validations = 0

for i, (source, expected, desc) in enumerate(tests):
    # Track validations before run
    validations_before = c4.speculator.validations
    mismatches_before = c4.speculator.mismatches

    try:
        result = c4.run_c(source)

        # Check if this test was validated
        was_validated = c4.speculator.validations > validations_before

        if result == expected:
            passed += 1
            if was_validated:
                total_validations += 1
        else:
            failed += 1

    except ValidationError as e:
        validation_errors += 1
        total_validations += 1
        # Don't break - continue to collect all statistics

    # Progress indicator
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(tests)} tests")

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

# Pass rate (Fast VM execution)
pass_rate = (passed / len(tests)) * 100
print("FAST VM EXECUTION (Speculator):")
print(f"  Total tests: {len(tests)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Pass rate: {pass_rate:.1f}%")
print()

# Match rate (Neural model validation)
total_validated = c4.speculator.validations
total_mismatches = c4.speculator.mismatches
match_count = total_validated - total_mismatches
match_rate = (match_count / total_validated * 100) if total_validated > 0 else 0

print("NEURAL MODEL VALIDATION:")
print(f"  Tests validated: {total_validated}/{len(tests)} ({total_validated/len(tests)*100:.1f}%)")
print(f"  Matches: {match_count}")
print(f"  Mismatches: {total_mismatches}")
print(f"  Match rate: {match_rate:.1f}%")
print()

if validation_errors > 0:
    print(f"  ValidationErrors raised: {validation_errors}")
    print(f"  (Tests would have failed here)")
    print()

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print()

if pass_rate >= 99:
    print("✓ Fast VM works correctly (high pass rate)")
else:
    print("✗ Fast VM has issues (low pass rate)")

if match_rate >= 99:
    print("✓ Neural model matches Fast VM (high match rate)")
elif match_rate >= 50:
    print("⚠ Neural model partially works (moderate match rate)")
else:
    print("✗ Neural model is broken (low match rate)")

print()
print(f"Current status: Fast VM at {pass_rate:.1f}%, Neural match at {match_rate:.1f}%")
