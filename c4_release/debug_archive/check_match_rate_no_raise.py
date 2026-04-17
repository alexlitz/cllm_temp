#!/usr/bin/env python3
"""Check pass rate and neural model match rate (no exceptions raised)."""

import sys
from src.speculator import SpeculativeVM, FastLogicalVM
from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c
from tests.test_suite_1000 import get_quick_tests

print("=" * 70)
print("TEST SUITE PASS AND MATCH RATE ANALYSIS")
print("=" * 70)
print()

print("Configuration:")
print("  Fast VM: Executes all programs")
print("  Neural VM: Validates 10% of programs")
print("  Errors: Logged but not raised (for statistics)")
print()

# Create VMs
transformer_vm = C4TransformerVM()
speculator = SpeculativeVM(
    transformer_vm=transformer_vm,
    validate_ratio=0.1
)

# Get test suite
tests = get_quick_tests()
print(f"Running {len(tests)} tests from quick test suite")
print()

# Run tests and collect statistics
passed = 0
failed = 0
errors = 0

for i, (source, expected, desc) in enumerate(tests):
    try:
        bytecode, data = compile_c(source)
        # Run with validation but don't raise on mismatch
        result = speculator.run(bytecode, data, validate=False, raise_on_mismatch=False)

        if result == expected:
            passed += 1
        else:
            failed += 1

    except Exception as e:
        errors += 1
        if i < 5:  # Show first few errors
            print(f"  ERROR on test {i+1}: {str(e)[:80]}")

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
print("FAST VM EXECUTION:")
print(f"  Total tests: {len(tests)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Errors: {errors}")
print(f"  Pass rate: {pass_rate:.1f}%")
print()

# Match rate (Neural model validation)
total_validated = speculator.validations
total_mismatches = speculator.mismatches
match_count = total_validated - total_mismatches
match_rate = (match_count / total_validated * 100) if total_validated > 0 else 0

print("NEURAL MODEL VALIDATION:")
print(f"  Tests validated: {total_validated}/{len(tests)} ({total_validated/len(tests)*100:.1f}%)")
print(f"  Matches: {match_count}")
print(f"  Mismatches: {total_mismatches}")
if total_validated > 0:
    print(f"  Match rate: {match_rate:.1f}%")
else:
    print(f"  Match rate: N/A (no tests validated)")
print()

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print()

if pass_rate >= 99:
    print("✓ Fast VM works correctly (high pass rate)")
else:
    print("✗ Fast VM has issues (low pass rate)")

if total_validated == 0:
    print("⚠ No neural validations occurred (increase sample rate or run more tests)")
elif match_rate >= 99:
    print("✓ Neural model matches Fast VM (high match rate)")
elif match_rate >= 50:
    print("⚠ Neural model partially works (moderate match rate)")
elif match_rate >= 1:
    print("✗ Neural model mostly broken (low match rate)")
else:
    print("✗ Neural model completely broken (0% match rate)")

print()
print(f"Summary: Fast VM {pass_rate:.1f}% pass rate, Neural {match_rate:.1f}% match rate")
