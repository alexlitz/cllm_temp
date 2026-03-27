#!/usr/bin/env python3
"""Test with 100% validation on 2 tests."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Testing with 100% Validation (2 tests)")
print("=" * 60)
print()

# Force 100% validation
c4 = BakedC4Transformer(validation_ratio=1.0)

print(f"Validation ratio: {c4.speculator.validate_ratio}")
print(f"Raise on mismatch: {c4.speculator.raise_on_mismatch}")
print()
print("Note: Neural VM is slow (~12 seconds per test)")
print()

# Test 1: Should match (both return 0)
print("Test 1: int main() { return 0; }")
print("  Expected: 0")
print("  Neural VM should return: 0")
print("  Validating...")

try:
    result = c4.run_c("int main() { return 0; }")
    print(f"  Result: {result}")
    print(f"  ✓ VALIDATION PASSED (match!)")
except ValidationError as e:
    print(f"  ✗ VALIDATION FAILED")
    print(f"  {str(e)[:200]}")

print()

# Test 2: Should fail (Fast VM=42, Neural VM=0)
print("Test 2: int main() { return 42; }")
print("  Expected: 42")
print("  Neural VM will return: 0")
print("  Validating...")

try:
    result = c4.run_c("int main() { return 42; }")
    print(f"  Result: {result}")
    print(f"  ⚠ No validation error (unexpected)")
except ValidationError as e:
    print(f"  ✓ VALIDATION FAILED (as expected - neural VM broken)")
    lines = str(e).split('\n')
    for line in lines[:5]:
        print(f"    {line.strip()}")

print()
print("=" * 60)
stats = c4.speculator.get_stats()
print("Final Statistics:")
print(f"  Validations: {stats['validations']}")
print(f"  Mismatches: {stats['mismatches']}")
print(f"  Match rate: {stats['match_rate']*100:.1f}%")
