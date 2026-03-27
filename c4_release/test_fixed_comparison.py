#!/usr/bin/env python3
"""Test that comparison is now working correctly."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Testing with fixed comparison logic")
print("=" * 60)

# Test that should match (both return 0)
code_zero = "int main() { return 0; }"

print(f"\nTest 1: {code_zero}")
print("  Expected: Fast VM = 0, Neural VM = 0, Should MATCH")

c4 = BakedC4Transformer(use_speculator=True, validate_neural=True,
                        validation_sample_rate=1.0)

try:
    result = c4.run_c(code_zero)
    print(f"  Result: {result}")
    print(f"  Status: PASSED (no ValidationError)")
    print(f"  → Match! Both returned 0")
except ValidationError as e:
    print(f"  Status: ValidationError raised")
    print(f"  → Mismatch detected")
    print(f"  Details: {str(e)[:150]}...")

# Test that shouldn't match (Fast=42, Neural=0)
code_fortytwo = "int main() { return 42; }"

print(f"\nTest 2: {code_fortytwo}")
print("  Expected: Fast VM = 42, Neural VM = 0, Should MISMATCH")

try:
    result = c4.run_c(code_fortytwo)
    print(f"  Result: {result}")
    print(f"  Status: PASSED (no ValidationError)")
    print(f"  → Unexpected match!")
except ValidationError as e:
    print(f"  Status: ValidationError raised")
    print(f"  → Mismatch detected (correct)")
    print(f"  Details: {str(e)[:150]}...")

print()
print("=" * 60)
print("Statistics:")
print(f"  Validations: {c4.speculator.validations}")
print(f"  Mismatches: {c4.speculator.mismatches}")
print(f"  Matches: {c4.speculator.validations - c4.speculator.mismatches}")

if c4.speculator.validations > 0:
    match_rate = (c4.speculator.validations - c4.speculator.mismatches) / c4.speculator.validations * 100
    print(f"  Match rate: {match_rate:.1f}%")
