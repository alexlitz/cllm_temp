#!/usr/bin/env python3
"""Test that validation is enabled by default and fails appropriately."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Testing Default Validation")
print("=" * 60)

# Create with default settings (should have validation enabled)
c4 = BakedC4Transformer()

print(f"Validation ratio: {c4.speculator.validate_ratio}")
print(f"Raise on mismatch: {c4.speculator.raise_on_mismatch}")
print()

# Test programs
test_cases = [
    ("int main() { return 0; }", 0, "Should MATCH"),
    ("int main() { return 42; }", 42, "Should FAIL validation"),
    ("int main() { return 1; }", 1, "Should FAIL validation"),
]

passed = 0
failed = 0

for source, expected, description in test_cases:
    print(f"Test: {source}")
    print(f"  Expected: {expected}")
    print(f"  {description}")

    try:
        result = c4.run_c(source)
        passed += 1
        print(f"  Result: {result} ✓ PASSED")
    except ValidationError as e:
        failed += 1
        print(f"  ✗ VALIDATION FAILED (neural VM broken)")
        # Extract key info from error
        lines = str(e).split('\n')
        for line in lines[:4]:
            print(f"    {line.strip()}")
    except Exception as e:
        failed += 1
        print(f"  ✗ ERROR: {type(e).__name__}: {e}")
    print()

print("=" * 60)
print(f"Results:")
print(f"  Passed (no validation error): {passed}")
print(f"  Failed (validation error): {failed}")
print()

if c4.speculator:
    stats = c4.speculator.get_stats()
    print(f"Speculator statistics:")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Validations: {stats['validations']}")
    print(f"  Mismatches: {stats['mismatches']}")
    if stats['validations'] > 0:
        print(f"  Match rate: {stats['match_rate']*100:.1f}%")

print()
if failed > 0:
    print("✓ SUCCESS: Validation is working - it catches neural VM failures!")
else:
    print("⚠ WARNING: No validation failures detected")
