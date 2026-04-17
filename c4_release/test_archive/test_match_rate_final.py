#!/usr/bin/env python3
"""Test actual match rate with working neural VM."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Testing Match Rate with Fixed Neural VM")
print("=" * 60)

# Create with 100% validation to test all programs
c4 = BakedC4Transformer(
    use_speculator=True,
    validate_neural=True,
    validation_sample_rate=1.0  # Validate everything
)

test_cases = [
    ("int main() { return 0; }", 0),
    ("int main() { return 1; }", 1),
]

matches = 0
mismatches = 0

for source, expected in test_cases:
    print(f"\nTest: {source}")
    print(f"  Expected: {expected}")

    try:
        result = c4.run_c(source)
        matches += 1
        print(f"  Result: {result}")
        print(f"  Status: MATCHED ✓")
    except ValidationError as e:
        mismatches += 1
        print(f"  Status: MISMATCH ✗")
        # Extract Fast and Neural results from error message
        lines = str(e).split('\n')
        for line in lines:
            if 'Fast VM result:' in line or 'Neural VM result:' in line:
                print(f"  {line.strip()}")

print()
print("=" * 60)
print(f"Results:")
print(f"  Matches: {matches}/{len(test_cases)}")
print(f"  Mismatches: {mismatches}/{len(test_cases)}")

if c4.speculator:
    print(f"\nSpeculator stats:")
    print(f"  Validations: {c4.speculator.validations}")
    print(f"  Mismatches: {c4.speculator.mismatches}")
    if c4.speculator.validations > 0:
        match_rate = (c4.speculator.validations - c4.speculator.mismatches) / c4.speculator.validations * 100
        print(f"  Match rate: {match_rate:.1f}%")
