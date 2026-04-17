#!/usr/bin/env python3
"""Test SpeculativeVM with validation on return 0."""

import sys
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Testing SpeculativeVM validation")
print("=" * 60)

# Create with validation enabled and 100% sample rate
c4 = BakedC4Transformer(
    use_speculator=True,
    validate_neural=True,
    validation_sample_rate=1.0  # Always validate
)

test_cases = [
    ("int main() { return 0; }", 0, "Should MATCH (both return 0)"),
    ("int main() { return 1; }", 1, "Should MISMATCH (Fast=1, Neural=0)"),
]

matches = 0
mismatches = 0

for source, expected, description in test_cases:
    print(f"\nTest: {source}")
    print(f"  Expected: {expected}")
    print(f"  {description}")

    try:
        result = c4.run_c(source)
        matches += 1
        print(f"  Result: {result}")
        print(f"  Status: MATCHED ✓")
    except ValidationError as e:
        mismatches += 1
        print(f"  Status: MISMATCH ✗")
        # Parse the error message to get Fast and Neural results
        lines = str(e).split('\n')
        for line in lines:
            if 'Fast VM result:' in line or 'Neural VM result:' in line:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"  Status: ERROR - {type(e).__name__}")
        print(f"  {str(e)[:100]}")
        mismatches += 1

print()
print("=" * 60)
print(f"Results:")
print(f"  Matches: {matches}")
print(f"  Mismatches: {mismatches}")

if c4.speculator:
    stats = c4.speculator.get_stats()
    print(f"\nSpeculator stats:")
    print(f"  Validations: {stats.get('validations', 0)}")
    print(f"  Mismatches: {stats.get('mismatches', 0)}")
    if stats.get('validations', 0) > 0:
        match_rate = (stats['validations'] - stats['mismatches']) / stats['validations'] * 100
        print(f"  Match rate: {match_rate:.1f}%")
