#!/usr/bin/env python3
"""Test actual match rate with the working neural VM."""

from src.baked_c4 import BakedC4Transformer

print("Testing Match Rate")
print("=" * 60)

# Create with 100% validation
c4 = BakedC4Transformer(
    use_speculator=True,
    validation_ratio=1.0
)

test_cases = [
    ("int main() { return 0; }", 0),
]

print("Note: Neural VM takes ~12 seconds per validation\n")

for source, expected in test_cases:
    print(f"Test: {source}")
    print(f"  Expected: {expected}")
    print(f"  Running...")

    result = c4.run_c(source)

    print(f"  Result: {result}")
    print(f"  Match: {result == expected}")

print()
print("=" * 60)
print("Speculator Statistics:")
stats = c4.speculator.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

if stats['validations'] > 0:
    print()
    match_rate = stats['match_rate'] * 100
    print(f"MATCH RATE: {match_rate:.1f}%")

    if match_rate > 0:
        print("\n✓ Neural VM matches Fast VM on return 0 programs!")
    else:
        print("\n✗ No matches found")
