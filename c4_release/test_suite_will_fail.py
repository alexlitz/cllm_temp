#!/usr/bin/env python3
"""Demonstrate that the test suite will now fail with validation enabled."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("=" * 60)
print("DEMONSTRATING TEST FAILURE WITH VALIDATION")
print("=" * 60)
print()

# This is exactly what tests/run_1000_tests.py does
print("Creating BakedC4Transformer(use_speculator=True)")
print("(uses default: validate_neural=True, validation_sample_rate=0.1)")
print()

c4 = BakedC4Transformer(use_speculator=True)

print(f"Settings:")
print(f"  use_speculator: {c4.use_speculator}")
print(f"  validate_neural: {c4.validate_neural}")
print(f"  validation_sample_rate: {c4.speculator.validate_ratio}")
print()

# Run 20 simple tests to trigger validation
test_cases = [
    ("int main() { return 0; }", 0),
    ("int main() { return 1; }", 1),
    ("int main() { return 2; }", 2),
    ("int main() { return 3; }", 3),
    ("int main() { return 4; }", 4),
    ("int main() { return 5; }", 5),
    ("int main() { return 6; }", 6),
    ("int main() { return 7; }", 7),
    ("int main() { return 8; }", 8),
    ("int main() { return 9; }", 9),
    ("int main() { return 10; }", 10),
    ("int main() { return 11; }", 11),
    ("int main() { return 12; }", 12),
    ("int main() { return 13; }", 13),
    ("int main() { return 14; }", 14),
    ("int main() { return 15; }", 15),
    ("int main() { return 16; }", 16),
    ("int main() { return 17; }", 17),
    ("int main() { return 18; }", 18),
    ("int main() { return 19; }", 19),
]

print("Running 20 tests (with 10% validation, ~2 should be validated):")
print()

passed = 0
validation_errors = 0

for i, (code, expected) in enumerate(test_cases):
    try:
        result = c4.run_c(code)
        if result == expected:
            passed += 1
            print(f"  [{i+1:2d}] PASS (validation skipped)")
        else:
            print(f"  [{i+1:2d}] FAIL: got {result}, expected {expected}")
    except ValidationError as e:
        validation_errors += 1
        print(f"  [{i+1:2d}] VALIDATION FAILED!")
        print(f"       -> {str(e)[:100]}...")
        print()
        print("=" * 60)
        print("TEST SUITE WOULD FAIL HERE")
        print("=" * 60)
        break

print()
print(f"Results before failure:")
print(f"  Passed: {passed}/{i+1}")
print(f"  Validation errors: {validation_errors}")
print()

if validation_errors > 0:
    print("✓ SUCCESS: Validation caught the broken neural model!")
    print("✓ Test suite will now fail instead of passing incorrectly")
else:
    print("Note: No validation triggered in this run (random sampling)")
    print("      Run with more tests to guarantee validation")
