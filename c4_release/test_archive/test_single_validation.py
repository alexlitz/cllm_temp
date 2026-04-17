#!/usr/bin/env python3
"""Run a single test to show validation failure."""

import sys
print("Running Single Test with 100% Validation")
print("=" * 60)

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Creating BakedC4Transformer...")
sys.stdout.flush()

c4 = BakedC4Transformer()

print(f"Validation settings:")
print(f"  validate_ratio: {c4.speculator.validate_ratio}")
print(f"  raise_on_mismatch: {c4.speculator.raise_on_mismatch}")
print()

# Test a simple program that should fail validation
test_code = "int main() { return 42; }"
print(f"Running test: {test_code}")
print(f"  Expected Fast VM result: 42")
print(f"  Expected Neural VM result: 0 (broken)")
print()
print("Validating... (this will take ~12 seconds)")
sys.stdout.flush()

try:
    result = c4.run_c(test_code)
    print(f"\n✗ UNEXPECTED: Got result {result}")
    print("   Validation should have failed!")
    sys.exit(1)
except ValidationError as e:
    print(f"\n✓ SUCCESS: ValidationError raised!")
    print()
    print("Error details:")
    print(str(e))
    print()
    print("=" * 60)
    print("TEST RESULTS:")
    print("  • Validation is enabled (100%)")
    print("  • ValidationError was raised")
    print("  • Test FAILED as expected (neural VM is broken)")
    print()
    print("This is CORRECT behavior!")
    sys.exit(0)
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
